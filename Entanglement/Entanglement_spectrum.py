import netket as nk
import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
import sys
import os
import jax
import jax.numpy as jnp
from jax.nn.initializers import normal
import flax
import re

# Add path to project root
sys.path.append("/scratch/f/F.Conoscenti/Thesis_QSL")
try:
    from ViT_Heisenberg.ViT_model_ent import ViT_ent
    from HFDS_Heisenberg.entanglement_model.HFDS_model_spin_ent import HiddenFermion_ent
    from ViT_Heisenberg.ViT_model import ViT_sym
    from HFDS_Heisenberg.HFDS_model_spin import HiddenFermion
except ImportError:
    pass

def get_unique_path(directory, filename):
    base, ext = os.path.splitext(filename)
    counter = 1
    new_filename = filename
    while os.path.exists(os.path.join(directory, new_filename)):
        new_filename = f"{base}_{counter}{ext}"
        counter += 1
    return os.path.join(directory, new_filename)

def compute_entanglement_spectrum_2d(L, indices_A, psi=None):
    """
    Computes the entanglement spectrum for any arbitrary subsystem (e.g., in a 2D lattice).
    
    Args:
        L (int): Total number of spins.
        indices_A (list or np.ndarray): List of integer indices belonging to Subsystem A.
        psi (np.ndarray, optional): The full 2^L state vector.
        
    Returns:
        entanglement_spectrum (np.ndarray): The values xi_i = -ln(lambda_i), sorted.
        eigenvalues (np.ndarray): The valid positive eigenvalues lambda_i of rho_A.
    """
    
    # ---------------------------------------------------------
    # Step 1: Obtain and normalize the full state vector
    # ---------------------------------------------------------
    if psi is not None:
        psi = np.array(psi)
        norm = np.linalg.norm(psi)
        if norm > 1e-12:
            psi = psi / norm
        else:
            print(f"Warning: psi norm is {norm}, returning empty spectrum.")
            return np.array([]), np.array([])
    else:
        raise ValueError("Must provide 'psi'.")

    # ---------------------------------------------------------
    # Step 2: Separate indices for A and B
    # ---------------------------------------------------------
    N_A = len(indices_A)
    N_B = L - N_A
    
    # Find all spins that belong to Subsystem B
    indices_B = [i for i in range(L) if i not in indices_A]
    
    # ---------------------------------------------------------
    # Step 3: Tensor Permutation (The crucial step for 2D)
    # ---------------------------------------------------------
    # Reshape the vector into an L-dimensional tensor, shape (2, 2, ..., 2)
    psi_tensor = np.reshape(psi, [2] * L)
    
    # Reorder the axes so that Subsystem A's spins come first, followed by B's
    permuted_axes = list(indices_A) + list(indices_B)
    psi_tensor_permuted = np.transpose(psi_tensor, permuted_axes)
    
    # Reshape into a bipartite matrix of size (2^N_A, 2^N_B)
    psi_bipartite = np.reshape(psi_tensor_permuted, (2**N_A, 2**N_B))

    # ---------------------------------------------------------
    # Step 4: SVD instead of Density Matrix
    # ---------------------------------------------------------
    # Calculate singular values of the bipartite matrix. 
    # This avoids squaring the condition number and is much more numerically stable.
    singular_values = scipy.linalg.svdvals(psi_bipartite)
    
    # The eigenvalues of rho_A are exactly the squares of the singular values of psi
    eigenvalues = singular_values ** 2

    # ---------------------------------------------------------
    # Step 5: Calculate the Entanglement Spectrum
    # ---------------------------------------------------------
    tolerance = 1e-14
    valid_eigenvalues = eigenvalues[eigenvalues > tolerance]
    
    entanglement_spectrum = -np.log(valid_eigenvalues)
    entanglement_spectrum = np.sort(entanglement_spectrum)

    return entanglement_spectrum, valid_eigenvalues

def load_trained_model(path, L, J2, hi_constrained, hi_full):
    if not os.path.exists(path):
        path = path.replace("/cluster/home/fconoscenti", "/scratch/f/F.Conoscenti")
    
    if not os.path.exists(path):
        print(f"Error: Path {path} does not exist.")
        return None, None, None

    j_path = os.path.join(path, f"J={J2}")
    if not os.path.exists(j_path):
        j_path = os.path.join(path, f"J2={J2}")
    
    if not os.path.exists(j_path):
        print(f"Error: J={J2} folder not found in {path}")
        return None, None, None

    seeds = [d for d in os.listdir(j_path) if d.startswith("seed_")]
    if not seeds:
        print("Error: No seed folder found.")
        return None, None, None
    seed_path = os.path.join(j_path, seeds[0])
    
    models_dir = os.path.join(seed_path, "models")
    if not os.path.exists(models_dir):
        print("Error: models folder not found.")
        return None, None, None
        
    mpack_files = [f for f in os.listdir(models_dir) if f.endswith(".mpack")]
    if not mpack_files:
        print("Error: No .mpack files found.")
        return None, None, None
        
    mpack_files.sort(key=lambda x: int(re.search(r"model_(\d+)", x).group(1)))
    last_model = mpack_files[-1]
    model_file_path = os.path.join(models_dir, last_model)
    print(f"Loading {model_file_path}...")
    
    with open(model_file_path, 'rb') as f:
        data = f.read()

    # --- Model Type Detection and Instantiation ---
    model, vstate, model_type = None, None, None
    is_vit = "ViT" in path
    is_hfds = "HFDS" in path or "hidd" in path

    if is_vit:
        model_type = "ViT"
        print("Detected ViT model.")
        try:
            num_layers = int(re.search(r"layers(\d+)", path).group(1))
            d_model = int(re.search(r"_d(\d+)", path).group(1))
            n_heads = int(re.search(r"heads(\d+)", path).group(1))
            patch_size = int(re.search(r"patch(\d+)", path).group(1))
            parity = "parityTrue" in path
            rotation = "rotTrue" in path
        except AttributeError as e:
            print(f"Error parsing ViT model parameters from path: {e}")
            return None, None, None

        model = ViT_sym(
            L=L, num_layers=num_layers, d_model=d_model, n_heads=n_heads,
            patch_size=patch_size, transl_invariant=True, parity=parity, rotation=rotation
        )
        lattice = nk.graph.Hypercube(length=L, n_dim=2, pbc=True)
        sampler = nk.sampler.MetropolisExchange(hilbert=hi_constrained, graph=lattice)
        vstate = nk.vqs.MCState(sampler, model, n_samples=16)

    elif is_hfds:
        model_type = "HFDS"
        print("Detected HFDS model.")
        try:
            n_hid = int(re.search(r"hidd(\d+)", path).group(1))
            layers = int(re.search(r"layers(\d+)", path).group(1))
            features = int(re.search(r"feat(\d+)", path).group(1))
            mf_init = re.search(r"Init([a-zA-Z]+)", path).group(1)
            dtype_str = re.search(r"type([a-zA-Z]+)", path).group(1)
            parity = "parityTrue" in path
            rotation = "rotTrue" in path
        except AttributeError as e:
            print(f"Error parsing HFDS model parameters from path: {e}")
            return None, None, None
            
        dtype = jnp.complex128 if dtype_str == "complex" else jnp.float64
        
        model = HiddenFermion(L=L, network="FFNN", n_hid=n_hid, layers=layers, features=features, MFinit=mf_init, hilbert=hi_constrained, parity=parity, rotation=rotation, dtype=dtype)
        
        lattice = nk.graph.Hypercube(length=L, n_dim=2, pbc=True)
        sampler = nk.sampler.MetropolisExchange(hilbert=hi_constrained, graph=lattice)
        vstate = nk.vqs.MCState(sampler, model, n_samples=16)

    else:
        print("Error: Could not determine model type from path.")
        return None, None, None
    
    try:
        vstate = flax.serialization.from_bytes(vstate, data)
        variables = vstate.variables
    except:
        variables = flax.serialization.from_bytes(model.init(jax.random.PRNGKey(0), jnp.zeros((1, L*L))), data)
            
    return model, variables, model_type

def plot_spectrum(ket_gs, vstate, L, J2, save_dir=None):
    """
    Plots the entanglement spectrum of a given vstate alongside the exact ground state,
    and computes the total Euclidean error and the relative error in three sectors.
    
    Args:
        ket_gs: Exact ground state wavefunction.
        vstate: NetKet variational state.
        L (int): Linear size of the lattice.
        J2 (float): Next-nearest neighbor coupling.
        save_dir (str, optional): Directory to save the plot.
        
    Returns:
        total_error (float): Total Euclidean distance between the spectra.
        sector_errors (list): Mean relative errors in the High, Mid, and Low sectors.
    """
    print(f"--- Plotting Entanglement Spectrum (L={L}) ---")
    N = L * L
    lattice = nk.graph.Hypercube(length=L, n_dim=2, pbc=True, max_neighbor_order=2)
    hi_full = nk.hilbert.Spin(s=1/2, N=N)
    # Use vstate's own Hilbert space (whatever total_sz sector it was trained in)
    # rather than assuming total_sz=0, so ket_gs/vstate embed into the correct sector.
    hi_constrained = vstate.hilbert

    # Subsystem A indices
    indices_A = []
    for y in range(L):
        for x in range(L // 2): 
            flat_index = y * L + x
            indices_A.append(flat_index)

    # 1. Exact Ground State
    ket_gs = ket_gs.flatten()
    
    psi_exact = np.zeros(hi_full.n_states, dtype=ket_gs.dtype)
    full_indices_constrained = hi_full.states_to_numbers(hi_constrained.all_states())
    psi_exact[full_indices_constrained] = ket_gs
    psi_exact /= np.linalg.norm(psi_exact)

    # 2. Vstate Wavefunction
    print("Computing Vstate Wavefunction...")
    if vstate.hilbert.size != N:
        raise ValueError("Hilbert space size mismatch")
    
    psi_c = vstate.to_array()
    if vstate.hilbert.n_states == hi_constrained.n_states:
        psi_vstate = np.zeros(hi_full.n_states, dtype=psi_c.dtype)
        psi_vstate[full_indices_constrained] = psi_c
    elif vstate.hilbert.n_states == hi_full.n_states:
        psi_vstate = psi_c
    else:
        raise ValueError("Unsupported Hilbert space")
    
    psi_vstate /= np.linalg.norm(psi_vstate)

    # 3. Compute Spectra
    print("Computing Entanglement Spectra...")
    _, evals_exact = compute_entanglement_spectrum_2d(N, indices_A, psi_exact)
    _, evals_vstate = compute_entanglement_spectrum_2d(N, indices_A, psi_vstate)

    # 4. Calculate Errors
    min_len = min(len(evals_exact), len(evals_vstate))
    if min_len == 0:
        print("Warning: Spectrum is empty.")
        return np.nan, [np.nan, np.nan, np.nan]
        
    diff = np.abs(evals_exact[:min_len] - evals_vstate[:min_len])
    total_error = np.linalg.norm(diff)
    
    # Sector errors (relative difference)
    denominator = evals_exact[:min_len]
    valid_indices = denominator > 1e-12
    relative_diff = np.full_like(diff, np.nan)
    relative_diff[valid_indices] = diff[valid_indices] / denominator[valid_indices]
    
    s1 = min_len // 3
    s2 = 2 * (min_len // 3)
    sectors = [(0, s1), (s1, s2), (s2, min_len)]
    
    sector_errors = []
    for start, end in sectors:
        seg_mean = np.nanmean(relative_diff[start:end]) if start < end else np.nan
        sector_errors.append(seg_mean)

    # 5. Plotting
    plt.figure(figsize=(10, 7))
    plt.semilogy(evals_exact, 'o-', label='Exact GS', markersize=4, alpha=0.8, color='red', zorder=10)
    plt.semilogy(evals_vstate, 's--', label='Variational State', markersize=4, alpha=0.8, color='blue')
    
    # Annotate errors
    stats_text = (
        f"Total Euclidean Error: {total_error:.3e}\n"
        f"High Sector Rel. Error: {sector_errors[0]:.3e}\n"
        f"Mid Sector Rel. Error: {sector_errors[1]:.3e}\n"
        f"Low Sector Rel. Error: {sector_errors[2]:.3e}"
    )
    props = dict(boxstyle='round', facecolor='white', alpha=0.8)
    plt.gca().text(0.05, 0.05, stats_text, transform=plt.gca().transAxes, fontsize=10, verticalalignment='bottom', bbox=props)

    plt.xlabel('Index')
    plt.ylabel(r'Eigenvalues $\lambda_i$')
    plt.legend()
    plt.grid(True, alpha=0.3)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = get_unique_path(save_dir, f"Entanglement_Spectrum_Vstate_L{L}.png")
        plt.savefig(save_path, dpi=300)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()
        
    plt.close()

    return evals_vstate, total_error, sector_errors

def _model_color(model_type):
    return 'orange' if model_type == 'ViT' else 'blue' if model_type == 'HFDS' else 'black'


def plot_random_init(evals_exact, evals_vit, evals_hfds, save_dir, L):
    plt.figure(figsize=(10, 7))
    plt.semilogy(evals_exact, 'o-', label='Exact GS', markersize=14, alpha=0.6, color='red', zorder=1)
    plt.semilogy(evals_vit, 's--', label='ViT (Random)', markersize=10, alpha=0.6, color='orange', zorder=2)
    plt.semilogy(evals_hfds, '^--', label='HFDS (Random)', markersize=10, alpha=0.6, color='blue', zorder=3)
    plt.xlabel('Index')
    plt.ylabel(r'Eigenvalues $\lambda_i$') 
    plt.legend()
    plt.grid(True, alpha=0.3)
    save_path = get_unique_path(save_dir, f"Entanglement_Spectrum_Random_L{L}.png")
    plt.savefig(save_path, dpi=300)
    print(f"Random init plot saved to {save_path}")
    plt.close()


def plot_trained_models(evals_exact, trained_results, save_dir, L):
    plt.figure(figsize=(10, 7))
    plt.semilogy(evals_exact, 'o-', label='Exact GS', markersize=14, alpha=0.6, color='red', zorder=1)
    for result in trained_results:
        color = _model_color(result['type'])
        zorder = 3 if result['type'] == 'HFDS' else 2
        label = f'{result["type"]}'
        plt.semilogy(result['evals'], '*-', label=label, markersize=10, alpha=0.8, color=color, zorder=zorder)

    fidelity_between_models = None
    if len(trained_results) == 2:
        fidelity_between_models = np.abs(np.vdot(trained_results[0]['psi'], trained_results[1]['psi']))**2

    """stats_text = [f"F({res['type']}, Exact) = {res['fidelity']:.5f}" for res in trained_results]
    if fidelity_between_models is not None:
        stats_text.append(f"F({trained_results[0]['type']}, {trained_results[1]['type']}) = {fidelity_between_models:.5f}")
    if stats_text:
        props = dict(boxstyle='round', facecolor='white', alpha=0.8)
        plt.gca().text(0.05, 0.05, "\n".join(stats_text), transform=plt.gca().transAxes, fontsize=10, verticalalignment='bottom', bbox=props)
    """
    plt.xlabel('Index')
    plt.ylabel(r'Eigenvalues $\lambda_i$')
    plt.legend()
    plt.grid(True, alpha=0.3)
    save_path = get_unique_path(save_dir, f"Entanglement_Spectrum_Trained_L{L}.png")
    plt.savefig(save_path, dpi=300)
    print(f"Trained models plot saved to {save_path}")
    plt.close()


def plot_spectrum_diff(evals_exact, trained_results, save_dir, L):
    plt.figure(figsize=(10, 7))
    for result in trained_results:
        evals_model = result['evals']
        min_len = min(len(evals_exact), len(evals_model))
        if min_len > 0:
            diff = np.abs(evals_exact[:min_len] - evals_model[:min_len])
            dist = np.linalg.norm(diff)
            color = _model_color(result['type'])
            label = f"{result['type']} (Eucl. Dist: {dist:.3e})"
            plt.semilogy(diff, 'o--', label=label, markersize=10, alpha=0.7, color=color)
    plt.xlabel('Index')
    plt.ylabel(r'Absolute Difference $|\lambda_i^{exact} - \lambda_i^{model}|$')
    plt.legend()
    plt.grid(True, alpha=0.3)
    save_path = get_unique_path(save_dir, f"Entanglement_Spectrum_Diff_L{L}.png")
    plt.savefig(save_path, dpi=300)
    print(f"Spectrum difference plot saved to {save_path}")
    plt.close()


def plot_spectrum_rel_diff(evals_exact, trained_results, save_dir, L):
    plt.figure(figsize=(10, 7))
    for result in trained_results:
        color = _model_color(result['type'])
        evals_model = result['evals']
        min_len = min(len(evals_exact), len(evals_model))
        if min_len > 0:
            denominator = evals_exact[:min_len]
            valid_indices = denominator > 1e-12
            diff = np.abs(denominator - evals_model[:min_len])
            relative_diff = np.full_like(diff, np.nan)
            relative_diff[valid_indices] = diff[valid_indices] / denominator[valid_indices]
            mean_rel_diff = np.nanmean(relative_diff)
            var_rel_diff = np.nanvar(relative_diff)
            label = f"{result['type']} (Mean: {mean_rel_diff:.3e}, Var: {var_rel_diff:.3e})"
            plt.semilogy(relative_diff, 'o--', label=label, markersize=10, alpha=0.7, color=color)
            plt.axhline(y=mean_rel_diff, color=color, linestyle=':', linewidth=10, alpha=0.9)
    plt.xlabel('Index')
    plt.ylabel(r'Relative Difference $|\lambda_i^{exact} - \lambda_i^{model}| / \lambda_i^{exact}$')
    plt.legend()
    plt.grid(True, alpha=0.3)
    save_path = get_unique_path(save_dir, f"Entanglement_Spectrum_Rel_Diff_L{L}.png")
    plt.savefig(save_path, dpi=300)
    print(f"Spectrum relative difference plot saved to {save_path}")
    plt.close()


def plot_spectrum_sectors(evals_exact, trained_results, save_dir, L):
    plt.figure(figsize=(10, 7))
    for result in trained_results:
        color = _model_color(result['type'])
        evals_model = result['evals']
        min_len = min(len(evals_exact), len(evals_model))
        if min_len > 0:
            denominator = evals_exact[:min_len]
            valid_indices = denominator > 1e-12
            diff = np.abs(denominator - evals_model[:min_len])
            relative_diff = np.full_like(diff, np.nan)
            relative_diff[valid_indices] = diff[valid_indices] / denominator[valid_indices]
            plt.plot(relative_diff, 'o', markersize=10, alpha=0.2, color=color)
            plt.ylim(0, 0.25)
            #plt.semilogy(relative_diff, 'o', markersize=10, alpha=0.2, color=color)
            n_sectors = 5
            boundaries = [int(round(i * min_len / n_sectors)) for i in range(n_sectors + 1)]
            sectors = [(boundaries[i], boundaries[i + 1]) for i in range(n_sectors)]
            means = []
            for start, end in sectors:
                seg_mean = np.nanmean(relative_diff[start:end]) if start < end else np.nan
                means.append(seg_mean)
                if not np.isnan(seg_mean):
                    plt.hlines(y=seg_mean, xmin=start, xmax=end-1, colors=color, linestyles='-', linewidth=4)
            means_str = ", ".join(f"{m:.1e}" for m in means)
            label = f"{result['type']}\nMeans: {means_str}"
            plt.plot([], [], color=color)
    plt.xlabel('Index')
    plt.ylabel(r'Relative Difference')
    plt.legend()
    plt.grid(True, alpha=0.3)
    save_path = get_unique_path(save_dir, f"Entanglement_Spectrum_Rel_Diff_Sectors_L{L}.png")
    plt.savefig(save_path, dpi=300)
    print(f"Spectrum relative difference sectors plot saved to {save_path}")
    plt.close()


def plot_combined_spectrum(evals_exact, trained_results, save_dir, L):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    
    # Top plot: trained models
    ax1.semilogy(evals_exact, 'o-', label='Exact GS', markersize=14, alpha=0.6, color='red', zorder=1)
    for result in trained_results:
        color = _model_color(result['type'])
        zorder = 3 if result['type'] == 'HFDS' else 2
        label = f'{result["type"]}'
        ax1.semilogy(result['evals'], '*-', label=label, markersize=10, alpha=0.8, color=color, zorder=zorder)

    ax1.set_ylabel(r'Eigenvalues $\lambda_i$')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Bottom plot: spectrum sectors
    for result in trained_results:
        color = _model_color(result['type'])
        evals_model = result['evals']
        min_len = min(len(evals_exact), len(evals_model))
        if min_len > 0:
            denominator = evals_exact[:min_len]
            valid_indices = denominator > 1e-12
            diff = np.abs(denominator - evals_model[:min_len])
            relative_diff = np.full_like(diff, np.nan)
            relative_diff[valid_indices] = diff[valid_indices] / denominator[valid_indices]
            ax2.plot(relative_diff, 'o', markersize=10, alpha=0.2, color=color)
            ax2.set_ylim(0, 0.25)
            n_sectors = 5
            boundaries = [int(round(i * min_len / n_sectors)) for i in range(n_sectors + 1)]
            sectors = [(boundaries[i], boundaries[i + 1]) for i in range(n_sectors)]
            for start, end in sectors:
                seg_mean = np.nanmean(relative_diff[start:end]) if start < end else np.nan
                if not np.isnan(seg_mean):
                    ax2.hlines(y=seg_mean, xmin=start, xmax=end-1, colors=color, linestyles='-', linewidth=4)
    
    ax2.set_xlabel('Index')
    ax2.set_ylabel(r'Relative Difference')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = get_unique_path(save_dir, f"Entanglement_Spectrum_Combined_L{L}.png")
    plt.savefig(save_path, dpi=300)
    print(f"Combined spectrum plot saved to {save_path}")
    plt.close()


def _resolve_save_dir():
    save_dir = "/cluster/home/fconoscenti/Thesis_QSL/Entanglement/plots"
    if not os.path.exists(save_dir):
        save_dir = "/scratch/f/F.Conoscenti/Thesis_QSL/Entanglement/plots"
    os.makedirs(save_dir, exist_ok=True)
    return save_dir


def _setup_hilbert(L, J2):
    N = L * L
    lattice = nk.graph.Hypercube(length=L, n_dim=2, pbc=True, max_neighbor_order=2)
    hi_full = nk.hilbert.Spin(s=1/2, N=N)
    hi_constrained = nk.hilbert.Spin(s=1/2, N=N, total_sz=0)
    ha = nk.operator.Heisenberg(hilbert=hi_constrained, graph=lattice, J=[1.0, J2], sign_rule=[False, False])
    all_states_constrained = hi_constrained.all_states()
    full_indices_constrained = hi_full.states_to_numbers(all_states_constrained)
    indices_A = [y * L + x for y in range(L) for x in range(L // 2)]
    return N, hi_full, hi_constrained, ha, all_states_constrained, full_indices_constrained, indices_A


def _compute_exact_state(hi_full, hi_constrained, ha, full_indices_constrained, N, indices_A):
    print("Computing Exact Ground State...")
    _, ket_gs = nk.exact.lanczos_ed(ha, compute_eigenvectors=True)
    ket_gs = ket_gs.flatten()
    psi_exact = np.zeros(hi_full.n_states, dtype=ket_gs.dtype)
    psi_exact[full_indices_constrained] = ket_gs
    psi_exact /= np.linalg.norm(psi_exact)
    _, evals_exact = compute_entanglement_spectrum_2d(N, indices_A, psi_exact)
    return psi_exact, evals_exact


def _compute_random_init_evals(L, N, hi_constrained, hi_full, all_states_constrained, full_indices_constrained, indices_A):
    key = jax.random.PRNGKey(42)
    dummy = jnp.zeros((1, N))

    print("Computing ViT (Random Init)...")
    vit_model = ViT_ent(num_layers=2, d_model=16, n_heads=4, patch_size=2, kernel_init=normal(stddev=10))
    params_vit = vit_model.init(key, dummy)
    psi_vit_c = np.array(jnp.exp(vit_model.apply(params_vit, all_states_constrained)))
    psi_vit = np.zeros(hi_full.n_states, dtype=psi_vit_c.dtype)
    psi_vit[full_indices_constrained] = psi_vit_c

    print("Computing HFDS (Random Init)...")
    hfds_model = HiddenFermion_ent(L=L, network="FFNN", n_hid=2, layers=1, features=32, MFinit="random", hilbert=hi_constrained, kernel_init=normal(stddev=10), dtype=jnp.complex128)
    params_hfds = hfds_model.init(key, dummy)
    psi_hfds_c = np.array(jnp.exp(hfds_model.apply(params_hfds, all_states_constrained)))
    psi_hfds = np.zeros(hi_full.n_states, dtype=psi_hfds_c.dtype)
    psi_hfds[full_indices_constrained] = psi_hfds_c

    _, evals_vit = compute_entanglement_spectrum_2d(N, indices_A, psi_vit)
    _, evals_hfds = compute_entanglement_spectrum_2d(N, indices_A, psi_hfds)
    return evals_vit, evals_hfds


def _load_trained_results(trained_model_paths, L, J2, hi_constrained, hi_full, all_states_constrained, full_indices_constrained, psi_exact, N, indices_A):
    if isinstance(trained_model_paths, str):
        trained_model_paths = [trained_model_paths]

    trained_results = []
    for path in trained_model_paths:
        print(f"--- Loading Trained Model from: {os.path.basename(path)} ---")
        model_trained, params_trained, model_type = load_trained_model(path, L, J2, hi_constrained, hi_full)

        if model_trained is None:
            print(f"Skipping path {path} as model could not be loaded.")
            continue

        n_params = nk.jax.tree_size(params_trained)

        if model_type not in ("HFDS", "ViT"):
            print(f"Could not evaluate wavefunction for model from {path}")
            continue

        log_psi = model_trained.apply(params_trained, all_states_constrained)
        log_psi = log_psi - jnp.max(log_psi.real)
        psi_c = np.array(jnp.exp(log_psi))
        psi_trained = np.zeros(hi_full.n_states, dtype=psi_c.dtype)
        psi_trained[full_indices_constrained] = psi_c
        psi_trained /= np.linalg.norm(psi_trained)

        fidelity_exact = np.abs(np.vdot(psi_exact, psi_trained))**2
        _, evals_trained = compute_entanglement_spectrum_2d(N, indices_A, psi_trained)
        trained_results.append({'type': model_type, 'path': path, 'evals': evals_trained, 'n_params': n_params, 'fidelity': fidelity_exact, 'psi': psi_trained})

    return trained_results


def run_spectrum_comparison(L=4, J2=0.5, trained_model_paths=None):
    print(f"--- Running Entanglement Spectrum Comparison (L={L}, J2={J2}) ---")

    save_dir = _resolve_save_dir()
    N, hi_full, hi_constrained, ha, all_states_constrained, full_indices_constrained, indices_A = _setup_hilbert(L, J2)
    psi_exact, evals_exact = _compute_exact_state(hi_full, hi_constrained, ha, full_indices_constrained, N, indices_A)

    evals_vit, evals_hfds = _compute_random_init_evals(L, N, hi_constrained, hi_full, all_states_constrained, full_indices_constrained, indices_A)
    plot_random_init(evals_exact, evals_vit, evals_hfds, save_dir, L)

    if not trained_model_paths:
        print("No trained model paths provided. Skipping trained model plots.")
        return

    trained_results = _load_trained_results(trained_model_paths, L, J2, hi_constrained, hi_full, all_states_constrained, full_indices_constrained, psi_exact, N, indices_A)
    if not trained_results:
        return

    # plot_trained_models(evals_exact, trained_results, save_dir, L)
    #plot_spectrum_diff(evals_exact, trained_results, save_dir, L)
    #plot_spectrum_rel_diff(evals_exact, trained_results, save_dir, L)
    # plot_spectrum_sectors(evals_exact, trained_results, save_dir, L)
    plot_combined_spectrum(evals_exact, trained_results, save_dir, L)

if __name__ == "__main__":
    """paths = [
        "/scratch/f/F.Conoscenti/Thesis_QSL/HFDS_Heisenberg/plot/4x4/layers1_hidd4_feat64_sample1024_lr0.02_iter1000_parityTrue_rotTrue_InitFermi_typecomplex",
        "/scratch/f/F.Conoscenti/Thesis_QSL/ViT_Heisenberg/plot/4x4/layers2_d16_heads4_patch2_sample1024_lr0.0075_iter4000_parityTrue_rotTrue_latest_model"
    ]"""

    """paths = ["/scratch/f/F.Conoscenti/Thesis_QSL/ViT_Heisenberg/plot/4x4/layers2_d16_heads4_patch2_sample1024_lr0.0075_iter20000_parityTrue_rotTrue_latest_model",
             "/scratch/f/F.Conoscenti/Thesis_QSL/HFDS_Heisenberg/plot/4x4/layers1_hidd4_feat32_sample1024_bcPBC_PBC_phi0.0_lr0.02_iter20000_parityTrue_rotTrue_InitFermi_typecomplex_phi"]    
    """

    paths = ["/scratch/f/F.Conoscenti/Thesis_QSL/ViT_Heisenberg/plot/4x4/layers2_d20_heads5_patch2_sample1024_lr0.0075_iter20000_parityTrue_rotTrue_QGT",
            #"/scratch/f/F.Conoscenti/Thesis_QSL/ViT_Heisenberg/plot/4x4/layers1_d16_heads4_patch2_sample1024_lr0.0075_iter20000_parityTrue_rotTrue_QGT",
            "/scratch/f/F.Conoscenti/Thesis_QSL/HFDS_Heisenberg/plot/4x4/layers1_hidd2_feat32_sample1024_bcPBC_PBC_lr0.02_iter20000_parityTrue_rotTrue_InitFermi_typecomplex",
            #"/scratch/f/F.Conoscenti/Thesis_QSL/ViT_Heisenberg/plot/4x4/layers1_d16_heads4_patch2_sample1024_lr0.0075_iter2000_parityTrue_rotTrue_latest_model"
    ]

    run_spectrum_comparison(L=4, J2=0.55, trained_model_paths=paths)