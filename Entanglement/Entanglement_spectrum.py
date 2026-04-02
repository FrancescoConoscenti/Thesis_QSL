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
        return None, None

    j_path = os.path.join(path, f"J={J2}")
    if not os.path.exists(j_path):
        j_path = os.path.join(path, f"J2={J2}")
    
    if not os.path.exists(j_path):
        print(f"Error: J={J2} folder not found in {path}")
        return None, None, None

    seeds = [d for d in os.listdir(j_path) if d.startswith("seed_")]
    if not seeds:
        print("Error: No seed folder found.")
        return None, None
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

def plot_spectrum(ket_gs, vstate, L, save_dir=None):
    """
    Plots the entanglement spectrum of a given vstate alongside the exact ground state,
    and computes the total Euclidean error and the relative error in three sectors.
    
    Args:
        ket_gs: Exact ground state wavefunction.
        vstate: NetKet variational state.
        L (int): Linear size of the lattice.
        J2 (float): Next-nearest neighbor coupling.
        indices_A (list): Indices of the subsystem A.
        save_dir (str, optional): Directory to save the plot.
        
    Returns:
        total_error (float): Total Euclidean distance between the spectra.
        sector_errors (list): Mean relative errors in the High, Mid, and Low sectors.
    """
    print(f"--- Plotting Entanglement Spectrum (L={L}) ---")
    N = L * L
    lattice = nk.graph.Hypercube(length=L, n_dim=2, pbc=True, max_neighbor_order=2)
    hi_full = nk.hilbert.Spin(s=1/2, N=N)
    hi_constrained = nk.hilbert.Spin(s=1/2, N=N, total_sz=0)

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
    plt.title(f'Entanglement Eigenvalues (L={L}, J2={J2})')
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

def run_spectrum_comparison(L=4, J2=0.5, trained_model_paths=None):
    print(f"--- Running Entanglement Spectrum Comparison (L={L}, J2={J2}) ---")

    N = L * L
    lattice = nk.graph.Hypercube(length=L, n_dim=2, pbc=True, max_neighbor_order=2)

    # Hilbert spaces
    hi_full = nk.hilbert.Spin(s=1/2, N=N)
    hi_constrained = nk.hilbert.Spin(s=1/2, N=N, total_sz=0)

    # Hamiltonian for Exact GS (using constrained space for efficiency)
    ha = nk.operator.Heisenberg(hilbert=hi_constrained, graph=lattice, J=[1.0, J2], sign_rule=[False, False])

    # 1. Exact Ground State
    print("Computing Exact Ground State...")
    E_gs, ket_gs = nk.exact.lanczos_ed(ha, compute_eigenvectors=True)
    ket_gs = ket_gs.flatten()

    # Embed Exact GS in full space
    psi_exact = np.zeros(hi_full.n_states, dtype=ket_gs.dtype)
    full_indices_constrained = hi_full.states_to_numbers(hi_constrained.all_states())
    psi_exact[full_indices_constrained] = ket_gs
    
    # Normalize exact state
    psi_exact /= np.linalg.norm(psi_exact)

    # 2. ViT (Random Init)
    print("Computing ViT (Random Init)...")
    vit_model = ViT_ent(num_layers=2, d_model=16, n_heads=4, patch_size=2, kernel_init=normal(stddev=10))
    # Initialize parameters
    key = jax.random.PRNGKey(42)
    dummy_input_c = jnp.zeros((1, N))
    params_vit = vit_model.init(key, dummy_input_c)

    # Compute state on constrained space
    all_states_constrained = hi_constrained.all_states()
    log_psi_vit_c = vit_model.apply(params_vit, all_states_constrained)
    psi_vit_c = np.array(jnp.exp(log_psi_vit_c))

    # Embed in full space
    psi_vit = np.zeros(hi_full.n_states, dtype=psi_vit_c.dtype)
    psi_vit[full_indices_constrained] = psi_vit_c

    # 3. HFDS (Random Init)
    print("Computing HFDS (Random Init)...")
    hfds_model = HiddenFermion_ent(L=L, network="FFNN", n_hid=2, layers=1, features=32, MFinit="random", hilbert=hi_constrained, kernel_init=normal(stddev=10), dtype=jnp.complex128)
    # Initialize parameters
    key = jax.random.PRNGKey(42)
    dummy_input_c = jnp.zeros((1, N))
    params_hfds = hfds_model.init(key, dummy_input_c)

    # Compute state on constrained space
    all_states_constrained = hi_constrained.all_states()
    log_psi_hfds = hfds_model.apply(params_hfds, all_states_constrained)
    psi_hfds_c = np.array(jnp.exp(log_psi_hfds))

    # Embed in full space
    psi_hfds = np.zeros(hi_full.n_states, dtype=psi_hfds_c.dtype)
    psi_hfds[full_indices_constrained] = psi_hfds_c

    # Define Subsystem A (Half vertical)
    indices_A = []
    for y in range(L):
        for x in range(L // 2): 
            flat_index = y * L + x
            indices_A.append(flat_index)

    # Compute Spectra
    _, evals_exact = compute_entanglement_spectrum_2d(N, indices_A, psi_exact)
    _, evals_vit = compute_entanglement_spectrum_2d(N, indices_A, psi_vit)
    _, evals_hfds = compute_entanglement_spectrum_2d(N, indices_A, psi_hfds)

    # --- PLOT 1: Random Init vs Exact ---
    plt.figure(figsize=(10, 7))
    plt.semilogy(evals_exact, 'o-', label='Exact GS', markersize=4, alpha=0.8, color='red', zorder=10)
    plt.semilogy(evals_vit, 's--', label='ViT (Random)', markersize=4, alpha=0.6)
    plt.semilogy(evals_hfds, '^--', label='HFDS (Random)', markersize=4, alpha=0.6)

    plt.xlabel('Index')
    plt.ylabel(r'Eigenvalues $\lambda_i$')
    plt.title(f'Entanglement Eigenvalues (Random Init) (L={L}, J2={J2})')
    plt.legend()
    plt.grid(True, alpha=0.3)

    save_dir = "/cluster/home/fconoscenti/Thesis_QSL/Entanglement/plots"
    if not os.path.exists(save_dir):
        save_dir = "/scratch/f/F.Conoscenti/Thesis_QSL/Entanglement/plots"
    os.makedirs(save_dir, exist_ok=True)

    save_path_random = get_unique_path(save_dir, f"Entanglement_Spectrum_Random_L{L}.png")
    plt.savefig(save_path_random, dpi=300)
    print(f"Random init plot saved to {save_path_random}")
    plt.close()

    # --- PLOT 2: Trained Models vs Exact ---
    if not trained_model_paths:
        print("No trained model paths provided. Skipping second plot.")
        return

    # 4. Trained Models
    trained_results = []
    if isinstance(trained_model_paths, str): # Handle single path for backward compatibility
        trained_model_paths = [trained_model_paths]

    for path in trained_model_paths:
        print(f"--- Loading Trained Model from: {os.path.basename(path)} ---")
        model_trained, params_trained, model_type = load_trained_model(path, L, J2, hi_constrained, hi_full)

        if model_trained is None:
            print(f"Skipping path {path} as model could not be loaded.")
            continue

        # Calculate number of parameters
        n_params = nk.jax.tree_size(params_trained)

        psi_trained = None
        if model_type == "HFDS" or model_type == "ViT":
            log_psi_trained = model_trained.apply(params_trained, all_states_constrained)
            log_psi_trained = log_psi_trained - jnp.max(log_psi_trained.real)
            psi_trained_c = np.array(jnp.exp(log_psi_trained))
            psi_trained = np.zeros(hi_full.n_states, dtype=psi_trained_c.dtype)
            psi_trained[full_indices_constrained] = psi_trained_c

        if psi_trained is not None:
            # Normalize trained state
            psi_trained /= np.linalg.norm(psi_trained)
            
            # Calculate fidelity with exact ground state
            fidelity_exact = np.abs(np.vdot(psi_exact, psi_trained))**2
            
            _, evals_trained = compute_entanglement_spectrum_2d(N, indices_A, psi_trained)
            trained_results.append({'type': model_type, 'path': path, 'evals': evals_trained, 'n_params': n_params, 'fidelity': fidelity_exact, 'psi': psi_trained})
        else:
            print(f"Could not evaluate wavefunction for model from {path}")

    # Plotting trained models
    plt.figure(figsize=(10, 7))
    plt.semilogy(evals_exact, 'o-', label='Exact GS', markersize=4, alpha=0.8, color='red', zorder=10)

    # Plot trained models
    if trained_results:
        # Use the 'plasma' colormap and offset the range to avoid the darkest colors,
        # ensuring better contrast with the black 'Exact GS' line.
        colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(trained_results)))
        for i, result in enumerate(trained_results):
            label = f'{result["type"]} ({result["n_params"]:,} params)'
            plt.semilogy(result['evals'], '*-', label=label, markersize=5, alpha=0.8, color=colors[i])

    # Calculate fidelity between trained models if exactly 2
    fidelity_between_models = None
    if len(trained_results) == 2:
        fidelity_between_models = np.abs(np.vdot(trained_results[0]['psi'], trained_results[1]['psi']))**2

    # Add fidelity information to the plot
    stats_text = []
    for res in trained_results:
        stats_text.append(f"F({res['type']}, Exact) = {res['fidelity']:.5f}")
    if fidelity_between_models is not None:
        stats_text.append(f"F({trained_results[0]['type']}, {trained_results[1]['type']}) = {fidelity_between_models:.5f}")
    
    if stats_text:
        full_text = "\n".join(stats_text)
        props = dict(boxstyle='round', facecolor='white', alpha=0.8)
        plt.gca().text(0.05, 0.05, full_text, transform=plt.gca().transAxes, fontsize=10, verticalalignment='bottom', bbox=props)

    plt.xlabel('Index')
    plt.ylabel(r'Eigenvalues $\lambda_i$')
    plt.title(f'Entanglement Eigenvalues (Trained Models) (L={L}, J2={J2})')
    plt.legend()
    plt.grid(True, alpha=0.3)

    save_path_trained = get_unique_path(save_dir, f"Entanglement_Spectrum_Trained_L{L}.png")
    plt.savefig(save_path_trained, dpi=300)
    print(f"Trained models plot saved to {save_path_trained}")
    plt.close()

    # --- PLOT 3: Spectrum Distance (Difference) ---
    if trained_results:
        plt.figure(figsize=(10, 7))
        
        for result in trained_results:
            evals_model = result['evals']
            min_len = min(len(evals_exact), len(evals_model))
            
            if min_len > 0:
                diff = np.abs(evals_exact[:min_len] - evals_model[:min_len])
                dist = np.linalg.norm(diff)
                
                label = f"{result['type']} (Eucl. Dist: {dist:.3e})"
                plt.semilogy(diff, 'o--', label=label, markersize=4, alpha=0.7)
        
        plt.xlabel('Index')
        plt.ylabel(r'Absolute Difference $|\lambda_i^{exact} - \lambda_i^{model}|$')
        plt.title(f'Entanglement Spectrum Error (L={L}, J2={J2})')
        plt.legend()
        plt.grid(True, alpha=0.3)

        save_path_diff = get_unique_path(save_dir, f"Entanglement_Spectrum_Diff_L{L}.png")
        plt.savefig(save_path_diff, dpi=300)
        print(f"Spectrum difference plot saved to {save_path_diff}")
        plt.close()

    # --- PLOT 4: Spectrum Distance (Relative Difference) ---
    if trained_results:
        plt.figure(figsize=(10, 7))
        
        # Get colors to match the mean line with the data points
        colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(trained_results)))

        for i, result in enumerate(trained_results):
            evals_model = result['evals']
            min_len = min(len(evals_exact), len(evals_model))
            
            if min_len > 0:
                denominator = evals_exact[:min_len]
                valid_indices = denominator > 1e-12
                
                diff = np.abs(denominator - evals_model[:min_len])
                relative_diff = np.full_like(diff, np.nan)
                relative_diff[valid_indices] = diff[valid_indices] / denominator[valid_indices]

                # Calculate mean and variance, ignoring NaNs
                mean_rel_diff = np.nanmean(relative_diff)
                var_rel_diff = np.nanvar(relative_diff)

                # Update label to include mean and variance
                label = f"{result['type']} (Mean: {mean_rel_diff:.3e}, Var: {var_rel_diff:.3e})"
                
                plt.semilogy(relative_diff, 'o--', label=label, markersize=4, alpha=0.7, color=colors[i])
                plt.axhline(y=mean_rel_diff, color=colors[i], linestyle=':', linewidth=2, alpha=0.9)
        
        plt.xlabel('Index')
        plt.ylabel(r'Relative Difference $|\lambda_i^{exact} - \lambda_i^{model}| / \lambda_i^{exact}$')
        plt.title(f'Entanglement Spectrum Relative Error (L={L}, J2={J2})')
        plt.legend()
        plt.grid(True, alpha=0.3)

        save_path_rel_diff = get_unique_path(save_dir, f"Entanglement_Spectrum_Rel_Diff_L{L}.png")
        plt.savefig(save_path_rel_diff, dpi=300)
        print(f"Spectrum relative difference plot saved to {save_path_rel_diff}")
        plt.close()

    # --- PLOT 5: Spectrum Distance (Relative Difference) with Sectors ---
    if trained_results:
        plt.figure(figsize=(10, 7))
        
        colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(trained_results)))

        for i, result in enumerate(trained_results):
            evals_model = result['evals']
            min_len = min(len(evals_exact), len(evals_model))
            
            if min_len > 0:
                denominator = evals_exact[:min_len]
                valid_indices = denominator > 1e-12
                
                diff = np.abs(denominator - evals_model[:min_len])
                relative_diff = np.full_like(diff, np.nan)
                relative_diff[valid_indices] = diff[valid_indices] / denominator[valid_indices]

                # Plot points
                plt.semilogy(relative_diff, 'o', markersize=2, alpha=0.2, color=colors[i])

                # Sectors: High (0 to 1/3), Mid (1/3 to 2/3), Low (2/3 to 1)
                s1 = min_len // 3
                s2 = 2 * (min_len // 3)
                sectors = [(0, s1), (s1, s2), (s2, min_len)]
                
                means = []
                for start, end in sectors:
                    seg_mean = np.nanmean(relative_diff[start:end]) if start < end else np.nan
                    means.append(seg_mean)
                    if not np.isnan(seg_mean):
                        plt.hlines(y=seg_mean, xmin=start, xmax=end-1, colors=colors[i], linestyles='-', linewidth=2)
                
                label = f"{result['type']}\nMeans: H={means[0]:.1e}, M={means[1]:.1e}, L={means[2]:.1e}"
                plt.plot([], [], color=colors[i], label=label)

        plt.xlabel('Index')
        plt.ylabel(r'Relative Difference')
        plt.title(f'Entanglement Spectrum Relative Error by Sector (L={L}, J2={J2})')
        plt.legend()
        plt.grid(True, alpha=0.3)

        save_path_sectors = get_unique_path(save_dir, f"Entanglement_Spectrum_Rel_Diff_Sectors_L{L}.png")
        plt.savefig(save_path_sectors, dpi=300)
        print(f"Spectrum relative difference sectors plot saved to {save_path_sectors}")
        plt.close()

if __name__ == "__main__":
    paths = [
        "/scratch/f/F.Conoscenti/Thesis_QSL/HFDS_Heisenberg/plot/4x4/layers1_hidd4_feat64_sample1024_lr0.02_iter1000_parityTrue_rotTrue_InitFermi_typecomplex",
        "/scratch/f/F.Conoscenti/Thesis_QSL/ViT_Heisenberg/plot/4x4/layers2_d16_heads4_patch2_sample1024_lr0.0075_iter4000_parityTrue_rotTrue_latest_model"
    ]
    run_spectrum_comparison(L=4, J2=0.5, trained_model_paths=paths)