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
        sampler = nk.sampler.MetropolisLocal(hilbert=hi_full)
        vstate = nk.vqs.MCState(sampler, model, n_samples=10)

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
        vstate = nk.vqs.MCState(sampler, model, n_samples=10)

    else:
        print("Error: Could not determine model type from path.")
        return None, None, None
    
    try:
        vstate = flax.serialization.from_bytes(vstate, data)
        variables = vstate.variables
    except:
        variables = flax.serialization.from_bytes(model.init(jax.random.PRNGKey(0), jnp.zeros((1, L*L))), data)
            
    return model, variables, model_type

def run_spectrum_comparison(L=4, J2=0.5, trained_model_path=None):
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
    
    # 2. ViT (Random Init)
    print("Computing ViT (Random Init)...")
    vit_model = ViT_ent(
        num_layers=2,
        d_model=16,
        n_heads=4,
        patch_size=2,
        kernel_init=normal(stddev=10)
    )
    # Initialize parameters
    key = jax.random.PRNGKey(42)
    dummy_input = jnp.zeros((1, N))
    params_vit = vit_model.init(key, dummy_input)
    
    # Compute full state
    all_states_full = hi_full.all_states()
    log_psi_vit = vit_model.apply(params_vit, all_states_full)
    psi_vit = np.array(jnp.exp(log_psi_vit))
    
    # 3. HFDS (Random Init)
    print("Computing HFDS (Random Init)...")
    hfds_model = HiddenFermion_ent(
        L=L,
        network="FFNN",
        n_hid=2,
        layers=1,
        features=32,
        MFinit="random",
        hilbert=hi_constrained,
        kernel_init=normal(stddev=10),
        dtype=jnp.complex128
    )
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
    
    # 4. Trained HFDS (Optional)
    psi_trained = None
    trained_model_type = None
    if trained_model_path:
        print("Computing Trained Model...")
        model_trained, params_trained, model_type = load_trained_model(trained_model_path, L, J2, hi_constrained, hi_full)
        trained_model_type = model_type

        if model_trained and model_type == "HFDS":
            log_psi_trained = model_trained.apply(params_trained, all_states_constrained)
            log_psi_trained = log_psi_trained - jnp.max(log_psi_trained.real)
            psi_trained_c = np.array(jnp.exp(log_psi_trained))
            psi_trained = np.zeros(hi_full.n_states, dtype=psi_trained_c.dtype)
            psi_trained[full_indices_constrained] = psi_trained_c

        elif model_trained and model_type == "ViT":
            all_states_full = hi_full.all_states()
            log_psi_trained = model_trained.apply(params_trained, all_states_full)
            log_psi_trained = log_psi_trained - jnp.max(log_psi_trained.real)
            psi_trained = np.array(jnp.exp(log_psi_trained))
        else:
            print("Trained model could not be loaded or evaluated.")

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
    
    evals_trained_spec = None
    if psi_trained is not None:
        _, evals_trained_spec = compute_entanglement_spectrum_2d(N, indices_A, psi_trained)
    
    # Plotting
    plt.figure(figsize=(8, 6))
    plt.semilogy(evals_exact, 'o-', label='Exact GS', markersize=4, alpha=0.8)
    plt.semilogy(evals_vit, 's--', label='ViT (Random)', markersize=4, alpha=0.8)
    plt.semilogy(evals_hfds, '^--', label='HFDS (Random)', markersize=4, alpha=0.8)
    if evals_trained_spec is not None:
        label = f'{trained_model_type} (Trained)'
        plt.semilogy(evals_trained_spec, '*-', label=label, markersize=4, alpha=0.8)
    
    plt.xlabel('Index')
    plt.ylabel(r'Eigenvalues $\lambda_i$')
    plt.title(f'Entanglement Eigenvalues Comparison (L={L}, J2={J2})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    save_dir = "/cluster/home/fconoscenti/Thesis_QSL/Entanglement/plots"
    if not os.path.exists(save_dir):
        save_dir = "/scratch/f/F.Conoscenti/Thesis_QSL/Entanglement/plots"
    os.makedirs(save_dir, exist_ok=True)
    
    save_path = get_unique_path(save_dir, f"Entanglement_Spectrum_Comparison_L{L}.png")
    plt.savefig(save_path, dpi=300)
    print(f"Plot saved to {save_path}")
    plt.close()

if __name__ == "__main__":
    path = "/cluster/home/fconoscenti/Thesis_QSL/HFDS_Heisenberg/plot/4x4/layers1_hidd4_feat64_sample1024_lr0.02_iter1000_parityTrue_rotTrue_InitFermi_typecomplex"
    run_spectrum_comparison(L=4, J2=0.5, trained_model_path=path)