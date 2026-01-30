import jax
import jax.numpy as jnp
import netket as nk
from netket import jax as nkjax
from netket.stats import statistics as mpi_statistics
from functools import partial
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import gc
sys.path.append("/scratch/f/F.Conoscenti/Thesis_QSL")
from ViT_Heisenberg.ViT_model_ent import ViT_ent
from HFDS_Heisenberg.entanglement_model.HFDS_model_spin_ent import HiddenFermion_ent

def compute_renyi2_entropy(vstate, partition_indices=None, n_samples=1024, chunk_size=None):
    """
    Calculates the Second Renyi Entropy (S2) of a NetKet Variational State 
    using the Swap Operator (Replica) method.

    Args:
        vstate: A NetKet Variational State (e.g., MCState).
        partition_indices (array): Indices of the sites in partition A. 
                                   If None, defaults to the first half of the system.
        n_samples (int): Total number of samples to use for the estimation.
        chunk_size (int): Batch size for evaluating the network. If None, 
                          NetKet tries to determine it automatically.
    
    Returns:
        mean_entropy (float): The estimated S2 entropy.
        error_entropy (float): The standard error of the estimation.
    """
    
    if chunk_size is None:
        chunk_size = 1024

    # 1. Determine Partition (Default to Half-System)
    N = vstate.hilbert.size
    if partition_indices is None:
        partition_indices = jnp.arange(N // 2)
    else:
        partition_indices = jnp.array(partition_indices)

    # 2. Generate Samples
    # We need two independent "replicas" of the system. We sample a large batch
    # and split it into two halves (samples_1 and samples_2).
    #print(f"Sampling {n_samples} configurations...")
    samples = vstate.sample(n_samples=n_samples)
    
    n_chains = samples.shape[0]
    n_samples_per_chain = samples.shape[1]
    total_samples = n_chains * n_samples_per_chain

    # Logic to split samples into two replicas
    if n_chains > 1:
        # Split by chains (e.g., chains 0-255 vs 256-511)
        samples_1 = samples[:(n_chains // 2), :]
        samples_2 = samples[(n_chains // 2):, :]
    else:
        # If only 1 chain, split by time samples
        samples_1 = samples[:, :(n_samples_per_chain // 2)]
        samples_2 = samples[:, (n_samples_per_chain // 2):]

    # Flatten the batches for easier processing
    sigma_A = samples_1.reshape(-1, N)
    sigma_B = samples_2.reshape(-1, N)

    # 3. Create Swapped Configurations
    # Sigma_tilde has the configuration of Replica A, but with the partition sites swapped with Replica B
    sigma_A_tilde = jnp.copy(sigma_A)
    sigma_B_tilde = jnp.copy(sigma_B)
    
    # Perform the swap on the partition indices
    sigma_A_tilde = sigma_A_tilde.at[:, partition_indices].set(sigma_B[:, partition_indices])
    sigma_B_tilde = sigma_B_tilde.at[:, partition_indices].set(sigma_A[:, partition_indices])

    # 4. Define the JIT-compiled Entropy Calculation
    # We use nkjax.apply_chunked to prevent OOM errors if the sample size is huge
    @partial(nkjax.apply_chunked, in_axes=(None, None, 0, 0, 0, 0), chunk_size=chunk_size)
    def compute_log_overlap(params, model_state, sA, sB, sA_tilde, sB_tilde):
        # Access the underlying apply function of the model
        afun = vstate._apply_fun
        W = {"params": params, **model_state}
        
        # Calculate log_psi for all 4 configurations
        log_psi_A = afun(W, sA)
        log_psi_B = afun(W, sB)
        log_psi_At = afun(W, sA_tilde)
        log_psi_Bt = afun(W, sB_tilde)
        
        # The estimator for Tr(rho^2) is the expectation value of:
        # Psi(A_tilde) * Psi(B_tilde) / (Psi(A) * Psi(B))
        return jnp.exp(log_psi_At + log_psi_Bt - log_psi_A - log_psi_B)

    # 5. Execute Calculation
    #print("Computing swap amplitudes...")
    overlaps = compute_log_overlap(
        vstate.parameters, 
        vstate.model_state, 
        sigma_A, 
        sigma_B, 
        sigma_A_tilde, 
        sigma_B_tilde
    )

    # 6. Compute Statistics
    # NetKet's mpi_statistics handles averaging across different MPI nodes if applicable
    stats = mpi_statistics(overlaps)
    
    # S2 = -ln(Mean(Overlap))
    # We take the real part because the overlap should be real for Hermitian systems,
    # though small imaginary parts may exist due to noise.
    entropy_mean = -jnp.log(stats.mean).real
    
    # Propagate error: Delta(ln x) = Delta(x) / x
    entropy_error = float(jnp.sqrt(stats.variance / total_samples) / stats.mean.real)

    # Normalize by max entropy
    max_entropy = len(partition_indices) * np.log(2)
    entropy_mean /= max_entropy
    entropy_error /= max_entropy

    return entropy_mean, entropy_error

def clean_up():
    try:
        jax.clear_caches()
    except AttributeError:
        pass
    gc.collect()

def check_zero_variance_output(vstate):
    # Sample a small batch to check outputs
    samples = vstate.sample(n_samples=16)
    samples = samples.reshape(-1, vstate.hilbert.size)
    log_psi = vstate.log_value(samples)
    
    print(f"\n[Zero Variance Check]")
    print(f"Log Psi (first 5): {log_psi[:5]}")
    print(f"Max |Log Psi|: {jnp.max(jnp.abs(log_psi))}")
    print(f"Is zero? {jnp.allclose(log_psi, 0j, atol=1e-7)}\n")

    # Check parameters for non-zero values
    print("Scanning parameters for non-zeros...")
    params = vstate.parameters
    
    def scan_params(params, path=""):
        if hasattr(params, 'items'):
            for k, v in params.items():
                scan_params(v, f"{path}/{k}")
        else:
            # Leaf node (array). Check kernels and attention matrices.
            # Note: LayerNorm 'scale' is expected to be 1.0, so we skip it or check for != 1.
            if "kernel" in path or "alpha" in path or "V" in path or "W" in path:
                max_val = jnp.max(jnp.abs(params))
                if max_val > 1e-7:
                    print(f"  -> Non-zero param at '{path}': max {max_val}")

    scan_params(params)
    print("Scan complete.\n")

def plot_rbm_sweep(N, variances, alphas, n_samples=2000, n_seeds=1):
    print("Starting RBM parameter sweep...")
    save_dir = os.path.join(os.path.dirname(__file__), "plots")
    os.makedirs(save_dir, exist_ok=True)
    hi = nk.hilbert.Spin(s=0.5, N=N)
    entropy_grid = np.zeros((len(alphas), len(variances)))

    for i, alpha in enumerate(alphas):
        for j, var in enumerate(variances):
            s2_accum = 0.0
            for seed in range(n_seeds):
                stddev = np.sqrt(var)
                initializer = jax.nn.initializers.normal(stddev=stddev)
                ma = nk.models.RBM(alpha=alpha, kernel_init=initializer, hidden_bias_init=initializer, visible_bias_init=initializer)

                sampler = nk.sampler.MetropolisLocal(hi)
                vstate = nk.vqs.MCState(sampler, ma, n_samples=1024, seed=seed)
                vstate.init_parameters()

                if var == 0 and seed == 0:
                    check_zero_variance_output(vstate)

                s2, _ = compute_renyi2_entropy(vstate, n_samples=n_samples)
                s2_accum += s2
                clean_up()

            entropy_grid[i, j] = s2_accum / n_seeds

    plt.figure(figsize=(10, 6))
    dv = (variances[1] - variances[0]) / 2. if len(variances) > 1 else 0.5
    da = (np.array(alphas)[1] - np.array(alphas)[0]) / 2. if len(alphas) > 1 else 0.5
    extent = [np.min(variances) - dv, np.max(variances) + dv, np.min(alphas) - da, np.max(alphas) + da]
    im = plt.imshow(entropy_grid, cmap='viridis', origin='lower', extent=extent, aspect='auto', interpolation='none')
    plt.colorbar(im, label='Renyi-2 Entropy')
    plt.xlabel('Variance of Initialization')
    plt.ylabel('Alpha (RBM density)')
    plt.title(f'Entanglement vs Initialization Variance and Model Density (N={N}, n_samples={n_samples}, n_seeds={n_seeds})')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(save_dir, 'Entanglement_Sweep_RBM1.png'))
    print(f"Plot saved to {os.path.join(save_dir, 'Entanglement_Sweep_RBM.png')} (RBM)")

def plot_vit_sweep(N, variances, d_models, n_samples=2000, n_seeds=1):
    print("Starting ViT sweep...")
    num_layers = 2
    n_heads = 4
    patch_size = 2
    save_dir = os.path.join(os.path.dirname(__file__), "plots")
    os.makedirs(save_dir, exist_ok=True)
    hi = nk.hilbert.Spin(s=0.5, N=N)
    entropy_grid = np.zeros((len(d_models), len(variances)))

    for i, d in enumerate(d_models):
        for j, var in enumerate(variances):
            s2_accum = 0.0
            for seed in range(n_seeds):
                stddev = np.sqrt(var)
                if var > 0:
                    initializer = jax.nn.initializers.normal(stddev=stddev)
                else:
                    initializer = jax.nn.initializers.zeros
                ma = ViT_ent(num_layers=num_layers, d_model=d, n_heads=n_heads, patch_size=patch_size, transl_invariant=True, kernel_init=initializer)
                sampler = nk.sampler.MetropolisLocal(hi)
                vstate = nk.vqs.MCState(sampler, ma, n_samples=1024, seed=seed)
                vstate.init_parameters()

                if var == 0 and seed == 0:
                    check_zero_variance_output(vstate)

                s2, _ = compute_renyi2_entropy(vstate, n_samples=n_samples)
                s2_accum += s2
                clean_up()

            entropy_grid[i, j] = s2_accum / n_seeds

    plt.figure(figsize=(10, 6))
    dv = (variances[1] - variances[0]) / 2. if len(variances) > 1 else 0.5
    dd = (np.array(d_models)[1] - np.array(d_models)[0]) / 2. if len(d_models) > 1 else 0.5
    extent = [np.min(variances) - dv, np.max(variances) + dv, np.min(d_models) - dd, np.max(d_models) + dd]
    im = plt.imshow(entropy_grid, cmap='viridis', origin='lower', extent=extent, aspect='auto', interpolation='none')
    plt.colorbar(im, label='Renyi-2 Entropy')
    plt.xlabel('Variance (Approx)')
    plt.ylabel('d_model')
    plt.title(f'Entanglement ViT (N={N}, n_samples={n_samples}, n_seeds={n_seeds}, num_layers={num_layers}, n_heads={n_heads}, patch_size={patch_size})')
    plt.savefig(os.path.join(save_dir, 'Entanglement_Sweep_ViT1.png'))
    print(f"Plot saved to {os.path.join(save_dir, 'Entanglement_Sweep_ViT.png')}")

def plot_hfds_sweep(N, variances, n_hids, n_samples=2000, n_seeds=1):
    print("Starting HFDS sweep...")
    save_dir = os.path.join(os.path.dirname(__file__), "plots")
    os.makedirs(save_dir, exist_ok=True)
    hi = nk.hilbert.Spin(s=0.5, N=N)
    entropy_grid = np.zeros((len(n_hids), len(variances)))

    for i, nh in enumerate(n_hids):
        for j, var in enumerate(variances):
            s2_accum = 0.0
            for seed in range(n_seeds):
                stddev = np.sqrt(var)
                if var > 0:
                    initializer = jax.nn.initializers.normal(stddev=stddev)
                else:
                    initializer = jax.nn.initializers.zeros
                L_side = int(np.sqrt(N))
                ma = HiddenFermion_ent(L=L_side, network="FFNN", n_hid=nh, layers=1, features=16, MFinit="Fermi", hilbert=hi, kernel_init=initializer)
                sampler = nk.sampler.MetropolisLocal(hi)
                vstate = nk.vqs.MCState(sampler, ma, n_samples=1024, seed=seed)
                vstate.init_parameters()

                if var == 0 and seed == 0:
                    check_zero_variance_output(vstate)

                s2, _ = compute_renyi2_entropy(vstate, n_samples=n_samples)
                s2_accum += s2
                clean_up()

            entropy_grid[i, j] = s2_accum / n_seeds

    plt.figure(figsize=(10, 6))
    dv = (variances[1] - variances[0]) / 2. if len(variances) > 1 else 0.5
    dnh = (np.array(n_hids)[1] - np.array(n_hids)[0]) / 2. if len(n_hids) > 1 else 0.5
    extent = [np.min(variances) - dv, np.max(variances) + dv, np.min(n_hids) - dnh, np.max(n_hids) + dnh]
    im = plt.imshow(entropy_grid, cmap='viridis', origin='lower', extent=extent, aspect='auto', interpolation='none')
    plt.colorbar(im, label='Renyi-2 Entropy')
    plt.xlabel('Variance (Approx)')
    plt.ylabel('n_hiddens')
    plt.title(f'Entanglement HFDS (N={N}, n_samples={n_samples}, n_seeds={n_seeds})')
    plt.savefig(os.path.join(save_dir, 'Entanglement_Sweep_HFDS1.png'))
    print(f"Plot saved to {os.path.join(save_dir, 'Entanglement_Sweep_HFDS.png')}")

def plot_rbm_scaling(N_values, variances_lines, n_seeds=10, n_samples=2000):
    print("\nStarting RBM Scaling sweep...")
    save_dir = os.path.join(os.path.dirname(__file__), "plots")
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(10, 6))
    alpha_fixed = 10

    for var in variances_lines:
        entropies = []
        for N in N_values:
            s2_accum = 0.0
            for seed in range(n_seeds):
                hi = nk.hilbert.Spin(s=0.5, N=N)
                stddev = np.sqrt(var)
                if var > 0:
                    initializer = jax.nn.initializers.normal(stddev=stddev)
                else:
                    initializer = jax.nn.initializers.zeros
                
                ma = nk.models.RBM(alpha=alpha_fixed, kernel_init=initializer, hidden_bias_init=initializer, visible_bias_init=initializer)
                
                sampler = nk.sampler.MetropolisLocal(hi)
                vstate = nk.vqs.MCState(sampler, ma, n_samples=1024, seed=seed)
                vstate.init_parameters()
                
                s2, _ = compute_renyi2_entropy(vstate, n_samples=n_samples)
                s2_accum += s2
                clean_up()
            entropies.append(s2_accum / n_seeds)
        
        plt.plot(N_values, entropies, marker='o', label=f'RBM Var={var}')
    plt.xlabel('Number of Spins (N)')
    plt.ylabel('Renyi-2 Entropy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(save_dir, 'Entanglement_Scaling_N_RBM1.png'))
    print(f"Plot saved to {os.path.join(save_dir, 'Entanglement_Scaling_N_RBM.png')}")

def plot_vit_scaling(N_values, variances_lines, n_seeds=10, n_samples=2000):
    print("\nStarting ViT Scaling sweep...")
    num_layers = 2
    n_heads = 4
    d_models = 64
    patch_size = 2
    save_dir = os.path.join(os.path.dirname(__file__), "plots")
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(10, 6))
    for var in variances_lines:
        entropies_vit = []
        stddev = np.sqrt(var)
        if var > 0:
            initializer = jax.nn.initializers.normal(stddev=stddev)
        else:
            initializer = jax.nn.initializers.zeros
        for N in N_values:
            s2_accum = 0.0
            for seed in range(n_seeds):
                hi = nk.hilbert.Spin(s=0.5, N=N)
                ma = ViT_ent(num_layers=num_layers, d_model=d_models, n_heads=n_heads, patch_size=patch_size, transl_invariant=True, kernel_init=initializer)
                sampler = nk.sampler.MetropolisLocal(hi)
                vstate = nk.vqs.MCState(sampler, ma, n_samples=1024, seed=seed)
                vstate.init_parameters()

                s2, _ = compute_renyi2_entropy(vstate, n_samples=n_samples)
                s2_accum += s2
                clean_up()

            entropies_vit.append(s2_accum / n_seeds)

        plt.plot(N_values, entropies_vit, marker='s', linestyle='--', label=f'Var={var}')
    plt.xlabel('Number of Spins (N)')
    plt.ylabel('Renyi-2 Entropy')
    plt.title(f'ViT Entanglement Scaling (num_layers={num_layers}, n_heads={n_heads}, d_model={d_models}, n_samples={n_samples}, n_seeds={n_seeds})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(save_dir, 'Entanglement_Scaling_N_ViT1.png'))
    print(f"Plot saved to {os.path.join(save_dir, 'Entanglement_Scaling_N_ViT.png')}")

def plot_hfds_scaling(N_values, variances_lines, n_seeds=10, n_samples=2000):
    print("\nStarting HFDS Scaling sweep...")
    n_hids = 4
    hidden_features = 64
    save_dir = os.path.join(os.path.dirname(__file__), "plots")
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(10, 6))
    for var in variances_lines:
        entropies_hfds = []
        stddev = np.sqrt(var)
        if var > 0:
            initializer = jax.nn.initializers.normal(stddev=stddev)
        else:
            initializer = jax.nn.initializers.zeros
        valid_N = []
        for N in N_values:
            L_side = int(np.sqrt(N))
            if L_side * L_side != N: continue
            valid_N.append(N)
            
            s2_accum = 0.0
            for seed in range(n_seeds):
                hi = nk.hilbert.Spin(s=0.5, N=N)
                ma = HiddenFermion_ent(L=L_side, network="FFNN", n_hid=n_hids, layers=1, features=hidden_features, MFinit="Fermi", hilbert=hi, kernel_init=initializer)
                sampler = nk.sampler.MetropolisLocal(hi)
                vstate = nk.vqs.MCState(sampler, ma, n_samples=1024, seed=seed)
                vstate.init_parameters()

                s2, _ = compute_renyi2_entropy(vstate, n_samples=n_samples)
                s2_accum += s2
                clean_up()

            entropies_hfds.append(s2_accum / n_seeds)
        
        plt.plot(valid_N, entropies_hfds, marker='^', linestyle=':', label=f'Var={var}')
    plt.xlabel('Number of Spins (N)')
    plt.ylabel('Renyi-2 Entropy')   
    plt.title(f'HFDS Entanglement Scaling (n_hid={n_hids}, hid_feat={hidden_features}, n_samples={n_samples}, n_seeds={n_seeds})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(save_dir, 'Entanglement_Scaling_N_HFDS1.png'))
    print(f"Plot saved to {os.path.join(save_dir, 'Entanglement_Scaling_N_HFDS.png')}")

# --- Example Usage ---
if __name__ == "__main__":
    # Define system
    N = 16
    n_samples = 1024#32768*2 #200k
    n_seeds = 1
    
    # Parameters to vary
    variances = np.linspace(0, 1, 3)
    alphas = [2 , 10]#,  6,  8,  10,  12,  14,  16,  18,  20, 22, 24, 26, 28, 30, 32, 34 ,36 ,38 ,40]
    d_models = [4, 20]#, 16, 20, 22, 24, 26, 28, 30, 32, 34 ,36 ,38 ,40]
    n_hids = [1 ,4]#, 4 ,5 ,6 ,7 ,8 ,9 ,10 ,11 ,12 ,13 ,14 ,15 ,16 ,17 ,18 ,19 ,20]

    #plot_rbm_sweep(N, variances, alphas, n_samples=n_samples, n_seeds=n_seeds)
    plot_vit_sweep(N, variances, d_models, n_samples=n_samples, n_seeds=n_seeds)
    #plot_hfds_sweep(N, variances, n_hids, n_samples=n_samples, n_seeds=n_seeds)

    N_values = [ 16, 36, 64, 100, 144]
    variances_lines = [0.1, 0.2, 0.5, 1.0, 2.0]
    
    #plot_rbm_scaling(N_values, variances_lines, n_seeds=n_seeds, n_samples=n_samples)
    #plot_vit_scaling(N_values, variances_lines, n_seeds=n_seeds, n_samples=n_samples)
    #plot_hfds_scaling(N_values, variances_lines, n_seeds=n_seeds, n_samples=n_samples)