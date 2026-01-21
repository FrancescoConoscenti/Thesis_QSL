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
sys.path.append("/scratch/f/F.Conoscenti/Thesis_QSL")
from ViT_Heisenberg.ViT_model import ViT
from HFDS_Heisenberg.HFDS_model_spin import HiddenFermion

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

    return entropy_mean, entropy_error

# --- Example Usage ---
if __name__ == "__main__":
    # Define system
    N = 16
    hi = nk.hilbert.Spin(s=0.5, N=N)
    
    # Parameters to vary
    variances = np.linspace(0, 2, 5)
    alphas = [1, 2, 4, 8]
    
    results_var = []
    results_alpha = []
    results_entropy = []

    # --- RBM Sweep ---
    print("Starting parameter sweep...")
    for alpha in alphas:
        for var in variances:
            stddev = np.sqrt(var)
            initializer = jax.nn.initializers.normal(stddev=stddev)
            ma = nk.models.RBM(alpha=alpha, kernel_init=initializer, hidden_bias_init=initializer, visible_bias_init=initializer)
            
            sampler = nk.sampler.MetropolisLocal(hi)
            vstate = nk.vqs.MCState(sampler, ma, n_samples=1024)
            vstate.init_parameters()
            
            s2, _ = compute_renyi2_entropy(vstate, n_samples=2000)
            
            results_var.append(var)
            results_alpha.append(alpha)
            results_entropy.append(s2)
            #print(f"Alpha={alpha}, Var={var:.2f} -> S2={s2:.4f}")
    
    # Plotting
    plt.figure(figsize=(10, 6))
    sc = plt.scatter(results_var, results_alpha, c=results_entropy, cmap='viridis', s=150, marker='s')
    plt.colorbar(sc, label='Renyi-2 Entropy')
    plt.xlabel('Variance of Initialization')
    plt.ylabel('Alpha (RBM density)')
    plt.title('Entanglement vs Initialization Variance and Model Density')
    plt.grid(True, alpha=0.3)
    plt.savefig('Entanglement_Sweep.png')
    print("Plot saved to Entanglement_Sweep.png (RBM)")

    # --- ViT Sweep ---
    print("Starting ViT sweep...")
    d_models = [2, 4, 8]
    results_vit_var = []
    results_vit_d = []
    results_vit_entropy = []
    
    for d in d_models:
        for var in variances:
            stddev = np.sqrt(var)
            ma = ViT(num_layers=2, d_model=d, n_heads=2, patch_size=1)
            sampler = nk.sampler.MetropolisLocal(hi)
            vstate = nk.vqs.MCState(sampler, ma, n_samples=1024)
            vstate.init_parameters()
            if var > 0:
                 vstate.parameters = jax.tree_util.tree_map(lambda x: x * stddev, vstate.parameters)
            elif var == 0:
                 vstate.parameters = jax.tree_util.tree_map(lambda x: jnp.zeros_like(x), vstate.parameters)

            s2, _ = compute_renyi2_entropy(vstate, n_samples=2000)
            results_vit_var.append(var)
            results_vit_d.append(d)
            results_vit_entropy.append(s2)

    plt.figure(figsize=(10, 6))
    sc = plt.scatter(results_vit_var, results_vit_d, c=results_vit_entropy, cmap='viridis', s=150, marker='s')
    plt.colorbar(sc, label='Renyi-2 Entropy')
    plt.xlabel('Variance (Approx)')
    plt.ylabel('d_model')
    plt.title('Entanglement ViT')
    plt.savefig('Entanglement_Sweep_ViT.png')

    # --- HFDS Sweep ---
    print("Starting HFDS sweep...")
    n_hids = [1, 2, 4]
    results_hfds_var = []
    results_hfds_nhid = []
    results_hfds_entropy = []
    
    for nh in n_hids:
        for var in variances:
            stddev = np.sqrt(var)
            L_side = int(np.sqrt(N))
            ma = HiddenFermion(L=L_side, network="FFNN", n_hid=nh, layers=1, features=16, MFinit="random", hilbert=hi)
            sampler = nk.sampler.MetropolisLocal(hi)
            vstate = nk.vqs.MCState(sampler, ma, n_samples=1024)
            vstate.init_parameters()
            if var > 0:
                 vstate.parameters = jax.tree_util.tree_map(lambda x: x * stddev, vstate.parameters)
            elif var == 0:
                 vstate.parameters = jax.tree_util.tree_map(lambda x: jnp.zeros_like(x), vstate.parameters)

            s2, _ = compute_renyi2_entropy(vstate, n_samples=2000)
            results_hfds_var.append(var)
            results_hfds_nhid.append(nh)
            results_hfds_entropy.append(s2)

    plt.figure(figsize=(10, 6))
    sc = plt.scatter(results_hfds_var, results_hfds_nhid, c=results_hfds_entropy, cmap='viridis', s=150, marker='s')
    plt.colorbar(sc, label='Renyi-2 Entropy')
    plt.xlabel('Variance (Approx)')
    plt.ylabel('n_hid')
    plt.title('Entanglement HFDS')
    plt.savefig('Entanglement_Sweep_HFDS.png')

############################################################################

    # --- New Plot: Entropy vs N for different Variances ---
    print("\nStarting N vs Entropy sweep...")
    N_values = [ 16, 36, 64 ]
    variances_lines = [0.1, 1.0]
    alpha_fixed = 4
    
    plt.figure(figsize=(10, 6))
    
    # RBM Scaling
    for var in variances_lines:
        entropies = []
        for N in N_values:
            hi = nk.hilbert.Spin(s=0.5, N=N)
            stddev = np.sqrt(var)
            initializer = jax.nn.initializers.normal(stddev=stddev)
            ma = nk.models.RBM(alpha=alpha_fixed, kernel_init=initializer, hidden_bias_init=initializer, visible_bias_init=initializer)
            
            sampler = nk.sampler.MetropolisLocal(hi)
            vstate = nk.vqs.MCState(sampler, ma, n_samples=1024)
            vstate.init_parameters()
            
            s2, _ = compute_renyi2_entropy(vstate, n_samples=2000)
            entropies.append(s2)
            #print(f"Var={var}, N={N} -> S2={s2:.4f}")
        
        plt.plot(N_values, entropies, marker='o', label=f'RBM Var={var}')

###############################################################################

    # ViT Scaling
    entropies_vit = []
    for N in N_values:
        hi = nk.hilbert.Spin(s=0.5, N=N)
        ma = ViT(num_layers=2, d_model=8, n_heads=2, patch_size=1)
        sampler = nk.sampler.MetropolisLocal(hi)
        vstate = nk.vqs.MCState(sampler, ma, n_samples=1024)
        vstate.init_parameters()
        s2, _ = compute_renyi2_entropy(vstate, n_samples=2000)
        entropies_vit.append(s2)
    plt.plot(N_values, entropies_vit, marker='s', linestyle='--', label='ViT (d=8)')

#####################################################################################

    # HFDS Scaling
    entropies_hfds = []
    for N in N_values:
        L_side = int(np.sqrt(N))
        hi = nk.hilbert.Spin(s=0.5, N=N)
        ma = HiddenFermion(L=L_side, network="FFNN", n_hid=2, layers=1, features=16, MFinit="random", hilbert=hi)
        sampler = nk.sampler.MetropolisLocal(hi)
        vstate = nk.vqs.MCState(sampler, ma, n_samples=1024)
        vstate.init_parameters()
        s2, _ = compute_renyi2_entropy(vstate, n_samples=2000)
        entropies_hfds.append(s2)
    plt.plot(N_values, entropies_hfds, marker='^', linestyle=':', label='HFDS (h=2)')
    
    plt.xlabel('Number of Spins (N)')
    plt.ylabel('Renyi-2 Entropy')
    plt.title(f'Entanglement Scaling with System Size (Alpha={alpha_fixed})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('Entanglement_Scaling_N.png')
    print("Plot saved to Entanglement_Scaling_N.png")