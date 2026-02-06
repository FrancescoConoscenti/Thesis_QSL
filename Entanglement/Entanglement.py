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
from scipy.optimize import curve_fit

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

def get_square_subsystem(L, l):
    """
    Create a square subsystem partition.
    
    Args:
        L: total lattice size (L×L)
        l: subsystem linear size (l×l square in corner)
    
    Returns:
        jnp.array of site indices in the subsystem
    """
    indices = []
    for row in range(l):
        for col in range(l):
            indices.append(row * L + col)
    return jnp.array(indices)


def compute_entanglement_scaling(vstate, L_total, n_samples=2048, min_size=2):
    """
    Compute Renyi-2 entropy for growing square subsystems and fit to area law.
    
    Args:
        vstate: NetKet variational state
        L_total: total lattice linear size (L×L system)
        n_samples: number of samples for entropy estimation
        min_size: minimum subsystem size to consider
    
    Returns:
        results: dict with perimeters, entropies, errors, and fit parameters
    """
    subsystem_sizes = range(min_size, L_total - 1)
    perimeters = []
    entropies = []
    errors = []
    
    #print("Computing entanglement entropy for different subsystem sizes...")
    #print("-" * 60)
    
    for l in subsystem_sizes:
        partition = get_square_subsystem(L_total, l)
        perimeter = 4 * l
        
        S2, err = compute_renyi2_entropy(vstate, partition, n_samples=n_samples)
        
        perimeters.append(perimeter)
        entropies.append(S2)
        errors.append(err)
        
        print(f"Subsystem size: {l}×{l} | Perimeter: {perimeter} | "
              f"S2 = {S2:.4f} ± {err:.4f}")
    
    # Linear fit: S2 = alpha * perimeter + beta
    def linear_model(x, alpha, beta):
        return alpha * x + beta
    
    perimeters = np.array(perimeters)
    entropies = np.array(entropies)
    errors = np.array(errors)
    
    # Perform weighted fit (using errors as weights)
    popt, pcov = curve_fit(linear_model, perimeters, entropies, 
                           sigma=errors, absolute_sigma=True)
    alpha, beta = popt
    alpha_err, beta_err = np.sqrt(np.diag(pcov))
    
    # Calculate gamma_topo = -beta
    gamma_topo = -beta
    gamma_topo_err = beta_err
    
    #print("\n" + "=" * 60)
    print("FIT RESULTS: S2 = α × perimeter + β")
    #print("=" * 60)
    print(f"α (area law coefficient) = {alpha:.4f} ± {alpha_err:.4f}")
    print(f"β (constant offset)      = {beta:.4f} ± {beta_err:.4f}")
    print(f"γ_topo = -β              = {gamma_topo:.4f} ± {gamma_topo_err:.4f}")
    #print("=" * 60)
    
    if gamma_topo > 2 * gamma_topo_err and gamma_topo > 0:
        print("⚠️  LONG-RANGE ENTANGLEMENT DETECTED (γ_topo > 0)")
        print("    → Possible topological order or quantum spin liquid phase")
    elif abs(gamma_topo) < 2 * gamma_topo_err:
        print("✓  SHORT-RANGE ENTANGLEMENT (γ_topo ≈ 0)")
        print("    → Conventional ordered or gapped phase")
    else:
        print("?  γ_topo < 0 (unphysical or finite-size effects)")
    
    print("=" * 60)
    
    return {
        'perimeters': perimeters,
        'entropies': entropies,
        'errors': errors,
        'alpha': alpha,
        'alpha_err': alpha_err,
        'beta': beta,
        'beta_err': beta_err,
        'gamma_topo': gamma_topo,
        'gamma_topo_err': gamma_topo_err
    }

def plot_entanglement_scaling(results, save_path=None):
    """
    Plot S2 vs perimeter with linear fit.
    
    Args:
        results: dict from analyze_entanglement_scaling
        J2_value: J2/J1 ratio for the plot title
        save_path: if provided, save figure to this path
    """
    perimeters = results['perimeters']
    entropies = results['entropies']
    errors = results['errors']
    alpha = results['alpha']
    beta = results['beta']
    gamma_topo = results['gamma_topo']
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot data with error bars
    ax.errorbar(perimeters, entropies, yerr=errors, 
                fmt='o', markersize=8, capsize=5, 
                label='Computed S₂', color='steelblue')
    
    # Plot fit line
    perimeter_fit = np.linspace(perimeters.min(), perimeters.max(), 100)
    entropy_fit = alpha * perimeter_fit + beta
    ax.plot(perimeter_fit, entropy_fit, '--', 
            label=f'Fit: S₂ = {alpha:.3f}L + {beta:.3f}', 
            color='coral', linewidth=2)
    
    # Add zero line for reference
    ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
    
    ax.set_xlabel('Perimeter (L)', fontsize=12)
    ax.set_ylabel('Rényi-2 Entropy (S₂)', fontsize=12)
    
    title = 'Entanglement Entropy Scaling'
    ax.set_title(title, fontsize=14)
    
    # Add text box with gamma_topo
    textstr = f'γ_topo = {gamma_topo:.3f} ± {results["gamma_topo_err"]:.3f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=props)
    
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nPlot saved to: {save_path}")
    
    plt.show()



def clean_up():
    try:
        jax.clear_caches()
    except AttributeError:
        pass
    gc.collect()
