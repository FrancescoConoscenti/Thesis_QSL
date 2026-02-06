import jax
import netket as nk
import gzip
import jax.numpy as jnp
import flax 
from flax import linen as nn
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict

def compute_sign_complexity(vstate, n_samples: int = 32768, batch_size: int = 100):
    """
    Memory-efficient batched version of sign complexity computation.
    
    Useful when dealing with large systems or many samples that don't fit in memory.
    
    Args:
        vstate: NetKet variational state
        n_samples: Total number of MCMC samples
        batch_size: Number of samples to process at once
    
    Returns:
        Same dictionary as compute_sign_complexity()
    """
    #print("Computing sign complexity...")
    n_batches = (n_samples + batch_size - 1) // batch_size
    all_sensitivities = []
    total_flips = 0
    total_checks = 0
    
    
    for batch_idx in range(n_batches):
        current_batch_size = min(batch_size, n_samples - batch_idx * batch_size)
        
        # Sample batch
        samples = vstate.sample(n_samples=current_batch_size)
        if samples.ndim == 3:
            samples = samples.reshape(-1, samples.shape[-1])
        if samples.ndim == 1:
            samples = samples.reshape(1, -1)
        
        n_configs, n_spins = samples.shape
        
        #print(f"Batch {batch_idx + 1}/{n_batches}: {n_configs} samples")
        
        # Process each configuration in batch
        for config in samples:
            log_psi_original = vstate.log_value(config.reshape(1, -1))
            sign_original = jnp.sign(log_psi_original.real)
            
            sign_changes = 0
            for spin_idx in range(n_spins):
                config_flipped = config.at[spin_idx].multiply(-1)
                
                log_psi_flipped = vstate.log_value(config_flipped.reshape(1, -1))
                sign_flipped = jnp.sign(log_psi_flipped.real)
                
                if sign_flipped != sign_original:
                    sign_changes += 1
                    total_flips += 1
                
                total_checks += 1
            
            sensitivity = sign_changes / n_spins
            all_sensitivities.append(float(sensitivity))
    
    # Compute statistics
    all_sensitivities = np.array(all_sensitivities)
    
    mean_sensitivity = float(np.mean(all_sensitivities))
    std_sensitivity = float(np.std(all_sensitivities))
    min_sensitivity = float(np.min(all_sensitivities))
    max_sensitivity = float(np.max(all_sensitivities))

    
    # Print interpretation
    #print("\n" + "="*60)
    print("SIGN COMPLEXITY ANALYSIS")
    #print("="*60)
    print(f"Average local sensitivity: {mean_sensitivity:.4f} ± {std_sensitivity:.4f}")
    print(f"Overall sign flip rate: {total_flips / total_checks:.4f}")
    #print("="*60)
    
    return  mean_sensitivity, std_sensitivity, min_sensitivity, max_sensitivity


if __name__ == "__main__":
    # Define a simple system
    L = 4
    g = nk.graph.Hypercube(length=L, n_dim=1, pbc=True)
    hi = nk.hilbert.Spin(s=1/2, N=g.n_nodes)

    # Define RBM
    ma = nk.models.RBM(alpha=1, param_dtype=complex)

    # Define sampler
    sa = nk.sampler.MetropolisLocal(hi)

    # Define variational state
    vstate = nk.vqs.MCState(sa, ma, n_samples=128)

    # Compute sign complexity
    compute_sign_complexity(vstate, n_samples=100, batch_size=10)
