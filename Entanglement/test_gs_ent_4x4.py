import sys
import os

# Add project root to path
sys.path.append("/scratch/f/F.Conoscenti/Thesis_QSL")

import netket as nk
import jax
import jax.numpy as jnp
from flax import linen as nn
import numpy as np
import matplotlib.pyplot as plt

from Entanglement.Entanglement import compute_renyi2_entropy

class HashableArray:
    def __init__(self, array):
        self.array = array
    
    def __hash__(self):
        return id(self.array)
    
    def __eq__(self, other):
        return self.array is other.array

class ExactModel(nn.Module):
    ket_gs_wrapper: HashableArray
    hilbert: nk.hilbert.AbstractHilbert

    @nn.compact
    def __call__(self, x):
        # Add a dummy parameter to satisfy NetKet's requirement for a 'params' key
        self.param('dummy', lambda rng: jnp.array(0.0))

        # x shape: (batch, N)
        ket_gs = self.ket_gs_wrapper.array
        indices = self.hilbert.states_to_numbers(x)
        vals = ket_gs[indices]
        return jnp.log(vals.astype(complex))

def run_test():
    L = 4
    J2_values = np.linspace(0.0, 1.0, 11)
    s2_values = []
    s2_errors = []
    
    # Setup system
    lattice = nk.graph.Hypercube(length=L, n_dim=2, pbc=True, max_neighbor_order=2)
    hilbert = nk.hilbert.Spin(s=1/2, N=lattice.n_nodes, total_sz=0)
    
    print(f"System: {L}x{L} Heisenberg")

    for J2 in J2_values:
        print(f"\nProcessing J2 = {J2:.2f}")
        hamiltonian = nk.operator.Heisenberg(hilbert=hilbert, graph=lattice, J=[1.0, J2], sign_rule=[False, False])
        
        # Exact Diagonalization
        E, ket_gs = nk.exact.lanczos_ed(hamiltonian, compute_eigenvectors=True)
        ket_gs = ket_gs[:, 0]
        print(f"  Exact Energy: {E[0]:.6f}")
        
        # Convert to JAX array
        ket_gs_jax = jnp.array(ket_gs)
        
        # Define Model
        model = ExactModel(ket_gs_wrapper=HashableArray(ket_gs_jax), hilbert=hilbert)
        
        # Sampler
        sampler = nk.sampler.MetropolisExchange(hilbert=hilbert, graph=lattice, d_max=2)
        
        # Variational State
        vstate = nk.vqs.MCState(sampler, model, n_samples=10000)
        
        # Compute Entropy
        s2, s2_err = compute_renyi2_entropy(vstate, n_samples=65536)
        print(f"  Renyi-2 Entropy (S2): {s2:.6f} ± {s2_err:.6f}")
        
        s2_values.append(s2)
        s2_errors.append(s2_err)

    # Plotting
    plt.figure(figsize=(8, 6))
    plt.errorbar(J2_values, s2_values, yerr=s2_errors, fmt='o-', capsize=5, label='Exact GS (ED + Swap)')
    plt.xlabel(r'$J_2$')
    plt.ylabel(r'Renyi Entropy $S_2$')
    plt.title(f'Entanglement Entropy vs $J_2$ ({L}x{L} Heisenberg)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    output_file = "Entanglement_vs_J2_Exact_4x4.png"
    plt.savefig(output_file)
    print(f"\nPlot saved to {output_file}")

if __name__ == "__main__":
    run_test()
