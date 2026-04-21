import jax
import jax.numpy as jnp
import flax.linen as nn
import netket as nk
import numpy as np

# Import initialization functions from your existing scripts
import sys
from typing import Any
sys.path.append("/scratch/f/F.Conoscenti/Thesis_QSL")
from HFDS_Heisenberg.MF_Init import init_orbitals_mf_phi
from HFDS_Heisenberg.Init_orbitals import compute_orbital_selection

class FreeFermionSlaterDeterminant(nn.Module):
    """
    A Neural Quantum State representing a pure Free Fermion Slater Determinant.
    """
    L: int
    phi: float = 0.0
    bounds: Any = ("PBC", "PBC")
    dtype: type = jnp.complex128

    def setup(self):
        self.N_sites = self.L * self.L
        self.n_elecs = self.N_sites  # Half-filling (total_sz = 0)

        # Wrapper to use your init_orbitals_mf_phi with Flax's param initialization
        def _init_mf(key, shape, dtype):
            return init_orbitals_mf_phi(L=self.L, bounds=self.bounds, phi=self.phi, dtype=dtype)
        
        # Define the orbitals as a parameter so they can be optimized if needed.
        # Shape is (2*N_sites, n_elecs) representing spin-up and spin-down available states.
        self.orbitals = self.param(
            'orbitals_mf', 
            _init_mf, 
            (2 * self.N_sites, self.n_elecs), 
            self.dtype
        )

    def calc_psi(self, x):
        """Calculates log(psi) for a single spin configuration."""
        # 1. Select the rows corresponding to occupied orbitals for this specific config
        Phi_occupied = compute_orbital_selection(x, self.orbitals, self.N_sites)
        
        # 2. Compute the log-determinant of the occupied NxN matrix
        sign, log_det = jnp.linalg.slogdet(Phi_occupied)
        
        # Combine sign and amplitude into a complex log value
        return log_det + jnp.log(sign + 0j)

    def __call__(self, x):
        """Evaluates the NQS over a batch of configurations."""
        # Vectorize the single-sample function over the batch axis (axis=0)
        return jax.vmap(self.calc_psi)(x)


if __name__ == "__main__":
    # --- 1. Physical Parameters ---
    L = 4
    J2 = 0.5
    phi = 0.0
    N_sites = L * L

    print(f"Setting up Free Fermion Slater Determinant for L={L}, J2={J2}, phi={phi}")

    # --- 2. Define Lattice, Hilbert Space, and Hamiltonian ---
    lattice = nk.graph.Hypercube(length=L, n_dim=2, pbc=True, max_neighbor_order=2)
    hilbert = nk.hilbert.Spin(s=0.5, N=lattice.n_nodes, total_sz=0)
    
    ha = nk.operator.Heisenberg(
        hilbert=hilbert, 
        graph=lattice, 
        J=[1.0, J2], 
        sign_rule=[False, False]
    ).to_jax_operator()

    # --- 3. Initialize the Model ---
    model = FreeFermionSlaterDeterminant(L=L, phi=phi, bounds=("PBC", "PBC"), dtype=jnp.complex128)

    # --- 4. Setup Sampler and Variational State ---
    # We use MetropolisExchange since we enforce total_sz = 0 (particle conservation)
    sampler = nk.sampler.MetropolisExchange(
        hilbert=hilbert,
        graph=lattice,
        d_max=2,
        n_chains=16,
        sweep_size=lattice.n_nodes,
    )

    vstate = nk.vqs.MCState(
        sampler=sampler,
        model=model,
        n_samples=2048,
        n_discard_per_chain=128,
        seed=jax.random.PRNGKey(42)
    )

    print(f"Number of parameters in the model: {nk.jax.tree_size(vstate.parameters)}")

    # --- 5. Evaluate the Free Fermion Energy ---
    # Since the orbitals are initialized to the Fermi sea (MF), this expectation
    # value corresponds to the pure mean-field Free Fermion state.
    print("Sampling configurations to compute initial expectation value...")
    E_mf = vstate.expect(ha)
    
    # Heisenberg energy per site (NetKet adds a factor of 4 for Pauli matrices)
    E_per_site = E_mf.mean.real / (N_sites * 4)
    print(f"Free Fermion Mean-Field Energy per site: {E_per_site:.6f}")
    
    
