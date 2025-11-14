#%%
from tenpy.models.model import CouplingMPOModel
from tenpy.models.lattice import Square
from tenpy.networks.site import SpinSite
from tenpy.networks.mps import MPS
from tenpy.algorithms import dmrg
from tenpy.tools import hdf5_io
from numpy import linspace
from random import shuffle
import numpy as np
from tenpy.networks.site import SpinHalfSite
from tenpy.algorithms.exact_diag import get_full_wavefunction
from DMRG.Plotting import *
from DMRG.Observable.Corr_Struct import Correlations_Structure_Factor

from netket.experimental.driver import VMC_SR

from jax import numpy as jnp
import os
import netket as nk
import pickle
import gzip
import flax

#%%
class J1J2Heisenberg(CouplingMPOModel):
    """A TeNPy model for the J1-J2 Heisenberg model on a square lattice."""
    def init_sites(self, model_params):
        # Enforce spin-flip parity conservation instead of Sz conservation.
        # 'parity' here corresponds to the eigenvalue of the spin-flip operator.
        # The ground state is expected to be in the even (p=+1) sector.
        return SpinHalfSite()

    def init_lattice(self, model_params):
        Lx = model_params.get('Lx', 4)
        Ly = model_params.get('Ly', 4)
        site = self.init_sites(model_params=model_params)
        return Square(Lx=Lx, Ly=Ly, site=site, bc='periodic', bc_MPS='finite')

    def init_terms(self, model_params):
        J1 = model_params.get('J1', 1.0)
        J2 = model_params.get('J2', 0.0)

        # nearest neighbors (J1)
        for (u1, u2, dx) in self.lat.pairs['nearest_neighbors']:
            self.add_coupling(J1, u1, 'Sz', u2, 'Sz', dx)
            self.add_coupling(1/2*J1, u1, 'Sp', u2, 'Sm', dx, plus_hc=True)
            

        # next-nearest neighbors (J2)
        for (u1, u2, dx) in self.lat.pairs['next_nearest_neighbors']:
            self.add_coupling(J2, u1, 'Sz', u2, 'Sz', dx)
            self.add_coupling(1/2*J2, u1, 'Sp', u2, 'Sm', dx, plus_hc=True)
            

def DMRG_vstate_optimization(hamiltonian, model_params, filename=None):
    if filename and os.path.exists(filename):
        print(f"--- Loading pre-trained DMRG state from {filename} ---")
        with gzip.open(filename, 'rb') as f:
            DMRG_vstate = pickle.load(f)
        print("--- DMRG state loaded. ---")
        Correlations_Structure_Factor(DMRG_vstate, model_params, hamiltonian)
        return DMRG_vstate

    print("\n--- Starting DMRG Optimization ---")
    sites = hamiltonian.lat.mps_sites()

    # Create a Neel state (alternating up/down) as the initial product state
    prod_state = ['up', 'down'] * (hamiltonian.lat.N_sites // 2)
    if hamiltonian.lat.N_sites % 2 == 1:
        prod_state.append('up')

    DMRG_vstate = MPS.from_product_state(sites, prod_state)
    dmrg_params = {
        'max_sweeps' : 20,
        'trunc_params' : {'chi_max' : 1024, 'svd_min': 1e-10},
        'chi_list' : {5:128, 10:256, 15:512, 20:1024}
        }

    engine = dmrg.TwoSiteDMRGEngine(DMRG_vstate, hamiltonian, dmrg_params)
    E0, DMRG_vstate = engine.run()

    # Extract energies and bond dimensions from all sweeps
    energies_DMRG = engine.sweep_stats['E']
    bond_dims = engine.sweep_stats['max_chi']
    sweeps = range(len(energies_DMRG))
    energies_per_site = [E / hamiltonian.lat.N_sites for E in energies_DMRG]

    plot_DMRG_energies(energies_per_site, bond_dims, sweeps, model_params)    

    Correlations_Structure_Factor(DMRG_vstate, model_params, hamiltonian)

    if filename:
        output_dir = os.path.dirname(filename)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        print(f"--- Saving trained DMRG state to {filename} ---")
        with gzip.open(filename, 'wb') as f:
            pickle.dump(DMRG_vstate, f)
        print("--- DMRG state saved. ---")

    return DMRG_vstate

#%%

def RBM_vstate_optimization(model_params, filename=None):

    # --- RBM Model and Sampler Definition ---
    L = model_params['Lx']
    N_sites = L * L
    hi = nk.hilbert.Spin(s=1/2, N=N_sites)
    RBM_model = nk.models.RBM(alpha=3, param_dtype=jnp.complex128)
    n_iter = 150

    # --- RBM Part (already existing) ---
    print("\n--- Starting NetKet RBM Calculation ---")

    # Extract lattice parameters from model_params
    L = model_params['Lx'] # Assuming Lx == Ly for a square lattice
    N_sites = L * L
    # Define the lattice for the Hamiltonian
    lattice = nk.graph.Hypercube(length=L, n_dim=2, pbc=True, max_neighbor_order=2)
    # 2. Define NetKet Hamiltonian
    # The J1-J2 Heisenberg Hamiltonian with J1 and J2 couplings
    # NOTE: NetKet's Heisenberg is defined with Pauli matrices (σ), while TeNPy uses Spin operators (S=σ/2).
    # This means H_nk = 4 * H_tenpy. We must divide by 4 for a correct comparison.
    ha = 0.25 * nk.operator.Heisenberg(hilbert=hi, graph=lattice, J=[1.0, model_params['J2']], sign_rule=[False, False])
    """
    ha = nk.operator.LocalOperator(hi, dtype=float)
    for i, j in lattice.edges():
        ha += 0.25 * nk.operator.spin.sigmaz(hi, i) * nk.operator.spin.sigmaz(hi, j)
    """
    
    # 4. Define a Sampler for RBM
    RBM_sampler = nk.sampler.MetropolisLocal(hilbert=hi)

    # 5. Initialize the Variational State
    RBM_vstate = nk.vqs.MCState(
        sampler=RBM_sampler,
        model=RBM_model,
        n_samples=1024, # Number of samples for Monte Carlo estimation
        n_discard_per_chain=16, # Number of discarded samples at the beginning of each chain
        seed=42 # For reproducibility
    )

    # Construct the filename with iteration count before the extension
    iter_filename = None
    if filename:
        name, ext = os.path.splitext(filename)
        iter_filename = f"{name}_iter{n_iter}{ext}"

    if iter_filename and os.path.exists(iter_filename):
        print(f"--- Loading pre-trained RBM parameters from {iter_filename} ---")
        with open(iter_filename, "rb") as f:
            RBM_vstate.variables = flax.serialization.from_bytes(
                RBM_vstate.variables, f.read()
            )
        print("--- RBM parameters loaded. ---")
        print("--- NetKet RBM Calculation Finished ---")
        return RBM_vstate, ha


    
    # 6. Define an Optimizer
    RBM_optimizer = nk.optimizer.Sgd(learning_rate=0.01)

    # 7. Define the VMC Driver (using Stochastic Reconfiguration for better convergence)
    RBM_vmc = VMC_SR(
        hamiltonian=ha,
        optimizer=RBM_optimizer,
        variational_state=RBM_vstate,
        diag_shift=0.001 # Small regularization for SR to improve stability
    )

    # 8. Run VMC optimization
    rbm_log = nk.logging.RuntimeLog() # Log to store optimization data
    RBM_vmc.run(n_iter=n_iter, out=rbm_log)

    # 9. Extract and print final energy
    final_RBM_energy = rbm_log.data["Energy"]["Mean"].real[-1]
    print(f"Final RBM Energy: {final_RBM_energy:.6f}")
    print(f"Final RBM Energy per site: {final_RBM_energy / N_sites:.6f}")

    if iter_filename:
        output_dir = os.path.dirname(iter_filename)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        print(f"--- Saving trained RBM parameters to {iter_filename} ---")
        with open(iter_filename, "wb") as f:
            bytes_out = flax.serialization.to_bytes(RBM_vstate.variables)
            f.write(bytes_out)
        print("--- RBM parameters saved. ---")

    print("--- NetKet RBM Calculation Finished ---")

    return RBM_vstate, ha