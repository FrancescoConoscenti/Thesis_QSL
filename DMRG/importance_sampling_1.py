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
from scipy.sparse.linalg import eigsh


from jax import numpy as jnp
import matplotlib.pyplot as plt
import netket as nk
import flax
import os

from DMRG.QSL_DMRG import *
from DMRG.Fidelities import Fidelity_exact, Fidelity_sampled
from Elaborate.Sign_Obs import Marshall_Sign_full_hilbert, Marshall_Sign_exact, Amp_overlap_configs, Sign_overlap
import tempfile
import shutil

#%%
if __name__ == "__main__":

    model_params = {
        'Lx': 4,
        'Ly': 4,
        'J1': 1.0,
        'J2': 0.5
    }

    n_samples = 1024
    n_iter = 500
    N_sites = model_params['Lx'] **2

    # --- Define file paths for saved models ---
    model_storage_dir = "DMRG/trained_models"
    os.makedirs(model_storage_dir, exist_ok=True)
    dmrg_filename = os.path.join(model_storage_dir, f"dmrg_L{model_params['Lx']}_J2_{model_params['J2']}.pkl.gz")
    rbm_filename = os.path.join(model_storage_dir, f"rbm_L{model_params['Lx']}_J2_{model_params['J2']}.mpack")


    # --- DMRG ---
    hamiltonian = J1J2Heisenberg(model_params=model_params)
    DMRG_vstate = DMRG_vstate_optimization(hamiltonian, model_params, filename=dmrg_filename)
    
    # --- RBM ---
    RBM_vstate, ha = RBM_vstate_optimization(model_params, n_iter, filename=rbm_filename) # ha is the NetKet Hamiltonian for exact diagonalization
    
    # --- Importance Sampling ---

    ops_z = ['Sigmaz'] * N_sites  # or just 'Sigmaz' if measuring all sites
    samples = np.zeros((n_samples, N_sites), dtype=int)
    psi_DMRG_sampled = np.zeros(n_samples, dtype=np.complex128) 
    for n in range(n_samples):
        sigmas, psi_DMRG = DMRG_vstate.sample_measurements(first_site=0, last_site=N_sites-1, ops = ops_z, complex_amplitude=True)
        samples[n, :] = sigmas
        psi_DMRG_sampled[n] = psi_DMRG
        

    # Convert samples from {-1, 1} basis (NetKet/Sigmaz) to {0, 1} basis (TenPy/DMRG)
    # This is crucial for functions expecting TenPy-like configurations.
    # +1 -> 0 (up), -1 -> 1 (down)
    samples_netket = samples # Keep original for RBM logpsi if needed
    samples_dmrg_01_basis = ((1 - np.asarray(samples_netket)) / 2).astype(int)
    

    # Evaluate RBM amplitudes on the sampled configurations (NetKet format)
    # Note: logpsi_netket expects samples in {-1, 1} basis, so use samples_netket
    logval = RBM_vstate.log_value(samples)
    logpsi_RBM_sampled = np.array(logval, dtype=np.complex128)
    psi_RBM_sampled = np.exp(logpsi_RBM_sampled)

    # --- Compute full wavefunction arrays for fidelity calculations ---
    dmrg_array = get_full_wavefunction(DMRG_vstate, undo_sort_charge=True)
    RBM_array = RBM_vstate.to_array()

    # --- Fidelity with DMRG and RBM ---
    fidelity_exact_rbm_dmrg = Fidelity_exact(RBM_vstate, DMRG_vstate)
    print("\nFidelity exact (RBM_Full vs DMRG_Full):", fidelity_exact_rbm_dmrg)
    fidelity_sampled_rbm_dmrg = Fidelity_sampled(psi_DMRG_sampled, psi_RBM_sampled)
    print("\nFidelity sampled (RBM_sampled vs DMRG_sampled):", fidelity_sampled_rbm_dmrg)
