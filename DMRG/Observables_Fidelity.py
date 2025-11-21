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
from Elaborate.Sign_Obs import *
import tempfile
import shutil
from Elaborate.Sign_Obs_MCMC import *
from Elaborate.Sign_Obs import *



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
    hi = RBM_vstate.hilbert
    
    # --- Importance Sampling ---

    ops_z = ['Sigmaz'] * N_sites  # or just 'Sigmaz' if measuring all sites
    samples = np.zeros((n_samples, N_sites), dtype=int)
    psi_DMRG_sampled = np.zeros(n_samples, dtype=np.complex128) 
    for n in range(n_samples):
        sigmas, psi_DMRG = DMRG_vstate.sample_measurements(first_site=0, last_site=N_sites-1, ops = ops_z, complex_amplitude=True)
        samples[n, :] = sigmas
        psi_DMRG_sampled[n] = psi_DMRG

    samples_netket = samples # Keep original for RBM logpsi if needed
    samples_dmrg_01_basis = ((1 - np.asarray(samples_netket)) / 2).astype(int)

    # --- Observables ---
    E_gs, ket_gs = nk.exact.lanczos_ed(ha, compute_eigenvectors=True)

    sign_exact_gs, signs = Marshall_Sign_exact(ket_gs, hi)
    sign_full_RBM, signs = Marshall_Sign_full_hilbert_one(RBM_vstate, hi)

    RBM_vstate.n_samples = n_samples
    SignObs = MarshallSignObs(hi)
    sign_MCMC = RBM_vstate.expect(SignObs)

    # --- Importance Sampled Sign ---
    SignObs = MarshallSignObs(hi)
    kernel = nk.vqs.get_local_kernel(RBM_vstate, SignObs)
    sigma_template, args_template = nk.vqs.get_local_kernel_arguments(RBM_vstate, SignObs)

    logpsi_vals = RBM_vstate.log_value(samples_netket)

    local_vals = kernel(logpsi_vals, RBM_vstate.parameters, samples_netket, args_template)

    print(sign_exact_gs, sign_full_RBM, sign_MCMC.mean, local_vals.mean())

    
