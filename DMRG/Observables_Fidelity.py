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
from DMRG.plot.Plotting import *
from DMRG.Observable.Corr_Struct import Correlations_Structure_Factor

from netket.experimental.driver import VMC_SR
from scipy.sparse.linalg import eigsh


from jax import numpy as jnp
import matplotlib.pyplot as plt
import netket as nk
import flax
import os

from DMRG.DMRG import *
from DMRG.Fidelities import *
from Elaborate.Sign_Obs import *
from Elaborate.Sign_Obs_MCMC import *
from Elaborate.Sign_Obs import *



#%%
if __name__ == "__main__":

    model_params = {
        'Lx': 4,
        'Ly': 4,
        'J1': 1.0,
        'J2': 0.0
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


    # --- Fidelity DMRG and Exact gs ---
    E_gs, ket_gs = nk.exact.lanczos_ed(ha, compute_eigenvectors=True)
    fidelity_exact_dmrg_gs = fidelity_DMRG_exact(DMRG_vstate, ket_gs[:,0])
    print("\nFidelity exact (DMRG vs gs):", fidelity_exact_dmrg_gs)

    # --- Fidelity DMRG and RBM ---
    Fidelity_exact_rbm_dmrg = Fidelity_exact(RBM_vstate, DMRG_vstate)
    print("Fidelity exact (RBM_Full vs DMRG_Full):", Fidelity_exact_rbm_dmrg)

    logval = RBM_vstate.log_value(samples)
    logpsi_RBM_sampled = np.array(logval, dtype=np.complex128)
    psi_RBM_sampled = np.exp(logpsi_RBM_sampled)    
    fidelity_sampled_rbm_dmrg = Fidelity_sampled(psi_DMRG_sampled, psi_RBM_sampled)
    print("Fidelity sampled (RBM_sampled vs DMRG_sampled):", fidelity_sampled_rbm_dmrg)

    # --- Observables ---
    sign_exact_gs, signs = Marshall_Sign_exact(ket_gs, hi)
    print("\nExact GS Marshall Sign:", sign_exact_gs)
    sign_full_RBM, signs = Marshall_Sign_full_hilbert_one(RBM_vstate, hi)
    print("\nFull Hilbert RBM Marshall Sign:", sign_full_RBM)

    # --- MCMC Sign ---
    SignObs = MarshallSignObs(hi)
    RBM_vstate.n_samples = n_samples
    sign_MCMC = RBM_vstate.expect(SignObs)
    print("MCMC RBM Marshall Sign:", sign_MCMC.mean)

    # --- Importance Sampled Sign RBM ---
    SignObs = MarshallSignObs(hi)
    kernel = nk.vqs.get_local_kernel(RBM_vstate, SignObs)
    sigma_template, args_template = nk.vqs.get_local_kernel_arguments(RBM_vstate, SignObs)
    logpsi_vals = RBM_vstate.log_value(samples_netket)
    sign_RBM_samples = kernel(logpsi_vals, RBM_vstate.parameters, samples_netket, args_template)
    
    prob_samples = jnp.exp(2.0 * jnp.abs(logpsi_vals))
    expectation = jnp.sum(prob_samples * sign_RBM_samples.reshape(-1)) / jnp.sum(prob_samples) # weighted mean
    print("Importance Sampled RBM Marshall Sign:", expectation)

    # --- DMRG Sign Full Hilbert ---
    """
    sign_DMRG_full, psi_DMRG_full = Sign_DMRG_full_hilbert(DMRG_vstate, hi)
    prob_DMRG_full = np.abs(psi_DMRG_full) **2
    sign_DMRG_full = np.sum(prob_DMRG_full * sign_DMRG_full.reshape(-1)) / np.sum(prob_DMRG_full)
    print("\nFull Hilbert DMRG Marshall Sign:", sign_DMRG_full)
    """
    # ---DMRG Sign on sampled configurations---
    sign_DMRG_samples, psi_DMRG_sampled_1 = Sign_DMRG_samples(DMRG_vstate, samples_netket)
    prob_DMRG_samples = np.abs(psi_DMRG_sampled_1) **2
    sign_DMRG = np.sum(prob_DMRG_samples * sign_DMRG_samples.reshape(-1)) / np.sum(prob_DMRG_samples)
    print("Importance Sampled DMRG Marshall Sign:", sign_DMRG)

    # --- DMRG RBM Sign Overlap on sampled configurations ---
    Overlap_sign_samples = np.abs(np.sum(np.abs(psi_DMRG_sampled)**2 * sign_DMRG_samples.reshape(-1) * sign_RBM_samples.reshape(-1))) / np.sum(np.abs(psi_DMRG_sampled)**2)
    print("\nSign Overlap (DMRG vs RBM) on sampled configurations:", Overlap_sign_samples)


    # --- DMRG RBM Amplitudes Overlap on sampled configurations ---
    logpsi_vals = RBM_vstate.log_value(samples_netket)
    psi_RBM_samples = jnp.exp(logpsi_vals)
    # Calculate the norms (L2 norm) of each wavefunction
    norm_RBM = np.linalg.norm(psi_RBM_samples)
    norm_DMRG = np.linalg.norm(psi_DMRG_sampled)
    dot_product_mag = np.sum(np.abs(psi_RBM_samples) * np.abs(psi_DMRG_sampled))
    Overlap_amp_samples = dot_product_mag / (norm_RBM * norm_DMRG)
    print("\nAmp Overlap (DMRG vs RBM) on sampled configurations:", Overlap_amp_samples)

    
