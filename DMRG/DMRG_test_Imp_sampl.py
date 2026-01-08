import numpy as np
from DMRG.plot.Plotting import *

from jax import numpy as jnp
import matplotlib.pyplot as plt
import netket as nk
import flax
import os
import pickle
import gzip
import argparse

from DMRG.DMRG import *
from DMRG.Fidelities import *
from DMRG.Observable.Corr_Struct import Correlations_Structure_Factor

from Elaborate.Sign_Obs_MCMC import *
from Elaborate.Sign_Obs import *
from Elaborate.Sign_Obs_Importance import *




#%%
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--J2", type=float, default=0.5, help="J2 coupling")
    args = parser.parse_args()
    J2 = args.J2

    model_params = {
        'Lx': 6,
        'Ly': 6,
        'J1': 1.0,
        'J2': J2,
        'conserve': 'Sz'
    }

    n_samples = 1024
    n_iter = 500
    N_sites = model_params['Lx'] **2

    # --- Define file paths for saved models ---
    model_storage_dir = "/cluster/home/fconoscenti/Thesis_QSL/DMRG/trained_models"
    os.makedirs(model_storage_dir, exist_ok=True)
    dmrg_filename = os.path.join(model_storage_dir, f"dmrg_L{model_params['Lx']}_J2_{model_params['J2']}.pkl.gz")
    rbm_filename = os.path.join(model_storage_dir, f"rbm_L{model_params['Lx']}_J2_{model_params['J2']}.mpack")
    samples_filename = os.path.join(model_storage_dir, f"samples_L{model_params['Lx']}_J2_{model_params['J2']}.pkl")


    # --- DMRG ---
    hamiltonian = J1J2Heisenberg(model_params=model_params)
    
    if os.path.exists(dmrg_filename):
        print(f"Loading DMRG state from {dmrg_filename}")
        with gzip.open(dmrg_filename, 'rb') as f:
            DMRG_vstate = pickle.load(f)
        energy_per_site = [0.0]
        Correlations_Structure_Factor(DMRG_vstate, model_params, hamiltonian)
    else:
        DMRG_vstate, energy_per_site = DMRG_vstate_optimization(hamiltonian, model_params, filename=dmrg_filename)

        Correlations_Structure_Factor(DMRG_vstate, model_params, hamiltonian)
        results = {"final_energy_DMRG": energy_per_site[-1]}
        
        results_filename = os.path.join(model_storage_dir, f"final_energy_L{model_params['Lx']}_J2_{model_params['J2']}.pkl")
        with open(results_filename, "wb") as f:
            pickle.dump(results, f)
        print(f"Results saved to {results_filename}")
    
    
    # --- RBM ---
    #RBM_vstate, ha = RBM_vstate_optimization(model_params, n_iter, filename=rbm_filename) # ha is the NetKet Hamiltonian for exact diagonalization
    #hi = RBM_vstate.hilbert
    
    # --- Importance Sampling ---
    if os.path.exists(samples_filename):
        print(f"Loading samples from {samples_filename}")
        with open(samples_filename, 'rb') as f:
            data = pickle.load(f)
            samples = data['samples']
            psi_DMRG_sampled = data['psi_DMRG_sampled']
    else:
        print("Generating importance samples...")
        samples, psi_DMRG_sampled = importance_Sampling_DMRG(DMRG_vstate, n_samples, N_sites)
        print(f"Saving samples to {samples_filename}")
        with open(samples_filename, 'wb') as f:
            pickle.dump({'samples': samples, 'psi_DMRG_sampled': psi_DMRG_sampled}, f)
    


    
    """    
    samples_netket = samples # Keep original for RBM logpsi if needed
    samples_dmrg_01_basis = ((1 - np.asarray(samples_netket)) / 2).astype(int)


    # --- Full Fidelity DMRG and Exact gs ---
    #E_gs, ket_gs = nk.exact.lanczos_ed(ha, compute_eigenvectors=True)
    #fidelity_exact_dmrg_gs = fidelity_DMRG_exact(DMRG_vstate, ket_gs[:,0])
    #print("\nFidelity exact (DMRG vs gs):", fidelity_exact_dmrg_gs)

    # --- Full Fidelity DMRG and RBM ---
    #Fidelity_exact_rbm_dmrg = Fidelity_exact(RBM_vstate, DMRG_vstate)
    #print("Fidelity exact (RBM_Full vs DMRG_Full):", Fidelity_exact_rbm_dmrg)

    logval = RBM_vstate.log_value(samples)
    logpsi_RBM_sampled = np.array(logval, dtype=np.complex128)
    psi_RBM_sampled = np.exp(logpsi_RBM_sampled)    
    fidelity_sampled_rbm_dmrg = Fidelity_sampled(psi_DMRG_sampled, psi_RBM_sampled)
    print("Fidelity sampled (RBM_sampled vs DMRG_sampled):", fidelity_sampled_rbm_dmrg)

    # --- Observables ---
    #sign_exact_gs, signs = Marshall_Sign_exact(ket_gs, hi)
    #print("\nExact GS Marshall Sign:", sign_exact_gs)
    #sign_full_RBM, signs = Marshall_Sign_full_hilbert_one(RBM_vstate, hi)
    #print("\nFull Hilbert RBM Marshall Sign:", sign_full_RBM)

    # --- MCMC Sign ---
    SignObs = MarshallSignObs(hi)
    RBM_vstate.n_samples = n_samples
    sign_MCMC = RBM_vstate.expect(SignObs)
    print("MCMC RBM Marshall Sign:", sign_MCMC.mean)

    # --- Importance Sampled Sign RBM ---
    expectation = sign_NQS_importance_Sampled(RBM_vstate, samples_netket, hi)
    print("Importance Sampled RBM Marshall Sign (function):", expectation)

    # --- DMRG Sign Full Hilbert ---
    #sign_DMRG_full, psi_DMRG_full, sign_DMRG_full_expectation = Sign_DMRG_full_hilbert(DMRG_vstate, hi)
    #print("\nFull Hilbert DMRG Marshall Sign:", sign_DMRG_full_expectaion)

    # ---DMRG Sign on sampled configurations---
    sign_DMRG = sign_DMRG_importance_Sampled(DMRG_vstate, samples_netket)
    print("Importance Sampled DMRG Marshall Sign (function):", sign_DMRG)


    # --- DMRG RBM Sign Overlap on sampled configurations ---
    Overlap_sign_samples = sign_overlap_Importance_Sampled(RBM_vstate, DMRG_vstate, samples_netket, hi)
    print("Sign Overlap (DMRG vs RBM) on sampled configurations (function):", Overlap_sign_samples)

    # --- DMRG RBM Amplitudes Overlap on sampled configurations ---
    Overlap_amp_samples = amp_overlap_Importance_Sampled(RBM_vstate, psi_DMRG_sampled, samples_netket)
    print("Amp Overlap (DMRG vs RBM) on sampled configurations (function):", Overlap_amp_samples)


    """
