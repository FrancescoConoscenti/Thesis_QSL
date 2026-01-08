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
import re

from DMRG.DMRG import *
from DMRG.Fidelities import *

from Elaborate.Sign_Obs_MCMC import MarshallSignObs
from Elaborate.Sign_Obs import MarshallSignOperator
from Elaborate.Sign_Obs_Importance import *

from HFDS_Heisenberg.HFDS_model_spin import HiddenFermion

try:
    from ViT_Heisenberg.ViT_model import ViT
except ImportError:
    print("Warning: ViT model not found.")


#%%
def Observable_Importance_sampling(J2, NQS_path=None, vstate=None):

    # --- Parameters ---
    Lx = 6
    Ly = 6
    model_params = {
        'Lx': Lx,
        'Ly': Ly,
        'J1': 1.0,
        'J2': J2,
        'conserve': 'Sz'
    }
    N_sites = Lx * Ly
    n_samples = 1024
    model_storage_dir = "DMRG/trained_models"
    os.makedirs(model_storage_dir, exist_ok=True)

    # --- Paths ---


    # --- DMRG ---
    hamiltonian = J1J2Heisenberg(model_params=model_params)
    dmrg_filename = f"DMRG/trained_models/dmrg_L{Lx}_J2_{J2}.pkl.gz"
    
    if os.path.exists(dmrg_filename):
        print(f"Loading DMRG state from {dmrg_filename}")
        with gzip.open(dmrg_filename, 'rb') as f:
            DMRG_vstate = pickle.load(f)
        energy_per_site = [0.0] # Placeholder if not re-running
    #else:
    #    DMRG_vstate, energy_per_site = DMRG_vstate_optimization(hamiltonian, model_params, filename=dmrg_filename)

    # --- NQS Loading ---
    hi = nk.hilbert.Spin(s=1/2, N=N_sites)
    HFDS_vstate = None
    ViT_vstate = None

    if vstate is not None:
        print("Using provided vstate.")
        # Attempt to identify model type
        model_cls_str = str(type(vstate.model))
        if 'HiddenFermion' in model_cls_str:
            HFDS_vstate = vstate
        elif 'ViT' in model_cls_str:
            ViT_vstate = vstate
        else:
            print(f"Warning: Could not identify vstate model type ({model_cls_str}).")

    elif NQS_path is not None:
        # Check for HFDS
        match_hfds = re.search(r'layers(\d+)_hidd(\d+)_feat(\d+)', NQS_path)
        # Check for ViT
        match_vit = re.search(r'layers(\d+)_d(\d+)_heads(\d+)_patch(\d+)', NQS_path)

        if match_hfds and 'HiddenFermion' in globals():
            n_layers = int(match_hfds.group(1))
            n_hidden = int(match_hfds.group(2))
            n_features = int(match_hfds.group(3))
            print(f"Parsed HFDS params: layers={n_layers}, hidden={n_hidden}, features={n_features}")
            model = HiddenFermion(n_layers=n_layers, n_hidden=n_hidden, n_features=n_features)
            sampler = nk.sampler.MetropolisLocal(hi)
            HFDS_vstate = nk.vqs.MCState(sampler, model, n_samples=n_samples)
            
            if os.path.exists(NQS_path):
                print(f"Loading HFDS parameters from {NQS_path}")
                with open(NQS_path, 'rb') as f:
                    HFDS_vstate.variables = flax.serialization.from_bytes(HFDS_vstate.variables, f.read())
        
        elif match_vit and 'ViT' in globals():
            n_layers = int(match_vit.group(1))
            d_model = int(match_vit.group(2))
            n_heads = int(match_vit.group(3))
            patch_size = int(match_vit.group(4))
            print(f"Parsed ViT params: layers={n_layers}, d={d_model}, heads={n_heads}, patch={patch_size}")
            model_vit = ViT(n_layers=n_layers, d_model=d_model, n_heads=n_heads, patch_size=patch_size)
            sampler_vit = nk.sampler.MetropolisLocal(hi)
            ViT_vstate = nk.vqs.MCState(sampler_vit, model_vit, n_samples=n_samples)
            
            if os.path.exists(NQS_path):
                print(f"Loading ViT parameters from {NQS_path}")
                with open(NQS_path, 'rb') as f:
                    ViT_vstate.variables = flax.serialization.from_bytes(ViT_vstate.variables, f.read())
        else:
            print(f"Could not identify model type from path: {NQS_path}")

    # --- Importance Sampling ---
    samples, psi_DMRG_sampled = importance_Sampling_DMRG(DMRG_vstate, n_samples, N_sites)
    
    samples_netket = samples # Keep original for RBM logpsi if needed
    samples_dmrg_01_basis = ((1 - np.asarray(samples_netket)) / 2).astype(int)

    if HFDS_vstate is not None:
        logval = HFDS_vstate.log_value(samples)
        logpsi_RBM_sampled = np.array(logval, dtype=np.complex128)
        psi_RBM_sampled = np.exp(logpsi_RBM_sampled)    
        fidelity_sampled_rbm_dmrg = Fidelity_sampled(psi_DMRG_sampled, psi_RBM_sampled)
        print("Fidelity sampled (RBM_sampled vs DMRG_sampled):", fidelity_sampled_rbm_dmrg)

    if ViT_vstate is not None:
        logval_vit = ViT_vstate.log_value(samples)
        logpsi_ViT_sampled = np.array(logval_vit, dtype=np.complex128)
        psi_ViT_sampled = np.exp(logpsi_ViT_sampled)
        fidelity_sampled_vit_dmrg = Fidelity_sampled(psi_DMRG_sampled, psi_ViT_sampled)
        print("Fidelity sampled (ViT_sampled vs DMRG_sampled):", fidelity_sampled_vit_dmrg)

    # --- MCMC Sign ---
    SignObs = MarshallSignObs(hi)

    if HFDS_vstate is not None:
        HFDS_vstate.n_samples = n_samples
        sign_HFDS_MCMC = HFDS_vstate.expect(SignObs)
        print("MCMC HFDS Marshall Sign:", sign_HFDS_MCMC.mean)
        
        # --- Importance Sampled Sign HFDS ---
        sign_HFDS_Imp = sign_NQS_importance_Sampled(HFDS_vstate, samples_netket, hi)
        print("Importance Sampled HFDS Marshall Sign (function):", sign_HFDS_Imp)

    if ViT_vstate is not None:
        ViT_vstate.n_samples = n_samples
        sign_ViT_MCMC = ViT_vstate.expect(SignObs)
        print("MCMC ViT Marshall Sign:", sign_ViT_MCMC.mean)
        
        sign_ViT_Imp  = sign_NQS_importance_Sampled(ViT_vstate, samples_netket, hi)
        print("Importance Sampled ViT Marshall Sign (function):",  sign_ViT_Imp)


    # ---DMRG Sign on sampled configurations---
    sign_DMRG = sign_DMRG_importance_Sampled(DMRG_vstate, samples_netket)
    print("Importance Sampled DMRG Marshall Sign (function):", sign_DMRG)


    if HFDS_vstate is not None:
        # --- DMRG RBM Sign Overlap on sampled configurations ---
        Overlap_sign_HFDS_DMRG = sign_overlap_Importance_Sampled(HFDS_vstate, DMRG_vstate, samples_netket, hi)
        print("Sign Overlap (DMRG vs HFDS) on sampled configurations (function):", Overlap_sign_HFDS_DMRG)

        # --- DMRG HFDS Amplitudes Overlap on sampled configurations ---
        Overlap_amp_HFDS_DMRG = amp_overlap_Importance_Sampled(HFDS_vstate, psi_DMRG_sampled, samples_netket)
        print("Amp Overlap (DMRG vs HFDS) on sampled configurations (function):", Overlap_amp_HFDS_DMRG)

    if ViT_vstate is not None:
        Overlap_sign_ViT_DMRG = sign_overlap_Importance_Sampled(ViT_vstate, DMRG_vstate, samples_netket, hi)
        print("Sign Overlap (DMRG vs ViT) on sampled configurations (function):", Overlap_sign_ViT_DMRG)

        Overlap_amp_ViT_DMRG = amp_overlap_Importance_Sampled(ViT_vstate, psi_DMRG_sampled, samples_netket)
        print("Amp Overlap (DMRG vs ViT) on sampled configurations (function):", Overlap_amp_ViT_DMRG)


    results = {
        "final_energy_DMRG": energy_per_site[-1],
        "sign_DMRG_Imp": sign_DMRG,
    }
    
    if HFDS_vstate is not None:
        results.update({
            "fidelity_sampled_NQS_DMRG": fidelity_sampled_rbm_dmrg,
            "sign_NQS_MCMC": sign_HFDS_MCMC.mean,
            "sign_NQS_Imp": sign_HFDS_Imp,
            "Overlap_sign_NQS_DMRG": Overlap_sign_HFDS_DMRG,
            "Overlap_amp_NQS_DMRG": Overlap_amp_HFDS_DMRG
        })

    if ViT_vstate is not None:
        results.update({
            "fidelity_sampled_NQS_DMRG": fidelity_sampled_vit_dmrg,
            "sign_NQS_MCMC": sign_ViT_MCMC.mean,
            "sign_NQS_Imp": sign_ViT_Imp,
            "Overlap_sign_NQS_DMRG": Overlap_sign_ViT_DMRG,
            "Overlap_amp_NQS_DMRG": Overlap_amp_ViT_DMRG
        })
    
    results_filename = os.path.join(model_storage_dir, f"results_L{model_params['Lx']}_J2_{model_params['J2']}.pkl")
    with open(results_filename, "wb") as f:
        pickle.dump(results, f)
    print(f"Results saved to {results_filename}")

    return results
