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


def Fidelity_vs_Iterations(folder, vstate, params):
    print("\n--- Calculating Fidelity vs Iterations ---")
    L = params['L']
    J2 = params['J2']
    N_sites = L * L
    n_samples = 1024

    samples_filename = f"DMRG/trained_models/samples_L{L}_J2_{J2}.pkl"
    
    if os.path.exists(samples_filename):
        print(f"Loading samples from {samples_filename}")
        with open(samples_filename, 'rb') as f:
            data = pickle.load(f)
            samples = data['samples']
            psi_DMRG_sampled = data['psi_DMRG_sampled']
    else:
        # Load DMRG State
        dmrg_filename = f"DMRG/trained_models/dmrg_L{L}_J2_{J2}.pkl.gz"
        if not os.path.exists(dmrg_filename):
            print(f"DMRG file not found: {dmrg_filename}")
            return

        print(f"Loading DMRG state from {dmrg_filename}")
        with gzip.open(dmrg_filename, 'rb') as f:
            DMRG_vstate = pickle.load(f)

        # Sample from DMRG
        print(f"Generating {n_samples} samples from DMRG...")
        ops_z = ['Sigmaz'] * N_sites
        samples = np.zeros((n_samples, N_sites), dtype=int)
        psi_DMRG_sampled = np.zeros(n_samples, dtype=np.complex128)
        
        for n in range(n_samples):
            sigmas, psi_DMRG = DMRG_vstate.sample_measurements(first_site=0, last_site=N_sites-1, ops=ops_z, complex_amplitude=True)
            samples[n, :] = sigmas
            psi_DMRG_sampled[n] = psi_DMRG
            
        print(f"Saving samples to {samples_filename}")
        with open(samples_filename, 'wb') as f:
            pickle.dump({'samples': samples, 'psi_DMRG_sampled': psi_DMRG_sampled}, f)

    # Iterate over NQS models
    models_dir = os.path.join(folder, "models")
    if not os.path.exists(models_dir):
        print("Models directory not found.")
        return

    files = [f for f in os.listdir(models_dir) if f.endswith(".mpack")]
    files_with_iter = []
    for f in files:
        match = re.search(r"model_(\d+)", f)
        if match:
            files_with_iter.append((int(match.group(1)), f))
    
    files_with_iter.sort(key=lambda x: x[0])
    
    iterations = []
    fidelities = []
    amp = []
    sign = []
    
    for n_iter, filename in files_with_iter:
        filepath = os.path.join(models_dir, filename)
        with open(filepath, 'rb') as f:
            vstate.variables = flax.serialization.from_bytes(vstate.variables, f.read())
        
        log_values = vstate.log_value(samples)
        psi_RBM_sampled = np.exp(np.array(log_values))
        
        fid = Fidelity_sampled(psi_DMRG_sampled, psi_RBM_sampled)
        iterations.append(n_iter)
        fidelities.append(fid)

        amp_overlap = Amplitude_Overlap_sampled(psi_DMRG_sampled, psi_RBM_sampled)
        sign_overlap = Sign_Overlap_sampled(psi_DMRG_sampled, psi_RBM_sampled)
        amp.append(amp_overlap)
        sign.append(sign_overlap)   
        

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, fidelities, 'o-', label='Fidelity (Sampled)')
    plt.plot(iterations, amp, 's-', label='Amplitude Overlap')
    plt.plot(iterations, sign, '^-', label='Sign Overlap')
    plt.xlabel("Iterations")
    plt.ylabel("Overlap / Fidelity")
    plt.title(f"Fidelity & Overlaps vs Iterations (L={L}, J2={J2})")
    plt.grid(True)
    plt.legend()
    
    plot_dir = os.path.join(folder, "Fidelity_plot")
    os.makedirs(plot_dir, exist_ok=True)
    plt.savefig(os.path.join(plot_dir, "fidelity_vs_iter.png"))
    plt.close()
    
    np.savetxt(os.path.join(plot_dir, "fidelity_vs_iter.txt"), np.column_stack((iterations, fidelities, amp, sign)), header="Iter Fidelity Amp_Overlap Sign_Overlap")


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
