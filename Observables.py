
import argparse
import os
import sys
import pickle
import re
import jax
import netket as nk
import gzip
import flax
from flax import linen as nn
import numpy as np
import matplotlib.pyplot as plt

# Add path to project root
sys.path.append("/scratch/f/F.Conoscenti/Thesis_QSL")

# Imports
from ViT_Heisenberg.ViT_model import ViT_sym
from HFDS_Heisenberg.HFDS_model_spin import HiddenFermion
from Elaborate.Statistics.Energy import *
from Elaborate.Statistics.Corr_Struct import *
from Elaborate.Statistics.Error_Stat import *
from Elaborate.Statistics.count_params import *
from Elaborate.Plotting.Sign_vs_iteration import *
from Elaborate.Plotting.S_matrix_vs_iteration import *
from Elaborate.Sign_Obs import *
from DMRG.DMRG_NQS_Imp_sampl import Observable_Importance_sampling
from DMRG.Fidelities import Fidelity_sampled, Sign_Overlap_sampled, Amplitude_Overlap_sampled

# Mock class for log if not available
class MockLog:
    def __init__(self, data):
        self.data = data

def parse_model_path(model_path):
    params = {}
    # Extract parameters from path
    params['L'] = 4
    if "6x6" in model_path: params['L'] = 6
    
    match_J = re.search(r"J=([\d\.]+)", model_path)
    params['J2'] = float(match_J.group(1)) if match_J else 0.5
    
    if "hidd" in model_path:
        params['model_type'] = 'HFDS'
        params['n_hid'] = int(re.search(r"hidd(\d+)", model_path).group(1))
        params['features'] = int(re.search(r"feat(\d+)", model_path).group(1))
        params['layers'] = int(re.search(r"layers(\d+)", model_path).group(1))
        
        match_init = re.search(r"Init((?:(?!_type)[a-zA-Z_])+)", model_path)
        params['MFinit'] = match_init.group(1) if match_init else "random"
        
        match_type = re.search(r"type([a-zA-Z]+)", model_path)
        params['dtype'] = match_type.group(1) if match_type else "complex"
    else:
        params['model_type'] = 'ViT'
        params['num_layers'] = int(re.search(r"layers(\d+)", model_path).group(1)) if re.search(r"layers(\d+)", model_path) else 2
        params['d_model'] = int(re.search(r"_d(\d+)", model_path).group(1)) if re.search(r"_d(\d+)", model_path) else 8
        params['n_heads'] = int(re.search(r"heads(\d+)", model_path).group(1)) if re.search(r"heads(\d+)", model_path) else 4
        params['patch_size'] = int(re.search(r"patch(\d+)", model_path).group(1)) if re.search(r"patch(\d+)", model_path) else 2
    
    return params

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

def run_observables(log, folder):
    folder_energy = os.path.join(folder, "Energy_plot")
    os.makedirs(folder_energy, exist_ok=True)

    os.makedirs(os.path.join(folder, "physical_obs"), exist_ok=True)
    os.makedirs(os.path.join(folder, "Sign_plot"), exist_ok=True)
    
    sys.stdout = open(os.path.join(folder, "output.txt"), "w")

    params = parse_model_path(folder)
    L = params['L']
    J2 = params['J2']
    print(f"Loaded params: {params}")

    # Setup Hilbert/Hamiltonian
    n_dim = 2
    lattice = nk.graph.Hypercube(length=L, n_dim=n_dim, pbc=True, max_neighbor_order=2)
    hilbert = nk.hilbert.Spin(s=1 / 2, N=lattice.n_nodes, total_sz=0)
    hamiltonian = nk.operator.Heisenberg(
        hilbert=hilbert, graph=lattice, J=[1.0, J2], sign_rule=[False, False]
    ).to_jax_operator()

    # Setup Model
    if params['model_type'] == 'ViT':
        model = ViT_sym(
            L=L,
            num_layers=params['num_layers'], 
            d_model=params['d_model'], 
            n_heads=params['n_heads'], 
            patch_size=params['patch_size'], 
            transl_invariant=True, 
            parity=True, 
            rotation=True
        )
    elif params['model_type'] == 'HFDS':
        dtype_ = jnp.float64 if params['dtype'] == "real" else jnp.complex128
        model = HiddenFermion(
            L=L,
            network="FFNN",
            n_hid=params['n_hid'],
            layers=params['layers'],
            features=params['features'],
            MFinit=params['MFinit'],
            hilbert=hilbert,
            stop_grad_mf=False,
            stop_grad_lower_block=False,
            bounds="PBC",
            parity=True,
            rotation=True,
            dtype=dtype_
        )

    # Setup Sampler and VState
    sampler = nk.sampler.MetropolisExchange(
        hilbert=hilbert,
        graph=lattice,
        d_max=2,
        n_chains=1024,
        sweep_size=lattice.n_nodes,
    )
    
    # Initialize with dummy parameters to get structure
    key = jax.random.PRNGKey(0)
    vstate = nk.vqs.MCState(
        sampler=sampler,
        model=model,
        n_samples=1024,
        n_discard_per_chain=16,
        seed=key
    )
    
    # Load trained parameters
    models_dir = os.path.join(folder, "models")
    if os.path.exists(models_dir):
        files = [f for f in os.listdir(models_dir) if f.endswith(".mpack")]
        if files:
            # Sort by iteration number
            files.sort(key=lambda x: int(re.search(r"model_(\d+)", x).group(1)))
            last_model = files[-1]
            with open(os.path.join(models_dir, last_model), 'rb') as f:
                vstate.variables = flax.serialization.from_bytes(vstate.variables, f.read())
            print(f"Loaded model: {last_model}")
        else:
            print("No model files found in models directory.")
            sys.stdout.close()
            return
    else:
        print(f"Models directory not found at {models_dir}")
        sys.stdout.close()
        return


    # --- Observables Calculation ---
    
    # Correlation function
    vstate.n_samples = 1024
    Corr_Struct(lattice, vstate, L, folder, hilbert)
    
    if L == 4:
        # Exact
        E_exact, ket_gs = Exact_gs(L, J2, hamiltonian, J1J2=True, spin=True)
    elif L == 6:
        E_exact = Exact_gs_en_6x6(J2)
        ket_gs = None

    if log is not None:
        E_vs_final = Energy(log, L, folder_energy, E_exact=E_exact)
        rel_err_E = Relative_Error(E_vs_final, E_exact, L)

    if log is None:
        E_vs_final = vstate.expect(hamiltonian).mean.real
        rel_err_E = Relative_Error(E_vs_final, E_exact, L)
        
    # Magn
    Magnetization(vstate, lattice, hilbert)
    
    if log is not None:
        # Variance
        variance = Variance(log, folder_energy)
        
        # Vscore
        Vscore(L, variance, E_vs_final)
    
    # count Params
    if params['model_type'] == 'ViT':
        count_params = vit_param_count(params['n_heads'], params['num_layers'], params['patch_size'], params['d_model'], L*L)
        print(f"params={count_params}")
    elif params['model_type'] == 'HFDS':
        hidden_fermion_param_count(L*L, params['n_hid'], L, L, params['layers'], params['features'])

    if L == 4 and ket_gs is not None:
        # Fidelity
        fidelity = Fidelity(vstate, ket_gs)
        print(f"Fidelity <vstate|exact> = {fidelity}")

        configs, sign_vstate_config, weight_exact, weight_vstate = plot_Sign_single_config(ket_gs, vstate, hilbert, 3, L, folder, one_avg = "one")
        configs, sign_vstate_config, weight_exact, weight_vstate = plot_Weight_single(ket_gs, vstate, hilbert, 8, L, folder, one_avg = "one")
        amp_overlap, fidelity, sign_vstate, sign_exact, sign_overlap = plot_Sign_Err_Amplitude_Err_Fidelity(ket_gs, vstate, hilbert, folder, one_avg = "one")
        amp_overlap, sign_vstate, sign_exact, sign_overlap = plot_Sign_Err_vs_Amplitude_Err_with_iteration(ket_gs, vstate, hilbert, folder, one_avg = "one")
        sorted_weights, sorted_amp_overlap, sorted_sign_overlap = plot_Overlap_vs_Weight(ket_gs, vstate, hilbert, folder, "one")
        eigenvalues, rank = plot_S_matrix_eigenvalues(vstate, folder, hilbert,  part_training = "end", one_avg = "one")

        variables = {
                'E_exact': E_exact,
                'E_vs_final': E_vs_final,
                'rel_err_E': rel_err_E,
                'sign_vstate': sign_vstate,
                'sign_exact': sign_exact,
                'fidelity': fidelity,
                'configs': configs,
                'sign_vstate_config': sign_vstate_config,
                'weight_exact': weight_exact,
                'weight_vstate': weight_vstate,
                'amp_overlap': amp_overlap,
                'sign_overlap': sign_overlap,
                'eigenvalues_S': eigenvalues,
                'rank_S': rank,
                'params': count_params
            }
        
        with open(os.path.join(folder, "variables.pkl"), 'wb') as f:
            pickle.dump(variables, f)                   

    elif L == 6:
        print("6x6")
        Fidelity_vs_Iterations(folder, vstate, params)
        eigenvalues, rank = plot_S_matrix_eigenvalues(vstate, folder, hilbert,  part_training = "end", one_avg = "one")
        
        variables = {

                'E_exact': E_exact,
                'E_vs_final': E_vs_final,
                'rel_err_E': rel_err_E,
                'eigenvalues_S': eigenvalues,
                'rank_S': rank,
                'params': count_params
        }

        results = Observable_Importance_sampling(J2, NQS_path=None, vstate=vstate)
        
        variables.update({
                'final_energy_DMRG': results['final_energy_DMRG'],
                'sign_DMRG_Imp': results['sign_DMRG_Imp'],
                'fidelity_sampled_NQS_DMRG': results['fidelity_sampled_NQS_DMRG'],
                'sign_NQS_MCMC': results['sign_NQS_MCMC'],
                'sign_NQS_Imp': results['sign_NQS_Imp'],
                'Overlap_sign_NQS_DMRG': results['Overlap_sign_NQS_DMRG'],
                'Overlap_amp_NQS_DMRG': results['Overlap_amp_NQS_DMRG'],
        })

        with open(os.path.join(folder, "variables.pkl"), 'wb') as f:
            pickle.dump(variables, f)                   


    sys.stdout.close()

if __name__ == "__main__":

    model_path = "/scratch/f/F.Conoscenti/Thesis_QSL/ViT_Heisenberg/plot/6x6/layers2_d16_heads4_patch2_sample1024_lr0.0075_iter500_parityTrue_rotTrue_latest_model"
    log = None

    if not os.path.exists(model_path):
        model_path = model_path.replace("/cluster/home/fconoscenti/Thesis_QSL", "/scratch/f/F.Conoscenti/Thesis_QSL")

    if os.path.exists(model_path):
        j_folders = [f for f in os.listdir(model_path) if f.startswith("J=") and os.path.isdir(os.path.join(model_path, f))]
        try:
            j_folders.sort(key=lambda x: float(x.split('=')[1]))
        except:
            j_folders.sort()

        for j_folder in j_folders:
            j_path = os.path.join(model_path, j_folder)
            seed_folders = [f for f in os.listdir(j_path) if f.startswith("seed_") and os.path.isdir(os.path.join(j_path, f))]
            try:
                seed_folders.sort(key=lambda x: int(x.split('_')[1]))
            except:
                seed_folders.sort()

            for seed_folder in seed_folders:
                full_path = os.path.join(j_path, seed_folder)
                print(f"Running observables for: {full_path}")
                original_stdout = sys.stdout
                try:
                    run_observables(log, full_path)
                except Exception as e:
                    sys.stdout = original_stdout
                    print(f"Error processing {full_path}: {e}")
                finally:
                    sys.stdout = original_stdout
    else:
        print(f"Model path not found: {model_path}")
