
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
from Elaborate.Statistics.Energy import Energy, Exact_gs_en_6x6
from Elaborate.Statistics.Corr_Struct import Corr_Struct, Corr_Struct_Exact
from Elaborate.Statistics.Error_Stat import Relative_Error, Variance, Vscore, Magnetization, Exact_gs
from Elaborate.Statistics.count_params import vit_param_count, hidden_fermion_param_count
from Elaborate.Plotting.Sign_vs_iteration import *
from Elaborate.Plotting.S_matrix_vs_iteration import plot_S_matrix_eigenvalues, calculate_relevant_eigenvalues, Plot_S_matrix_histogram
from Elaborate.Sign_Obs_MCMC import MarshallSignObs
from DMRG.DMRG_NQS_Imp_sampl import Observable_Importance_sampling, Fidelity_vs_Iterations
from DMRG.Fidelities import Fidelity_sampled, Sign_Overlap_sampled, Amplitude_Overlap_sampled

from Entanglement.Entanglement import compute_renyi2_entropy

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
                data = f.read()
                try:
                    vstate = flax.serialization.from_bytes(vstate, data)
                except KeyError:
                    vstate.variables = flax.serialization.from_bytes(vstate.variables, data)
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

    variables = {}
    
    # Correlation function
    vstate.n_samples = 1024
    R = Corr_Struct(lattice, vstate, L, folder, hilbert)

    variables.update({
        'R': R,
    })
    print(f"Correlation Ratio R = {R}")
    

    if L == 4:
        # Exact
        E_exact, ket_gs = Exact_gs(L, J2, hamiltonian, J1J2=True, spin=True)
    elif L == 6:
        E_exact = Exact_gs_en_6x6(J2)
        ket_gs = None
    else:
        E_exact = None
        ket_gs = None


    if log is not None:
        E_vs_final = Energy(log, L, folder_energy, E_exact=E_exact)

    if log is None:
        E_vs_final = vstate.expect(hamiltonian).mean.real
        

    # Magn
    Magnetization(vstate, lattice, hilbert)
    
    if log is not None:
        # Variance
        variance = Variance(log, folder_energy)
        
        # Vscore
        vscore = Vscore(L, variance, E_vs_final)
    
    # count Params
    if params['model_type'] == 'ViT':
        count_params = vit_param_count(params['n_heads'], params['num_layers'], params['patch_size'], params['d_model'], L*L)
        
    elif params['model_type'] == 'HFDS':
        count_params = hidden_fermion_param_count(L*L, params['n_hid'], L, L, params['layers'], params['features'])
        

    variables.update({
        'E_vs_final': E_vs_final,
        'params': count_params,
        'vscore': vscore,
        'variance': variance
    })

    print(f"Final Energy per site: {E_vs_final} (Exact: {E_exact})")
    print(f"Variance: {variance}")
    print(f"V-score: {vscore}")
    print(f"Number of parameters: {count_params}")

    # Renyi Entropy S2
    n_samples = 4096
    s2, s2_error = compute_renyi2_entropy(vstate, n_samples=n_samples)

    variables.update({
            's2': s2,
            's2_error': s2_error,
    })
    print(f"Renyi S2 = {s2} ± {s2_error} (n_samples={n_samples})")
    

    #QGT
    all_eigenvalues, relevant_count_first, mean_rest_ratio, mean_rest_norm = calculate_relevant_eigenvalues(vstate, folder, hilbert, threshold_ratio_rest=1e-2)
    Plot_S_matrix_histogram(all_eigenvalues, folder, one_avg = "one")
    plot_S_matrix_eigenvalues(vstate, folder, hilbert, one_avg = "one")
    variables.update({
            'eigenvalues_S': all_eigenvalues,
            'mean_rest_ratio': mean_rest_ratio,
            'mean_rest_norm': mean_rest_norm,
            'relevant_count_first': relevant_count_first,
    })
    
    print(f"QGT relevant eigenvalues - first: {relevant_count_first}, mean rest ratio: {mean_rest_ratio}, mean rest norm: {mean_rest_norm}")

    #Sign MCMC
    n_samples = 4096
    vstate.n_samples = n_samples
    sign_op = MarshallSignObs(hilbert)
    sign_MCMC = vstate.expect(sign_op)

    variables.update({
            'sign_vstate_MCMC': sign_MCMC.mean,
    })

    print(f"Marshall Sign (MCMC): {sign_MCMC.mean} ± {sign_MCMC.variance**0.5}")
        

    with open(os.path.join(folder, "variables.pkl"), 'wb') as f:
        pickle.dump(variables, f) 

 
    ################################################################################################

    
    #Rel Err
    if L == 6 or L ==4:
        rel_err_E = Relative_Error(E_vs_final, E_exact, L)

    variables.update({
        'rel_err_E': rel_err_E,
        'E_exact': E_exact
    })


    if L == 4 and ket_gs is not None:
        # Fidelity
        fidelity = Fidelity(vstate, ket_gs)
        print(f"Fidelity <vstate|exact> = {fidelity}")

        configs, sign_vstate_config, weight_exact, weight_vstate = plot_Sign_single_config(ket_gs, vstate, hilbert, 3, L, folder, one_avg = "one")
        configs, sign_vstate_config, weight_exact, weight_vstate = plot_Weight_single(ket_gs, vstate, hilbert, 8, L, folder, one_avg = "one")
        amp_overlap, fidelity, sign_vstate, sign_exact, sign_overlap = plot_Sign_Err_Amplitude_Err_Fidelity(ket_gs, vstate, hilbert, folder, one_avg = "one")
        amp_overlap, sign_vstate, sign_exact, sign_overlap = plot_Sign_Err_vs_Amplitude_Err_with_iteration(ket_gs, vstate, hilbert, folder, one_avg = "one")
        sorted_weights, sorted_amp_overlap, sorted_sign_overlap = plot_Overlap_vs_Weight(ket_gs, vstate, hilbert, folder, "one")
        eigenvalues, rel_1, rel_2, rel_3 = plot_S_matrix_eigenvalues(vstate, folder, hilbert, one_avg = "one")

        variables.update({

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
                'rank_S': rel_1,
                'params': count_params
            })
        
        with open(os.path.join(folder, "variables.pkl"), 'wb') as f:
            pickle.dump(variables, f)                   

    elif L == 6:
        print("6x6")
        
        """
        #DMRG Observables via Importance Samplings
        results = Observable_Importance_sampling(J2, NQS_path=None, vstate=vstate)
        Fidelity_vs_Iterations(folder, vstate, params)

        variables.update({
                'final_energy_DMRG': results['final_energy_DMRG'],
                'sign_DMRG_Imp': results['sign_DMRG_Imp'],
                'fidelity_sampled_NQS_DMRG': results['fidelity_sampled_NQS_DMRG'],
                'sign_NQS_MCMC': results['sign_NQS_MCMC'],
                'sign_NQS_Imp': results['sign_NQS_Imp'],
                'Overlap_sign_NQS_DMRG': results['Overlap_sign_NQS_DMRG'],
                'Overlap_amp_NQS_DMRG': results['Overlap_amp_NQS_DMRG'],
        })
        """

        with open(os.path.join(folder, "variables.pkl"), 'wb') as f:
            pickle.dump(variables, f)                   

    sys.stdout.close()

if __name__ == "__main__":

    model_path = "/scratch/f/F.Conoscenti/Thesis_QSL/ViT_Heisenberg/plot/6x6/layers2_d24_heads4_patch2_sample1024_lr0.0075_iter500_parityTrue_rotTrue_latest_model"
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
