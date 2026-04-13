
import argparse
import os
import sys

# Force JAX to use CPU to avoid CUDA initialization errors
#os.environ["JAX_PLATFORM_NAME"] = "cpu"

import pickle
import re
import gc
from Elaborate.Statistics import count_params
import jax
import netket as nk
import gzip
import jax.numpy as jnp
import flax
from flax import linen as nn
import numpy as np
import matplotlib.pyplot as plt

# Add path to project root
sys.path.append("/scratch/f/F.Conoscenti/Thesis_QSL")

# Imports
from ViT_Heisenberg.ViT_model import ViT_sym
from HFDS_Heisenberg.HFDS_model_spin import HiddenFermion
from Elaborate.Statistics.Energy import Energy, Exact_gs_en_6x6, plot_energy
from Elaborate.Statistics.Corr_Struct import Corr_Struct, Corr_Struct_Exact
from Elaborate.Statistics.Error_Stat import Relative_Error, Variance, Vscore, Magnetization, Exact_gs, Autocorrelation_time, Rhat
from Elaborate.Statistics.count_params import vit_param_count, hidden_fermion_param_count
from Elaborate.Plotting.Old.Sign_vs_iteration import *
from Elaborate.Plotting.QGT.QGT_vs_iteration import plot_S_matrix_eigenvalues, calculate_relevant_eigenvalues, Plot_S_matrix_histogram, Plot_S_matrix_eigenvalues, plot_S_matrix_spectrum
from Elaborate.Sign_Obs_MCMC import MarshallSignObs
from DMRG.DMRG_NQS_Imp_sampl import Observable_Importance_sampling, Fidelity_vs_Iterations
from DMRG.Fidelities import Fidelity_sampled, Sign_Overlap_sampled, Amplitude_Overlap_sampled
from Elaborate.Sign_complexity import compute_sign_complexity
from Entanglement.Entanglement import compute_entanglement_scaling, plot_entanglement_scaling, compute_renyi2_entropy
from Entanglement.Entanglement_spectrum import plot_spectrum, compute_entanglement_spectrum_2d

from Hamiltonian import build_heisenberg_apbc, build_heisenberg_twisted

# Mock class for log if not available
class MockLog:
    def __init__(self, data):
        self.data = data

def parse_model_path(model_path):
    params = {}
    # Extract parameters from path
    if "4x4" in model_path: params['L'] = 4
    if "6x6" in model_path: params['L'] = 6
    if "8x8" in model_path: params['L'] = 8
    if "10x10" in model_path: params['L'] = 10
    
    match_J = re.search(r"J=([\d\.]+)", model_path)
    params['J2'] = float(match_J.group(1)) if match_J else 0.5
    
    match_phi = re.search(r"_phi([\d\.]+)_", model_path)
    if match_phi:
        params['phi'] = float(match_phi.group(1))
    else:
        params['phi'] = 0.0
    
    if "hidd" in model_path:
        params['model_type'] = 'HFDS'
        params['n_hid'] = int(re.search(r"hidd(\d+)", model_path).group(1))
        params['features'] = int(re.search(r"feat(\d+)", model_path).group(1))
        params['layers'] = int(re.search(r"layers(\d+)", model_path).group(1))
        
        match_init = re.search(r"Init((?:(?!_type)[a-zA-Z_])+)", model_path)
        params['MFinit'] = match_init.group(1) if match_init else "random"
        
        match_type = re.search(r"type([a-zA-Z]+)", model_path)
        params['dtype'] = match_type.group(1) if match_type else "complex"

        match_bc = re.search(r"bc([A-Z]+)_([A-Z]+)", model_path)
        if match_bc:
            params['bc_x'] = match_bc.group(1)
            params['bc_y'] = match_bc.group(2)
        else:
            match_bc_single = re.search(r"bc([A-Z]+)", model_path)
            if match_bc_single:
                params['bc_x'] = match_bc_single.group(1)
                params['bc_y'] = match_bc_single.group(1)
    else:
        params['model_type'] = 'ViT'
        params['num_layers'] = int(re.search(r"layers(\d+)", model_path).group(1)) if re.search(r"layers(\d+)", model_path) else 2
        params['d_model'] = int(re.search(r"_d(\d+)", model_path).group(1)) if re.search(r"_d(\d+)", model_path) else 8
        params['n_heads'] = int(re.search(r"heads(\d+)", model_path).group(1)) if re.search(r"heads(\d+)", model_path) else 4
        params['patch_size'] = int(re.search(r"patch(\d+)", model_path).group(1)) if re.search(r"patch(\d+)", model_path) else 2
    
    return params


def setup_environment(folder):
    folder_energy = os.path.join(folder, "Energy_plot")
    os.makedirs(folder_energy, exist_ok=True)
    os.makedirs(os.path.join(folder, "physical_obs"), exist_ok=True)
    os.makedirs(os.path.join(folder, "Sign_plot"), exist_ok=True)
    sys.stdout = open(os.path.join(folder, "output.txt"), "a")
    return folder_energy

def setup_system(L, J2, params=None):
    n_dim = 2
    # We must keep the graph periodic to have edges at the boundary
    lattice = nk.graph.Hypercube(length=L, n_dim=n_dim, pbc=[True, True], max_neighbor_order=2)
    hilbert = nk.hilbert.Spin(s=1 / 2, N=lattice.n_nodes, total_sz=0)
    
    bc_x = params.get("bc_x", "PBC") if params else "PBC"
    bc_y = params.get("bc_y", "PBC") if params else "PBC"
    phi = params.get("phi", 0.0) if params else 0.0

    """if bc_x == "PBC" and bc_y == "PBC":
        hamiltonian = nk.operator.Heisenberg(hilbert=hilbert, graph=lattice, J=[1.0, J2], sign_rule=[False, False]).to_jax_operator()
    else:
        hamiltonian = build_heisenberg_apbc(L, L, J1=1.0, J2=J2, apbc_x=(bc_x == "APC"), apbc_y=(bc_y == "APC")).to_jax_operator()"""
    
    print(f"Building Hamiltonian with L={L}, J2={J2}, bc_x={bc_x}, bc_y={bc_y}, phi={phi}")
    hamiltonian = build_heisenberg_twisted(L, L, J1=1.0, J2=J2, phi=phi, apbc_y=(bc_y == "APC")).to_jax_operator()
  
    return lattice, hilbert, hamiltonian

def setup_model(params, hilbert, L):
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
            bounds=(params.get('bc_x', 'PBC'), params.get('bc_y', 'PBC')),
            parity=True,
            rotation=True,
            dtype=dtype_
        )
    return model

def load_vstate(folder, sampler, model):
    key = jax.random.PRNGKey(0)
    vstate = nk.vqs.MCState(
        sampler=sampler,
        model=model,
        n_samples=1024,
        n_discard_per_chain=16,
        seed=key,
        chunk_size=512
    )
    
    models_dir = os.path.join(folder, "models")
    if os.path.exists(models_dir):
        files = [f for f in os.listdir(models_dir) if f.endswith(".mpack")]
        if files:
            files.sort(key=lambda x: int(re.search(r"model_(\d+)", x).group(1)))
            last_model = files[-1]
            with open(os.path.join(models_dir, last_model), 'rb') as f:
                data = f.read()
                try:
                    vstate = flax.serialization.from_bytes(vstate, data)
                except KeyError:
                    vstate.variables = flax.serialization.from_bytes(vstate.variables, data)
            print(f"Loaded model: {last_model}")
            return vstate
    
    print(f"Models directory not found or empty at {models_dir}")
    return None

def compute_correlations(vstate, lattice, L, folder, hilbert):
    vstate.n_samples = 1024
    R = Corr_Struct(lattice, vstate, L, folder, hilbert)
    print(f"Correlation Ratio R = {R}")
    return R

def compute_exact_energy(L, J2, hamiltonian):
    E_exact = None
    ket_gs = None
    if L == 4:
        E_exact, ket_gs = Exact_gs(L, J2, hamiltonian, J1J2=True, spin=True)
    elif L == 6:
        E_exact = Exact_gs_en_6x6(J2)
    elif L == 10:
        # Dictionary of known exact energies for 10x10
        gs_10x10 = {0.4: -0.52371, 0.5: -0.4976921, 0.55: -0.485434, 0.6: -0.47604}
        # Find closest J2
        for j_val, e_val in gs_10x10.items():
            if abs(J2 - j_val) < 1e-5:
                E_exact = e_val
                break

    if E_exact is not None:
        print(f"Exact ground state energy per site: {E_exact}")
    return E_exact, ket_gs

def compute_energy_stats(log, L, folder, folder_energy, E_exact, vstate, hamiltonian):
    # The physical Heisenberg Hamiltonian is H_phys = J * Sum(S_i.S_j).
    # NetKet's operator is H_nk = J * Sum(sigma_i.sigma_j).
    # Since S = sigma/2, H_nk = 4 * H_phys. This factor is used for normalization.
    ENERGY_NORMALIZATION_FACTOR = 4.0
    N_SITES = L * L

    if log is not None:
        E_vs_final_per_site, energy_per_iterations = Energy(log, L)
        if E_exact is not None:
            plot_energy(folder, energy_per_iterations, E_exact=E_exact)
        else:
            plot_energy(folder, energy_per_iterations, E_last=E_vs_final_per_site)
        variance_per_site = Variance(log, L, folder_energy)
        vscore = Vscore(L, variance_per_site, E_vs_final_per_site)
        tau_final = Autocorrelation_time(log, folder_energy)
        rhat_final = Rhat(log, folder_energy)
    else:
        E_vs = vstate.expect(hamiltonian)
        E_vs_final_per_site = E_vs.mean.real / (N_SITES * ENERGY_NORMALIZATION_FACTOR)
        variance_per_site = E_vs.variance.real / (N_SITES * ENERGY_NORMALIZATION_FACTOR**2)
        vscore = Vscore(L, variance_per_site, E_vs_final_per_site)
    
    print(f"Final Energy from VMC: {E_vs_final_per_site}")
    print("Variance = ", variance_per_site)
    print("Vscore = ", vscore)
    
    return E_vs_final_per_site, variance_per_site, vscore


def compute_param_count(params, L):
    if params['model_type'] == 'ViT':
        count = vit_param_count(params['n_heads'], params['num_layers'], params['patch_size'], params['d_model'], L*L)
    elif params['model_type'] == 'HFDS':
        count = hidden_fermion_param_count(L*L, params['n_hid'], L, L, params['layers'], params['features'])
    print(f"Number of parameters: {count}")
    return count

def compute_entropy(vstate, n_samples=65536):
    s2, s2_error = compute_renyi2_entropy(vstate, n_samples=n_samples)
    print(f"Renyi S2 = {s2} ± {s2_error} (n_samples={n_samples})")
    return s2, s2_error

def compute_sign(vstate, hilbert, n_samples=32768):
    vstate.n_samples = n_samples
    sign_op = MarshallSignObs(hilbert)
    sign_MCMC = vstate.expect(sign_op)
    print(f"Marshall Sign (MCMC): {sign_MCMC.mean} ± {sign_MCMC.error_of_mean} (n_samples={n_samples})")
    return sign_MCMC.mean, sign_MCMC.variance


def compute_qgt(vstate, folder, hilbert):
    try:
        # Attempt to free memory before heavy dense matrix allocation
        gc.collect()
        if hasattr(jax, 'clear_caches'):
            jax.clear_caches()
            
        all_eigenvalues, relevant_count_first, mean_rest_ratio, mean_rest_norm, mean_rest_norm_12 = calculate_relevant_eigenvalues(vstate, folder, hilbert, threshold_ratio_rest=1e-2)
        
        Plot_S_matrix_histogram(all_eigenvalues, folder, one_avg = "one")
        
        if all_eigenvalues:
            indices_to_plot = sorted([int(k.split('_')[1]) for k in all_eigenvalues.keys()])
            plot_S_matrix_spectrum(all_eigenvalues, indices_to_plot, folder, mean_rest_norm, len(indices_to_plot))
            
        print(f"QGT relevant eigenvalues - first: {relevant_count_first}, mean rest ratio: {mean_rest_ratio}, mean rest norm: {mean_rest_norm}")
        return {
            'eigenvalues_S': all_eigenvalues,
            'mean_rest_ratio': mean_rest_ratio,
            'mean_rest_norm': mean_rest_norm,
            'mean_rest_norm_12': mean_rest_norm_12,
            'relevant_count_first': relevant_count_first,
        }
    except Exception as e:
        print(f"⚠️ Skipping QGT calculation due to error (likely OOM): {e}")
        # Attempt to restore variables if they were moved
        return {}

def compute_L4_observables(vstate, ket_gs, hilbert, L, folder, count_params):
    fidelity = Fidelity(vstate, ket_gs)
    print(f"Fidelity <vstate|exact> = {fidelity}")

    configs, sign_vstate_config, weight_exact, weight_vstate = plot_Sign_single_config(ket_gs, vstate, hilbert, 3, L, folder, one_avg = "one")
    configs, sign_vstate_config, weight_exact, weight_vstate = plot_Weight_single(ket_gs, vstate, hilbert, 8, L, folder, one_avg = "one")
    amp_overlap, fidelity, sign_vstate, sign_exact, sign_overlap = plot_Sign_Err_Amplitude_Err_Fidelity(ket_gs, vstate, hilbert, folder, one_avg = "one")
    amp_overlap, sign_vstate, sign_exact, sign_overlap = plot_Sign_Err_vs_Amplitude_Err_with_iteration(ket_gs, vstate, hilbert, folder, one_avg = "one")
    sector_amp_err, sector_sign_err = plot_Sector_Overlap_err_vs_iteration(ket_gs, vstate, hilbert, folder, one_avg = "one")
    sorted_weights, sorted_amp_overlap, sorted_sign_overlap = plot_Overlap_vs_Weight(ket_gs, vstate, hilbert, folder, "one")
    eigenvalues, rel_1, rel_2, rel_3, rel_4 = plot_S_matrix_eigenvalues(vstate, folder, hilbert, one_avg = "one")
    spectrum, total_error, sector_errors = plot_spectrum(ket_gs, vstate, L, save_dir=folder+"/physical_obs/spectrum")
    #Plot_Energy_Fidelity(log, fidelity, folder, one_avg="one", L =4, plot_variance=False, fidelity_var=None)
    
    return {
        'sign_vstate': sign_vstate,
        'sign_exact': sign_exact,
        'fidelity': fidelity,
        'configs': configs,
        'sign_vstate_config': sign_vstate_config,
        'weight_exact': weight_exact,
        'weight_vstate': weight_vstate,
        'amp_overlap': amp_overlap,
        'sign_overlap': sign_overlap,
        'sector_amp_err': sector_amp_err,
        'sector_sign_err': sector_sign_err,
        'eigenvalues_S': eigenvalues,
        'rank_S': rel_1,
        'spectrum': spectrum,
        'total_error_spectrum': total_error,
        'sector_errors_spectrum': sector_errors,
        'params': count_params
    }

def compute_L6_observables(vstate, J2, folder, params):
    print("6x6")
    results = Observable_Importance_sampling(J2, NQS_path=None, vstate=vstate)
    Fidelity_vs_Iterations(folder, vstate, params)
    return {
        'final_energy_DMRG': results['final_energy_DMRG'],
        'sign_DMRG_Imp': results['sign_DMRG_Imp'],
        'fidelity_sampled_NQS_DMRG': results['fidelity_sampled_NQS_DMRG'],
        'sign_NQS_MCMC': results['sign_NQS_MCMC'],
        'sign_NQS_Imp': results['sign_NQS_Imp'],
        'Overlap_sign_NQS_DMRG': results['Overlap_sign_NQS_DMRG'],
        'Overlap_amp_NQS_DMRG': results['Overlap_amp_NQS_DMRG'],
    }


def save_variables(folder, variables):
    with open(os.path.join(folder, "variables.pkl"), 'wb') as f:
        pickle.dump(variables, f)



def run_observables(log, folder):
    folder_energy = setup_environment(folder)
    params = parse_model_path(folder)
    L = params['L']
    J2 = params['J2']
    print(f"Loaded params: {params}")

    lattice, hilbert, hamiltonian = setup_system(L, J2, params)
    model = setup_model(params, hilbert, L)
    
    sampler = nk.sampler.MetropolisExchange(
        hilbert=hilbert,
        graph=lattice,
        d_max=2,
        n_chains=1024,
        sweep_size=lattice.n_nodes,
    )

    vstate = load_vstate(folder, sampler, model)
    if vstate is None:
        sys.stdout.close()
        return

    # Load existing variables
    variables = {}
    variables_path = os.path.join(folder, "variables.pkl")
    if os.path.exists(variables_path):
        with open(variables_path, 'rb') as f:
            variables = pickle.load(f)

    ################################################################################################à
    
    # 1. Correlations
    """R = compute_correlations(vstate, lattice, L, folder, hilbert)
    variables['R'] = R
    save_variables(folder, variables)
    """

    # 2. Exact Energy
    E_exact, ket_gs = compute_exact_energy(L, J2, hamiltonian)
    if E_exact is not None:
        print(f"Exact ground state energy per site for L={L}, J2={J2}: {E_exact}")
    
    # 3. Energy Stats
    E_vs_final_per_site, variance_per_site, vscore = compute_energy_stats(log, L, folder, folder_energy, E_exact, vstate, hamiltonian)
    print(f"Final Energy per site: {E_vs_final_per_site}")
    print(f"Variance per site: {variance_per_site}")
    print(f"Vscore: {vscore}")

    if E_exact is not None:
        variables['E_exact'] = E_exact
        variables['rel_err_E'] = abs((E_vs_final_per_site - E_exact) / E_exact)

    
    
    # 5. Param Count
    count_params = compute_param_count(params, L)

    variables.update({
        'E_vs_final': E_vs_final_per_site,
        'params': count_params,
        'vscore': vscore,
        'variance': variance_per_site,
        'E_exact': E_exact,
        'rel_err_E': abs((E_vs_final_per_site - E_exact) / E_exact) if E_exact is not None else None
    })

    save_variables(folder, variables)
    
    # 6. Entanglement Entropy
    """n_samples_entropy = 524288//2
    s2, s2_error = compute_entropy(vstate, n_samples=n_samples_entropy)
    variables.update({
        's2': s2,
        's2_error': s2_error
    })
    save_variables(folder, variables)"""
    
    #6. Entanglement Scaling
    """results = compute_entanglement_scaling(vstate, L, n_samples=65536*2) 
    plot_entanglement_scaling(results, save_path=folder+"/physical_obs/entanglement_scaling.png")
    variables.update({'entanglement_scaling': results})
    save_variables(folder, variables)"""
    

    #7. Sign
    """n_samples_sign = 32768
    sign_mean, sign_var = compute_sign(vstate, hilbert, n_samples=n_samples_sign)
    
    variables.update({
        'sign_vstate_MCMC': sign_mean,
        'sign_vstate_MCMC_variance': sign_var
    })
    save_variables(folder, variables)
    
    """
    
    # 10. System specific observables
    
    """if L == 4 and ket_gs is not None:
        l4_vars = compute_L4_observables(vstate, ket_gs, hilbert, L, folder, count_params)
        variables.update(l4_vars)
        save_variables(folder, variables)
    elif L == 6:
        l6_vars = compute_L6_observables(vstate, J2, folder, params)
        variables.update(l6_vars)
        save_variables(folder, variables)
    """

    # 9. QGT
    """qgt_vars = compute_qgt(vstate, folder, hilbert)
    variables.update(qgt_vars)
    save_variables(folder, variables)"""

    print("\n\n#######################################################################################################\n\n")

    sys.stdout.close()


if __name__ == "__main__":

    #model_path = "/scratch/f/F.Conoscenti/Thesis_QSL/HFDS_Heisenberg/plot/8x8/layers1_hidd4_feat32_sample4096_bcPBC_PBC_lr0.02_iter400_parityTrue_rotTrue_InitFermi_typecomplex_QGT"
    #model_path = "/scratch/f/F.Conoscenti/Thesis_QSL/ViT_Heisenberg/plot/8x8/QGT/layers2_d24_heads6_patch2_sample8192_lr0.0075_iter200_parityTrue_rotTrue_QGT"
    model_path="/scratch/f/F.Conoscenti/Thesis_QSL/HFDS_Heisenberg/plot/8x8/QGT/layers1_hidd4_feat32_sample8192_bcPBC_PBC_phi0.0_lr0.02_iter200_parityTrue_rotTrue_InitFermi_typecomplex_phi"
    #model_path = "/scratch/f/F.Conoscenti/Thesis_QSL/HFDS_Heisenberg/plot/8x8/layers1_hidd6_feat32_sample2048_bcPBC_PBC_lr0.02_iter10000_parityTrue_rotTrue_InitFermi_typecomplex_QGT"
    #model_path = "/scratch/f/F.Conoscenti/Thesis_QSL/ViT_Heisenberg/plot/6x6/layers2_d24_heads6_patch2_sample2048_lr0.0075_iter10000_parityTrue_rotTrue_QGT"
    #model_path = "/scratch/f/F.Conoscenti/Thesis_QSL/HFDS_Heisenberg/plot/8x8/layers1_hidd6_feat32_sample2048_bcPBC_PBC_lr0.02_iter10000_parityTrue_rotTrue_InitFermi_typecomplex_QGT"
    log = None

    if not os.path.exists(model_path):
        model_path = model_path.replace("/cluster/home/fconoscenti/Thesis_QSL", "/scratch/f/F.Conoscenti/Thesis_QSL")

    if os.path.exists(model_path):
        if os.path.exists(os.path.join(model_path, "log.pkl")):
            try:
                with open(os.path.join(model_path, "log.pkl"), "rb") as f:
                    log = pickle.load(f)
            except Exception as e:
                print(f"Could not load existing log: {e}")

        # Check if the path is already a specific J folder
        if os.path.basename(os.path.normpath(model_path)).startswith("J=") or os.path.basename(os.path.normpath(model_path)).startswith("J2="):
            j_paths = [model_path]
        else:
            j_folders = [f for f in os.listdir(model_path) if (f.startswith("J=") or f.startswith("J2=")) and os.path.isdir(os.path.join(model_path, f))]
            try:
                j_folders.sort(key=lambda x: float(x.split('=')[1]))
            except:
                j_folders.sort()
            j_paths = [os.path.join(model_path, f) for f in j_folders]

        for j_path in j_paths:
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
