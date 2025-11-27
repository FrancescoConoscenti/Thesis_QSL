import pickle
import os
import numpy as np
import os
from pathlib import Path
import netket as nk

import re
import jax.numpy as jnp

from Elaborate.Statistics.Error_Stat import *
from Elaborate.Plotting.Sign_vs_iteration import *
from Elaborate.Sign_Obs import *
#from Elaborate.Plotting.apply_function_models import *


def average_models_seeds(folder, Js):

    for J in Js:
        base_dir = folder+f"/J={J}"
        # --- Automatically detect all seed directories ---
        seed_dirs = sorted([d for d in os.listdir(base_dir) if d.startswith("seed_")])
        if not seed_dirs:
            raise FileNotFoundError("No seed directories found in base directory.")

        # --- Load all variables.pkl files ---
        data_list = []
        for seed_dir in seed_dirs:
            seed_path = os.path.join(base_dir, seed_dir, "variables")
            if not os.path.exists(seed_path):
                #seed_path = os.path.join(base_dir, seed_dir, "variables.pkl") 
                if not os.path.exists(seed_path):
                    print(f"Warning: {seed_path} not found, skipping.")
                    continue
            with open(seed_path, "rb") as f:
                loaded_data = pickle.load(f)
                data_list.append(loaded_data)

        if not data_list:
            raise FileNotFoundError("No valid variables.pkl files found.")

        # --- Calculate sign_err_var ---
        # Truncate arrays to the minimum length before stacking to avoid shape errors.
        sign_vstate_arrays = [np.array(d['sign_vstate']) for d in data_list]
        min_len_sign = min(len(arr) for arr in sign_vstate_arrays)
        truncated_sign_arrays = [arr[:min_len_sign] for arr in sign_vstate_arrays]
        sign_vstate_full_values = np.stack(truncated_sign_arrays)

        sign_exact_values = np.stack([np.array(d['sign_exact']) for d in data_list])[:, np.newaxis]
        sign_err_values = np.abs(np.abs(sign_vstate_full_values) - np.abs(sign_exact_values))
        sign_err_var = np.var(sign_err_values, axis=0)

        # --- Compute average and variance for each variable ---
        keys = data_list[0].keys()
        results = {}

        for key in keys:
            try:
                all_data = [np.array(d[key]) for d in data_list]
                first_item = all_data[0]

                # Handle different data shapes: 0D (scalar), 1D, and 2D arrays
                if first_item.ndim == 0:
                    # Case 1: Scalar data (e.g., 'sign_exact')
                    values_stack = np.stack(all_data)
                elif first_item.ndim == 1:
                    # Case 2: 1D array data (e.g., 'fidelity' over iterations)
                    min_len = min(len(arr) for arr in all_data)
                    truncated_arrays = [arr[:min_len] for arr in all_data]
                    values_stack = np.stack(truncated_arrays)
                elif first_item.ndim == 2:
                    # Case 3: 2D array data (e.g., 'sign_vstate_config')
                    # We truncate along the second axis (iterations)
                    min_len_inner = min(arr.shape[1] for arr in all_data)
                    truncated_arrays = [arr[:, :min_len_inner] for arr in all_data]
                    values_stack = np.stack(truncated_arrays)
                else:
                    print(f"Skipping key '{key}': Unsupported array dimension {first_item.ndim}.")
                    continue

                if key == "configs":
                    results[key] = data_list[0][key]
                else:
                    if key == "sign_vstate_full":
                        mean_val = np.mean(np.abs(values_stack), axis=0)
                        var_val = np.var(np.abs(values_stack), axis=0)  
                    else:
                        mean_val = np.mean(values_stack, axis=0)
                        var_val = np.var(values_stack, axis=0)

                    results[key + "_mean"] = mean_val
                    results[key + "_var"] = var_val

            except Exception as e:
                print(f"Skipping key '{key}' (non-numeric or incompatible type): {e}")
        results["sign_err_var"] = sign_err_var

        # --- Save the averaged and variance data ---
        output_path = os.path.join(base_dir, "variables_average")
        with open(output_path, "wb") as f:
            pickle.dump(results, f)

        print(f"\n✅ Averaged and variance data saved to: {output_path}")


def avarage_plots_seeds(folder, Js, plot_variance=True):

    L=4

    for J in Js:
        
        loaded_data = pickle.load(open(f'{folder}/J={J}/variables_average', 'rb')) 

        # sign_vstate_MCMC = loaded_data['sign_vstate_MCMC'] 
        #sign_vstate_full_mean = loaded_data['sign_vstate_full_mean']
        #sign_vstate_full_var = loaded_data['sign_vstate_full_var']
        sign_exact_full_mean = loaded_data['sign_exact_mean']
        sign_exact_full_var = loaded_data['sign_exact_var']
        fidelity_mean = loaded_data['fidelity_mean']
        fidelity_var = loaded_data['fidelity_var']
        configs = loaded_data['configs']
        sign_vstate_config_mean = loaded_data['sign_vstate_config_mean']
        sign_vstate_config_var = loaded_data['sign_vstate_config_var']
        weight_exact_mean = loaded_data['weight_exact_mean'] 
        weight_exact_var = loaded_data['weight_exact_var']
        weight_vstate_mean = loaded_data['weight_vstate_mean']
        weight_vstate_var = loaded_data['weight_vstate_var']  
        amp_overlap_mean = loaded_data['amp_overlap_mean']
        amp_overlap_var = loaded_data['amp_overlap_var']
        sign_overlap_mean = loaded_data['sign_overlap_mean']
        sign_overlap_var = loaded_data['sign_overlap_var']
        sign_err_var = loaded_data['sign_err_var']

        number_states_avg = sign_vstate_config_mean.shape[0]

        # Find the first available seed directory to use as a base for saving plots.
        # This avoids hardcoding "seed_1".
        base_dir = Path(folder) / f"J={J}"
        try:
            first_seed_dir = next(d for d in base_dir.iterdir() if d.is_dir() and d.name.startswith("seed_"))
            folder_to_plot = str(first_seed_dir)
        except StopIteration:
            print(f"Warning: No seed directories found in {base_dir}. Cannot generate average plots.")
            continue

        #sign_vstate_MCMC, sign_vstate_full = plot_Sign_full_MCMC(marshall_op, vstate, str(second_level), 64, hi)
        #Plot_Sign_Fidelity(sign_vstate_full_mean, sign_exact_full_mean, fidelity_mean, folder_to_plot, one_avg= "avg", plot_variance=plot_variance, sign_vstate_full_var=sign_vstate_full_var, fidelity_var=fidelity_var)
        #Plot_Sign_single_config(configs, sign_vstate_config_mean, sign_vstate_full_mean, sign_exact_full_mean, weight_exact_mean, weight_vstate_mean, number_states_avg, folder_to_plot, one_avg= "avg", plot_variance=plot_variance, sign_vstate_full_var=sign_vstate_full_var)
        #Plot_Weight_single(configs, sign_vstate_config_mean, weight_exact_mean, weight_vstate_mean, number_states_avg, folder_to_plot, one_avg = "avg", plot_variance=plot_variance, weight_vstate_var=weight_vstate_var)
        #Plot_Amp_overlap_configs(amp_overlap_mean, folder_to_plot, one_avg = "avg", plot_variance=plot_variance, error_var=amp_overlap_var)
        #Plot_Sign_Err_Amplitude_Err_Fidelity(amp_overlap_mean, fidelity_mean, sign_overlap_mean, folder_to_plot, one_avg = "avg", plot_variance=plot_variance, error_var=amp_overlap_var, fidelity_var=fidelity_var, sign_err_var=sign_err_var) # Existing call
        Plot_Sign_Err_vs_Amplitude_Err_with_iteration(amp_overlap_mean, sign_overlap_mean, folder_to_plot, one_avg="avg", plot_variance=plot_variance, amplitude_overlap_var=amp_overlap_var, sign_overlap_var=sign_overlap_var)
        

if __name__ == "__main__":

    model_path = "/scratch/f/F.Conoscenti/Thesis_QSL/HFDS_Heisenberg/plot/spin_new/layers1_hidd1_feat2_sample256_lr0.025_iter2_parityTrue_rotTrue_transFalse_InitFermi_typecomplex"
    #model_path = "/scratch/f/F.Conoscenti/Thesis_QSL/HFDS_Heisenberg/plot/spin_new/layers1_hidd1_feat2_sample256_lr0.025_iter2_parityTrue_rotTrue_transFalse_InitG_MF_typecomplex"
    #model_path = "/scratch/f/F.Conoscenti/Thesis_QSL/HFDS_Heisenberg/plot/spin_new/layers1_hidd1_feat2_sample256_lr0.025_iter2_parityTrue_rotTrue_transFalse_Initrandom_typecomplex"
    Js = [0.0, 0.2, 0.5, 0.7, 1]

    average_models_seeds(model_path, Js)
    avarage_plots_seeds(model_path, Js, plot_variance=True)