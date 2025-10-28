import pickle
import os
import numpy as np
import os
from pathlib import Path
import netket as nk

import re
import jax.numpy as jnp

from Elaborate.Error_Stat import *
from Elaborate.Sign_vs_iteration import *
from Elaborate.Sign_Obs import *
from Elaborate.apply_function_models import *


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
                print(f"Warning: {seed_path} not found, skipping.")
                continue
            with open(seed_path, "rb") as f:
                loaded_data = pickle.load(f)
                data_list.append(loaded_data)

        if not data_list:
            raise FileNotFoundError("No valid variables.pkl files found.")

        # --- Calculate sign_err_var ---
        sign_vstate_full_values = np.stack([np.array(d['sign_vstate_full']) for d in data_list])
        sign_exact_values = np.stack([np.array(d['sign_exact']) for d in data_list])[:, np.newaxis]
        sign_err_values = np.abs(np.abs(sign_vstate_full_values) - np.abs(sign_exact_values))
        sign_err_var = np.var(sign_err_values, axis=0)
        # --- Compute average and variance for each variable ---
        keys = data_list[0].keys()
        results = {}

        for key in keys:
            try:
                values = [np.array(d[key]) for d in data_list]
                values_stack = np.stack(values)
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
        sign_vstate_full_mean = loaded_data['sign_vstate_full_mean']
        sign_vstate_full_var = loaded_data['sign_vstate_full_var']
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
        error_mean = loaded_data['error_mean']
        error_var = loaded_data['error_var']
        sign_err_var = loaded_data['sign_err_var']

        number_states_avg = sign_vstate_config_mean.shape[0]

        folder_to_plot = folder + f"/J={J}/seed_1"

        #sign_vstate_MCMC, sign_vstate_full = plot_Sign_full_MCMC(marshall_op, vstate, str(second_level), 64, hi)
        Plot_Sign_Fidelity(sign_vstate_full_mean, sign_exact_full_mean, fidelity_mean, folder_to_plot, one_avg= "avg", plot_variance=plot_variance, sign_vstate_full_var=sign_vstate_full_var, fidelity_var=fidelity_var)
        Plot_Sign_single_config(configs, sign_vstate_config_mean, sign_vstate_full_mean, sign_exact_full_mean, weight_exact_mean, weight_vstate_mean, number_states_avg, folder_to_plot, one_avg= "avg", plot_variance=plot_variance, sign_vstate_full_var=sign_vstate_full_var)
        Plot_Weight_single(configs, sign_vstate_config_mean, weight_exact_mean, weight_vstate_mean, number_states_avg, folder_to_plot, one_avg = "avg", plot_variance=plot_variance, weight_vstate_var=weight_vstate_var)
        Plot_MSE_configs(error_mean, folder_to_plot, one_avg = "avg", plot_variance=plot_variance, error_var=error_var)
        Plot_Sign_Err_Amplitude_Err_Fidelity(error_mean, fidelity_mean, sign_vstate_full_mean, sign_exact_full_mean, folder_to_plot, one_avg = "avg", plot_variance=plot_variance, error_var=error_var, fidelity_var=fidelity_var, sign_err_var=sign_err_var)


if __name__ == "__main__":

    model_path = "/scratch/f/F.Conoscenti/Thesis_QSL/ViT_Heisenberg/plot/J1J2/layers1_d16_heads4_patch2_sample1024_lr0.0075_iter3000_symmTrue_new"
    Js = [ 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

    average_models_seeds(model_path, Js)
    avarage_plots_seeds(model_path, Js, plot_variance=True)