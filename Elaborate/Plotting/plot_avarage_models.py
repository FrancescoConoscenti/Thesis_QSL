import pickle
import os
import numpy as np
from pathlib import Path
import netket as nk

import re
import matplotlib.pyplot as plt

import jax.numpy as jnp

from Elaborate.Statistics.Error_Stat import *
from Elaborate.Plotting.Sign_vs_iteration import *
from Elaborate.Sign_Obs import *
#from Elaborate.Plotting.apply_function_models import *


def average_models_seeds(folder):
    try:
        j_subfolders = sorted(
            [p for p in Path(folder).iterdir() if p.is_dir() and p.name.startswith("J=")],
            key=lambda p: float(p.name.split('=')[1])
        )
    except (ValueError, IndexError):
        print(f"Warning: Could not correctly parse J-subfolders in {folder}. Skipping model.")
        return

    for j_folder in j_subfolders:
        base_dir = str(j_folder)
        if not j_folder.is_dir():
            continue

        # --- Automatically detect all seed directories ---
        seed_dirs = sorted([d for d in os.listdir(base_dir) if d.startswith("seed_")])
        if not seed_dirs:
            raise FileNotFoundError(f"No seed directories found in base directory: {base_dir}")

        # --- Load all variables.pkl files ---
        data_list = []
        for seed_dir in seed_dirs:
            # try both "variables" and "variables.pkl"
            seed_path = os.path.join(base_dir, seed_dir, "variables")
            if not os.path.exists(seed_path):
                seed_path = os.path.join(base_dir, seed_dir, "variables.pkl")
            if not os.path.exists(seed_path):
                print(f"Warning: {seed_path} not found, skipping {seed_dir}.")
                continue
            with open(seed_path, "rb") as f:
                loaded_data = pickle.load(f)
                data_list.append(loaded_data)

        if not data_list:
            raise FileNotFoundError(f"No valid variables files found under {base_dir} for seeds: {seed_dirs}")

        # --- Calculate sign_err_var (keep your logic) ---
        # Make sure keys exist before indexing
        if all('sign_vstate' in d for d in data_list) and all('sign_exact' in d for d in data_list):
            sign_vstate_arrays = [np.array(d['sign_vstate']) for d in data_list]
            min_len_sign = min(len(arr) for arr in sign_vstate_arrays)
            truncated_sign_arrays = [arr[:min_len_sign] for arr in sign_vstate_arrays]
            sign_vstate_full_values = np.stack(truncated_sign_arrays)

            sign_exact_values = np.stack([np.array(d['sign_exact']) for d in data_list])[:, np.newaxis]
            sign_err_values = np.abs(np.abs(sign_vstate_full_values) - np.abs(sign_exact_values))
            sign_err_var = np.var(sign_err_values, axis=0)
        else:
            sign_err_var = None
            print("Warning: 'sign_vstate' or 'sign_exact' missing in some seeds; sign_err_var set to None.")

        # --- Compute average and variance for each variable ---
        # We'll iterate all keys that appear in ANY file and try to handle them robustly
        all_keys = sorted({k for d in data_list for k in d.keys()})
        results = {}

        for key in all_keys:
            # skip configs copying later
            try:
                # gather only numeric-like arrays for this key
                values = []
                for d in data_list:
                    if key not in d:
                        # skip seeds that don't have the key
                        continue
                    values.append(np.array(d[key]))

                if not values:
                    continue

                # Convert scalars to 1D arrays for consistent stacking
                if values[0].ndim == 0:
                    values_stack = np.stack(values)   # shape (n_seeds,)
                elif values[0].ndim == 1:
                    min_len = min(arr.shape[0] for arr in values)
                    truncated = [arr[:min_len] for arr in values]
                    values_stack = np.stack(truncated)  # shape (n_seeds, min_len)
                elif values[0].ndim == 2:
                    min_len_inner = min(arr.shape[1] for arr in values)
                    truncated = [arr[:, :min_len_inner] for arr in values]
                    values_stack = np.stack(truncated)  # shape (n_seeds, dim0, min_len_inner)
                else:
                    # unsupported dimension, skip
                    print(f"Skipping key '{key}': unsupported ndim {values[0].ndim}")
                    continue

                # Avoid double-suffixing if key already ends with _mean/_var
                base_key = key
                if base_key.endswith("_mean"):
                    base_key = base_key[:-5]
                if base_key.endswith("_var"):
                    base_key = base_key[:-4]

                # special-case configs: keep original (non-numeric)
                if base_key == "configs":
                    results["configs"] = data_list[0].get("configs")
                    continue
                
                # E_exact is a constant, so just copy it from the first seed
                if base_key == "E_exact":
                    results["E_exact"] = data_list[0].get("E_exact")
                    continue

                # For sign_vstate we want stats on absolute values (as in your original code)
                if base_key == "sign_vstate":
                    mean_val = np.mean(np.abs(values_stack), axis=0)
                    var_val = np.var(np.abs(values_stack), axis=0)
                # Special handling for Energy_iter to get initial energy
                elif base_key == "Energy_iter":
                    # We only care about the first element (initial energy).
                    # values_stack can be 1D (if Energy_iter was scalar) or 2D.
                    if values_stack.ndim == 2:
                        initial_energies = values_stack[:, 0]
                    else: # ndim == 1
                        initial_energies = values_stack
                    results["E_init_mean"] = np.mean(initial_energies)
                    results["E_init_var"] = np.var(initial_energies)
                    continue # Skip creating Energy_iter_mean/var
                else:
                    mean_val = np.mean(values_stack, axis=0)
                    var_val = np.var(values_stack, axis=0)

                results[f"{base_key}_mean"] = mean_val
                results[f"{base_key}_var"] = var_val

                print(f"{base_key}_mean",mean_val)

            except Exception as e:
                print(f"Skipping key '{key}' (error while processing): {e}")

        # include sign_err_var if computed
        results["sign_err_var"] = sign_err_var

        # Save the averaged and variance data (use .pkl for clarity)
        output_path = os.path.join(base_dir, "variables_average.pkl")
        with open(output_path, "wb") as f:
            pickle.dump(results, f)

        # diagnostic print: show top-level keys saved
        print(f"\n✅ Averaged data saved to: {output_path}")
        print("Saved keys:", ", ".join(sorted(results.keys())))


def avarage_plots_seeds(folder, plot_variance=True):

    L=4

    try:
        j_subfolders = sorted(
            [p for p in Path(folder).iterdir() if p.is_dir() and p.name.startswith("J=")],
            key=lambda p: float(p.name.split('=')[1])
        )
    except (ValueError, IndexError):
        print(f"Warning: Could not correctly parse J-subfolders in {folder}. Skipping plots.")
        return

    for j_folder in j_subfolders:
        avg_file_path = j_folder / "variables_average.pkl"
        if not avg_file_path.exists():
            # Try the old path for backward compatibility
            avg_file_path = j_folder / "variables_average"
        if not avg_file_path.exists():
            print(f"Warning: Average data file not found, skipping plots for {j_folder.name}: {avg_file_path}")
            continue
        
        with open(avg_file_path, 'rb') as f:
            loaded_data = pickle.load(f)

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
        base_dir = j_folder
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
        
def output_variables_avarage(folder):
    try:
        j_subfolders = sorted(
            [p for p in Path(folder).iterdir() if p.is_dir() and p.name.startswith("J=")],
            key=lambda p: float(p.name.split('=')[1])
        )
    except (ValueError, IndexError):
        print(f"Warning: Could not correctly parse J-subfolders in {folder}. Skipping output.")
        return

    for j_folder in j_subfolders:
        avg_file_path = j_folder / "variables_average.pkl"
        if not avg_file_path.exists():
            # Try the old path for backward compatibility
            avg_file_path = j_folder / "variables_average"
        if not avg_file_path.exists():
            print(f"Warning: Average data file not found, skipping output for {j_folder.name}: {avg_file_path}")
            continue
        
        with open(avg_file_path, 'rb') as f:
            loaded_data = pickle.load(f)

        E_init_mean = loaded_data['E_init_mean']
        E_exact = loaded_data['E_exact']
        fidelity_mean = loaded_data['fidelity_mean']

        print(f"{j_folder.name}: E_init_mean = {E_init_mean}, E_exact = {E_exact}, Fidelity_mean = {fidelity_mean}")


if __name__ == "__main__":

    # Define a list of model paths to process
    model_paths = [
        "HFDS_Heisenberg/plot/spin_new/layers1_hidd1_feat1_sample256_lr0.025_iter1_parityTrue_rotTrue_InitG_MF_typecomplex"    
        ]       
    
    # Loop through each model path and process it
    for model_path in model_paths:
        print(f"\n{'='*20} Processing Model: {Path(model_path).name} {'='*20}")
        average_models_seeds(model_path)
        avarage_plots_seeds(model_path, plot_variance=True)
        print(f"{'='*20} Finished Processing Model: {Path(model_path).name} {'='*20}\n")

    for model_path in model_paths:
        print(f"\n{'='*20} Processing Model: {Path(model_path).name} {'='*20}")
        output_variables_avarage(model_path)
        print(f"{'='*20} Finished Processing Model: {Path(model_path).name} {'='*20}\n")