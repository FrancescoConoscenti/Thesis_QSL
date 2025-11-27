import os
import re
import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib.cm as cm
import argparse
import json

def plot_initial_fidelity_vs_Js(model_paths: list[str]):
    """
    Loads averaged fidelity data for different J values from multiple model directories
    and plots the fidelity at the 0-th iteration (initial state) vs. J on a single graph.

    Args:
        model_paths (list[str]): A list of paths to the main model directories,
                                  each containing J=... subfolders.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Use a colormap to get distinct colors for each model
    colors = cm.get_cmap('tab10', len(model_paths))
    all_max_y = []

    for model_idx, model_path in enumerate(model_paths):
        base_path = Path(model_path)
        if not base_path.is_dir():
            print(f"❌ Error: Provided path '{model_path}' is not a valid directory. Skipping.")
            continue

        # --- Data collection for current model ---
        j_values = []
        initial_fidelities = []
        initial_fidelity_errors = []

        # --- Find and process J folders ---
        j_folders = sorted([d for d in base_path.iterdir() if d.is_dir() and d.name.startswith("J=")])

        if not j_folders:
            print(f"🤷 No 'J=...' subdirectories found in '{model_path}'. Skipping.")
            continue

        for j_path in j_folders:
            try:
                j_value_str = j_path.name.split('=')[1]
                j_value = float(j_value_str)
            except (ValueError, IndexError):
                print(f"⚠️ Warning: Could not parse J value from folder name: {j_path.name}. Skipping.")
                continue

            avg_file_path = j_path / "variables_average"
            if not avg_file_path.exists():
                print(f"⚠️ Warning: '{avg_file_path}' not found. Skipping J={j_value}.")
                continue

            # --- Load data and extract initial fidelity ---
            with open(avg_file_path, "rb") as f:
                loaded_data = pickle.load(f)

            try:
                # Fidelity is stored over iterations. The first entry is at iteration 0.
                fidelity_mean = loaded_data['fidelity_mean']
                fidelity_var = loaded_data.get('fidelity_var') # Use .get for safety

                if len(fidelity_mean) > 0:
                    j_values.append(j_value)
                    initial_fidelities.append(fidelity_mean[0])
                    if fidelity_var is not None and len(fidelity_var) > 0:
                        initial_fidelity_errors.append(np.sqrt(fidelity_var[0]))
                    else:
                        initial_fidelity_errors.append(0) # No error bar if variance is missing

            except KeyError as e:
                print(f"⚠️ Warning: Missing key {e} in data for J={j_value}. Skipping.")
                continue

        # --- Plotting for current model ---
        if j_values:
            # --- Custom Label Generation ---
            model_label = ""
            if "HFDS_Heisenberg" in model_path:
                model_name = base_path.name
                hidd_match = re.search(r'hidd(\d+)', model_name)
                init_match = re.search(r'Init([A-Za-z0-9+]+)', model_name) # Updated regex to include '+' and numbers
                hidd_fermions = hidd_match.group(1) if hidd_match else '?'
                init_type = init_match.group(1) if init_match else '?'
                model_label = f"HFDS: {hidd_fermions} hidden, {init_type} Init"
            elif "ViT_Heisenberg" in model_path:
                model_label = "ViT"
            else:
                model_label = base_path.name # Fallback to full name
            ax.errorbar(j_values, initial_fidelities, yerr=initial_fidelity_errors,
                        fmt='o', linestyle='none', color=colors(model_idx), capsize=5, markersize=8,
                        label=model_label)
            
            if initial_fidelities:
                all_max_y.append(max(np.array(initial_fidelities) + np.array(initial_fidelity_errors)))

    # --- Final plot styling ---
    ax.set_xlabel("$J_2$", fontsize=12)
    ax.set_ylabel("Initial Fidelity $|\\langle \\Psi_{exact} | \\Psi_{init} \\rangle|^2$", fontsize=12)
    ax.set_title(f"Initial Fidelity of Fermi Sea State vs. $J_2$", fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(loc='best')
    #ax.set_yscale('log')
    
    # Adjust y-axis to ensure error bars are fully visible
    if all_max_y:
        overall_max_y = max(all_max_y)
        ax.set_ylim(bottom=0, top=min(1.0, overall_max_y * 1.1))
    else:
        ax.set_ylim(bottom=0, top=1.0)

    plt.tight_layout()

    # Generate a more generic save path for multiple models
    if len(model_paths) == 1:
        save_path = Path("/scratch/f/F.Conoscenti/Thesis_QSL/Elaborate/plot") / f"Initial_Fidelity_vs_J_{Path(model_paths[0]).name}.png"
    else:
        save_path = Path("/scratch/f/F.Conoscenti/Thesis_QSL/Elaborate/plot") / "Initial_Fidelity_vs_J_Comparison.png"

    plt.savefig(save_path, dpi=300)
    print(f"✅ Plot saved to {save_path}")
    plt.show()

def plot_initial_energy_vs_Js(model_paths: list[str]):
    """
    Loads energy data for different J values from multiple model directories
    and plots the energy at the 0-th iteration (initial state) vs. J on a single graph.

    Args:
        model_paths (list[str]): A list of paths to the main model directories,
                                  each containing J=... subfolders.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Use a colormap to get distinct colors for each model
    colors = cm.get_cmap('tab10', len(model_paths))

    for model_idx, model_path in enumerate(model_paths):
        base_path = Path(model_path)
        if not base_path.is_dir():
            print(f"❌ Error: Provided path '{model_path}' is not a valid directory. Skipping.")
            continue

        # --- Data collection for current model ---
        j_values = []
        initial_energies = []
        initial_energy_errors = []

        # --- Find and process J folders ---
        j_folders = sorted([d for d in base_path.iterdir() if d.is_dir() and d.name.startswith("J=")])

        if not j_folders:
            print(f"🤷 No 'J=...' subdirectories found in '{model_path}'. Skipping.")
            continue

        for j_path in j_folders:
            try:
                j_value_str = j_path.name.split('=')[1]
                j_value = float(j_value_str)
            except (ValueError, IndexError):
                print(f"⚠️ Warning: Could not parse J value from folder name: {j_path.name}. Skipping.")
                continue

            log_file_path = j_path / "log_average.json"
            if not log_file_path.exists():
                print(f"⚠️ Warning: '{log_file_path}' not found. Skipping J={j_value}.")
                continue

            # --- Load data and extract initial energy ---
            with open(log_file_path, "r") as f:
                log_data = [json.loads(line) for line in f]

            if log_data:
                initial_energy_data = log_data[0]['Energy']
                j_values.append(j_value)
                initial_energies.append(initial_energy_data['Mean'])
                initial_energy_errors.append(initial_energy_data.get('Sigma', 0))

        # --- Plotting for current model ---
        if j_values:
            # --- Custom Label Generation ---
            model_label = ""
            if "HFDS_Heisenberg" in model_path:
                model_name = base_path.name
                hidd_match = re.search(r'hidd(\d+)', model_name)
                init_match = re.search(r'Init([A-Za-z0-9+]+)', model_name)
                hidd_fermions = hidd_match.group(1) if hidd_match else '?'
                init_type = init_match.group(1) if init_match else '?'
                model_label = f"HFDS: {hidd_fermions} hidden, {init_type} Init"
            else:
                model_label = base_path.name # Fallback to full name
            
            ax.errorbar(j_values, initial_energies, yerr=initial_energy_errors,
                        fmt='o', linestyle='-', color=colors(model_idx), capsize=5, markersize=8,
                        label=model_label)

    # --- Final plot styling ---
    ax.set_xlabel("$J_2$", fontsize=12)
    ax.set_ylabel("Initial Energy", fontsize=12)
    ax.set_title(f"Initial Energy vs. $J_2$", fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(loc='best')
    plt.tight_layout()

    save_path = Path("/scratch/f/F.Conoscenti/Thesis_QSL/Elaborate/plot") / "Initial_Energy_vs_J_Comparison.png"
    plt.savefig(save_path, dpi=300)
    print(f"✅ Plot saved to {save_path}")
    plt.show()

if __name__ == '__main__':
    model_path =[]
    model_path.append("/scratch/f/F.Conoscenti/Thesis_QSL/HFDS_Heisenberg/plot/spin_new/layers1_hidd1_feat2_sample256_lr0.025_iter2_parityTrue_rotTrue_transFalse_InitFermi_typecomplex")
    model_path.append("/scratch/f/F.Conoscenti/Thesis_QSL/HFDS_Heisenberg/plot/spin_new/layers1_hidd1_feat2_sample256_lr0.025_iter2_parityTrue_rotTrue_transFalse_InitG_MF_typecomplex")
    model_path.append("/scratch/f/F.Conoscenti/Thesis_QSL/HFDS_Heisenberg/plot/spin_new/layers1_hidd1_feat2_sample256_lr0.025_iter2_parityTrue_rotTrue_transFalse_Initrandom_typecomplex")
    plot_initial_fidelity_vs_Js(model_path)
    plot_initial_energy_vs_Js(model_path)
