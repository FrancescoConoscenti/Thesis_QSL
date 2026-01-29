import os
import re
import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def get_data_from_average(model_folder, j_val, part_training='end'):
    """
    Reads params and relevant eigenvalues from variables_average.pkl
    """
    model_path = Path(model_folder)
    if not model_path.exists():
        return None, None, None

    # Find J folder
    target_j_folder = None
    for d in model_path.iterdir():
        if d.is_dir():
            # Check if folder name contains J=j_val or J2=j_val
            if f"J={j_val}" in d.name or f"J2={j_val}" in d.name:
                target_j_folder = d
                break
    
    if target_j_folder is None:
        return None, None, None

    avg_file = target_j_folder / "variables_average.pkl"
    if not avg_file.exists():
        avg_file = target_j_folder / "variables_average"
    
    if not avg_file.exists():
        return None, None, None

    try:
        with open(avg_file, "rb") as f:
            data = pickle.load(f)
        
        n_params = data.get('params')
        if n_params is None:
            n_params = data.get('count_params')
        
        # If params not in pickle, try to read from output.txt in the same J folder (if seeds exist)
        if n_params is None:
             # Try to find output.txt in seed folders
            for seed_folder in target_j_folder.iterdir():
                if seed_folder.is_dir() and seed_folder.name.startswith("seed_"):
                    output_file = seed_folder / "output.txt"
                    if output_file.exists():
                        with open(output_file, "r") as f:
                            content = f.read()
                            match = re.search(r"Total number of parameters:\s*(\d+)", content)
                            if match: 
                                n_params = int(match.group(1))
                                break
                            match = re.search(r"Number of parameters\s*=\s*(\d+)", content)
                            if match: 
                                n_params = int(match.group(1))
                                break
                            match = re.search(r"params=(\d+)", content)
                            if match: 
                                n_params = int(match.group(1))
                                break
        
        key_mean = f'number_relevant_S_eigenvalues_{part_training}_mean'
        key_var = f'number_relevant_S_eigenvalues_{part_training}_var'
        
        if key_mean not in data:
             # Try alternative keys
             if part_training == 'end' and 'rank_S_mean' in data:
                 key_mean = 'rank_S_mean'
                 key_var = 'rank_S_var'
             elif 'number_relevant_S_eigenvalues_mean' in data:
                 key_mean = 'number_relevant_S_eigenvalues_mean'
                 key_var = 'number_relevant_S_eigenvalues_var'

        n_eigs = data.get(key_mean)
        n_eigs_var = data.get(key_var)
        
        n_eigs_std = np.sqrt(n_eigs_var) if n_eigs_var is not None else 0.0
        
        return n_params, n_eigs, n_eigs_std

    except Exception as e:
        print(f"Error reading {avg_file}: {e}")
        return None, None, None

def get_model_style(model_path):
    path_str = str(model_path)
    if "ViT" in path_str:
        return 'o', 'tab:orange', "ViT"
    elif "HFDS" in path_str:
        return '^', 'tab:blue', "HFDS"
    return 's', 'gray', "Model"

def plot_relevant_eigenvalues_vs_params(model_paths, j_val, part_training='end'):
    """
    Plots the number of relevant S-matrix eigenvalues vs number of parameters for given models.
    """
    data = [] # List of (n_params, n_eigs, n_eigs_std, label, marker, color)

    for model_path in model_paths:
        n_params, n_eigs, n_eigs_std = get_data_from_average(model_path, j_val, part_training)
        
        if n_params is None:
            print(f"⚠️ Could not find parameter count for {Path(model_path).name}")
            continue
        
        if n_eigs is None:
            print(f"⚠️ Could not find relevant eigenvalues for {Path(model_path).name}")
            continue
            
        marker, color, type_label = get_model_style(model_path)
        # Create a label (simplified)
        label = f"{type_label} ({n_params} params)"
        data.append((n_params, n_eigs, n_eigs_std, label, marker, color))

    if not data:
        print("No data found to plot.")
        return

    # Sort by number of parameters
    data.sort(key=lambda x: x[0])
    
    plt.figure(figsize=(10, 7))
    
    # Plot points
    for x, y, yerr, label, marker, color in data:
        plt.errorbar(x, y, yerr=yerr, fmt=marker, color=color, label=label, capsize=5, markersize=10, alpha=0.8)

    plt.xlabel("Number of Parameters", fontsize=12)
    plt.ylabel(f"Number of Relevant S-Matrix Eigenvalues ({part_training})", fontsize=12)
    plt.title(f"Relevant S-Matrix Eigenvalues vs Parameters (J={j_val})", fontsize=14)
    plt.xscale("log")
    plt.grid(True, which="both", ls="--", alpha=0.5)
    
    #if len(data) > 0:
    #plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    
    plt.tight_layout()
    
    save_path = f"/cluster/home/fconoscenti/Thesis_QSL/Elaborate/plot/QGT/QGT_rank_vs_Params_J{j_val}_{part_training}.png"
    if not os.path.exists(os.path.dirname(save_path)):
        save_path = save_path.replace("/cluster/home/fconoscenti/Thesis_QSL", "/scratch/f/F.Conoscenti/Thesis_QSL")
        
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Plot saved to {save_path}")
    plt.show()

if __name__ == "__main__":
    # Example usage

    models = [
        "/cluster/home/fconoscenti/Thesis_QSL/ViT_Heisenberg/plot/6x6/layers2_d8_heads1_patch2_sample1024_lr0.0075_iter20_parityTrue_rotTrue_latest_model",
        "/cluster/home/fconoscenti/Thesis_QSL/ViT_Heisenberg/plot/6x6/layers1_d8_heads1_patch2_sample1024_lr0.0075_iter20_parityTrue_rotTrue_latest_model"
    ]

    
    valid_models = []
    for m in models:
        if os.path.exists(m):
            valid_models.append(m)

    plot_relevant_eigenvalues_vs_params(valid_models, j_val=0.5, part_training='end')
