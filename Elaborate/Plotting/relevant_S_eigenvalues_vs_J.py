import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import re

def get_model_label(model_path: str) -> str:
    """
    Creates a concise, readable label from a model's directory path.
    This helps in creating a clean plot legend.

    Args:
        model_path (str): The full path to the model's directory.

    Returns:
        str: A formatted label for the model.
    """
    model_name = Path(model_path).name
    if "ViT_Heisenberg" in model_path:
        mode_match = re.search(r'mode(exact_amp|exact_sign|psi)', model_name)
        if mode_match:
            return f"ViT ({mode_match.group(1)})"
        return "ViT"
    elif "HFDS_Heisenberg" in model_path:
        init_match = re.search(r'Init([A-Za-z_]+)', model_name)
        if init_match:
            return f"HFDS ({init_match.group(1)})"
        return "HFDS"
    return model_name

def plot_relevant_eigenvalues_histogram(model_paths: list[str], j_values: list[float], part_training: str, seed_id: int = 1):
    """
    Generates a grouped bar chart comparing the number of relevant S-matrix eigenvalues
    for different models across various J values.

    Args:
        model_paths (list[str]): A list of paths to the root directories of the models to compare.
        j_values (list[float]): A list of the J values to include on the x-axis.
        seed_id (int): The identifier for the single seed to plot (e.g., 0 for 'seed_0').
    """
    num_models = len(model_paths)
    data = {j: [] for j in j_values}
    model_labels = [get_model_label(p) for p in model_paths]

    # --- Data Collection ---
    for model_path in model_paths:
        for j_val in j_values:
            # Path to the single seed's variable file
            seed_file = Path(model_path) / f"J={j_val}" / f"seed_{seed_id}" / "variables"
            if not seed_file.exists():
                # Fallback for .pkl extension
                seed_file = seed_file.with_suffix('.pkl')

            if not seed_file.exists():
                print(f"Warning: Data file not found for J={j_val}, seed={seed_id} in {model_path}. Skipping.")
                data[j_val].append(np.nan) # Append NaN for missing data
                continue

            with open(seed_file, "rb") as f:
                loaded_data = pickle.load(f)

            if part_training == 'start':
                val = loaded_data.get('number_relevant_S_eigenvalues_start', np.nan)
            elif part_training == 'end':
                val = loaded_data.get('number_relevant_S_eigenvalues_end', np.nan)
            else:       
                raise ValueError(f"Invalid part_training value: {part_training}")
            
            data[j_val].append(val)

    # --- Plotting ---
    fig, ax = plt.subplots(figsize=(14, 8))
    
    x = np.arange(len(j_values))  # the label locations
    width = 0.8 / num_models  # the width of the bars
    
    for i, model_label in enumerate(model_labels):
        values = [data[j][i] for j in j_values]
        
        position = x - (num_models - 1) * width / 2 + i * width
        ax.bar(position, values, width, label=model_label)

    # --- Final plot styling ---
    ax.set_ylabel('Number of Relevant S-Matrix Eigenvalues (Single Seed)', fontsize=12)
    ax.set_xlabel('$J_2$', fontsize=12)
    ax.set_title(f'Comparison of Relevant S-Matrix Eigenvalues Across Models and $J_2$ {part_training} training', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(j_values)
    ax.legend(title="Models")
    ax.grid(True, axis='y', linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    
    # --- Save the plot ---
    if part_training == 'start':
        save_path = "/cluster/home/fconoscenti/Thesis_QSL/Elaborate/plot/relevant_S_eigenvalues_vs_J_start_HFDS.png"
    elif part_training == 'end':
        save_path = "/cluster/home/fconoscenti/Thesis_QSL/Elaborate/plot/relevant_S_eigenvalues_vs_J_end_HFDS.png"

    if not os.path.exists(os.path.dirname(save_path)):
        save_path = save_path.replace("/cluster/home/fconoscenti/Thesis_QSL", "/scratch/f/F.Conoscenti/Thesis_QSL")
    plt.savefig(save_path, dpi=300)
    print(f"✅ Plot saved to {save_path}")
    plt.show()


if __name__ == '__main__':
    # --- Define the models you want to compare (from a single seed run) ---
    # The path should point to the directory containing the J=... subfolders.
    # The script will append the rest of the path (J=..., seed_...).
    model_paths_to_compare = ["/cluster/home/fconoscenti/Thesis_QSL/HFDS_Heisenberg/plot/spin_new/layers1_hidd2_feat16_sample1024_lr0.025_iter100_parityTrue_rotTrue_Initrandom_typecomplex_modeexact_sign_to_plot",
                              "/cluster/home/fconoscenti/Thesis_QSL/HFDS_Heisenberg/plot/spin_new/layers1_hidd2_feat16_sample1024_lr0.025_iter100_parityTrue_rotTrue_Initrandom_typecomplex_modepsi_to_plot"]
    for i in range(len(model_paths_to_compare)):
        if not os.path.exists(model_paths_to_compare[i]):
            model_paths_to_compare[i] = model_paths_to_compare[i].replace("/cluster/home/fconoscenti/Thesis_QSL", "/scratch/f/F.Conoscenti/Thesis_QSL")
    j_values_to_plot = [0.0, 0.2, 0.4, 0.5, 0.6, 0.7]

    # --- Generate the plot for a single seed (e.g., seed_id=0) ---
    plot_relevant_eigenvalues_histogram(model_paths_to_compare, j_values_to_plot, part_training='start',seed_id=1)
    plot_relevant_eigenvalues_histogram(model_paths_to_compare, j_values_to_plot, part_training='end',seed_id=1)
