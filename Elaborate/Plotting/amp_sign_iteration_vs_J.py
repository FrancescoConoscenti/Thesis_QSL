import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import re

def plot_avg_overlap_vs_J(model_path: str):
    """
    Loads averaged data for different J values from a model's result directory
    and plots the Sign Overlap vs. Amplitude Overlap on a single graph.

    Each point on the scatter plot represents a saved model checkpoint during
    the optimization for a specific J value.

    Args:
        model_path (str): The path to the main model directory containing J=... subfolders.
    """
    base_path = Path(model_path)
    if not base_path.is_dir():
        print(f"Error: Provided path '{model_path}' is not a valid directory.")
        return

    # --- Plot setup ---
    fig, ax = plt.subplots(figsize=(10, 8))
    cmap = plt.get_cmap("tab10") # Use a colormap with more distinct colors

    # --- Find and process J folders ---
    j_folders = sorted([d for d in base_path.iterdir() if d.is_dir() and d.name.startswith("J=")])

    if not j_folders:
        print(f"No 'J=...' subdirectories found in '{model_path}'.")
        return

    # Filter out J=1.0 before assigning colors
    j_folders_to_plot = [p for p in j_folders if not p.name.endswith("=1.0")]
    if not j_folders_to_plot:
        print("No J folders to plot after filtering.")
        return

    for i, j_path in enumerate(j_folders):
        try:
            j_value_str = j_path.name.split('=')[1]
            j_value = float(j_value_str)
        except (ValueError, IndexError):
            print(f"Warning: Could not parse J value from folder name: {j_path.name}. Skipping.")
            continue

        # Skip plotting for J=1.0 as requested
        if j_value == 1.0:
            print(f"Skipping J={j_value} as requested.")
            continue

        avg_file_path = j_path / "variables_average"
        if not avg_file_path.exists():
            print(f"Warning: '{avg_file_path}' not found. Skipping J={j_value}.")
            continue

        # --- Load data ---
        with open(avg_file_path, "rb") as f:
            loaded_data = pickle.load(f)

        # --- Extract data for plotting ---
        try:
            amp_overlap = loaded_data['amp_overlap_mean']
            sign_overlap = loaded_data['sign_overlap_mean']
            amp_overlap_var = loaded_data.get('amp_overlap_var') # Use .get for safety
            sign_overlap_var = loaded_data.get('sign_overlap_var')
        except KeyError as e:
            print(f"Warning: Missing key {e} in data for J={j_value}. Skipping.")
            continue

        if amp_overlap_var is not None and sign_overlap_var is not None:
            xerr = np.sqrt(amp_overlap_var)
            yerr = np.sqrt(sign_overlap_var)
            # Plot error bars with higher transparency
            ax.errorbar(amp_overlap, sign_overlap, xerr=xerr, yerr=yerr, fmt='none',
                        ecolor=cmap(i % 10), elinewidth=1, capsize=3, alpha=0.4)
            # Plot points on top with lower transparency
            ax.scatter(amp_overlap, sign_overlap, marker='o', color=cmap(i % 10), alpha=0.9, label=f'J = {j_value}')
        else:
            ax.scatter(amp_overlap, sign_overlap, marker='o', color=cmap(i % 10), alpha=0.9, label=f'J = {j_value}')

    # --- Final plot styling ---
    ax.set_xlabel("Amplitude Overlap", fontsize=12)
    ax.set_ylabel("Sign Overlap", fontsize=12)
    ax.set_title("Sign Overlap vs. Amplitude Overlap for different J values", fontsize=14)
    ax.grid(True, alpha=0.4)
    ax.legend(loc='best', title="J₂ values")
    ax.set_xlim(0.5, 1)
    ax.set_ylim(0.5, 1)
    plt.tight_layout()

    save_path = "/scratch/f/F.Conoscenti/Thesis_QSL/Elaborate/plot/Avg_Sign_vs_Amp_Overlap_vs_J.png"
    plt.savefig(save_path, dpi=300)
    print(f"✅ Plot saved to {save_path}")
    plt.show()

if __name__ == '__main__':
    # Example usage:
    model_path = "/scratch/f/F.Conoscenti/Thesis_QSL/ViT_Heisenberg/plot/ViT_new/layers2_d16_heads4_patch2_sample1024_lr0.0075_iter3000_symmTrue_new"
    plot_avg_overlap_vs_J(model_path)
