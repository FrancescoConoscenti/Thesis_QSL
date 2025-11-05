import os
import re
import matplotlib.pyplot as plt
import numpy as np
import pickle
from pathlib import Path

def plot_S_eigenvalues_vs_Js(base_folder, Js, num_eigenvalues_to_plot=1000):
    """
    Reads S-matrix eigenvalues from saved pickle files for different J values
    and plots them on a single graph.

    Args:
        base_folder (str): The base directory containing J folders (e.g., 'J=0.2', 'J=0.5').
        Js (list): A list of J values to process.
    """
    plt.figure(figsize=(12, 7))
    
    # --- Recognize model type from the base_folder path ---
    if "HFDS_Heisenberg" in base_folder:
        model_type = "HFDS"
    elif "ViT_Heisenberg" in base_folder:
        model_type = "ViT"
    else:
        model_type = "UnknownModel"

    # Use a colormap that provides distinct colors for different J values
    colors = plt.cm.viridis(np.linspace(0, 1, len(Js))) 

    for i, J_val in enumerate(Js):
        j_path = Path(base_folder) / f"J={J_val}"
        
        # Find the first seed folder to load data from
        seed_folder = None
        for entry in j_path.iterdir():
            if entry.is_dir() and entry.name.startswith("seed_"):
                seed_folder = entry
                break
        
        if not seed_folder:
            print(f"Warning: No seed folder found for J={J_val} in {j_path}. Skipping.")
            continue

        eigenvalues_file = seed_folder / "Sign_plot" / "S_matrix_eigenvalues.pkl"

        if not eigenvalues_file.exists():
            print(f"Warning: Eigenvalues file not found for J={J_val} at {eigenvalues_file}. Skipping.")
            continue

        with open(eigenvalues_file, 'rb') as f:
            data = pickle.load(f)
            # The key might be 'eigenvalues' or 'eigenvalues_mean' depending on the saving script
            eigenvalues = data.get('eigenvalues', data.get('eigenvalues_mean'))

            if eigenvalues is None:
                print(f"Warning: Could not find 'eigenvalues' or 'eigenvalues_mean' key for J={J_val}. Skipping.")
                continue
            
            # Sort eigenvalues in descending order of magnitude
            sorted_eigenvalues = np.sort(np.abs(eigenvalues))[::-1] 

            # Select the top N eigenvalues to plot and create the corresponding x-axis
            eigenvalues_to_plot = sorted_eigenvalues[:num_eigenvalues_to_plot]
            indices = np.arange(len(eigenvalues_to_plot))
            
            # Plot the spectrum for the current J value using the sliced data
            plt.plot(indices, eigenvalues_to_plot, label=f'J = {J_val}', color=colors[i], alpha=0.8)

    # --- Final plot styling ---
    plt.xlabel("Eigenvalue Index (sorted by magnitude)", fontsize=12)
    plt.ylabel("Eigenvalue Magnitude", fontsize=12)
    plt.yscale("log")  # Use a log scale to see the full range of values
    plt.title(f"{model_type} S-Matrix Eigenvalue Spectrum vs. J", fontsize=14)
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.legend(title="J values")
    plt.tight_layout()
    
    # Save the plot
    save_path = f"/scratch/f/F.Conoscenti/Thesis_QSL/Elaborate/plot/S_eigenvalues_vs_J_{model_type}.png"
    plt.savefig(save_path, dpi=300)
    print(f"✅ Plot saved to {save_path}")
    plt.show()

if __name__ == '__main__':
    # Example usage:
    model_path = "/scratch/f/F.Conoscenti/Thesis_QSL/HFDS_Heisenberg/plot/spin_new/layers1_hidd4_feat32_sample1024_lr0.025_iter500_parityTrue_rotationFalse_InitFermi_typereal"
    Js_to_plot = [0.2, 0.5, 0.7, 1.0]
    plot_S_eigenvalues_vs_Js(model_path, Js_to_plot, num_eigenvalues_to_plot=1000)
