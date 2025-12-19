import os
import re
import matplotlib.pyplot as plt
import numpy as np
import pickle
from pathlib import Path

def plot_S_eigenvalues_vs_Js(base_folder, Js, num_eigenvalues_to_plot=1000):
    """
    For a single model, reads S-matrix eigenvalues from saved pickle files for different J values
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
        if j_path.exists():
            for entry in j_path.iterdir():
                if entry.is_dir() and entry.name.startswith("seed_"):
                    seed_folder = entry
                    break
        
        if not seed_folder:
            print(f"Warning: No seed folder found for J={J_val} in {j_path}. Skipping.")
            continue

        # Try to find the variables file
        variables_file = seed_folder / "variables"
        if not variables_file.exists():
            variables_file = seed_folder / "variables.pkl"
        
        eigenvalues = None
        
        if variables_file.exists():
            with open(variables_file, 'rb') as f:
                data = pickle.load(f)
                # Try to get eigenvalues_end (dict of iterations)
                eigenvalues_data = data.get('eigenvalues_end')
                
                if eigenvalues_data is None:
                     # Fallback to 'eigenvalues' if 'eigenvalues_end' is missing
                     eigenvalues_data = data.get('eigenvalues', data.get('eigenvalues_mean'))

                if isinstance(eigenvalues_data, dict):
                    # Extract the last iteration
                    try:
                        # Filter keys that match 'iter_{int}'
                        iter_keys = [k for k in eigenvalues_data.keys() if k.startswith('iter_')]
                        if iter_keys:
                            # Sort by iteration number
                            max_iter_key = max(iter_keys, key=lambda x: int(x.split('_')[1]))
                            eigenvalues = eigenvalues_data[max_iter_key]
                        else:
                             # Fallback if dictionary structure is different
                             eigenvalues = list(eigenvalues_data.values())[-1]
                    except Exception as e:
                        print(f"Error parsing eigenvalues dictionary for J={J_val}: {e}")
                elif eigenvalues_data is not None:
                    eigenvalues = eigenvalues_data

        # If still None, try the old path
        if eigenvalues is None:
             eigenvalues_file = seed_folder / "Sign_plot" / "S_matrix_eigenvalues.pkl"
             if eigenvalues_file.exists():
                 with open(eigenvalues_file, 'rb') as f:
                     data = pickle.load(f)
                     eigenvalues = data.get('eigenvalues', data.get('eigenvalues_mean'))

        if eigenvalues is None:
            print(f"Warning: Could not find eigenvalues for J={J_val}. Skipping.")
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
    plt.title(f"{model_type} S-Matrix Eigenvalue Spectrum vs. J (End Training)", fontsize=14)
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.legend(title="J values")
    plt.tight_layout()
    
    # Save the plot
    save_path = f"/cluster/home/fconoscenti/Thesis_QSL/Elaborate/plot/S_eigenvalues_vs_J_{model_type}_end_training.png"
    plt.savefig(save_path, dpi=300)
    print(f"✅ Plot saved to {save_path}")
    plt.show()

def plot_S_eigenvalues_models_vs_J(model_paths, Js, num_eigenvalues_to_plot=1000):
    """
    For each J value, creates a plot comparing the S-matrix eigenvalue spectrum
    across multiple models.

    Args:
        model_paths (list[str]): A list of base directories for different models.
        Js (list): A list of J values to process.
        num_eigenvalues_to_plot (int): The number of largest eigenvalues to plot.
    """
    for J_val in Js:
        plt.figure(figsize=(12, 7))
        ax = plt.gca()
        
        # Use a colormap for different models
        colors = plt.cm.rainbow(np.linspace(0, 1, len(model_paths)))

        for i, model_path in enumerate(model_paths):
            base_folder = Path(model_path)
            
            # --- Custom Label Generation ---
            if "modeamp" in model_path:
                model_type_label = "ViT exact amplitudes"
            elif "modesign" in model_path:
                model_type_label = "ViT exact sign"
            elif "modepsi" in model_path:
                model_type_label = "ViT psi"
            else:
                # Fallback label if no mode is specified in the path
                model_type_label = base_folder.name

            j_path = base_folder / f"J={J_val}"
            
            seed_folder = None
            if j_path.is_dir():
                for entry in j_path.iterdir():
                    if entry.is_dir() and entry.name.startswith("seed_"):
                        seed_folder = entry
                        break
            
            if not seed_folder:
                print(f"Warning: No seed folder found for J={J_val} in {j_path}. Skipping model.")
                continue

            # Check for both possible filenames
            eigenvalues_file = seed_folder / "Sign_plot" / "S_matrix_eigenvalues.pkl"
            if not eigenvalues_file.exists():
                eigenvalues_file = seed_folder / "variables.pkl" # Fallback for older scripts

            if not eigenvalues_file.exists():
                print(f"Warning: Eigenvalues file not found for J={J_val} in {seed_folder}. Skipping.")
                continue

            with open(eigenvalues_file, 'rb') as f:
                data = pickle.load(f)
                eigenvalues = data.get('eigenvalues', data.get('eigenvalues_mean'))

                if eigenvalues is None:
                    print(f"Warning: No eigenvalue data in {eigenvalues_file}. Skipping.")
                    continue
                
                sorted_eigenvalues = np.sort(np.abs(eigenvalues.flatten()))[::-1]
                eigenvalues_to_plot = sorted_eigenvalues[:num_eigenvalues_to_plot]
                indices = np.arange(len(eigenvalues_to_plot))
                
                ax.plot(indices, eigenvalues_to_plot, label=model_type_label, color=colors[i], alpha=0.8)

        ax.set_xlabel("Eigenvalue Index (sorted by magnitude)", fontsize=12)
        ax.set_ylabel("Eigenvalue Magnitude", fontsize=12)
        ax.set_yscale("log")
        ax.set_title(f"S-Matrix Eigenvalue Spectrum for J = {J_val}", fontsize=14)
        ax.grid(True, which="both", linestyle="--", alpha=0.5)
        ax.legend(title="Models", loc='upper right')
        plt.tight_layout()
        
        save_path = f"/cluster/home/fconoscenti/Thesis_QSL/Elaborate/plot/S_eigenvalues_J_{J_val}_model_comparison.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Plot for J={J_val} saved to {save_path}")
        plt.show()

def plot_S_eigenvalues_histogram_vs_Js(base_folder, Js, bins=50):
    """
    For a single model, reads S-matrix eigenvalues from saved pickle files for different J values
    and plots them as a histogram.
    x axis: log10(eigenvalues)
    y axis: number of eigenvalues for that interval

    Args:
        base_folder (str): The base directory containing J folders (e.g., 'J=0.2', 'J=0.5').
        Js (list): A list of J values to process.
        bins (int): Number of bins for the histogram.
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
        if j_path.exists():
            for entry in j_path.iterdir():
                if entry.is_dir() and entry.name.startswith("seed_"):
                    seed_folder = entry
                    break
        
        if not seed_folder:
            print(f"Warning: No seed folder found for J={J_val} in {j_path}. Skipping.")
            continue

        # Try to find the variables file
        variables_file = seed_folder / "variables"
        if not variables_file.exists():
            variables_file = seed_folder / "variables.pkl"
        
        eigenvalues = None
        
        if variables_file.exists():
            with open(variables_file, 'rb') as f:
                data = pickle.load(f)
                # Try to get eigenvalues_end (dict of iterations)
                eigenvalues_data = data.get('eigenvalues_end')
                
                if eigenvalues_data is None:
                     # Fallback to 'eigenvalues' if 'eigenvalues_end' is missing
                     eigenvalues_data = data.get('eigenvalues', data.get('eigenvalues_mean'))

                if isinstance(eigenvalues_data, dict):
                    # Extract the last iteration
                    try:
                        # Filter keys that match 'iter_{int}'
                        iter_keys = [k for k in eigenvalues_data.keys() if k.startswith('iter_')]
                        if iter_keys:
                            # Sort by iteration number
                            max_iter_key = max(iter_keys, key=lambda x: int(x.split('_')[1]))
                            eigenvalues = eigenvalues_data[max_iter_key]
                        else:
                             # Fallback if dictionary structure is different
                             eigenvalues = list(eigenvalues_data.values())[-1]
                    except Exception as e:
                        print(f"Error parsing eigenvalues dictionary for J={J_val}: {e}")
                elif eigenvalues_data is not None:
                    eigenvalues = eigenvalues_data

        # If still None, try the old path
        if eigenvalues is None:
             eigenvalues_file = seed_folder / "Sign_plot" / "S_matrix_eigenvalues.pkl"
             if eigenvalues_file.exists():
                 with open(eigenvalues_file, 'rb') as f:
                     data = pickle.load(f)
                     eigenvalues = data.get('eigenvalues', data.get('eigenvalues_mean'))

        if eigenvalues is None:
            print(f"Warning: Could not find eigenvalues for J={J_val}. Skipping.")
            continue
            
        # Compute log10 of eigenvalues
        # Filter out zeros or negative values (though magnitudes should be positive)
        magnitudes = np.abs(eigenvalues)
        # Filter small values to avoid log(0)
        valid_mask = magnitudes > 1e-30
        if not np.any(valid_mask):
            print(f"Warning: All eigenvalues effectively zero for J={J_val}. Skipping.")
            continue
            
        log_eigenvalues = np.log10(magnitudes[valid_mask])

        # Plot histogram
        plt.hist(log_eigenvalues, bins=bins, label=f'J = {J_val}', color=colors[i], alpha=0.4, histtype='stepfilled', edgecolor=colors[i])

    # --- Final plot styling ---
    plt.xlabel("log10(Eigenvalue Magnitude)", fontsize=12)
    plt.ylabel("Count", fontsize=12)
    plt.title(f"{model_type} S-Matrix Eigenvalue Histogram vs. J", fontsize=14)
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.legend(title="J values")
    plt.tight_layout()
    
    # Save the plot
    save_path = f"/cluster/home/fconoscenti/Thesis_QSL/Elaborate/plot/S_eigenvalues_histogram_vs_J_{model_type}.png"
    plt.savefig(save_path, dpi=300)
    print(f"✅ Plot saved to {save_path}")
    plt.show()

if __name__ == '__main__':
    
    #--- Plot for one model vs. different Js ---
    
    #model_path_single = "/cluster/home/fconoscenti/Thesis_QSL/HFDS_Heisenberg/plot/spin_new/layers1_hidd2_feat16_sample1024_lr0.025_iter100_parityTrue_rotTrue_Initrandom_typecomplex_modepsi_to_plot"
    model_path_single = "/cluster/home/fconoscenti/Thesis_QSL/ViT_Heisenberg/plot/Vision_new/layers2_d8_heads4_patch2_sample1024_lr0.0075_iter500_parityTrue_rotTrue_modepsi_sym3"
    Js_to_plot_single = [0.0, 0.2, 0.4, 0.5, 0.7]
    #plot_S_eigenvalues_vs_Js(model_path_single, Js_to_plot_single, num_eigenvalues_to_plot=1000)
    plot_S_eigenvalues_histogram_vs_Js(model_path_single, Js_to_plot_single, bins=200)
    
    
    """
    # --- Plot for multiple models vs. J (new function) ---
    model_paths_multi = ["/cluster/home/fconoscenti/Thesis_QSL/ViT_Heisenberg/plot/Vision_new/layers2_d8_heads4_patch2_sample1024_lr0.0075_iter500_parityTrue_rotTrue_modepsi_sym3"]

    Js_to_plot_multi = [0.0, 0.2, 0.4, 0.5, 0.6, 0.7]
    plot_S_eigenvalues_models_vs_J(model_paths_multi, Js_to_plot_multi, num_eigenvalues_to_plot=1000)
    """