import numpy as np
import matplotlib.pyplot as plt
import pickle
from pathlib import Path
import os
import flax
from Elaborate.S_matrix_Obs import *


def _count_relevant_eigenvalues(sorted_eigenvalues, threshold_ratio=1e3):
    """
    Counts the number of eigenvalues before a significant drop.
    The drop is detected when the ratio of two consecutive sorted eigenvalues
    exceeds the threshold.

    Args:
        sorted_eigenvalues (np.array): Eigenvalues sorted in descending order.
        threshold_ratio (float): The ratio that defines a significant drop.

    Returns:
        int: The number of eigenvalues before the first major drop.
    """
    for i in range(len(sorted_eigenvalues) - 1):
        current_eig = sorted_eigenvalues[i]
        next_eig = sorted_eigenvalues[i+1]

        if next_eig < 1e-18:  # Stop if we reach numerical precision limits
            return i + 1
        if (current_eig / next_eig) > threshold_ratio:
            return i + 1  # Return the count of eigenvalues before the drop
    return len(sorted_eigenvalues)  # No significant drop found

def calculate_relevant_eigenvalues(vstate, folder_path, hi, threshold_ratio_rest=100):
    models_dir = Path(folder_path) / "models"
    if not models_dir.is_dir():
        print(f"Warning: 'models' directory not found in {folder_path}. Skipping.")
        return {}, 0, 0, 0, 0

    model_files = sorted([f for f in os.listdir(models_dir) if f.startswith("model_") and f.endswith(".mpack")])
    num_models = len(model_files)
    if num_models == 0:
        print(f"Warning: No model files found in {models_dir}. Skipping.")
        return {}, 0, 0, 0, 0

    indices_to_plot = sorted(list(set(range(num_models))))
    all_eigenvalues = {}
    relevant_eigenvalues_counts = []
    relevant_counts_rest_ratio = []
    relevant_counts_rest_norm = []
    relevant_count_first = 0

    for i, model_idx in enumerate(indices_to_plot):
        model_file = models_dir / f"model_{model_idx}.mpack"
        with open(model_file, "rb") as f:
            data = f.read()
            try:
                vstate = flax.serialization.from_bytes(vstate, data)
            except KeyError:
                vstate.variables = flax.serialization.from_bytes(vstate.variables, data)

        S_matrix = compute_S_matrix_single_model(vstate, hi)
        eigenvalues = np.linalg.eigvalsh(S_matrix)
        all_eigenvalues[f'iter_{model_idx}'] = eigenvalues

        sorted_eigenval = np.sort(eigenvalues)[::-1]
        

        if i == 0:
            relevant_count_first = _count_relevant_eigenvalues(sorted_eigenval, threshold_ratio=1e3)
        else:
            # Metric 2: ratio with another threshold
            c_ratio = _count_relevant_eigenvalues(sorted_eigenval, threshold_ratio=threshold_ratio_rest)
            relevant_counts_rest_ratio.append(c_ratio)
            
            # Metric 3: normalized > 1e-16
            if sorted_eigenval[0] > 0:
                norm_eigs = sorted_eigenval / sorted_eigenval[0]
                c_norm = np.sum(norm_eigs > 1e-16)
            else:
                c_norm = 0
            relevant_counts_rest_norm.append(c_norm)
            

    mean_rest_ratio = np.mean(relevant_counts_rest_ratio) if relevant_counts_rest_ratio else 0
    mean_rest_norm = np.mean(relevant_counts_rest_norm) if relevant_counts_rest_norm else 0
    
    return all_eigenvalues, relevant_count_first, mean_rest_ratio, mean_rest_norm

def plot_S_matrix_spectrum(all_eigenvalues, indices_to_plot, folder_path, mean_relevant_eigenvalues, num_models):
    folder_save_QGT = folder_path+"/QGT_plot"
    os.makedirs(folder_save_QGT, exist_ok=True)
    
    plt.figure(figsize=(12, 7))
    
    # Setup colormap for 'all' case
    cmap = plt.get_cmap('viridis')
        
    for i, model_idx in enumerate(indices_to_plot):
        eigenvalues = all_eigenvalues[f'iter_{model_idx}']
        sorted_eigenval = np.sort(eigenvalues)[::-1]
        indices = np.arange(len(sorted_eigenval))
        
        # Use color gradient
        color = cmap(i / (len(indices_to_plot) - 1) if len(indices_to_plot) > 1 else 0)
        plt.plot(indices, sorted_eigenval, lw=1.5, color=color, alpha=0.5)
            
    plt.title(f"Eigenvalue Spectrum of S-matrix (Avg. Relevant Eigenvalues: {mean_relevant_eigenvalues:.2f})")
    plt.xlabel("Eigenvalue Index (sorted descending)")
    plt.ylabel("Eigenvalue Magnitude")
    plt.yscale("log")
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    
    # Add colorbar for 'all'
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=num_models-1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=plt.gca())
    cbar.set_label('Model Iteration')

    plt.tight_layout()

    save_plot_path = Path(folder_save_QGT) / "S_matrix_spectrum_vs_iteration.png"
        
    plt.savefig(save_plot_path, dpi=300)
    #print(f"✅ Plot saved to {save_plot_path}")
    plt.show()

def plot_S_matrix_eigenvalues(vstate, folder_path, hi, one_avg, threshold_ratio_rest=100):
    """
    Computes and plots the eigenvalues of the S-matrix.
    Can plot start, end, or all iterations.

    Args:
        vstate: The variational state.
        hi: The Hamiltonian instance.
        folder_path: Path to save the plots.
        one_avg: A string indicating the context (e.g., "one" for a single run).
    """

    all_eigenvalues, relevant_count_first, mean_rest_ratio, mean_rest_norm= calculate_relevant_eigenvalues(vstate, folder_path, hi, threshold_ratio_rest)
    
    if not all_eigenvalues:
        return None, None, None, None

    indices_to_plot = sorted([int(k.split('_')[1]) for k in all_eigenvalues.keys()])
    num_models = len(indices_to_plot)
    
    plot_S_matrix_spectrum(all_eigenvalues, indices_to_plot, folder_path, mean_rest_norm, num_models)

    # --- Save all computed eigenvalues ---
    save_data_path = Path(folder_path) / "Sign_plot" / "S_matrix_eigenvalues.pkl"
    with open(save_data_path, 'wb') as f:
        pickle.dump(all_eigenvalues, f)
    #print(f"S-matrix eigenvalues saved to {save_data_path}")


def Plot_S_matrix_eigenvalues(eigenvalues, folder_path, one_avg):

    #print("Plotting S-matrix eigenvalues...", eigenvalues.shape)

    sorted_eigenval = np.sort(eigenvalues)[::-1]
    indices = np.arange(len(sorted_eigenval))  # x-axis: eigenvalue index

    # Plot
    plt.figure(figsize=(8,4))
    plt.plot(indices, sorted_eigenval, lw=1.5, color='darkgreen', marker='.', markersize=3, linestyle='-')
    
    plt.title("Eigenvalue Spectrum of the S-matrix (Final Model)")
    plt.xlabel("eigenvalues index")
    plt.ylabel("eigenvalue")
    plt.yscale("log")
    if one_avg == "avg":
        folder_path = Path(folder_path)
        save_path = folder_path.parent /"plot_avg"/"S_matrix_spectrum.png"
        plt.savefig(save_path)
    if one_avg == "one":
        plt.savefig(f"{folder_path}/Sign_plot/S_matrix_spectrum.png")
    


    plt.show()

def Plot_S_matrix_histogram(eigenvalues, folder_path, one_avg, bins=50):

    plt.figure(figsize=(8,6))

    if isinstance(eigenvalues, dict):
        iter_keys = [k for k in eigenvalues.keys() if k.startswith('iter_')]
        if iter_keys:
            sorted_keys = sorted(iter_keys, key=lambda x: int(x.split('_')[1]))
            keys_to_plot = [sorted_keys[0]]
            if len(sorted_keys) > 1:
                keys_to_plot.append(sorted_keys[-1])
            
            colors = ['blue', 'red']
            labels = ['First Iteration', 'Last Iteration']

            for i, key in enumerate(keys_to_plot):
                magnitudes = np.abs(eigenvalues[key])
                valid_mask = magnitudes > 1e-30
                if np.any(valid_mask):
                    log_eigenvalues = np.log10(magnitudes[valid_mask])
                    plt.hist(log_eigenvalues, bins=bins, color=colors[i], alpha=0.5, label=f"{labels[i]} ({key})", edgecolor=colors[i], histtype='stepfilled')
            plt.legend()
    else:
        magnitudes = np.abs(eigenvalues)
        # Filter small values to avoid log(0)
        valid_mask = magnitudes > 1e-30
        if np.any(valid_mask):
            log_eigenvalues = np.log10(magnitudes[valid_mask])
            plt.hist(log_eigenvalues, bins=bins, color='skyblue', edgecolor='black', alpha=0.7)
    
    plt.title("Histogram of S-matrix Eigenvalues")
    plt.xlabel("log10(Eigenvalue)")
    plt.ylabel("Count")
    
    if one_avg == "avg":
        folder_path = Path(folder_path)
        save_path = folder_path.parent /"plot_avg"/"S_matrix_histogram.png"
        plt.savefig(save_path)
    if one_avg == "one":
        plt.savefig(f"{folder_path}/Sign_plot/S_matrix_histogram.png")

    plt.show()