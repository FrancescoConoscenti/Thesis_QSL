import numpy as np
import matplotlib.pyplot as plt
import pickle
from pathlib import Path
import os
import flax
from Elaborate.S_matrix_Obs import *
import jax
import gc
import scipy.linalg
import scipy.sparse.linalg

from Elaborate.S_matrix_Obs import compute_S_matrix_single_model
from Elaborate.S_matrix_Obs import compute_S_matrix_dense



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
    # Only evaluate and plot the first and last model iterations
    if len(indices_to_plot) > 1:
        indices_to_plot = [indices_to_plot[0], indices_to_plot[-1]]

    all_eigenvalues = {}
    relevant_eigenvalues_counts = []
    relevant_counts_rest_ratio = []
    relevant_counts_rest_norm = []
    relevant_counts_rest_norm_12 = []
    relevant_count_first = 0

    for i, model_idx in enumerate(indices_to_plot):
        model_file = models_dir / f"model_{model_idx}.mpack"
        with open(model_file, "rb") as f:
            data = f.read()
            try:
                vstate = flax.serialization.from_bytes(vstate, data)
            except KeyError:
                vstate.variables = flax.serialization.from_bytes(vstate.variables, data)

        jax.clear_caches()
        gc.collect()
        
        

        # Compute S-matrix for the current model
        S_dense = compute_S_matrix_single_model(vstate, hi)

        eigenvalues = scipy.linalg.eigvalsh(S_dense)
        all_eigenvalues[f'iter_{model_idx}'] = eigenvalues
        

        #compute eigenvalues using the dense S-matrix (fallback for small models)
        """
        S_matrix = compute_S_matrix_dense(vstate)

        
        # Use eigsh to compute only a subset of eigenvalues (e.g. top 4096) to save memory
        N = S_matrix.shape[0]
        # scipy's eigsh (ARPACK) requires k < N - 1 for LinearOperators
        k_eig = min(N - 2, 8192)
        
        if k_eig < 1:
            # Fallback for extremely small models where ARPACK cannot be used
            dense_S = np.zeros((N, N), dtype=np.complex128)
            for j in range(N):
                e_j = np.zeros(N)
                e_j[j] = 1.0
                dense_S[:, j] = S_matrix.matvec(e_j)
            eigenvalues = scipy.linalg.eigvalsh(dense_S)
        else:
            eigenvalues = scipy.sparse.linalg.eigsh(S_matrix, k=k_eig, which='LM', return_eigenvectors=False)
        all_eigenvalues[f'iter_{model_idx}'] = eigenvalues
        """

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
                c_norm_12 = np.sum(norm_eigs > 1e-12)
            else:
                c_norm = 0
                c_norm_12 = 0
            relevant_counts_rest_norm.append(c_norm)
            relevant_counts_rest_norm_12.append(c_norm_12)
            

    mean_rest_ratio = np.mean(relevant_counts_rest_ratio) if relevant_counts_rest_ratio else 0
    mean_rest_norm = np.mean(relevant_counts_rest_norm) if relevant_counts_rest_norm else 0
    mean_rest_norm_12 = np.mean(relevant_counts_rest_norm_12) if relevant_counts_rest_norm_12 else 0
    
    return all_eigenvalues, relevant_count_first, mean_rest_ratio, mean_rest_norm, mean_rest_norm_12

def plot_S_matrix_spectrum(all_eigenvalues, indices_to_plot, folder_path, num_models):
    folder_save_QGT = folder_path+"/QGT_plot"
    os.makedirs(folder_save_QGT, exist_ok=True)
    
    plt.figure(figsize=(12, 7))
    
    # Setup colormap for the 5 lines
    cmap = plt.get_cmap('viridis')
    
    # Force filtering to only look at iterations 0, 5, 10, 15, 20
    target_indices = [0, 5, 10, 15, 20]
    
    # Custom labels mapping to the 5 targeted iterations
    legend_labels = [
        "iter 0", 
        "iter 1000", 
        "iter 2000", 
        "iter 3000", 
        "iter 4000"
    ]
        
    for i, model_idx in enumerate(target_indices):
        # Safety check: ensure the iteration exists in your data before plotting
        if f'iter_{model_idx}' not in all_eigenvalues:
            continue
            
        eigenvalues = all_eigenvalues[f'iter_{model_idx}']
        sorted_eigenval = np.sort(eigenvalues)[::-1]
        indices = np.arange(len(sorted_eigenval))
        
        # Color gradient based on the 5 target lines
        color = cmap(i / (len(target_indices) - 1))
        
        # Plot with the custom label for the legend
        plt.plot(indices, sorted_eigenval, lw=3.5, color=color, alpha=0.8, label=legend_labels[i])
            
    plt.xlabel("Eigenvalue Index")
    plt.ylabel("Eigenvalue Magnitude")
    plt.yscale("log")
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    
    # REMOVED: Colorbar code
    # ADDED: Standard Legend
    plt.legend(loc="upper right")

    plt.tight_layout()

    # Determine suffix
    param_count = None
    try:
        with open(Path(folder_path)/"variables.pkl", "rb") as f:
            data = pickle.load(f)
            param_count = data.get("count_params", data.get("params", None))
    except:
        pass
        
    m_type = "Model"
    if "ViT" in str(folder_path): m_type = "ViT"
    elif "HFDS" in str(folder_path): m_type = "HFDS"
    
    suffix = f"_{m_type}"
    if param_count is not None:
        suffix += f"_{param_count}params"

    save_plot_path = Path(folder_save_QGT) / f"S_matrix_spectrum_vs_iteration{suffix}.png"
        
    plt.savefig(save_plot_path, dpi=300)
    print(f"✅ Plot saved to {save_plot_path}")
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

    all_eigenvalues, relevant_count_first, mean_rest_ratio, mean_rest_norm, mean_rest_norm_12 = calculate_relevant_eigenvalues(vstate, folder_path, hi, threshold_ratio_rest)
    
    if not all_eigenvalues:
        return None, None, None, None, None

    indices_to_plot = sorted([int(k.split('_')[1]) for k in all_eigenvalues.keys()])
    num_models = len(indices_to_plot)
    
    plot_S_matrix_spectrum(all_eigenvalues, indices_to_plot, folder_path, num_models)

    # --- Save all computed eigenvalues ---
    os.makedirs(Path(folder_path) / "QGT_plot", exist_ok=True)
    save_data_path = Path(folder_path) / "QGT_plot" / "S_matrix_eigenvalues.pkl"
    with open(save_data_path, 'wb') as f:
        pickle.dump(all_eigenvalues, f)
    #print(f"S-matrix eigenvalues saved to {save_data_path}")

    return all_eigenvalues, relevant_count_first, mean_rest_ratio, mean_rest_norm, mean_rest_norm_12


def Plot_S_matrix_eigenvalues(eigenvalues, folder_path, one_avg):

    print("Plotting S-matrix eigenvalues...", eigenvalues.shape)

    sorted_eigenval = np.sort(eigenvalues)[::-1]
    indices = np.arange(len(sorted_eigenval))  # x-axis: eigenvalue index

    # Plot
    # Determine suffix
    param_count = None
    try:
        with open(Path(folder_path)/"variables.pkl", "rb") as f:
            data = pickle.load(f)
            param_count = data.get("count_params", data.get("params", None))
    except:
        pass
        
    m_type = "Model"
    if "ViT" in str(folder_path): m_type = "ViT"
    elif "HFDS" in str(folder_path): m_type = "HFDS"
    
    suffix = f"_{m_type}"
    if param_count is not None:
        suffix += f"_{param_count}params"

    plt.figure(figsize=(8,4))
    plt.plot(indices, sorted_eigenval, lw=4.0, color='darkgreen', marker='.', markersize=3, linestyle='-')
    
    plt.title("Eigenvalue Spectrum of the S-matrix (Final Model)")
    plt.xlabel("eigenvalues index")
    plt.ylabel("eigenvalue")
    plt.yscale("log")
    if one_avg == "avg":
        folder_path = Path(folder_path)
        save_path = folder_path.parent /"plot_avg"/"S_matrix_spectrum.png"
        plt.savefig(save_path)
    if one_avg == "one":
        os.makedirs(f"{folder_path}/QGT_plot", exist_ok=True)
        plt.savefig(f"{folder_path}/QGT_plot/S_matrix_spectrum{suffix}.png")
    


    plt.show()

def Plot_S_matrix_histogram(eigenvalues, folder_path, one_avg, bins=50):

    plt.figure(figsize=(8,6))

    # Determine suffix
    param_count = None
    try:
        with open(Path(folder_path)/"variables.pkl", "rb") as f:
            data = pickle.load(f)
            param_count = data.get("count_params", data.get("params", None))
    except:
        pass
        
    m_type = "Model"
    if "ViT" in str(folder_path): m_type = "ViT"
    elif "HFDS" in str(folder_path): m_type = "HFDS"
    
    suffix = f"_{m_type}"
    if param_count is not None:
        suffix += f"_{param_count}params"

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
        os.makedirs(f"{folder_path}/QGT_plot", exist_ok=True)
        plt.savefig(f"{folder_path}/QGT_plot/S_matrix_histogram{suffix}.png")

    plt.show()

def plot_S_matrix_spectrum_2(all_eigenvalues1, all_eigenvalues2, indices_to_plot1, indices_to_plot2, folder_path, num_models1, num_models2):
    os.makedirs(folder_path, exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    
    cmap = plt.get_cmap('viridis')
    target_indices = [0, 5, 10, 15, 20]
    legend_labels = [
        "iter 0", 
        "iter 1000", 
        "iter 2000", 
        "iter 3000", 
        "iter 4000"
    ]

    # Plot for Model 1
    for i, model_idx in enumerate(target_indices):
        if f'iter_{model_idx}' in all_eigenvalues1:
            eigenvalues = all_eigenvalues1[f'iter_{model_idx}']
            sorted_eigenval = np.sort(eigenvalues)[::-1]
            indices = np.arange(len(sorted_eigenval))
            color = cmap(i / (len(target_indices) - 1))
            ax1.plot(indices, sorted_eigenval, lw=3.5, color=color, alpha=0.8, label=legend_labels[i])
            
    ax1.set_xlabel("Eigenvalue Index")
    ax1.set_ylabel("Eigenvalue Magnitude")
    ax1.set_yscale("log")
    ax1.set_xlim(0,20000)
    ax1.grid(True, which="both", linestyle="--", alpha=0.5)

    # Plot for Model 2
    for i, model_idx in enumerate(target_indices):
        if f'iter_{model_idx}' in all_eigenvalues2:
            eigenvalues = all_eigenvalues2[f'iter_{model_idx}']
            sorted_eigenval = np.sort(eigenvalues)[::-1]
            indices = np.arange(len(sorted_eigenval))
            color = cmap(i / (len(target_indices) - 1))
            ax2.plot(indices, sorted_eigenval, lw=3.5, color=color, alpha=0.8, label=legend_labels[i])

    ax2.set_xlabel("Eigenvalue Index")
    ax2.set_yscale("log")
    ax2.set_xlim(0,20000)
    ax2.grid(True, which="both", linestyle="--", alpha=0.5)

    # Shared Legend
    handles, labels = ax1.get_legend_handles_labels()
    if not handles:
        handles, labels = ax2.get_legend_handles_labels()
    ax2.legend(handles, labels, loc='upper right', fontsize=16)

    plt.tight_layout()
    
    save_plot_path = Path(folder_path) / "S_matrix_spectrum_comparison.png"
    plt.savefig(save_plot_path, dpi=300, bbox_inches='tight')
    print(f"✅ Plot saved to {save_plot_path}")
    plt.show()


def main():
    # Example usage
    folder_path1 = "/scratch/f/F.Conoscenti/Thesis_QSL/HFDS_Heisenberg/plot/6x6/layers1_hidd4_feat64_sample4096_lr0.02_iter500_parityTrue_rotTrue_InitFermi_typecomplex/J=0.5/seed_1"
    folder_path2 = "/scratch/f/F.Conoscenti/Thesis_QSL/ViT_Heisenberg/plot/6x6/layers2_d24_heads6_patch2_sample4096_lr0.0075_iter1000_parityTrue_rotTrue_latest_model/J=0.5/seed_1"
    folder_save_QGT = "/scratch/f/F.Conoscenti/Thesis_QSL/Elaborate/plot/QGT"

    #folder1
    variables_path = Path(folder_path1) / "variables.pkl"
    if variables_path.is_file():
        print(f"Loading variables from {variables_path}")
        with open(variables_path, 'rb') as f:
            variables = pickle.load(f)
    
    all_eigenvalues1 = variables.get('eigenvalues_S', None)
    indices_to_plot1 = sorted([int(k.split('_')[1]) for k in all_eigenvalues1.keys()])

    # folder2
    variables_path = Path(folder_path2) / "variables.pkl"
    if variables_path.is_file():
        print(f"Loading variables from {variables_path}")
        with open(variables_path, 'rb') as f:
            variables = pickle.load(f)
    
    all_eigenvalues2 = variables.get('eigenvalues_S', None)
    indices_to_plot2 = sorted([int(k.split('_')[1]) for k in all_eigenvalues2.keys()])

    
    #plot_S_matrix_spectrum(all_eigenvalues, indices_to_plot, folder_save_QGT,  len(indices_to_plot))
    plot_S_matrix_spectrum_2(all_eigenvalues1, all_eigenvalues2, indices_to_plot1, indices_to_plot2, folder_save_QGT, len(indices_to_plot1), len(indices_to_plot2))


if __name__ == "__main__":
    main()  