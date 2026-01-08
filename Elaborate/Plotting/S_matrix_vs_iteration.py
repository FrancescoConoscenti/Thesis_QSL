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

def plot_S_matrix_eigenvalues(vstate, folder_path, hi, part_training, one_avg):
    """
    Computes and plots the eigenvalues of the S-matrix at 5 key iterations:
    0%, 25%, 50%, 75%, and 100% of the training process.

    Args:
        vstate: The variational state.
        hi: The Hamiltonian instance.
        folder_path: Path to save the plots.
        one_avg: A string indicating the context (e.g., "one" for a single run).
    """

    folder_save_QGT = folder_path+"/QGT_plot"
    os.makedirs(folder_save_QGT, exist_ok=True)

    models_dir = Path(folder_path) / "models"
    if not models_dir.is_dir():
        print(f"Warning: 'models' directory not found in {folder_path}. Skipping.")
        return None, None

    # --- Determine the model iteration indices to plot ---
    model_files = sorted([f for f in os.listdir(models_dir) if f.startswith("model_") and f.endswith(".mpack")])
    num_models = len(model_files)
    if num_models == 0:
        print(f"Warning: No model files found in {models_dir}. Skipping.")
        return None, None

    # Define the 5 key iteration points
    if part_training == 'start':
        indices_to_plot = [
            0,  # First iteration
            1,
            2,
            3,
            4,
        ]
    if part_training == 'end':
        indices_to_plot = [
        4 *(num_models - 1) // 8,  # 2/4 iteration
        5*(num_models - 1) // 8,  # 1/4 iteration
        6 * (num_models - 1) // 8,  # 3/4 iteration
        7 * (num_models - 1) // 8,  # 4/4 iteration
        num_models - 1  # Last iteration
        ]

    # Ensure unique indices, especially for short runs
    indices_to_plot = sorted(list(set(indices_to_plot)))

    all_eigenvalues = {}
    eigenvalues_list_for_avg = []
    relevant_eigenvalues_counts = []
    plt.figure(figsize=(12, 7))

    print(f"Processing model at iterations: {indices_to_plot}")

    for i, model_idx in enumerate(indices_to_plot):
        model_file = models_dir / f"model_{model_idx}.mpack"
        with open(model_file, "rb") as f:
            vstate.variables = flax.serialization.from_bytes(
                vstate.variables, f.read()
            )

        # Compute S-matrix and its eigenvalues for the current model state
        S_matrix = compute_S_matrix_single_model(vstate, hi)
        eigenvalues = np.linalg.eigvalsh(S_matrix)
        all_eigenvalues[f'iter_{model_idx}'] = eigenvalues
        eigenvalues_list_for_avg.append(eigenvalues)

        # --- Plotting for the current iteration ---
        sorted_eigenval = np.sort(eigenvalues)[::-1]
        # --- Count relevant eigenvalues for this specific model ---
        count = _count_relevant_eigenvalues(sorted_eigenval)
        relevant_eigenvalues_counts.append(count)

        indices = np.arange(len(sorted_eigenval))
        if part_training == 'start':
            plt.plot(indices, sorted_eigenval, lw=1.5, color='green',label=f'Iteration {model_idx}' if i == 0 else "", alpha=0.5)
        elif part_training == 'end':
            plt.plot(indices, sorted_eigenval, lw=1.5, color='orange',label=f'Iteration {model_idx}' if i == 0 else "", alpha=0.5)



    # --- Calculate and report the mean of the relevant eigenvalue counts ---
    mean_relevant_eigenvalues = np.mean(relevant_eigenvalues_counts) if relevant_eigenvalues_counts else 0
    std_relevant_eigenvalues = np.std(relevant_eigenvalues_counts) if relevant_eigenvalues_counts else 0
    print(f"Mean number of relevant S-matrix eigenvalues (over 5 models): {mean_relevant_eigenvalues:.2f} ± {std_relevant_eigenvalues:.2f}")
    # Add this information to the plot title
    plt.title(f"Eigenvalue Spectrum of S-matrix (Avg. Relevant Eigenvalues: {mean_relevant_eigenvalues:.2f})")

    # --- Finalize and save the plot ---
    plt.title("Eigenvalue Spectrum of S-matrix at Different Training Iterations")
    plt.xlabel("Eigenvalue Index (sorted descending)")
    plt.ylabel("Eigenvalue Magnitude")
    plt.yscale("log")
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()

    if part_training == 'start':
        save_plot_path = Path(folder_save_QGT) / "S_matrix_spectrum_vs_iteration_start_training.png"
    elif part_training == 'end':
        save_plot_path = Path(folder_save_QGT) / "S_matrix_spectrum_vs_iteration_end_training.png"        
    plt.savefig(save_plot_path, dpi=300)
    print(f"✅ Plot saved to {save_plot_path}")
    plt.show()

    # --- Save all computed eigenvalues ---
    save_data_path = Path(folder_path) / "Sign_plot" / "S_matrix_eigenvalues_5_points.pkl"
    with open(save_data_path, 'wb') as f:
        pickle.dump(all_eigenvalues, f)
    print(f"S-matrix eigenvalues for 5 iterations saved to {save_data_path}")

    return all_eigenvalues, mean_relevant_eigenvalues

def Plot_S_matrix_eigenvalues(eigenvalues, folder_path, one_avg):

    print("Plotting S-matrix eigenvalues...", eigenvalues.shape)

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
    