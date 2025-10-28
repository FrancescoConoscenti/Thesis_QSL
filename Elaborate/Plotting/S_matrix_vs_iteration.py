import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from Elaborate.S_matrix_Obs import *


def plot_S_matrix_eigenvalues(vstate, folder_path, hi,  one_avg):
    """
    Plots the eigenvalues of the S-matrix for the variational state.

    Args:
        vstate: The variational state.
        hi: The Hamiltonian instance.
        folder_path: Path to save the plots.
        one_avg: Type of averaging to perform.
    """

    S_matrices = compute_S_matrix(vstate, folder_path, hi)
    eigenvalues, _ = np.linalg.eigh(S_matrices)

    Plot_S_matrix_eigenvalues(eigenvalues, folder_path, one_avg)

    return S_matrices, eigenvalues

def Plot_S_matrix_eigenvalues(eigenvalues, folder_path, one_avg):

    print("Plotting S-matrix eigenvalues...", eigenvalues.shape)

    sorted_eigenval = np.sort(eigenvalues, axis=-1)[:, ::-1]  # descending per row
    indices = np.arange(len(sorted_eigenval[1]))   # x-axis positions

    normalized_eigenvals = sorted_eigenval / np.max(np.abs(sorted_eigenval), axis=1, keepdims=True)

    # Choose a green colormap and pick evenly spaced colors from it
    cmap = plt.cm.Greens  # "Greens" goes from light to dark
    n_lines = normalized_eigenvals.shape[0]
    colors = [cmap(0.3 + 0.7*i/(n_lines-1)) for i in range(n_lines)]  # avoid very pale end

    # Plot
    plt.figure(figsize=(8,4))
    for i in range(normalized_eigenvals.shape[0]):
        plt.plot(indices, np.abs(normalized_eigenvals[i]), lw=1, color=colors[i])
    
    plt.title("Eigenvalues of the S-matrix for optimization iterations")
    plt.xlabel("eigenvalues index")
    plt.ylabel("eigenvalue (normalized)")
    plt.yscale("log")
    if one_avg == "avg":
        folder_path = Path(folder_path)
        save_path = folder_path.parent /"plot_avg"/"S_matrix_spectrum.png"
        plt.savefig(save_path)
    if one_avg == "one":
        plt.savefig(f"{folder_path}/Sign_plot/S_matrix_spectrum.png")
    
    plt.show()
    
