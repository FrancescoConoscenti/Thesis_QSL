import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy.sparse.linalg import eigsh
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

    S_matrix = compute_S_matrix(vstate, folder_path, hi)
    
    # Use eigsh to find the 1000 largest eigenvalues for each S-matrix
    # This is much faster than computing all eigenvalues with eigh for large matrices.
    #eigenvalues = np.array([eigsh(S, k=1000, which='LA', return_eigenvectors=False) for S in S_matrix ])

    eigenvalues = np.linalg.eigvalsh(S_matrix)
    
    Plot_S_matrix_eigenvalues(eigenvalues, folder_path, one_avg)

    # Calculate the mean of eigenvalues at each step
    #eigenvalues_mean = np.mean(eigenvalues, axis=1)

    # Save the eigenvalues to a file using pickle, similar to run_spin.py
    save_path = Path(folder_path) / "Sign_plot" / "S_matrix_eigenvalues.pkl"
    variables_to_save = {
        'eigenvalues_mean': eigenvalues
    }
    with open(save_path, 'wb') as f:
        pickle.dump(variables_to_save, f)
    print(f"S-matrix eigenvalues saved to {save_path} using pickle.")

    return S_matrix, eigenvalues

def Plot_S_matrix_eigenvalues(eigenvalues, folder_path, one_avg):

    print("Plotting S-matrix eigenvalues...", eigenvalues.shape)

    # Since we have eigenvalues for only one matrix, it's a 1D array.
    # Sort them in descending order.
    sorted_eigenval = np.sort(eigenvalues)[::-1]
    indices = np.arange(len(sorted_eigenval))  # x-axis: eigenvalue index

    # Normalize by the largest absolute eigenvalue
    normalized_eigenvals = sorted_eigenval / np.max(np.abs(sorted_eigenval))

    # Plot
    plt.figure(figsize=(8,4))
    plt.plot(indices, np.abs(normalized_eigenvals), lw=1.5, color='darkgreen', marker='.', markersize=3, linestyle='-')
    
    plt.title("Eigenvalues of the S-matrix for optimization iterations")
    plt.title("Eigenvalue Spectrum of the S-matrix (Final Model)")
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
    
