from tenpy.models.model import CouplingMPOModel
from tenpy.models.lattice import Square
from tenpy.networks.site import SpinSite
from tenpy.networks.mps import MPS
from tenpy.algorithms import dmrg
from tenpy.tools import hdf5_io
from numpy import linspace
from random import shuffle
import numpy as np
from tenpy.networks.site import SpinHalfSite
from tenpy.algorithms.exact_diag import get_full_wavefunction
from DMRG.Plotting import *
from DMRG.Observable.Corr_Struct import Correlations_Structure_Factor

from netket.experimental.driver import VMC_SR
from DMRG.QSL_DMRG import *
from DMRG.importance_sampling import *

import os
import numpy as np


import matplotlib.pyplot as plt


from jax import numpy as jnp
import netket as nk
import flax

def fidelity_RBM_exact(vstate, ket_gs):
    # Ensure vstate has access to the full Hilbert space for array conversion
    vstate_array = vstate.to_array(normalize=True)

    overlap = np.vdot(vstate_array, ket_gs)
    fidelity = np.abs(overlap)**2
    return fidelity

def fidelity_DMRG_exact(DMRG_vstate, exact_ket):

    dmrg_array = get_full_wavefunction(DMRG_vstate, undo_sort_charge=True)

    overlap = np.vdot(dmrg_array, exact_ket)
    fidelity = np.abs(overlap)**2
    return fidelity

def Fidelity_exact(RBM_vstate, DMRG_vstate):

    dmrg_array = get_full_wavefunction(DMRG_vstate, undo_sort_charge=True)
    RBM_array = RBM_vstate.to_array(normalize=True)

    overlap = np.vdot(RBM_array, dmrg_array)
    fidelity = np.abs(overlap)**2

    return fidelity


def Fidelity_sampled(psi_DMRG_sampled, psi_RBM_sampled):

    """
    The term np.mean(np.abs(psi_RBM_sampled)**2) is an estimate of Σ_s |ψ_DMRG(s)|^2 * |ψ_RBM(s)|^2.
    This is not the squared norm ⟨ψ_RBM | ψ_RBM⟩. 
    It's an overlap of probability distributions. 
    Therefore, psi_RBM_sampled_norm is not correctly normalizing ψ_RBM to unit norm.
    """
    # 1. Calculate the element-wise ratio X(sigma) = psi_RBM(sigma) / psi_DMRG(sigma)
    ratio = psi_RBM_sampled / psi_DMRG_sampled
    
    # 2. Estimate the overlap <psi_DMRG | psi_RBM>
    # This is the mean of the ratio: E[X]
    overlap_est = np.mean(ratio)
    # The numerator of the fidelity is the squared magnitude of the overlap.
    numerator_est = np.abs(overlap_est)**2 
    
    # 3. Estimate the RBM Norm Squared (Denominator): <psi_RBM | psi_RBM>
    # This is the mean of the squared magnitude of the ratio: E[|X|^2]
    denominator_est = np.mean(np.abs(ratio)**2)
    
    # 4. Calculate Fidelity: |E[X]|^2 / E[|X|^2].
    fidelity = numerator_est / denominator_est

    return fidelity

    
    """
    #RATIO MC
    # Assuming psi_DMRG_sampled and psi_RBM_sampled are the UNNORMALIZED amplitudes
    ratio = psi_RBM_sampled / psi_DMRG_sampled  # Element-wise division

    # Estimate of the UNNORMALIZED overlap <psi_DMRG | psi_RBM>
    # This is approximately the numerator in the formula above
    overlap_numerator_est = np.mean(ratio)

    # Estimate of the RBM norm squared <psi_RBM | psi_RBM>
    # This is approximately the denominator in the formula above
    norm_RBM_sq_est = np.mean(np.abs(ratio)**2)

    # Calculate the improved fidelity estimate
    fidelity_improved = np.abs(overlap_numerator_est)**2 / norm_RBM_sq_est

    return fidelity_improved
    """

def fidelity_DMRG_sampled_vs_full(psi_DMRG_sampled, samples_dmrg, dmrg_array, local_dims):
    """
    Calculates the fidelity between the sampled DMRG amplitudes and the full DMRG wavefunction.
    This serves as a sanity check for the importance sampling fidelity calculation, and should be close to 1.

    Args:
        psi_DMRG_sampled (np.ndarray): Amplitudes from the DMRG sampling process (proposal distribution).
        samples_dmrg (np.ndarray): The configurations {0,1} that were sampled.
        dmrg_array (np.ndarray): The full DMRG wavefunction array (target distribution).
        local_dims (list): List of local dimensions, e.g., [2, 2, ...].

    Returns:
        float: The calculated fidelity.
    """
    def ravel_configs_to_indices(configs_01, local_dims):
        if configs_01.ndim == 1:
            configs_01 = configs_01.reshape(1, -1)
        tuples = tuple(configs_01[:, i] for i in range(configs_01.shape[1]))
        return np.ravel_multi_index(tuples, dims=tuple(local_dims), order='C')


    # 1. Get the target amplitudes from the full array for each sampled configuration
    flat_indices = ravel_configs_to_indices(samples_dmrg, local_dims)
    psi_target_sampled = dmrg_array[flat_indices]

    # 2. Calculate the ratio X(s) = psi_target(s) / psi_proposal(s)
    # Here, both target and proposal are the DMRG wavefunction.
    ratio = psi_target_sampled / psi_DMRG_sampled

    # 3. Estimate fidelity using the ratio method
    overlap_est = np.mean(ratio)
    denominator_est = np.mean(np.abs(ratio)**2)
    fidelity = np.abs(overlap_est)**2 / denominator_est

    return fidelity

def fidelity_RBM_sampled_vs_full(psi_RBM_sampled, samples_dmrg, RBM_array, local_dims):
    """
    Calculates the fidelity between the sampled RBM amplitudes and the full RBM wavefunction.
    This is a cross-check where samples are drawn from DMRG, but used to evaluate RBM fidelity.

    Args:
        psi_RBM_sampled (np.ndarray): RBM amplitudes evaluated on configurations sampled from DMRG.
        samples_dmrg (np.ndarray): The configurations {0,1} that were sampled from DMRG.
        RBM_array (np.ndarray): The full RBM wavefunction array (target distribution).
        local_dims (list): List of local dimensions, e.g., [2, 2, ...].

    Returns:
        float: The calculated fidelity.
    """
    def ravel_configs_to_indices(configs_01, local_dims):
        if configs_01.ndim == 1:
            configs_01 = configs_01.reshape(1, -1)
        tuples = tuple(configs_01[:, i] for i in range(configs_01.shape[1]))
        return np.ravel_multi_index(tuples, dims=tuple(local_dims), order='C')

    # 1. Get the target amplitudes from the full array for each sampled configuration
    flat_indices = ravel_configs_to_indices(samples_dmrg, local_dims)
    psi_target_sampled = RBM_array[flat_indices]

    # 2. Calculate the ratio X(s) = psi_target(s) / psi_RBM_sampled(s)
    # NOTE: This is NOT an importance sampling ratio, as samples are not from RBM.
    # It's a direct comparison on a specific subset of configurations.
    overlap = np.vdot(psi_target_sampled, psi_RBM_sampled)
    norm_target_sq = np.vdot(psi_target_sampled, psi_target_sampled)
    norm_sampled_sq = np.vdot(psi_RBM_sampled, psi_RBM_sampled)

    fidelity = np.abs(overlap)**2 / (norm_target_sq * norm_sampled_sq)
    return fidelity.real

def plot_probability_distributions(ket_gs, samples_netket, samples_dmrg, B_list, RBM_vstate, dmrg_array, RBM_array, model_params, base_output_dir="DMRG/plot"):
    """
    Plots and compares four probability distributions:
    1. Exact probability distribution for unique sampled configurations.
    2. DMRG probability from full wavefunction (`dmrg_array`).
    3. RBM probability from full wavefunction (`RBM_array`).
    4. DMRG probability from MPS tensors (`B_list`).
    5. RBM probability from model evaluation (`log_value`).
    6. Distribution of samples sampled from DMRG (frequency of unique samples).

    Args:
        ket_gs (np.ndarray): Exact ground state vector (full Hilbert space amplitudes).
        hi (netket.hilbert.Hilbert): NetKet Hilbert space object.
        samples_netket (np.ndarray): Samples in NetKet {-1, 1} basis.
        samples_dmrg (np.ndarray): Samples in DMRG/TenPy {0, 1} basis.
        B_list (list): List of MPS tensors (B_list) from DMRG.
        RBM_vstate (netket.vqs.MCState): NetKet MCState object for the RBM.
        dmrg_array (np.ndarray): Full DMRG wavefunction from `get_full_wavefunction`.
        RBM_array (np.ndarray): Full RBM wavefunction from `to_array`.
        model_params (dict): Dictionary containing model parameters like 'Lx', 'J2'.
        base_output_dir (str): Base directory for saving plots (e.g., "DMRG/plot").
    """
    def ravel_configs_to_indices(configs_01, local_dims):
        # configs_01: (n, L) with values 0..d-1
        # local_dims: e.g. [2]*L
        if configs_01.ndim == 1:
            configs_01 = configs_01.reshape(1, -1)
        tuples = tuple(configs_01[:, i] for i in range(configs_01.shape[1]))
        return np.ravel_multi_index(tuples, dims=tuple(local_dims), order='C')

    # --- basic sizes and local dims ---
    L = model_params['Lx']**2
    local_dims = [2] * L  # spin-1/2 assumption
    n_samples = samples_dmrg.shape[0]

    # --- find unique configs in NetKet-sampled set (keep first-occurrence indices) ---
    unique_configs, unique_indices, inverse_indices, counts = np.unique(
        samples_dmrg, axis=0, return_index=True, return_inverse=True, return_counts=True
    )
    # unique_configs in order of first occurrence due to return_index
    n_unique = unique_configs.shape[0]

    # --- flat indices in full Hilbert space for each unique config ---
    flat_indices_unique = ravel_configs_to_indices(unique_configs, local_dims)  # shape (n_unique,)

    # --- extract & normalize distributions restricted to these unique configs ---
    # Exact
    P_exact = np.abs(ket_gs[flat_indices_unique])**2
    P_exact = P_exact / (P_exact.sum() + 1e-16)

    # DMRG full-wavefunction
    P_dmrg_full = np.abs(dmrg_array[flat_indices_unique])**2
    P_dmrg_full = P_dmrg_full / (P_dmrg_full.sum() + 1e-16)

    # RBM full-wavefunction (precomputed)
    P_rbm_full = np.abs(RBM_array[flat_indices_unique])**2
    P_rbm_full = P_rbm_full / (P_rbm_full.sum() + 1e-16)

    # DMRG from MPS tensors: psi_mps_from_config must accept array shape (n_configs, L) in 0/1 format
    psi_DMRG_from_B = psi_mps_from_config(B_list, unique_configs)   # shape (n_unique,)
    P_dmrg_from_B = np.abs(psi_DMRG_from_B)**2
    P_dmrg_from_B = P_dmrg_from_B / (P_dmrg_from_B.sum() + 1e-16)

    # RBM evaluation: logpsi_netket must accept configs in NetKet format {-1,+1}
    # We pass the NetKet-format configs corresponding to unique_configs
    unique_configs_netket = (unique_configs * 2 - 1)  # convert 0/1 -> -1/+1
    logpsi_rbm = logpsi_netket(RBM_vstate, unique_configs_netket)  # shape (n_unique,)
    psi_rbm_eval = np.exp(logpsi_rbm)
    P_rbm_eval = np.abs(psi_rbm_eval)**2
    P_rbm_eval = P_rbm_eval / (P_rbm_eval.sum() + 1e-16)

    # empirical frequencies from DMRG sampling:
    # We need counts for exactly the same unique configs; `counts` returned by np.unique above
    P_sampled = counts.astype(float) / counts.sum()

    # --- sort by exact prob descending so visuals line up ---
    order = np.argsort(P_exact)[::-1]
    P_exact_sorted = P_exact[order]
    P_DMRG_full_sorted = P_dmrg_full[order]
    P_RBM_full_sorted = P_rbm_full[order]
    P_DMRG_sorted = P_dmrg_from_B[order]
    P_RBM_sorted = P_rbm_eval[order]
    P_count_sorted = P_sampled[order]
    configs_sorted = unique_configs[order]
    counts_sorted = counts[order]

    # 4. Plotting
    fig, axes = plt.subplots(2, 3, figsize=(20, 10), sharex=True, sharey=False)
    fig.suptitle(f"Probability Distributions Comparison (L={model_params.get('Lx','N/A')}, J2={model_params.get('J2','N/A')})", fontsize=16)
    x_axis = np.arange(len(P_exact_sorted))

    # --- Subplot 1: Exact Probability ---
    axes[0, 0].bar(x_axis, P_exact_sorted, label='Exact', alpha=0.7, color='blue')
    axes[0, 0].set_title('Exact Probability Distribution')
    axes[0, 0].set_ylabel("Probability (log scale)")
    axes[0, 0].legend()
    axes[0, 0].grid(True, which="both", ls="--", alpha=0.6)

    # --- Subplot 2: DMRG from full array ---
    axes[0, 1].bar(x_axis, P_DMRG_full_sorted, label='DMRG (full array)', alpha=0.7, color='purple')
    axes[0, 1].set_title('DMRG Probability (from full array)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, which="both", ls="--", alpha=0.6)

    # --- Subplot 3: RBM from full array ---
    axes[0, 2].bar(x_axis, P_RBM_full_sorted, label='RBM (full array)', alpha=0.7, color='brown')
    axes[0, 2].set_title('RBM Probability (from full array)')
    axes[0, 2].legend()
    axes[0, 2].grid(True, which="both", ls="--", alpha=0.6)

     # --- Subplot 6: DMRG Sampled Counts (linear scale) ---
    axes[1, 0].bar(x_axis, counts_sorted, label='DMRG Sample Counts', alpha=0.7, color='red')
    axes[1, 0].set_title('DMRG Sampled Counts')
    axes[1, 0].set_xlabel("Unique Configuration Index (Sorted by Exact Prob.)")
    axes[1, 0].set_ylabel("Number of Samples")
    axes[1, 0].legend()
    axes[1, 0].grid(True, which="both", ls="--", alpha=0.6)
    axes[1, 0].set_yscale('log')

    # --- Subplot 4: DMRG from tensors ---
    axes[1, 1].bar(x_axis, P_DMRG_sorted, label='DMRG (tensors)', alpha=0.7, color='orange')
    axes[1, 1].set_title('DMRG Probability (from tensors)')
    axes[1, 1].set_xlabel("Unique Configuration Index (Sorted by Exact Prob.)")
    axes[1, 1].set_ylabel("Probability (log scale)")
    axes[1, 1].legend()
    axes[1, 1].grid(True, which="both", ls="--", alpha=0.6)

    # --- Subplot 5: RBM from model eval ---
    axes[1, 2].bar(x_axis, P_RBM_sorted, label='RBM (model eval)', alpha=0.7, color='green')
    axes[1, 2].set_title('RBM Probability (from model eval)')
    axes[1, 2].set_xlabel("Unique Configuration Index (Sorted by Exact Prob.)")
    axes[1, 2].legend()
    axes[1, 2].grid(True, which="both", ls="--", alpha=0.6)

   

    # Compute log y-limits from the probability arrays while avoiding zeros
    prob_arrays = [
        np.asarray(P_exact_sorted),
        np.asarray(P_DMRG_full_sorted),
        np.asarray(P_RBM_full_sorted),
        np.asarray(P_DMRG_sorted),
        np.asarray(P_RBM_sorted)
    ]
    # find the global positive min and global max across these arrays
    positive_vals = np.hstack([arr[arr > 0] for arr in prob_arrays if np.any(arr > 0)])
    if positive_vals.size == 0:
        # fallback to small positive number if everything is zero or negative (unlikely)
        ymin = 1e-12
        ymax = 1.0
    else:
        ymin = positive_vals.min() * 0.5  # a bit below the smallest positive value
        ymax = max(arr.max() for arr in prob_arrays)*1.1

    # Apply log scale and consistent y-limits to probability subplots (all except counts)
    prob_axes = [axes[0,0], axes[0,1], axes[0,2], axes[1,1], axes[1,2]]
    for ax in prob_axes:
        ax.set_yscale('log')
        ax.set_ylim(ymin, ymax)


    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Create output directory following the existing pattern
    Lx = model_params.get('Lx', 'N/A')
    J2 = model_params.get('J2', 'N/A')
    model_name_for_path = "DMRG_RBM_Comparison"
    final_output_dir = os.path.join(base_output_dir, model_name_for_path, f"J={J2}", "distributions")
    os.makedirs(final_output_dir, exist_ok=True)

    plot_filename = os.path.join(final_output_dir, f"probability_distributions_comparison_{n_samples}.png")
    plt.savefig(plot_filename, dpi=300)
    print(f"✅ Probability distribution comparison plot saved to {plot_filename}")
    plt.close()