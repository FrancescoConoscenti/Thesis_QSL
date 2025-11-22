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
from DMRG.Plotting import *


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

    # undo_sort_charge=True to ensure that the basis state ordering of the TenPy wavefunction
    #  matches the standard lexicographical ordering used by NetKet's `to_array()` method
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
    psi_DMRG_sampled_norm = psi_DMRG_sampled / np.sqrt(np.mean(np.abs(psi_DMRG_sampled)**2))
    psi_RBM_sampled_norm = psi_RBM_sampled / np.sqrt(np.mean(np.abs(psi_RBM_sampled)**2))

    # 1. Calculate the element-wise ratio X(sigma) = psi_RBM(sigma) / psi_DMRG(sigma)
    ratio = psi_RBM_sampled / psi_DMRG_sampled
    #ratio = psi_RBM_sampled / psi_DMRG_sampled

    
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


def Fidelity_sample_distr_vs_DMRG(P_DMRG_sampled, samples_dmrg_distr):
    overlap = np.mean(np.vdot(P_DMRG_sampled, samples_dmrg_distr))
    fidelity = np.abs(overlap)**2
    return fidelity


def ravel_configs_to_indices(configs_01, local_dims):
        if configs_01.ndim == 1:
            configs_01 = configs_01.reshape(1, -1)
        tuples = tuple(configs_01[:, i] for i in range(configs_01.shape[1]))
        return np.ravel_multi_index(tuples, dims=tuple(local_dims), order='C')


def Fidelity_RBM_sampled_vs_DMRG_exact(psi_RBM_sampled, samples_dmrg, dmrg_array, local_dims):

    # DMRG full-wavefunction
    # --- flat indices in full Hilbert space for each unique config ---
    flat_indices_unique = ravel_configs_to_indices(samples_dmrg, local_dims)  # shape (n_unique,)
    psi_DMRG_exact = dmrg_array[flat_indices_unique]
    psi_DMRG_exact = psi_DMRG_exact / (psi_DMRG_exact.sum() + 1e-16)


    # 1. Calculate the element-wise ratio X(sigma) = psi_RBM(sigma) / psi_DMRG(sigma)
    ratio = psi_RBM_sampled / psi_DMRG_exact
    
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


if __name__ == "__main__":

    model_params = {
        'Lx': 4,
        'Ly': 4,
        'J1': 1.0,
        'J2': 0.4
    }

    n_samples = 1024 # Number of samples for importance sampling
    n_iter_values = [40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300, 400, 500, 600, 700, 800, 900, 1000, 2000, 3000  ] # RBM training iterations to test
    N_sites = model_params['Lx'] **2

    # --- Define file paths for saved models ---
    model_storage_dir = "DMRG/trained_models"
    os.makedirs(model_storage_dir, exist_ok=True)
    dmrg_filename = os.path.join(model_storage_dir, f"dmrg_L{model_params['Lx']}_J2_{model_params['J2']}.pkl.gz")

    # --- DMRG (run once) ---
    hamiltonian = J1J2Heisenberg(model_params=model_params)
    DMRG_vstate = DMRG_vstate_optimization(hamiltonian, model_params, filename=dmrg_filename)
    B_list, local_dims = extract_mps_tensors(DMRG_vstate)
    DMRG_vstate.canonical_form()
    dmrg_array = get_full_wavefunction(DMRG_vstate, undo_sort_charge=True)

    # --- Importance Sampling from DMRG (run once) ---
    print(f"\n--- Generating {n_samples} samples from DMRG wavefunction ---")
    ops_z = ['Sigmaz'] * N_sites
    samples = np.zeros((n_samples, N_sites), dtype=int)
    psi_DMRG_sampled = np.zeros(n_samples, dtype=np.complex128)
    for n in range(n_samples):
        sigmas, psi_DMRG = DMRG_vstate.sample_measurements(first_site=0, last_site=N_sites-1, ops = ops_z, complex_amplitude=True)
        samples[n, :] = sigmas
        psi_DMRG_sampled[n] = psi_DMRG
    print("--- Sampling complete ---")

    samples_netket = samples  # Samples in {-1, 1} basis for NetKet
    samples_dmrg_01_basis = ((1 - np.asarray(samples_netket)) / 2).astype(int)

    # --- Get Exact Ground State (run once) ---
    # Define a dummy RBM to get the NetKet Hamiltonian `ha`
    _, ha = RBM_vstate_optimization(model_params, n_iter=1)
    E_gs_vals, ket_gs_matrix = nk.exact.lanczos_ed(ha, k=1, compute_eigenvectors=True)
    ket_gs = ket_gs_matrix[:, 0]

    # --- Loop over RBM training iterations ---
    exact_fidelities = []
    sampled_fidelities = []
    sampled_norm_fidelities = []
    new_fidelities = []

    for n_iter in n_iter_values:
        print(f"\n{'='*20} Running for n_iter = {n_iter} {'='*20}")

        # --- Calculate sample distribution for Fidelity_sampled_norm ---
        unique_configs, unique_indices, inverse_indices, counts = np.unique(
            samples_dmrg_01_basis, axis=0, return_index=True, return_inverse=True, return_counts=True
        )
        P_unique_samples = counts / counts.sum()
        P_sample_sort = P_unique_samples[inverse_indices]
        # Define a unique filename for each RBM training run
        rbm_filename = os.path.join(model_storage_dir, f"rbm_L{model_params['Lx']}_J2_{model_params['J2']}_iter{n_iter}.mpack")

        # --- RBM Training ---
        RBM_vstate, _ = RBM_vstate_optimization(model_params, n_iter, filename=rbm_filename)

        # --- Evaluate RBM amplitudes on the pre-sampled configurations ---
        logpsi_RBM_sampled = logpsi_netket(RBM_vstate, samples_netket)
        psi_RBM_sampled = np.exp(logpsi_RBM_sampled)

        # --- Compute full RBM wavefunction array ---
        RBM_array = RBM_vstate.to_array()

        # --- Calculate and store fidelities ---
        fidelity_exact_rbm_dmrg = Fidelity_exact(RBM_vstate, DMRG_vstate)
        print(f"\nFidelity exact (RBM_Full vs DMRG_Full): {fidelity_exact_rbm_dmrg:.6f}")
        exact_fidelities.append(fidelity_exact_rbm_dmrg)

        fidelity_sampled_rbm_dmrg = Fidelity_sampled(psi_DMRG_sampled, psi_RBM_sampled)
        print(f"Fidelity sampled (RBM_sampled vs DMRG_sampled): {fidelity_sampled_rbm_dmrg:.6f}")
        sampled_fidelities.append(fidelity_sampled_rbm_dmrg)


    # --- Final Plot: Fidelities vs. n_iter ---
    plt.figure(figsize=(10, 6))
    plt.plot(n_iter_values, exact_fidelities, 'o-', label='Fidelity (Exact)')
    plt.plot(n_iter_values, sampled_fidelities, 's--', label='Fidelity (Sampled)')
    plt.xlabel("Number of RBM Training Iterations (n_iter)")
    plt.ylabel("Fidelity")
    plt.title(f"Fidelity vs. RBM Training Iterations (L={model_params['Lx']}, J2={model_params['J2']})")
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.6)
    plt.xscale('log') # Use log scale for iterations if they span orders of magnitude

    # --- Save the final plot ---
    plot_output_dir = "DMRG/plot/Fidelity_vs_Iterations"
    os.makedirs(plot_output_dir, exist_ok=True)
    final_plot_filename = os.path.join(plot_output_dir, f"fidelity_vs_n_iter_L{model_params['Lx']}_J2_{model_params['J2']}.png")
    plt.savefig(final_plot_filename, dpi=300)
    print(f"\n✅ Final fidelity plot saved to {final_plot_filename}")
    plt.close()