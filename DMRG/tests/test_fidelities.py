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
from DMRG.plot.Plotting import *
from DMRG.Observable.Corr_Struct import Correlations_Structure_Factor

from netket.experimental.driver import VMC_SR
from DMRG.DMRG import *
from DMRG.plot.Plotting import *
from DMRG.Fidelities import *

import os
import numpy as np

import matplotlib.pyplot as plt

from jax import numpy as jnp
import netket as nk
import flax


if __name__ == "__main__":

    model_params = {
        'Lx': 4,
        'Ly': 4,
        'J1': 1.0,
        'J2': 0.6
    }

    n_samples_values = [1024, 2048, 4096] # Number of samples for importance sampling
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

    # -- 0.2- Get Exact Ground State (run once) ---
    # Define a dummy RBM to get the NetKet Hamiltonian `ha`
    _, ha = RBM_vstate_optimization(model_params, n_iter=1)
    E_gs_vals, ket_gs_matrix = nk.exact.lanczos_ed(ha, k=1, compute_eigenvectors=True)
    ket_gs = ket_gs_matrix[:, 0]

    # --- Prepare for plotting ---
    plt.figure(figsize=(10, 6))
    markers = ['s', 'D', '^']

    # --- Loop over different sample sizes ---
    for i, n_samples in enumerate(n_samples_values):
        print(f"\n{'='*25}\nRunning for n_samples = {n_samples}\n{'='*25}")

        # --- Importance Sampling from DMRG ---
        print(f"\n--- Generating {n_samples} samples from DMRG wavefunction ---")
        ops_z = ['Sigmaz'] * N_sites
        samples = np.zeros((n_samples, N_sites), dtype=int)
        psi_DMRG_sampled = np.zeros(n_samples, dtype=np.complex128)
        for n in range(n_samples):
            sigmas, psi_DMRG = DMRG_vstate.sample_measurements(first_site=0, last_site=N_sites-1, ops=ops_z, complex_amplitude=True)
            samples[n, :] = sigmas
            psi_DMRG_sampled[n] = psi_DMRG
        print("--- Sampling complete ---")

        samples_netket = samples  # Samples in {-1, 1} basis for NetKet
        samples_dmrg_01_basis = ((1 - np.asarray(samples_netket)) / 2).astype(int)


        # --- Loop over RBM training iterations ---
        sampled_fidelities = []
        if i == 0: # Calculate exact fidelity only once
            exact_fidelities = []

        for n_iter in n_iter_values:
            print(f"\n{'='*20} Running for n_iter = {n_iter} {'='*20}")

            # Define a unique filename for each RBM training run
            rbm_filename = os.path.join(model_storage_dir, f"rbm_L{model_params['Lx']}_J2_{model_params['J2']}_iter{n_iter}.mpack")

            # --- RBM Training ---
            RBM_vstate, _ = RBM_vstate_optimization(model_params, n_iter, filename=rbm_filename)

            # --- Evaluate RBM amplitudes on the pre-sampled configurations ---
            log_values = RBM_vstate.log_value(samples)
            logpsi_RBM_sampled = np.array(log_values, dtype=np.complex128)
            psi_RBM_sampled = np.exp(logpsi_RBM_sampled)

            # --- Calculate and store fidelities ---
            if i == 0:
                fidelity_exact_rbm_dmrg = Fidelity_exact(RBM_vstate, DMRG_vstate)
                print(f"\nFidelity exact (RBM_Full vs DMRG_Full): {fidelity_exact_rbm_dmrg:.6f}")
                exact_fidelities.append(fidelity_exact_rbm_dmrg)

            fidelity_sampled_rbm_dmrg = Fidelity_sampled(psi_DMRG_sampled, psi_RBM_sampled)
            print(f"Fidelity sampled (RBM_sampled vs DMRG_sampled): {fidelity_sampled_rbm_dmrg:.6f}")
            sampled_fidelities.append(fidelity_sampled_rbm_dmrg)

        # Plot sampled fidelity for the current n_samples
        plt.plot(n_iter_values, sampled_fidelities, marker=markers[i], linestyle='--', label=f'Fidelity (Sampled, n_samples={n_samples})')

    # --- Final Plot: Fidelities vs. n_iter ---
    plt.plot(n_iter_values, exact_fidelities, 'o-', label='Fidelity (Exact)')
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
        log_values = RBM_vstate.log_value(samples)
        logpsi_RBM_sampled = np.array(log_values, dtype=np.complex128)
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