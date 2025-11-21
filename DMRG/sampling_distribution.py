import numpy as np
from DMRG.QSL_DMRG import *
from DMRG.importance_sampling import *
from DMRG.Fidelities import *
import os
import matplotlib.pyplot as plt


if __name__ == "__main__":

    model_params = {
        'Lx': 4,
        'Ly': 4,
        'J1': 1.0,
        'J2': 0.5
    }

    # Define a list of sample sizes to test
    n_samples_values = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
    n_iter = 500
    N_sites = model_params['Lx'] **2

    # --- Define file paths for saved models ---
    model_storage_dir = "DMRG/trained_models"
    os.makedirs(model_storage_dir, exist_ok=True)
    dmrg_filename = os.path.join(model_storage_dir, f"dmrg_L{model_params['Lx']}_J2_{model_params['J2']}.pkl.gz")
    rbm_filename = os.path.join(model_storage_dir, f"rbm_L{model_params['Lx']}_J2_{model_params['J2']}.mpack")


    # --- DMRG ---
    hamiltonian = J1J2Heisenberg(model_params=model_params)
    DMRG_vstate = DMRG_vstate_optimization(hamiltonian, model_params, filename=dmrg_filename)
    B_list, local_dims = extract_mps_tensors(DMRG_vstate)
    DMRG_vstate.canonical_form()
    
    # --- RBM ---
    RBM_vstate, ha = RBM_vstate_optimization(model_params, n_iter, filename=rbm_filename) # ha is the NetKet Hamiltonian for exact diagonalization
    
    # --- Compute full wavefunction arrays for fidelity calculations ---
    dmrg_array = get_full_wavefunction(DMRG_vstate, undo_sort_charge=True)
    RBM_array = RBM_vstate.to_array()

    # --- Calculate Exact Fidelity (independent of sampling) ---
    fidelity_exact_rbm_dmrg = Fidelity_exact(RBM_vstate, DMRG_vstate)
    print(f"\nFidelity exact (RBM_Full vs DMRG_Full): {fidelity_exact_rbm_dmrg:.6f}")

    # --- Initialize lists to store results from the loop ---
    sampled_fidelities = []
    classical_fidelities = []

    # --- Loop over different numbers of samples ---
    for n_samples in n_samples_values:
        print(f"\n{'='*20} Running for n_samples = {n_samples} {'='*20}")

        # --- Importance Sampling ---
        ops_z = ['Sigmaz'] * N_sites
        samples = np.zeros((n_samples, N_sites), dtype=int)
        psi_DMRG_sampled = np.zeros(n_samples, dtype=np.complex128)
        for n in range(n_samples):
            sigmas, psi_DMRG = DMRG_vstate.sample_measurements(first_site=0, last_site=N_sites-1, ops=ops_z, complex_amplitude=True)
            samples[n, :] = sigmas
            psi_DMRG_sampled[n] = psi_DMRG

        # Convert samples to {0, 1} basis for DMRG and {-1, 1} for NetKet
        samples_netket = samples
        samples_dmrg_01_basis = ((1 - np.asarray(samples_netket)) / 2).astype(int)

        # Evaluate RBM amplitudes on the sampled configurations
        logpsi_RBM_sampled = logpsi_netket(RBM_vstate, samples_netket)
        psi_RBM_sampled = np.exp(logpsi_RBM_sampled)

        # --- Calculate and store sampled fidelities ---
        fidelity_sampled_rbm_dmrg = Fidelity_sampled(psi_DMRG_sampled, psi_RBM_sampled)
        print(f"Fidelity sampled (RBM vs DMRG): {fidelity_sampled_rbm_dmrg:.6f}")
        sampled_fidelities.append(fidelity_sampled_rbm_dmrg)

        # --- Compare DMRG probability distribution with the empirical sample distribution ---
        unique_configs, unique_indices, counts = np.unique(samples_dmrg_01_basis, axis=0, return_index=True, return_counts=True)
        P_samples = counts / counts.sum()
        psi_DMRG_unique = psi_DMRG_sampled[unique_indices]
        P_DMRG = np.abs(psi_DMRG_unique)**2
        P_DMRG /= P_DMRG.sum()

    # --- Final Plot: Fidelities vs. n_samples ---
    plt.figure(figsize=(10, 6))
    plt.axhline(y=fidelity_exact_rbm_dmrg, color='r', linestyle='--', label=f'Fidelity (Exact) = {fidelity_exact_rbm_dmrg:.4f}')
    plt.plot(n_samples_values, sampled_fidelities, 'o-', color='b', label='Fidelity (Sampled RBM vs DMRG)')

    plt.xlabel("Number of Samples (n_samples)")
    plt.ylabel("Fidelity")
    plt.title(f"Fidelity Convergence vs. Number of Samples (L={model_params['Lx']}, J2={model_params['J2']})")
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.6)
    plt.xscale('log')

    # --- Save the final plot ---
    plot_output_dir = "DMRG/plot/Fidelity_vs_Samples"
    os.makedirs(plot_output_dir, exist_ok=True)
    final_plot_filename = os.path.join(plot_output_dir, f"fidelity_vs_nsamples_L{model_params['Lx']}_J2_{model_params['J2']}.png")
    plt.savefig(final_plot_filename, dpi=300)
    print(f"\n✅ Final fidelity plot saved to {final_plot_filename}")
    plt.show()
