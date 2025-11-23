import os
import numpy as np
import matplotlib.pyplot as plt
import netket as nk

from DMRG.DMRG import J1J2Heisenberg, DMRG_vstate_optimization
from Elaborate.Sign_Obs_MCMC import Sign_DMRG_full_hilbert, Sign_DMRG_samples
from Elaborate.Sign_Obs import Marshall_Sign_exact

if __name__ == "__main__":

    # --- Simulation Parameters ---
    base_model_params = {
        'Lx': 4,
        'Ly': 4,
        'J1': 1.0,
    }

    # Define the different J2 values to test
    j2_values = [ 0.2, 0.5, 0.7]
    n_samples_values = [1024] # Number of samples for importance sampling
    N_sites = base_model_params['Lx'] * base_model_params['Ly']

    # --- Define file paths for saved models ---
    model_storage_dir = "DMRG/trained_models"
    os.makedirs(model_storage_dir, exist_ok=True)

    # --- Initialize lists to store results ---
    full_hilbert_signs = []
    exact_gs_signs = []
    sampled_signs_all = {n: [] for n in n_samples_values} # Dict to store results for each sample size

    # --- Loop over J2 values ---
    for j2 in j2_values:
        print(f"\n{'='*20} Running for J2 = {j2} {'='*20}")

        # Update model params for the current J2
        current_model_params = base_model_params.copy()
        current_model_params['J2'] = j2

        # Define a unique filename for the DMRG state
        dmrg_filename = os.path.join(model_storage_dir, f"dmrg_L{current_model_params['Lx']}_J2_{j2}.pkl.gz")

        # --- DMRG Optimization/Loading ---
        hamiltonian = J1J2Heisenberg(model_params=current_model_params)
        DMRG_vstate = DMRG_vstate_optimization(hamiltonian, current_model_params, filename=dmrg_filename)
        
        # Create a NetKet Hilbert space for the sign functions
        hi = nk.hilbert.Spin(s=1/2, N=N_sites)

        # --- Exact Diagonalization ---
        # Create the NetKet Hamiltonian to find the exact ground state
        # Note: The 0.25 factor accounts for the difference between NetKet's Pauli (sigma) and TenPy's Spin (S=sigma/2) operators
        graph = nk.graph.Hypercube(length=current_model_params['Lx'], n_dim=2, pbc=True, max_neighbor_order=2)
        ha_nk = 0.25 * nk.operator.Heisenberg(hilbert=hi, graph=graph, J=[1.0, j2], sign_rule=[False, False])
        E_gs, ket_gs = nk.exact.lanczos_ed(ha_nk, compute_eigenvectors=True)
        
        sign_exact_gs, _ = Marshall_Sign_exact(ket_gs, hi)
        exact_gs_signs.append(np.abs(sign_exact_gs))
        print(f"Exact GS Marshall Sign:          {sign_exact_gs:.6f}")

        """
        # --- Calculate Full Hilbert Space Sign ---
        signs_full, psi_DMRG_full = Sign_DMRG_full_hilbert(DMRG_vstate, hi)
        prob_DMRG_full = np.abs(psi_DMRG_full)**2
        sign_DMRG_full = np.abs(np.sum(prob_DMRG_full * signs_full.reshape(-1)) / np.sum(prob_DMRG_full))
        full_hilbert_signs.append(sign_DMRG_full)
        print(f"Full Hilbert DMRG Marshall Sign: {sign_DMRG_full:.6f}")
        """

        # --- Loop over different sample sizes ---
        for n_samples in n_samples_values:
            print(f"--- Generating {n_samples} samples ---")
            # --- Generate Samples from DMRG ---
            ops_z = ['Sigmaz'] * N_sites
            samples = np.zeros((n_samples, N_sites), dtype=int)
            psis_DMRG_sampled = np.zeros((n_samples), dtype=complex)
            for n in range(n_samples):
                sigmas, psi_DMRG_sampled = DMRG_vstate.sample_measurements(first_site=0, last_site=N_sites-1, ops=ops_z, complex_amplitude=True)
                samples[n, :] = sigmas
                psis_DMRG_sampled[n] = psi_DMRG_sampled
            samples_netket = samples

            # --- Calculate Sampled Sign ---
            sign_DMRG_samples, psis_DMRG_sampled_recalculated = Sign_DMRG_samples(DMRG_vstate, samples_netket)
            if not np.allclose(psis_DMRG_sampled, psis_DMRG_sampled_recalculated):
                print("Warning: Recalculated amplitudes do not match sampled amplitudes!")

            prob_DMRG_samples = np.abs(psi_DMRG_sampled)**2
            sign_DMRG_sampled_final = 1/n_samples * np.abs(np.sum(prob_DMRG_samples * sign_DMRG_samples.reshape(-1)) / np.sum(prob_DMRG_samples))
            sampled_signs_all[n_samples].append(sign_DMRG_sampled_final)
            print(f"Sampled DMRG Marshall Sign (n={n_samples}): {sign_DMRG_sampled_final:.6f}")

    # --- Plotting the results ---
    plt.figure(figsize=(10, 6))
    #.plot(j2_values, full_hilbert_signs, 'o-', label='Full Hilbert Sign DMRG', markersize=8, linewidth=2)
    plt.plot(j2_values, exact_gs_signs, '^-', label='Exact GS Sign (Lanczos)', markersize=8, linewidth=2, color='green')
    for n_samples, signs in sampled_signs_all.items():
        plt.plot(j2_values, signs, 's--', label=f'Sampled Sign DMRG (n={n_samples})', markersize=6)

    plt.xlabel("J2 Value")
    plt.ylabel("Marshall Sign")
    plt.title(f"DMRG Marshall Sign vs. J2 (L={base_model_params['Lx']})")
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.6)
    plt.xticks(j2_values)

    plot_output_dir = "DMRG/plot/DMRG_Sign_Analysis"
    os.makedirs(plot_output_dir, exist_ok=True)
    final_plot_filename = os.path.join(plot_output_dir, f"dmrg_sign_vs_j2_L{base_model_params['Lx']}.png")
    plt.savefig(final_plot_filename, dpi=300)
    print(f"\n✅ Final sign plot saved to {final_plot_filename}")
    plt.show()
