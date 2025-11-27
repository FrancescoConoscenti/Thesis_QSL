import os
import numpy as np
import matplotlib.pyplot as plt
from jax import numpy as jnp
import netket as nk

from DMRG.DMRG import RBM_vstate_optimization, J1J2Heisenberg, DMRG_vstate_optimization
from Elaborate.Sign_Obs import Marshall_Sign_exact, Marshall_Sign_full_hilbert_one
from Elaborate.Sign_Obs_MCMC import Sign_DMRG_samples

if __name__ == "__main__":

    model_params = {
        'Lx': 4,
        'Ly': 4,
        'J1': 1.0,
        'J2': 0.5
    }

    # --- Simulation Parameters ---
    # Define the different numbers of training iterations to test
    n_iter_values = [40, 50, 60, 80, 100, 120, 140, 160, 180, 200, 300, 400, 500, 700, 1000]
    n_samples_values = [1024] # Number of samples for importance sampling
    N_sites = model_params['Lx'] * model_params['Ly']

    # --- Define file paths for saved models ---
    model_storage_dir = "DMRG/trained_models"
    os.makedirs(model_storage_dir, exist_ok=True)

    # --- DMRG Ground State and Sampling (Done once) ---
    print("--- Obtaining DMRG ground state and generating samples ---")
    dmrg_filename = os.path.join(model_storage_dir, f"dmrg_L{model_params['Lx']}_J2_{model_params['J2']}.pkl.gz")
    hamiltonian = J1J2Heisenberg(model_params=model_params)
    DMRG_vstate = DMRG_vstate_optimization(hamiltonian, model_params, filename=dmrg_filename)

    # --- Get NetKet Hamiltonian and Hilbert space (Done once) ---
    _, ha = RBM_vstate_optimization(model_params, n_iter=1) # Dummy call to get Hamiltonian
    hi = ha.hilbert

    # --- Calculate DMRG full wavefunction (Done once) ---
    _, psi_DMRG_full = Sign_DMRG_samples(DMRG_vstate, hi.all_states())
    norm_DMRG_full = np.linalg.norm(psi_DMRG_full)


    # --- Prepare for plotting ---
    plt.figure(figsize=(12, 7))
    markers = ['s', 'D', '^', 'v']
    
    # --- Initialize lists to store results ---
    amp_overlap_full_rbm_dmrg = []

    # --- Loop over different sample sizes ---
    for i, n_samples in enumerate(n_samples_values):
        print(f"\n{'='*25}\nRunning for n_samples = {n_samples}\n{'='*25}")

        # --- Importance Sampling from DMRG ---
        print(f"\n--- Generating {n_samples} samples from DMRG wavefunction ---")
        ops_z = ['Sigmaz'] * N_sites
        samples_netket = np.zeros((n_samples, N_sites), dtype=int)
        psi_DMRG_sampled = np.zeros(n_samples, dtype=np.complex128)
        for n in range(n_samples):
            sigmas, psi_DMRG = DMRG_vstate.sample_measurements(first_site=0, last_site=N_sites-1, ops=ops_z, complex_amplitude=True)
            samples_netket[n, :] = sigmas
            psi_DMRG_sampled[n] = psi_DMRG
        print("--- Sampling complete ---")

        amp_overlap_sampled_rbm_dmrg = []
        # --- Loop over RBM training iterations ---
        for n_iter in n_iter_values:
            print(f"\n{'='*20} Running for n_iter = {n_iter} {'='*20}")

            # Define a unique filename for each RBM training run
            rbm_filename = os.path.join(model_storage_dir, f"rbm_L{model_params['Lx']}_J2_{model_params['J2']}_iter{n_iter}.mpack")

            # --- RBM Training/Loading ---
            # The function will load the model if it exists, or train and save it otherwise.
            RBM_vstate, _ = RBM_vstate_optimization(model_params, n_iter, filename=rbm_filename)

            # --- 1. RBM, DMRG full Hilbert space amplitude overlap (only calculate once) ---
            if i == 0:
                psi_RBM_full = RBM_vstate.to_array(normalize=False)
                norm_RBM_full = np.linalg.norm(psi_RBM_full)
                dot_product_mag_full = np.sum(np.abs(psi_RBM_full) * np.abs(psi_DMRG_full))
                overlap_full = dot_product_mag_full / (norm_RBM_full * norm_DMRG_full)
                amp_overlap_full_rbm_dmrg.append(overlap_full)
                print(f"Amplitude Overlap (Full RBM vs DMRG): {overlap_full:.6f}")

            # --- 2. RBM, DMRG importance samples amplitude overlap ---
            logpsi_vals = RBM_vstate.log_value(samples_netket)
            psi_RBM_samples = jnp.exp(logpsi_vals)
            
            weights = np.abs(psi_RBM_samples) / np.abs(psi_DMRG_sampled)
            numerator = np.mean(weights)
            norm_RBM_sq = np.mean(weights**2)
            overlap_sampled = numerator / np.sqrt(norm_RBM_sq)
            amp_overlap_sampled_rbm_dmrg.append(overlap_sampled)
            print(f"Amplitude Overlap (Sampled RBM vs DMRG, n={n_samples}): {overlap_sampled:.6f}")

        # Plot sampled overlap for the current n_samples
        plt.plot(n_iter_values, amp_overlap_sampled_rbm_dmrg, marker=markers[i], linestyle='--', label=f'Sampled Overlap (n_samples={n_samples})')

    # --- Final Plot: Overlaps vs. n_iter ---
    plt.plot(n_iter_values, amp_overlap_full_rbm_dmrg, 'o-', label='Full Hilbert Space Overlap')
    plt.xlabel("Number of RBM Training Iterations (n_iter)")
    plt.ylabel("Amplitude Overlap")
    plt.title(f"Amplitude Overlap vs. RBM Training (L={model_params['Lx']}, J2={model_params['J2']})")
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.6)
    plt.xscale('log')
    plt.ylim(bottom=0, top=1.05)

    # --- Save the final plot ---
    plot_output_dir = "DMRG/plot/Amp_Overlap_vs_Iterations"
    os.makedirs(plot_output_dir, exist_ok=True)
    final_plot_filename = os.path.join(plot_output_dir, f"amp_overlap_vs_n_iter_L{model_params['Lx']}_J2_{model_params['J2']}_multisample.png")
    plt.savefig(final_plot_filename, dpi=300)
    print(f"\n✅ Final amplitude overlap plot saved to {final_plot_filename}")
    plt.show()

"""
This block is now part of the loops above.

    # --- Loop over RBM training iterations ---
    for n_iter in n_iter_values:
        print(f"\n{'='*20} Running for n_iter = {n_iter} {'='*20}")

        # Define a unique filename for each RBM training run
        rbm_filename = os.path.join(model_storage_dir, f"rbm_L{model_params['Lx']}_J2_{model_params['J2']}_iter{n_iter}.mpack")

        # --- RBM Training/Loading ---
        # The function will load the model if it exists, or train and save it otherwise.
        RBM_vstate, _ = RBM_vstate_optimization(model_params, n_iter, filename=rbm_filename)

        # --- 1. RBM, DMRG full Hilbert space amplitude overlap ---
        psi_RBM_full = RBM_vstate.to_array(normalize=False)
        norm_RBM_full = np.linalg.norm(psi_RBM_full)
        dot_product_mag_full = np.sum(np.abs(psi_RBM_full) * np.abs(psi_DMRG_full))
        overlap_full = dot_product_mag_full / (norm_RBM_full * norm_DMRG_full)
        sign_overlap_full_rbm_dmrg.append(overlap_full.real)
        print(f"Amplitude Overlap (Full RBM vs DMRG): {overlap_full.real:.6f}")

        # --- 2. RBM, DMRG importance samples amplitude overlap ---
        logpsi_vals = RBM_vstate.log_value(samples_netket)
        psi_RBM_samples = jnp.exp(logpsi_vals)
        # Calculate the norms (L2 norm) of each wavefunction on the sampled configurations
        norm_RBM = np.linalg.norm(psi_RBM_samples)
        norm_DMRG = np.linalg.norm(psi_DMRG_sampled)
        dot_product_mag = np.sum(np.abs(psi_RBM_samples) * np.abs(psi_DMRG_sampled))
        overlap_sampled = dot_product_mag / (norm_RBM * norm_DMRG)
        sign_overlap_sampled_rbm_dmrg.append(overlap_sampled.real)
        print(f"Amplitude Overlap (Sampled RBM vs DMRG): {overlap_sampled.real:.6f}")

        # --- 3. Exact diagonalized gs, NQS (RBM) sign overlap ---
        
        prob_exact_gs = np.abs(ket_gs)**2
        overlap_exact = np.abs(np.sum(prob_exact_gs * signs_full.reshape(-1) * sign_RBM_full.reshape(-1))) / np.sum(prob_exact_gs)
        sign_overlap_full_rbm_exact.append(overlap_exact)
        print(f"Sign Overlap (Full RBM vs Exact GS): {overlap_exact:.6f}")
"""
