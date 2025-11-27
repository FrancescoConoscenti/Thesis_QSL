import os
import numpy as np
import matplotlib.pyplot as plt
from jax import numpy as jnp
import netket as nk

from DMRG.DMRG import RBM_vstate_optimization, J1J2Heisenberg, DMRG_vstate_optimization
from Elaborate.Sign_Obs import Marshall_Sign_exact, Marshall_Sign_full_hilbert_one
from Elaborate.Sign_Obs_MCMC import MarshallSignObs, Sign_DMRG_samples

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
    n_samples = 1024 # Number of samples for MCMC sign calculation
    N_sites = model_params['Lx'] * model_params['Ly']

    # --- Define file paths for saved models ---
    model_storage_dir = "DMRG/trained_models"
    os.makedirs(model_storage_dir, exist_ok=True)

    # --- DMRG Ground State and Sampling (Done once) ---
    print("--- Obtaining DMRG ground state and generating samples ---")
    dmrg_filename = os.path.join(model_storage_dir, f"dmrg_L{model_params['Lx']}_J2_{model_params['J2']}.pkl.gz")
    hamiltonian = J1J2Heisenberg(model_params=model_params)
    DMRG_vstate = DMRG_vstate_optimization(hamiltonian, model_params, filename=dmrg_filename)

    # Generate a fixed set of samples from the DMRG wavefunction
    ops_z = ['Sigmaz'] * N_sites
    samples_netket = np.zeros((n_samples, N_sites), dtype=int)
    psi_DMRG_sampled = np.zeros(n_samples, dtype=np.complex128)
    for n in range(n_samples):
        sigmas, psi_DMRG = DMRG_vstate.sample_measurements(first_site=0, last_site=N_sites-1, ops=ops_z, complex_amplitude=True)
        samples_netket[n, :] = sigmas
        psi_DMRG_sampled[n] = psi_DMRG
    print("--- DMRG sampling complete ---")

    # --- Get Exact Ground State and its Sign (Done once) ---
    print("\n--- Calculating Exact Ground State ---")
    _, ha = RBM_vstate_optimization(model_params, n_iter=1) # Dummy call to get Hamiltonian
    E_gs, ket_gs = nk.exact.lanczos_ed(ha, compute_eigenvectors=True)
    sign_exact_gs_val, signs_full = Marshall_Sign_exact(ket_gs, ha.hilbert)
    print(f"Exact GS Marshall Sign: {sign_exact_gs_val:.6f}")

    # --- Calculate DMRG sign on full Hilbert space (Done once) ---
    sign_DMRG_full, psi_DMRG_full = Sign_DMRG_samples(DMRG_vstate, ha.hilbert.all_states())
    prob_DMRG_full = np.abs(psi_DMRG_full)**2

    # --- Initialize lists to store results ---
    sign_overlap_full_rbm_dmrg = []
    sign_overlap_sampled_rbm_dmrg = []
    sign_overlap_full_rbm_exact = []

    # --- Loop over RBM training iterations ---
    for n_iter in n_iter_values:
        print(f"\n{'='*20} Running for n_iter = {n_iter} {'='*20}")

        # Define a unique filename for each RBM training run
        rbm_filename = os.path.join(model_storage_dir, f"rbm_L{model_params['Lx']}_J2_{model_params['J2']}_iter{n_iter}.mpack")

        # --- RBM Training/Loading ---
        # The function will load the model if it exists, or train and save it otherwise.
        RBM_vstate, _ = RBM_vstate_optimization(model_params, n_iter, filename=rbm_filename)
        hi = RBM_vstate.hilbert

        # --- 1. RBM, DMRG full Hilbert space sign overlap ---
        sign_RBM_full, _ = Marshall_Sign_full_hilbert_one(RBM_vstate, hi)
        overlap_full = np.abs(np.sum(prob_DMRG_full * sign_DMRG_full.reshape(-1) * sign_RBM_full.reshape(-1))) / np.sum(prob_DMRG_full)
        sign_overlap_full_rbm_dmrg.append(overlap_full)
        print(f"Sign Overlap (Full RBM vs DMRG): {overlap_full:.6f}")

        # --- 2. RBM, DMRG importance samples sign overlap ---
        SignObs = MarshallSignObs(hi)
        kernel = nk.vqs.get_local_kernel(RBM_vstate, SignObs)
        _, args_template = nk.vqs.get_local_kernel_arguments(RBM_vstate, SignObs)
        logpsi_vals = RBM_vstate.log_value(samples_netket)
        sign_RBM_samples = kernel(logpsi_vals, RBM_vstate.parameters, samples_netket, args_template)
        
        sign_DMRG_samples, _ = Sign_DMRG_samples(DMRG_vstate, samples_netket)
        
        prob_DMRG_sampled = np.abs(psi_DMRG_sampled)**2
        overlap_sampled = np.abs(np.sum(prob_DMRG_sampled * sign_DMRG_samples.reshape(-1) * sign_RBM_samples.reshape(-1))) / np.sum(prob_DMRG_sampled)
        sign_overlap_sampled_rbm_dmrg.append(overlap_sampled)
        print(f"Sign Overlap (Sampled RBM vs DMRG): {overlap_sampled:.6f}")

        # --- 3. Exact diagonalized gs, NQS (RBM) sign overlap ---
        """
        prob_exact_gs = np.abs(ket_gs)**2
        overlap_exact = np.abs(np.sum(prob_exact_gs * signs_full.reshape(-1) * sign_RBM_full.reshape(-1))) / np.sum(prob_exact_gs)
        sign_overlap_full_rbm_exact.append(overlap_exact)
        print(f"Sign Overlap (Full RBM vs Exact GS): {overlap_exact:.6f}")
        """

    # --- Plotting the results ---
    plt.figure(figsize=(12, 7))
    plt.plot(n_iter_values, sign_overlap_full_rbm_dmrg, 'o-', label='Sign Overlap (Full RBM vs DMRG)')
    plt.plot(n_iter_values, sign_overlap_sampled_rbm_dmrg, 's--', label=f'Sign Overlap (Sampled RBM vs DMRG, n={n_samples})')
    #plt.plot(n_iter_values, sign_overlap_full_rbm_exact, '^-', label='Sign Overlap (Full RBM vs Exact GS)')

    plt.xlabel("Number of RBM Training Iterations (n_iter)")
    plt.ylabel("Sign Overlap")
    plt.title(f"Sign Structure Overlap vs. RBM Training (L={model_params['Lx']}, J2={model_params['J2']})")
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.6)
    plt.xscale('log')
    plt.ylim(bottom=0, top=1.05)

    # --- Save the final plot ---
    plot_output_dir = "DMRG/plot/Sign_Overlap_vs_Iterations"
    os.makedirs(plot_output_dir, exist_ok=True)
    final_plot_filename = os.path.join(plot_output_dir, f"sign_overlap_vs_n_iter_L{model_params['Lx']}_J2_{model_params['J2']}.png")
    plt.savefig(final_plot_filename, dpi=300)
    print(f"\n✅ Final sign plot saved to {final_plot_filename}")
    plt.show()
