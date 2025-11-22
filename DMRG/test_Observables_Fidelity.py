import os
import numpy as np
import matplotlib.pyplot as plt
from jax import numpy as jnp
import netket as nk

from DMRG.QSL_DMRG import RBM_vstate_optimization, J1J2Heisenberg, DMRG_vstate_optimization
from Elaborate.Sign_Obs import Marshall_Sign_exact
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
    n_samples = 4096*2 # Number of samples for MCMC sign calculation
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
    sign_exact_gs, _ = Marshall_Sign_exact(ket_gs, ha.hilbert)
    print(f"Exact GS Marshall Sign: {sign_exact_gs:.6f}")

    # --- Initialize lists to store results ---
    mcmc_signs = []
    dmrg_sampled_rbm_signs = []

    # --- Loop over RBM training iterations ---
    for n_iter in n_iter_values:
        print(f"\n{'='*20} Running for n_iter = {n_iter} {'='*20}")

        # Define a unique filename for each RBM training run
        rbm_filename = os.path.join(model_storage_dir, f"rbm_L{model_params['Lx']}_J2_{model_params['J2']}_iter{n_iter}.mpack")

        # --- RBM Training/Loading ---
        # The function will load the model if it exists, or train and save it otherwise.
        RBM_vstate, _ = RBM_vstate_optimization(model_params, n_iter, filename=rbm_filename)
        hi = RBM_vstate.hilbert

        # --- Calculate MCMC Sign ---
        SignObs = MarshallSignObs(hi)
        RBM_vstate.n_samples = n_samples
        sign_mcmc_stats = RBM_vstate.expect(SignObs)
        mcmc_signs.append(sign_mcmc_stats.mean)
        print(f"MCMC Marshall Sign:       {sign_mcmc_stats.mean:.6f} ± {sign_mcmc_stats.error_of_mean:.6f}")

        # --- Calculate RBM Sign on DMRG Samples ---
        kernel = nk.vqs.get_local_kernel(RBM_vstate, SignObs)
        _, args_template = nk.vqs.get_local_kernel_arguments(RBM_vstate, SignObs)
        logpsi_vals = RBM_vstate.log_value(samples_netket)
        sign_RBM_samples = kernel(logpsi_vals, RBM_vstate.parameters, samples_netket, args_template)

        # This is the expectation value <Sign>_RBM estimated with samples from DMRG
        # using the ratio for importance sampling.
        ratio = jnp.exp(2.0 * (logpsi_vals - jnp.log(psi_DMRG_sampled)))
        expectation = jnp.sum(ratio * sign_RBM_samples.reshape(-1)) / jnp.sum(ratio)
        dmrg_sampled_rbm_signs.append(expectation)
        print(f"RBM Sign on DMRG Samples: {expectation.real:.6f}")

    # --- Plotting the results ---
    plt.figure(figsize=(12, 7))
    plt.errorbar(n_iter_values, [s.real for s in mcmc_signs], yerr=[s.imag for s in mcmc_signs], fmt='s--', label='RBM Sign (MCMC Sampling)', markersize=8, capsize=5)
    plt.plot(n_iter_values, [s.real for s in dmrg_sampled_rbm_signs], 'o-', label='RBM Sign (DMRG Importance Sampling)', markersize=8, linewidth=2)
    plt.axhline(y=-sign_exact_gs, color='green', linestyle='-', linewidth=2, label=f'Exact GS Sign = {sign_exact_gs:.4f}')

    plt.xlabel("Number of RBM Training Iterations (n_iter)")
    plt.ylabel("Marshall Sign")
    plt.title(f"Marshall Sign vs. RBM Training Iterations (L={model_params['Lx']}, J2={model_params['J2']})")
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.6)
    plt.xscale('log')
    all_signs = [s.real for s in mcmc_signs] + [s.real for s in dmrg_sampled_rbm_signs] + [sign_exact_gs]
    # Adjust y-limits for better visualization, avoiding NaN/Inf
    valid_signs = [s for s in all_signs if np.isfinite(s)]
    if valid_signs:
        plt.ylim(bottom=min(valid_signs) - 0.1, top=max(valid_signs) + 0.1)

    # --- Save the final plot ---
    plot_output_dir = "DMRG/plot/Sign_vs_Iterations"
    os.makedirs(plot_output_dir, exist_ok=True)
    final_plot_filename = os.path.join(plot_output_dir, f"sign_vs_n_iter_L{model_params['Lx']}_J2_{model_params['J2']}.png")
    plt.savefig(final_plot_filename, dpi=300)
    print(f"\n✅ Final sign plot saved to {final_plot_filename}")
    plt.show()
