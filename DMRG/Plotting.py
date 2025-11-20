import matplotlib.pyplot as plt
import os
import numpy as np

def plot_DMRG_energies(energies, bond_dims, sweeps, model_params):
    """
    Plots DMRG energy convergence and saves the figure.
    """
    Lx = model_params.get('Lx', 'N/A')
    J2 = model_params.get('J2', 'N/A')

    # --- Plot convergence with bond dimension annotations ---
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(sweeps, energies, 'o-', label='Energy')

    # Annotate each point with the bond dimension
    for i, E in enumerate(energies):
        ax.annotate(f'χ={bond_dims[i]}', (sweeps[i], E), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=8)

    ax.set_xlabel('Sweep')
    ax.set_ylabel('Energy per Site')
    ax.set_title(f'DMRG Energy Convergence (L={Lx} J2={J2})')
    ax.grid(True)
    
    model_name = f"DMRG_L{Lx}.png"
    output_dir = f"DMRG/plot/{model_name}/J={J2}"
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "Energies")
    fig.savefig(save_path, dpi=300)
    print(f"✅ DMRG plot saved to {save_path}")
    plt.show()

    print(f"Final ground state energy: {energies[-1]:.6f}")

def plot_correlation_function(corr_r, model_params):
    """Plots the real-space spin-spin correlation function C(r)."""
    Lx = model_params.get('Lx', 'N/A')
    Ly = model_params.get('Ly', 'N/A')
    J2 = model_params.get('J2', 'N/A')

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(corr_r.real, origin='lower', cmap='viridis')
    fig.colorbar(im, ax=ax, label="C(r)")
    ax.set_xlabel("dx", fontsize=12)
    ax.set_ylabel("dy", fontsize=12)
    ax.set_title(f"Spin-Spin Correlation C(r) (Lx={Lx}, Ly={Ly}, J2={J2})", fontsize=14)
    ax.set_xticks(np.arange(Lx)) # Ensure ticks are correct for Lx
    ax.set_yticks(np.arange(Ly)) # Ensure ticks are correct for Ly
    plt.tight_layout()

    filename = f"DMRG_L{Lx}.png"
    output_dir = f"DMRG/plot/{filename}/J={J2}"
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "Correlation")
    fig.savefig(save_path, dpi=300)
    plt.close(fig)

def plot_structure_factor(S_q, model_params):
    """Plots the static spin structure factor S(q)."""
    Lx = model_params.get('Lx', 'N/A')
    Ly = model_params.get('Ly', 'N/A')
    J2 = model_params.get('J2', 'N/A')

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(np.abs(S_q), origin='lower', cmap='viridis')
    fig.colorbar(im, ax=ax, label="|S(q)|")
    ax.set_xlabel(r"$q_x$", fontsize=12)
    ax.set_ylabel(r"$q_y$", fontsize=12)
    ax.set_title(f"Structure Factor S(q) (Lx={Lx}, Ly={Ly}, J2={J2})", fontsize=14)
    ax.set_xticks([0, Lx // 2, Lx - 1])
    ax.set_xticklabels(['0', r'$\pi$', r'$2\pi$'])
    ax.set_yticks([0, Ly // 2, Ly - 1])
    ax.set_yticklabels(['0', r'$\pi$', r'$2\pi$'])
    plt.tight_layout()

    filename = f"DMRG_L{Lx}.png"
    output_dir = f"DMRG/plot/{filename}/J={J2}"
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "Structure_Factor")
    fig.savefig(save_path, dpi=300)
    plt.close(fig)


def plot_probability_distributions(ket_gs, samples_netket, samples_dmrg, B_list, RBM_vstate, dmrg_array, RBM_array, model_params, base_output_dir="DMRG/plot", fidelity_exact=None, fidelity_sampled=None):
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
        fidelity_exact (float, optional): Exact fidelity to display in the title.
        fidelity_sampled (float, optional): Sampled fidelity to display in the title.
    """

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
    
    title = f"Probability Distributions Comparison (L={model_params.get('Lx','N/A')}, J2={model_params.get('J2','N/A')})"
    if fidelity_exact is not None: title += f"\nFidelity (Exact): {fidelity_exact:.6f}"
    if fidelity_sampled is not None: title += f" | Fidelity (Sampled): {fidelity_sampled:.6f}"
    fig.suptitle(title, fontsize=16)
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

     # --- Subplot 6: DMRG Sampled Frequencies ---
    axes[1, 0].bar(x_axis, P_count_sorted, label='DMRG Sample Frequencies', alpha=0.7, color='red')
    axes[1, 0].set_title('DMRG Sampled Frequencies')
    axes[1, 0].set_xlabel("Unique Configuration Index (Sorted by Exact Prob.)")
    axes[1, 0].set_ylabel("Normalized Frequency")
    axes[1, 0].legend()
    axes[1, 0].grid(True, which="both", ls="--", alpha=0.6)

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
    prob_axes = [axes[0,0], axes[0,1], axes[0,2], axes[1,0], axes[1,1], axes[1,2]]
    for ax in prob_axes:
        ax.set_yscale('log')
        ax.set_ylim(ymin, ymax)
    axes[1,0].set_ylabel("Normalized Frequency (log scale)")


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

def plot_full_hilbert_distributions(ket_gs, samples_dmrg, B_list, RBM_vstate, dmrg_array, RBM_array, model_params, base_output_dir="DMRG/plot", fidelity_exact=None, fidelity_sampled=None):
    """
    Plots and compares probability distributions over the ENTIRE Hilbert space.
    1. Exact probability distribution.
    2. DMRG probability from full wavefunction (`dmrg_array`).
    3. RBM probability from full wavefunction (`RBM_array`).
    4. DMRG probability from MPS tensors (`B_list`).
    5. RBM probability from model evaluation (`log_value`).
    6. Distribution of samples from DMRG (frequency of unique samples).

    Args:
        ket_gs (np.ndarray): Exact ground state vector (full Hilbert space amplitudes).
        samples_dmrg (np.ndarray): Samples in DMRG/TenPy {0, 1} basis.
        B_list (list): List of MPS tensors (B_list) from DMRG.
        RBM_vstate (netket.vqs.MCState): NetKet MCState object for the RBM.
        dmrg_array (np.ndarray): Full DMRG wavefunction from `get_full_wavefunction`.
        RBM_array (np.ndarray): Full RBM wavefunction from `to_array`.
        model_params (dict): Dictionary containing model parameters like 'Lx', 'J2'.
        base_output_dir (str): Base directory for saving plots.
        fidelity_exact (float, optional): Exact fidelity to display in the title.
        fidelity_sampled (float, optional): Sampled fidelity to display in the title.
    """

    # --- Basic sizes and configurations ---
    N = model_params['Lx'] * model_params['Ly']
    hilbert_size = 2**N

    # --- Extract and normalize full probability distributions ---
    # 1. Exact
    P_exact = np.abs(ket_gs)**2
    P_exact /= P_exact.sum()

    # 2. DMRG from full wavefunction
    P_dmrg_full = np.abs(dmrg_array)**2
    P_dmrg_full /= P_dmrg_full.sum()

    # 3. RBM from full wavefunction
    P_rbm_full = np.abs(RBM_array)**2
    P_rbm_full /= P_rbm_full.sum()

    # --- Sort all distributions by exact probability descending ---
    order = np.argsort(P_exact)[::-1]

    P_exact_sorted = P_exact[order]
    P_dmrg_full_sorted = P_dmrg_full[order]
    P_rbm_full_sorted = P_rbm_full[order]

    # --- Process DMRG samples for the counts plot ---
    # Find unique sampled configs and their counts
    unique_sampled_configs, counts = np.unique(samples_dmrg, axis=0, return_counts=True)

    # Get the flat indices for these unique sampled configs
    local_dims = [2] * N
    flat_indices_sampled = ravel_configs_to_indices(unique_sampled_configs, local_dims)

    # We need to place these counts on the x-axis which is sorted by P_exact.
    # We create an inverse map from flat_index -> sorted_index.
    inverse_order = np.argsort(order)
    sorted_indices_for_samples = inverse_order[flat_indices_sampled]
 
    # Create a full normalized frequency array for plotting, placing frequencies at the correct sorted positions
    total_samples = samples_dmrg.shape[0]
    frequencies = counts.astype(float) / total_samples
    freq_full_sorted = np.zeros(hilbert_size, dtype=float)
    freq_full_sorted[sorted_indices_for_samples] = frequencies

    # DMRG from MPS tensors: psi_mps_from_config must accept array shape (n_configs, L) in 0/1 format
    psi_DMRG_from_B = psi_mps_from_config(B_list, unique_sampled_configs)   # shape (n_unique,)
    P_dmrg_from_B = np.abs(psi_DMRG_from_B)**2
    P_dmrg_from_B = P_dmrg_from_B / (P_dmrg_from_B.sum() + 1e-16)
    # Create a full-sized array and place the probabilities at the correct sorted indices
    P_dmrg_from_B_sorted = np.zeros(hilbert_size, dtype=float)
    P_dmrg_from_B_sorted[sorted_indices_for_samples] = P_dmrg_from_B

    # RBM evaluation: logpsi_netket must accept configs in NetKet format {-1,+1}
    # We pass the NetKet-format configs corresponding to unique_configs
    unique_configs_netket = (unique_sampled_configs * 2 - 1)  # convert 0/1 -> -1/+1
    logpsi_rbm = logpsi_netket(RBM_vstate, unique_configs_netket)  # shape (n_unique,)
    psi_rbm_eval = np.exp(logpsi_rbm)
    P_rbm_eval = np.abs(psi_rbm_eval)**2
    if P_rbm_eval.sum() > 1e-16:
        P_rbm_eval = P_rbm_eval / P_rbm_eval.sum()
    # Create a full-sized array and place the probabilities at the correct sorted indices
    P_rbm_eval_sorted = np.zeros(hilbert_size, dtype=float)
    P_rbm_eval_sorted[sorted_indices_for_samples] = P_rbm_eval

    # --- Plotting ---
    fig, axes = plt.subplots(2, 3, figsize=(20, 10), sharex=True, sharey=True)
    
    title = f"Full Hilbert Space Probability Distributions (L={model_params.get('Lx','N/A')}, J2={model_params.get('J2','N/A')})"
    if fidelity_exact is not None: title += f"\nFidelity (Exact): {fidelity_exact:.6f}"
    if fidelity_sampled is not None: title += f" | Fidelity (Sampled): {fidelity_sampled:.6f}"
    fig.suptitle(title, fontsize=16)
    x_axis = np.arange(hilbert_size)

    # --- Subplot 1: Exact Probability ---
    axes[0, 0].bar(x_axis, P_exact_sorted, label='Exact', alpha=0.7, color='blue')
    axes[0, 0].set_title('Exact Probability Distribution')
    axes[0, 0].set_ylabel("Probability (log scale)")
    axes[0, 0].legend()

    # --- Subplot 2: DMRG from full array ---
    axes[0, 1].bar(x_axis, P_dmrg_full_sorted, label='DMRG (full array)', alpha=0.7, color='purple')
    axes[0, 1].set_title('DMRG Probability (from full array)')
    axes[0, 1].legend()

    # --- Subplot 3: RBM from full array ---
    axes[0, 2].bar(x_axis, P_rbm_full_sorted, label='RBM (full array)', alpha=0.7, color='brown')
    axes[0, 2].set_title('RBM Probability (from full array)')
    axes[0, 2].legend()

     # --- Subplot 6: DMRG Sampled Counts ---
    axes[1, 0].bar(x_axis, freq_full_sorted, label='DMRG Sample Frequencies', alpha=0.7, color='red')
    axes[1, 0].set_title('DMRG Sampled Frequencies')
    axes[1, 0].set_xlabel("State Index (Sorted by Exact Prob.)")
    axes[1, 0].set_ylabel("Normalized Frequency")
    axes[1, 0].legend()

    # --- Subplot 4: DMRG from tensors ---
    axes[1, 1].bar(x_axis, P_dmrg_from_B_sorted, label='DMRG (tensors)', alpha=0.7, color='orange')
    axes[1, 1].set_xlabel("State Index (Sorted by Exact Prob.)")
    axes[1, 1].set_ylabel("Probability (log scale)")
    axes[1, 1].set_title('DMRG Probability (from tensors)')
    axes[1, 1].legend()

    # --- Subplot 5: RBM from model eval ---
    axes[1, 2].bar(x_axis, P_rbm_eval_sorted, label='RBM (model eval)', alpha=0.7, color='green')
    axes[1, 2].set_xlabel("State Index (Sorted by Exact Prob.)")
    axes[1, 2].set_ylabel("Probability (log scale)")
    axes[1, 2].set_title('RBM Probability (from model eval)')
    axes[1, 2].legend()


    # --- Apply log scale and consistent y-limits ---
    all_probs = np.concatenate([P_exact_sorted, P_dmrg_full_sorted, P_rbm_full_sorted, P_dmrg_from_B_sorted, P_rbm_eval_sorted, freq_full_sorted])
    positive_probs = all_probs[all_probs > 1e-16]
    if positive_probs.size > 0:
        ymin = positive_probs.min() * 0.5
        ymax = all_probs.max() * 1.1
    else:
        ymin, ymax = 1e-12, 1.0

    # Apply log scale to all probability plots
    prob_axes = [axes[0,0], axes[0,1], axes[0,2], axes[1,0], axes[1,1], axes[1,2]]
    for ax in prob_axes:
        ax.set_yscale('log')
        ax.set_ylim(ymin, ymax)
        ax.grid(True, which="both", ls="--", alpha=0.6)


    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # --- Save the plot ---
    Lx = model_params.get('Lx', 'N/A')
    J2 = model_params.get('J2', 'N/A')
    model_name_for_path = "DMRG_RBM_Comparison"
    final_output_dir = os.path.join(base_output_dir, model_name_for_path, f"J={J2}", "distributions")
    os.makedirs(final_output_dir, exist_ok=True)

    plot_filename = os.path.join(final_output_dir, f"full_hilbert_distributions_{total_samples}.png")
    plt.savefig(plot_filename, dpi=300)
    print(f"✅ Full Hilbert space distribution plot saved to {plot_filename}")
    plt.close()


"""
    # 1. Identify unique configurations and their counts from `samples`
    unique_configs, inverse_indices, counts = np.unique(samples_netket, axis=0, return_inverse=True, return_counts=True)
    num_unique_configs = len(unique_configs)

    # 2. Calculate probabilities for these `unique_configs`

    # 2.1. Exact Probability Distribution
    all_configs_hi = hi.all_states()
    # Create a mapping from configuration (tuple) to its index in the full Hilbert space
    # This can be slow for very large Hilbert spaces, but necessary for exact comparison.
    config_to_idx_map = {tuple(np.asarray(c)): i for i, c in enumerate(all_configs_hi)}
    
    exact_indices_for_unique = []
    for uc in unique_configs:
        key = tuple(uc)
        if key in config_to_idx_map:
            exact_indices_for_unique.append(config_to_idx_map[key])
        else:
            # This case should ideally not happen if unique_configs are valid states in hi
            # Handle if a sampled config is not in the Hilbert space (e.g., due to total_sz constraint)
            print(f"Warning: Sampled configuration {uc} not found in Hilbert space.")
            exact_indices_for_unique.append(-1) # Placeholder for invalid index

    # Filter out invalid indices and corresponding unique_configs
    valid_indices_mask = np.array(exact_indices_for_unique) != -1
    unique_configs_valid = unique_configs[valid_indices_mask]
    exact_indices_for_unique_valid = np.array(exact_indices_for_unique)[valid_indices_mask]
    counts_valid = counts[valid_indices_mask]

    if len(unique_configs_valid) == 0:
        print("No valid unique configurations to plot.")
        return

    P_exact_unique = np.abs(ket_gs[exact_indices_for_unique_valid])**2
    # Normalize P_exact_unique to sum to 1
    P_exact_unique /= np.sum(P_exact_unique)

    # 2.2. DMRG Probability from full wavefunction
    P_DMRG_full_unique = np.abs(dmrg_array[exact_indices_for_unique_valid])**2
    P_DMRG_full_unique /= np.sum(P_DMRG_full_unique)

    # 2.3. RBM Probability from full wavefunction
    P_RBM_full_unique = np.abs(RBM_array[exact_indices_for_unique_valid])**2
    P_RBM_full_unique /= np.sum(P_RBM_full_unique)



    # 2.2. DMRG Normalized Probability Distribution
    # We need to find the corresponding {0,1} configs for the valid unique_configs
    # A simple way is to find the first occurrence of each unique_config_valid in samples_netket
    # and get the corresponding sample from samples_dmrg.
    _, unique_indices = np.unique(samples_netket, axis=0, return_index=True)
    unique_configs_dmrg = samples_dmrg[unique_indices][valid_indices_mask]

    psi_DMRG_unique = psi_mps_from_config(B_list, unique_configs_dmrg)
    P_DMRG_unique = np.abs(psi_DMRG_unique)**2
    P_DMRG_unique /= np.sum(P_DMRG_unique)

    # 2.3. RBM Normalized Probability Distribution
    logpsi_RBM_unique = logpsi_netket(RBM_vstate, unique_configs_valid)
    psi_RBM_unique = np.exp(logpsi_RBM_unique)
    P_RBM_unique = np.abs(psi_RBM_unique)**2
    P_RBM_unique /= np.sum(P_RBM_unique)

    # 2.4. Distribution of samples sampled from DMRG (normalized frequencies)
    P_sampled_from_DMRG_unique = counts_valid / np.sum(counts_valid)

    # 3. Sort the unique configurations by Exact Probability (descending)
    sort_indices = np.argsort(P_exact_unique)[::-1]

    P_exact_sorted = P_exact_unique[sort_indices]
    P_DMRG_full_sorted = P_DMRG_full_unique[sort_indices]
    P_RBM_full_sorted = P_RBM_full_unique[sort_indices]
    P_DMRG_sorted = P_DMRG_unique[sort_indices]
    P_RBM_sorted = P_RBM_unique[sort_indices]
    P_sampled_from_DMRG_sorted = P_sampled_from_DMRG_unique[sort_indices]
    
    """