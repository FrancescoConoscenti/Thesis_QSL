import jax
import jax.numpy as jnp
import netket as nk
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from netket.operator.spin import sigmax, sigmay, sigmaz
from scipy.sparse.linalg import eigsh

# --- Add paths to import local modules ---
from Elaborate.Statistics.Corr_Struct import Corr_Struct_exact, Dimer_Corr_Struct_exact


def calculate_R(S_q, L):
    """
    Calculates the peak sharpness ratio R = 1 - S(Q+dq)/S(Q).

    Args:
        S_q (np.ndarray): The 2D static structure factor.
        L (int): The linear size of the lattice.

    Returns:
        R (float): The calculated sharpness ratio, averaged over x and y directions.
    """
    # Find the peak momentum Q by finding the maximum of S(q)
    # We ignore q=(0,0) which is just total magnetization
    S_q_copy = S_q.copy()
    S_q_copy[0, 0] = 0
    peak_idx = np.unravel_index(np.argmax(S_q_copy), S_q.shape)
    Q_val = S_q[peak_idx]

    # Get neighboring momenta Q + dq
    # dq_x corresponds to moving one step in the q_x direction
    neighbor_x_idx = ((peak_idx[0] + 1) % L, peak_idx[1])
    neighbor_y_idx = (peak_idx[0], (peak_idx[1] + 1) % L)

    S_Q_plus_dq_x = S_q[neighbor_x_idx]
    S_Q_plus_dq_y = S_q[neighbor_y_idx]

    # Calculate R in each direction and average them
    R_x = 1 - (S_Q_plus_dq_x / Q_val)
    R_y = 1 - (S_Q_plus_dq_y / Q_val)
    R_avg = (R_x + R_y) / 2

    return R_avg


def main():
    # --- Simulation Parameters ---
    lattice_sizes = [4]
    J_values = np.linspace(0.45, 0.7, 15)  # J2 values from 0.4 to 0.7
    seed = 42

    # Create two separate figures and axes for the plots
    fig_spin, ax_spin = plt.subplots(figsize=(10, 7))
    fig_dimer, ax_dimer = plt.subplots(figsize=(10, 7))

    for L in lattice_sizes:
        R_values = []
        R_dimer_values = []
        print(f"\nStarting QSL regime analysis for L={L} and J2 values: {J_values}")

        for J2 in J_values:
            print(f"\n--- Processing L={L}, J2 = {J2:.2f} ---")

            # 1. Setup NetKet objects
            lattice = nk.graph.Hypercube(length=L, n_dim=2, pbc=True, max_neighbor_order=2)
            hi = nk.hilbert.Spin(s=1/2, N=lattice.n_nodes, total_sz=0)

            ha = nk.operator.Heisenberg(hilbert=hi, graph=lattice, J=[1.0, J2], sign_rule=[False, False])

            # 2. Solve with Exact Diagonalization
            print("Solving with Lanczos method...")
            try:
                E_gs, ket_gs = nk.exact.lanczos_ed(ha, compute_eigenvectors=True)
                E_exact = E_gs[0]
            except Exception as e:
                print(f"Could not perform Lanczos for L={L}: {e}")
                print("This is expected for larger lattices. Skipping this size.")
                break # Break from J2 loop and go to next L

            print(f"Lanczos complete. Ground state energy: {E_exact:.6f}")            
        
            # 3. Calculate the Structure Factor S(q)
            S_q_spin = Corr_Struct_exact(lattice, ket_gs, L, hi)
            print("Spin structure factor calculated.")

            S_q_dimer = Dimer_Corr_Struct_exact(lattice, ket_gs, L, hi)
            print("Dimer structure factor calculated.")

            # 4. Calculate the peak sharpness R
            R = calculate_R(np.abs(S_q_spin), L)
            R_values.append(R)
            print(f"Calculated R = {R:.4f}")

            R_dimer = calculate_R(np.abs(S_q_dimer), L)
            R_dimer_values.append(R_dimer)
            print(f"Calculated R_dimer = {R_dimer:.4f}")

        # 5. Plot R vs J2 for the current lattice size
        if R_values: # Only plot if we have data
            ax_spin.plot(J_values[:len(R_values)], R_values, marker='o', linestyle='-', label=f'Spin R (L={L})', color='blue')
        if R_dimer_values:
            ax_dimer.plot(J_values[:len(R_dimer_values)], R_dimer_values, marker='s', linestyle='--', label=f'Dimer R (L={L})', color='red')

    # --- Final plot styling for Spin R ---
    ax_spin.set_xlabel("$J_2$", fontsize=14)
    ax_spin.set_ylabel("Peak Sharpness R (Spin)", fontsize=14)
    ax_spin.set_title("Spin Structure Factor Sharpness vs. $J_2$", fontsize=16)
    ax_spin.grid(True, linestyle='--', alpha=0.6)
    ax_spin.legend()
    

    # --- Final plot styling for Dimer R ---
    ax_dimer.set_xlabel("$J_2$", fontsize=14)
    ax_dimer.set_ylabel("Peak Sharpness R (Dimer)", fontsize=14)
    ax_dimer.set_title("Dimer Structure Factor Sharpness vs. $J_2$", fontsize=16)
    ax_dimer.grid(True, linestyle='--', alpha=0.6)
    ax_dimer.legend()

    # --- Save and show the plots ---
    output_dir = "/scratch/f/F.Conoscenti/Thesis_QSL/Elaborate/plot"
    os.makedirs(output_dir, exist_ok=True)
    
    save_path_spin = os.path.join(output_dir, "QSL_regime_Spin_R_vs_J2.png")
    fig_spin.savefig(save_path_spin, dpi=300)
    print(f"\n✅ Spin R plot saved to {save_path_spin}")

    save_path_dimer = os.path.join(output_dir, "QSL_regime_Dimer_R_vs_J2.png")
    fig_dimer.savefig(save_path_dimer, dpi=300)
    print(f"✅ Dimer R plot saved to {save_path_dimer}")

    plt.show()


if __name__ == "__main__":
    main()