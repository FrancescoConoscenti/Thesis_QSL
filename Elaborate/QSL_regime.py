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

"""
def calculate_R(S_q, L):
    
    #Calculates the peak sharpness ratio R = 1 - S(Q+dq)/S(Q).

    #Args:
    #    S_q (np.ndarray): The 2D static structure factor.
    #    L (int): The linear size of the lattice.

    #Returns:
    #    R (float): The calculated sharpness ratio, averaged over x and y directions.
    
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
    lattice_sizes = [ 6]
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
            print("Solving with SciPy sparse diagonalization (eigsh)...")
            try:
                # Convert the NetKet operator to a SciPy sparse matrix
                ha_sparse = ha.to_sparse()
                # Use eigsh to find the ground state (k=1 eigenvalue with smallest algebraic value 'SA')
                E_gs, ket_gs = eigsh(ha_sparse, k=1, which='SA', return_eigenvectors=True)
                E_exact = E_gs[0] # The ground state energy

            except Exception as e:
                print(f"Could not perform sparse diagonalization for L={L}: {e}")
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
    
"""


#!/usr/bin/env python3
"""
ED J1-J2 Heisenberg (no NetKet), restricted to total magnetization Sz_tot = 0.

Features:
 - Builds the Sz basis with exactly N_up = N/2.
 - Constructs the J1 (nearest-neighbour) and J2 (next-nearest) Heisenberg Hamiltonian:
      H = sum_{<i,j>} J_{ij} ( S_i·S_j )
   with S_i·S_j = Sz_i Sz_j + 1/2 (S+_i S-_j + S-_i S+_j).
 - Diagonalizes (sparse) using eigsh (k=1, which='SA').
 - Computes spin structure factor S_spin(q) = (1/N) sum_{ij} e^{i q·(r_i-r_j)} <S_i·S_j>.
 - Computes a dimer structure factor built from horizontal and vertical bond operators
   B_r = S_r·S_{r+e_x} (and similarly for e_y). Then we compute correlations <B_r B_s>
   and their Fourier transform (averaged over x- and y-bond families).
 - Computes peak sharpness R = 1 - S(Q+dq)/S(Q) averaged over x/y neighbor directions.
"""

import numpy as np
import math
import itertools
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh, LinearOperator
import matplotlib.pyplot as plt
import os
import sys
from collections import defaultdict
from typing import List, Tuple

# ---------------------------
# Utilities: lattice & bonds
# ---------------------------

def square_lattice_sites(L: int):
    """Return list of 2D positions (x,y) for LxL periodic square lattice and map to site index."""
    coords = []
    for y in range(L):
        for x in range(L):
            coords.append((x, y))
    return coords

def site_index(x: int, y: int, L: int) -> int:
    return (y % L) * L + (x % L)

def nn_bonds(L:int) -> List[Tuple[int,int]]:
    """Nearest-neighbour bonds on periodic LxL."""
    bonds = []
    for y in range(L):
        for x in range(L):
            i = site_index(x, y, L)
            # right and up to avoid duplication
            jx = site_index(x+1, y, L)
            jy = site_index(x, y+1, L)
            bonds.append((i, jx))
            bonds.append((i, jy))
    # This duplicates each NN bond twice (once from each endpoint). We'll deduplicate:
    bonds_set = set(tuple(sorted(b)) for b in bonds)
    return [tuple(b) for b in sorted(bonds_set)]

def nnn_bonds(L:int) -> List[Tuple[int,int]]:
    """Next nearest neighbours (diagonals) for J2 on square lattice.

    We'll include both diagonals (x+1,y+1) and (x+1,y-1).
    """
    bonds = []
    for y in range(L):
        for x in range(L):
            i = site_index(x, y, L)
            d1 = site_index(x+1, y+1, L)
            d2 = site_index(x+1, y-1, L)
            bonds.append(tuple(sorted((i, d1))))
            bonds.append(tuple(sorted((i, d2))))
    bonds_set = set(bonds)
    return [tuple(b) for b in sorted(bonds_set)]

# ---------------------------
# Basis in Sz=0 sector
# ---------------------------

def build_sz0_basis(N: int):
    """
    Return:
      - basis_bits: list of ints (bitstrings) with exactly N/2 up spins (1)
      - bit_to_index: dict mapping bit -> index in basis
    """
    if N % 2 != 0:
        raise ValueError("N must be even to have Sz=0 sector (equal number of up/down).")
    n_up = N // 2
    # Use combinations to generate bitstrings with n_up ones.
    basis_bits = []
    for comb in itertools.combinations(range(N), n_up):
        # set bits at positions in comb (we'll use LSB=site0 convention)
        b = 0
        for p in comb:
            b |= (1 << p)
        basis_bits.append(b)
    bit_to_index = {b: idx for idx, b in enumerate(basis_bits)}
    return basis_bits, bit_to_index

# ---------------------------
# Build Hamiltonian in reduced basis
# ---------------------------

def build_heisenberg_hamiltonian(N: int, bonds_J: List[Tuple[Tuple[int,int], float]],
                                 basis_bits: List[int], bit_to_index: dict):
    """
    Construct sparse Hamiltonian H in Sz=0 basis.
    bonds_J is list of ((i,j), J_ij) pairs.
    Returns scipy.sparse.csr_matrix of shape (dim, dim).
    """
    dim = len(basis_bits)
    row = []
    col = []
    data = []

    for idx, bits in enumerate(basis_bits):
        # Precompute spin orientations: +1 for up, -1 for down (Sz = +/- 1/2 correspond to +/-1)
        # But careful: physical Sz eigenvalue is +/- 1/2; we store spin signs +/-1 for quick products
        # Sz_i Sz_j = (sign_i * sign_j) * (1/4)
        signs = [(1 if (bits >> s) & 1 else -1) for s in range(N)]

        diag_elem = 0.0  # accumulate diagonal contributions for this basis state

        # For off-diagonal flip contributions we will append entries
        for (i, j), J in bonds_J:
            si = signs[i]
            sj = signs[j]
            # Sz Sz term:
            diag_elem += J * ( (si * sj) * 0.25 )  # because Sz eigenvalues are +/- 1/2

            # Off-diagonal: 1/2 * (S+_i S-_j + S-_i S+_j)
            # This flips i and j if they are opposite; matrix element = J * 1/2
            if si != sj:
                # flip bits i and j
                flipped = bits ^ ((1 << i) | (1 << j))
                jidx = bit_to_index.get(flipped, None)
                if jidx is not None:
                    # off-diagonal element
                    val = J * 0.5
                    row.append(jidx)   # note: we will create (jidx, idx) entry because H |psi> uses H_ji * c_i
                    col.append(idx)
                    data.append(val)
                else:
                    # Flipped state is not in the Sz0 sector => should not happen, because flipping opposite spins preserves total Sz.
                    pass

        # Write diagonal
        if abs(diag_elem) > 0 or True:
            row.append(idx)
            col.append(idx)
            data.append(diag_elem)

    H = sp.coo_matrix((data, (row, col)), shape=(dim, dim))
    # symmetrize in case we have asymmetry due to insertion order (H is Hermitian and real)
    H = 0.5 * (H + H.T)
    return H.tocsr()

# ---------------------------
# Apply local operators to a state vector (in reduced basis)
# ---------------------------

def apply_SzSz_on_state(i: int, j: int, psi: np.ndarray, basis_bits: List[int]):
    """
    Compute phi = Sz_i Sz_j |psi>. Since SzSz is diagonal in Sz basis, it's trivial:
    (Sz_i Sz_j) |b> = (sign_i * sign_j) * 1/4 * |b>.
    So phi_k = coef_k * psi_k, easy to compute.
    """
    N = max(basis_bits).bit_length()
    signs_list = []
    for bits in basis_bits:
        si = 1 if (bits >> i) & 1 else -1
        sj = 1 if (bits >> j) & 1 else -1
        signs_list.append( (si * sj) * 0.25 )
    return psi * np.array(signs_list, dtype=psi.dtype)

def apply_SplusSminus_on_state(i: int, j: int, psi: np.ndarray, basis_bits: List[int], bit_to_index: dict):
    """
    Compute phi = S+_i S-_j |psi>. This operator flips i from down->up and j from up->down.
    If the basis state b has spin_i = down (0) and spin_j = up (1), then flipped state exists
    and the coefficient is 1 (for S+ S-). Return phi vector in the reduced basis order.
    """
    dim = len(basis_bits)
    phi = np.zeros(dim, dtype=psi.dtype)
    for idx, bits in enumerate(basis_bits):
        # need i: down (0), j: up (1)
        if (((bits >> i) & 1) == 0) and (((bits >> j) & 1) == 1):
            flipped = bits | (1 << i)   # set i to 1
            flipped &= ~(1 << j)        # set j to 0
            jidx = bit_to_index.get(flipped, None)
            if jidx is not None:
                # amplitude contribution: +1 * psi[idx] (S+ S- matrix element = 1)
                phi[jidx] += psi[idx]
    return phi

def apply_SminusSplus_on_state(i: int, j: int, psi: np.ndarray, basis_bits: List[int], bit_to_index: dict):
    """
    Compute phi = S-_i S+_j |psi>.
    """
    dim = len(basis_bits)
    phi = np.zeros(dim, dtype=psi.dtype)
    for idx, bits in enumerate(basis_bits):
        # need i: up (1), j: down (0)
        if (((bits >> i) & 1) == 1) and (((bits >> j) & 1) == 0):
            flipped = bits & ~(1 << i)  # set i to 0
            flipped = flipped | (1 << j) # set j to 1
            jidx = bit_to_index.get(flipped, None)
            if jidx is not None:
                phi[jidx] += psi[idx]
    return phi

def apply_Si_dot_Sj_on_state(i:int, j:int, psi: np.ndarray, basis_bits: List[int], bit_to_index: dict):
    """Compute phi = (S_i·S_j) |psi> with S_i·S_j = SzSz + 0.5*(S+S- + S-S+)."""
    # SzSz (diagonal)
    phi = apply_SzSz_on_state(i, j, psi, basis_bits)
    # S+_i S-_j
    phi += 0.5 * apply_SplusSminus_on_state(i, j, psi, basis_bits, bit_to_index)
    # S-_i S+_j
    phi += 0.5 * apply_SminusSplus_on_state(i, j, psi, basis_bits, bit_to_index)
    return phi

# ---------------------------
# Structure factors
# ---------------------------

def compute_spin_structure_factor(psi: np.ndarray, N: int, basis_bits: List[int], bit_to_index: dict, L: int):
    """
    Compute S(q) on the LxL grid of momenta.
    S(q) = (1/N) sum_{i,j} e^{i q·(r_i - r_j)} <S_i · S_j>
    Returns S_q as LxL numpy array (real).
    """
    coords = square_lattice_sites(L)
    # Precompute expectation values <S_i·S_j>
    # We'll compute a symmetric matrix corr_ij
    corr = np.zeros((N, N), dtype=np.complex128)

    # Precompute phi for each bond application and take <psi|phi> to get expectation.
    # This is O(N^2 * dim) in work; manageable for small N/dim.
    for i in range(N):
        for j in range(N):
            phi = apply_Si_dot_Sj_on_state(i, j, psi, basis_bits, bit_to_index)
            corr_val = np.vdot(psi, phi)  # <psi| S_i·S_j |psi>
            corr[i, j] = corr_val

    # Now Fourier transform to S(q)
    S_q = np.zeros((L, L), dtype=np.float64)
    for qy in range(L):
        for qx in range(L):
            qvec = (2*np.pi*qx / L, 2*np.pi*qy / L)
            sum_ = 0+0j
            for i in range(N):
                xi, yi = coords[i]
                for j in range(N):
                    xj, yj = coords[j]
                    rdot = qvec[0]*(xi - xj) + qvec[1]*(yi - yj)
                    sum_ += np.exp(1j * rdot) * corr[i, j]
            S_q[qx, qy] = (sum_.real) / N
    return S_q

def build_dimer_bonds(L:int):
    """Return list of bonds for 'dimers' separated by orientation.
    We return two lists: horizontal bonds and vertical bonds.
    Each bond is (site_i, site_j).
    """
    horiz = []
    vert = []
    for y in range(L):
        for x in range(L):
            i = site_index(x, y, L)
            jx = site_index(x+1, y, L)
            jy = site_index(x, y+1, L)
            horiz.append((i, jx))
            vert.append((i, jy))
    return horiz, vert

def compute_dimer_structure_factor(psi: np.ndarray, N: int, basis_bits: List[int], bit_to_index: dict, L:int):
    """
    Compute averaged dimer structure factor:
      - build bond operator B_r = S_r·S_{r+e_x} for horizontal bonds
      - compute correlations <B_r B_s>, Fourier transform across bond positions
      - do same for vertical bonds
      - return average S_dimer(q) = 0.5*(S_horiz(q) + S_vert(q))
    We map each bond to a coordinate (use the position of site r).
    """
    coords = square_lattice_sites(L)
    horiz_bonds, vert_bonds = build_dimer_bonds(L)
    # For each bond compute expectation value of B_r B_s
    n_bonds = len(horiz_bonds)  # equals N
    # Build list of bond operators as pairs (i,j)
    def bond_corr_matrix(bond_list):
        M = len(bond_list)
        corr = np.zeros((M, M), dtype=np.complex128)
        for a, (i, j) in enumerate(bond_list):
            # build phi_a = B_a |psi>  (B_a = S_i·S_j)
            phi_a = apply_Si_dot_Sj_on_state(i, j, psi, basis_bits, bit_to_index)
            for b, (k, l) in enumerate(bond_list):
                # corr_ab = <psi| B_a^\dagger B_b |psi> ; B are Hermitian so B_a^\dagger = B_a
                # So we need <psi| B_a B_b |psi> = <phi_a | B_b |psi> = <psi| B_b |phi_a>
                # We'll compute <psi| B_b |phi_a> by applying B_b to phi_a then vdot with psi
                phi_ab = apply_Si_dot_Sj_on_state(k, l, phi_a, basis_bits, bit_to_index)
                corr[a, b] = np.vdot(psi, phi_ab)
        return corr

    corr_h = bond_corr_matrix(horiz_bonds)
    corr_v = bond_corr_matrix(vert_bonds)

    # Fourier transform across bond positions using bond "origin" coordinates
    S_q_h = np.zeros((L, L), dtype=np.float64)
    S_q_v = np.zeros((L, L), dtype=np.float64)

    for qy in range(L):
        for qx in range(L):
            qvec = (2*np.pi*qx / L, 2*np.pi*qy / L)
            sum_h = 0+0j
            sum_v = 0+0j
            for a, (i, j) in enumerate(horiz_bonds):
                xa, ya = coords[i]
                for b, (k, l) in enumerate(horiz_bonds):
                    xb, yb = coords[k]
                    rdot = qvec[0]*(xa - xb) + qvec[1]*(ya - yb)
                    sum_h += np.exp(1j * rdot) * corr_h[a, b]
            for a, (i, j) in enumerate(vert_bonds):
                xa, ya = coords[i]
                for b, (k, l) in enumerate(vert_bonds):
                    xb, yb = coords[k]
                    rdot = qvec[0]*(xa - xb) + qvec[1]*(ya - yb)
                    sum_v += np.exp(1j * rdot) * corr_v[a, b]
            S_q_h[qx, qy] = (sum_h.real) / n_bonds
            S_q_v[qx, qy] = (sum_v.real) / n_bonds

    S_q = 0.5 * (S_q_h + S_q_v)
    return S_q

# ---------------------------
# Peak sharpness R
# ---------------------------

def calculate_R(S_q, L):
    S_q_copy = S_q.copy()
    S_q_copy[0, 0] = 0
    peak_idx = np.unravel_index(np.argmax(S_q_copy), S_q.shape)
    Q_val = S_q[peak_idx]
    neighbor_x_idx = ((peak_idx[0] + 1) % L, peak_idx[1])
    neighbor_y_idx = (peak_idx[0], (peak_idx[1] + 1) % L)
    S_Q_plus_dq_x = S_q[neighbor_x_idx]
    S_Q_plus_dq_y = S_q[neighbor_y_idx]
    R_x = 1 - (S_Q_plus_dq_x / Q_val) if Q_val != 0 else 0.0
    R_y = 1 - (S_Q_plus_dq_y / Q_val) if Q_val != 0 else 0.0
    return 0.5 * (R_x + R_y)

# ---------------------------
# Main script (replacing your NetKet script)
# ---------------------------

def main():
    lattice_sizes = [4, 6]   # set to 4 for ED; change to 5 or 6 only if you have enormous memory and CPU
    J_values = np.linspace(0.45, 0.7, 6)  # fewer points for speed in ED
    threshold_dim = 2_000_000_000_000  # refuse ED if sector dimension > threshold_dim

    for L in lattice_sizes:
        N = L * L
        dim = math.comb(N, N//2)
        print(f"L={L}, N={N}, Sz=0 sector dim = {dim}")
        if dim > threshold_dim:
            print(f"Dimension {dim} > threshold {threshold_dim}. Skipping ED for L={L}.")
            continue

        # build basis
        basis_bits, bit_to_index = build_sz0_basis(N)
        print("Constructed Sz=0 basis: dim =", len(basis_bits))

        # Bonds
        bonds1 = nn_bonds(L)
        bonds2 = nnn_bonds(L)
        bonds_J = [ (b, 1.0) for b in bonds1 ] + [ (b, None) for b in bonds2 ]  # we'll set J2 per loop

        # Prepare plotting lists
        R_spin_vals = []
        R_dimer_vals = []
        J_record = []

        for J2 in J_values:
            print(f"\nProcessing J2 = {J2:.4f}")

            # Build list of ((i,j), J) with J for J1=1.0, J2 given
            bonds_J_full = []
            for b in bonds1:
                bonds_J_full.append((b, 1.0))
            for b in bonds2:
                bonds_J_full.append((b, J2))

            # Build Hamiltonian
            print("Building Hamiltonian (sparse)...")
            H = build_heisenberg_hamiltonian(N, bonds_J_full, basis_bits, bit_to_index)
            print("H shape:", H.shape, "nnz:", H.nnz)

            if H.shape[0] == 0:
                print("Empty Hamiltonian? Skipping")
                continue

            # Diagonalize (ground state) with eigsh
            try:
                k = 1
                print("Running eigsh...")
                E_vals, V = eigsh(H, k=k, which='SA')
                E0 = E_vals[0]
                gs = V[:, 0]
                # normalize (eigsh returns normalized but numerical)
                gs = gs / np.linalg.norm(gs)
                print(f"Ground state energy: {E0:.8f}")
            except Exception as e:
                print("eigsh failed:", e)
                break

            # Compute structure factors
            print("Computing spin structure factor S(q) ...")
            S_q_spin = compute_spin_structure_factor(gs, N, basis_bits, bit_to_index, L)
            print("Computing dimer structure factor S_dimer(q) ...")
            S_q_dimer = compute_dimer_structure_factor(gs, N, basis_bits, bit_to_index, L)

            R_spin = calculate_R(np.abs(S_q_spin), L)
            R_dimer = calculate_R(np.abs(S_q_dimer), L)
            print(f"R_spin = {R_spin:.6f}, R_dimer = {R_dimer:.6f}")

            R_spin_vals.append(R_spin)
            R_dimer_vals.append(R_dimer)
            J_record.append(J2)

        # Plot
        if R_spin_vals:
            fig, ax = plt.subplots(figsize=(8,6))
            ax.plot(J_record, R_spin_vals, '-o', label=f'Spin R (L={L})')
            ax.plot(J_record, R_dimer_vals, '-s', label=f'Dimer R (L={L})')
            ax.set_xlabel("$J_2$")
            ax.set_ylabel("Peak Sharpness R")
            ax.set_title(f"Spin & Dimer R vs J2 (L={L})")
            ax.grid(True, linestyle='--', alpha=0.6)
            ax.legend()
            out_dir = "./Elaborate/plot/QSL_regime_R_vs_J2_no_netket"
            os.makedirs(out_dir, exist_ok=True)
            fname = os.path.join(out_dir, f"R_vs_J2_L{L}.png")
            fig.savefig(fname, dpi=200)
            print(f"Saved plot to {fname}")
            plt.show()

if __name__ == "__main__":
    main()
