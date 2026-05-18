"""
Gutzwiller-projected BCS mean-field state for the J1-J2 Heisenberg model
on the square lattice, following Ferrari & Becca, PRB 102, 014417 (2020).

Ansatz:
  |Psi_0> = P_G |Phi_0>

where |Phi_0> is the ground state of the auxiliary BCS Hamiltonian

  H_0 = sum_{R,R',sigma} t_{R,R'} c^dag_{R,sigma} c_{R',sigma}
       + sum_{R,R'} Delta_{R,R'} c^dag_{R,up} c^dag_{R',down} + H.c.

with:
  - s-wave hopping t at 1st neighbors
  - d_{x^2-y^2} pairing Delta_1 at 1st neighbors
  - d_{x^2-y^2} pairing Delta_4 at 4th neighbors  (2a along axis)
  - d_{xy}      pairing Delta_5 at 5th neighbors  (diagonal of 2x1 plaquette)

and P_G = prod_R n_R (2 - n_R) is the Gutzwiller projector onto the
singly-occupied sector (no empty or doubly-occupied sites).

The BCS Hamiltonian is diagonalized in momentum space via a Bogoliubov
transformation. The mean-field ground state |Phi_0> is the BCS vacuum
expressed in real space, and Gutzwiller projection is implemented by
Monte Carlo sampling (or, for small systems, exactly).

Reference:
  Hu, Becca, Parola, Sorella, PRB 88, 060402(R) (2013)  [Ansatz details]
  Ferrari & Becca, PRB 102, 014417 (2020)               [Level crossing study]
"""

import numpy as np
from itertools import combinations


# ---------------------------------------------------------------------------
# Lattice helpers
# ---------------------------------------------------------------------------

def make_square_lattice(Lx, Ly):
    """Return site indices and periodic neighbor tables for an Lx x Ly lattice."""
    N = Lx * Ly
    def idx(x, y):
        return (x % Lx) * Ly + (y % Ly)

    sites = [(x, y) for x in range(Lx) for y in range(Ly)]

    # Neighbor lists for the BCS Hamiltonian
    # nn1: 1st neighbors (±x, ±y)
    # nn4: 4th neighbors (±2x, ±2y)  [along axis at distance 2]
    # nn5: 5th neighbors (±x±2y, ±2x±y) [L-shaped knight-move]
    nn1, nn4, nn5 = [], [], []
    for x, y in sites:
        i = idx(x, y)
        for dx, dy in [(1, 0), (0, 1)]:   # store each bond once
            nn1.append((i, idx(x + dx, y + dy), dx, dy))
        for dx, dy in [(2, 0), (0, 2)]:
            nn4.append((i, idx(x + dx, y + dy), dx, dy))
        for dx, dy in [(1, 2), (2, 1)]:
            nn5.append((i, idx(x + dx, y + dy), dx, dy))

    return N, sites, idx, nn1, nn4, nn5


def make_k_points(Lx, Ly):
    """Return array of momenta (kx, ky) for the Lx x Ly BZ."""
    kx = 2 * np.pi * np.arange(Lx) / Lx
    ky = 2 * np.pi * np.arange(Ly) / Ly
    kpts = np.array([(kx[m], ky[n]) for m in range(Lx) for n in range(Ly)])
    return kpts


# ---------------------------------------------------------------------------
# BCS Hamiltonian in k-space
# ---------------------------------------------------------------------------

def epsilon_k(kx, ky, t):
    """s-wave hopping dispersion (measured from mu=0, half-filling)."""
    return -2.0 * t * (np.cos(kx) + np.cos(ky))


def Delta_k(kx, ky, Delta1, Delta4, Delta5):
    """
    Singlet pairing gap in k-space.

    d_{x^2-y^2} at 1st neighbors:
        Delta_1 * (cos kx - cos ky)

    d_{x^2-y^2} at 4th neighbors (distance 2 along axis):
        Delta_4 * (cos 2kx - cos 2ky)

    d_{xy} at 5th neighbors (knight-move):
        Delta_5 * 4 * sin kx sin ky * cos(kx + ky)   [structure factor]
        More precisely for 5th-neighbor d_xy:
        Delta_5 * (sin(kx+2ky) + sin(2kx+ky) - sin(kx-2ky) - sin(2kx-ky))
        which equals 4 Delta_5 sin kx sin ky (simplification for d_xy symmetry).
    """
    d1 = Delta1 * (np.cos(kx) - np.cos(ky))
    d4 = Delta4 * (np.cos(2 * kx) - np.cos(2 * ky))
    # 5th-neighbor d_xy structure factor
    d5 = Delta5 * (
        np.sin(kx + 2 * ky) + np.sin(2 * kx + ky)
        - np.sin(kx - 2 * ky) - np.sin(2 * kx - ky)
    )
    return d1 + d4 + d5


def bcs_bogoliubov(kpts, t, Delta1, Delta4, Delta5):
    """
    Diagonalize the BCS Hamiltonian at each k point.

    H_BdG(k) = [ eps_k    Delta_k ]
                [ Delta_k* -eps_k  ]

    Returns:
      E_k   : Bogoliubov energies E_k = sqrt(eps_k^2 + |Delta_k|^2)  shape (Nk,)
      u_k   : coherence factors u_k (|Phi_0> = prod_{E_k>0} (u_k + v_k c^dag_up c^dag_down) |vac>)
      v_k   : coherence factors v_k
    """
    eps = np.array([epsilon_k(kx, ky, t) for kx, ky in kpts])
    Delta = np.array([Delta_k(kx, ky, Delta1, Delta4, Delta5) for kx, ky in kpts])

    E = np.sqrt(eps**2 + np.abs(Delta)**2)

    # u_k^2 = (1 + eps_k/E_k)/2,  v_k^2 = (1 - eps_k/E_k)/2
    u = np.sqrt(np.clip((1.0 + eps / np.where(E > 1e-14, E, 1e-14)) / 2.0, 0, 1))
    v = np.sqrt(np.clip((1.0 - eps / np.where(E > 1e-14, E, 1e-14)) / 2.0, 0, 1))

    # Phase of v: follow Delta_k's phase so that v_k = |v_k| e^{i phi_k}
    phase = np.angle(Delta)
    v = v * np.exp(1j * phase)

    return E, u, v


# ---------------------------------------------------------------------------
# Pair wave function f(R-R') in real space
# ---------------------------------------------------------------------------

def compute_pair_wavefunction(Lx, Ly, t, Delta1, Delta4, Delta5):
    """
    Compute the pair wave function

       f(r) = (1/N) sum_k [v_k / u_k] e^{i k.r}

    which encodes all correlations in the BCS state.
    The Gutzwiller-projected state amplitude for a given spin configuration
    {R_up, R_down} is  det[ f(R_i^up - R_j^down) ].

    Returns:
      f_mat[i, j] = f(r_i - r_j)  for all site pairs (i, j),  shape (N, N).
    """
    N = Lx * Ly
    kpts = make_k_points(Lx, Ly)
    E, u, v = bcs_bogoliubov(kpts, t, Delta1, Delta4, Delta5)

    # ratio v/u (set to 0 where u ~ 0, i.e. fully paired modes)
    ratio = np.where(np.abs(u) > 1e-12, v / u, 0.0 + 0j)

    # Real-space positions
    sites = [(x, y) for x in range(Lx) for y in range(Ly)]

    # f(r) via inverse DFT
    f_r = np.zeros(N, dtype=complex)
    for ik, (kx, ky) in enumerate(kpts):
        for ir, (x, y) in enumerate(sites):
            f_r[ir] += ratio[ik] * np.exp(1j * (kx * x + ky * y))
    f_r /= N

    # Build f_mat[i, j] = f(r_i - r_j)
    # Index displacement modulo lattice
    def site_to_idx(x, y):
        return (x % Lx) * Ly + (y % Ly)

    f_mat = np.zeros((N, N), dtype=complex)
    for i, (xi, yi) in enumerate(sites):
        for j, (xj, yj) in enumerate(sites):
            dr_x = (xi - xj) % Lx
            dr_y = (yi - yj) % Ly
            f_mat[i, j] = f_r[site_to_idx(dr_x, dr_y)]

    return f_mat, f_r


# ---------------------------------------------------------------------------
# Gutzwiller-projected amplitude
# ---------------------------------------------------------------------------

def gutzwiller_amplitude(up_sites, dn_sites, f_mat):
    """
    Compute the Gutzwiller-projected BCS amplitude for a given spin
    configuration.

    For a configuration with N/2 up-spins at positions {R_i^up} and
    N/2 down-spins at {R_j^dn}, the amplitude is:

       <{R_up, R_dn} | Psi_0> = det[ f(R_i^up - R_j^dn) ]_{i,j=1..N/2}

    This is the Capriotti-Sorella-Becca determinantal formula.

    Args:
      up_sites  : list of site indices for up-spins, length Nup
      dn_sites  : list of site indices for down-spins, length Ndn
      f_mat     : pair wave function matrix (N x N)

    Returns:
      amplitude : complex scalar
    """
    submat = f_mat[np.ix_(up_sites, dn_sites)]
    return np.linalg.det(submat)


# ---------------------------------------------------------------------------
# Small exact Gutzwiller projection (brute-force, for tiny systems)
# ---------------------------------------------------------------------------

def build_projected_state_exact(Lx, Ly, t, Delta1, Delta4, Delta5):
    """
    Build the full Gutzwiller-projected state vector in the S^z=0 sector
    by evaluating det[f_{ij}] for every basis configuration.

    Only feasible for very small systems (N <= 16).

    Returns:
      psi   : dict {config: amplitude}, where config = (up_sites_tuple, dn_sites_tuple)
      norm  : float, norm of the state
    """
    N = Lx * Ly
    assert N <= 16, "Exact enumeration only feasible for N <= 16."
    Nup = N // 2  # half-filling, S^z = 0

    f_mat, _ = compute_pair_wavefunction(Lx, Ly, t, Delta1, Delta4, Delta5)

    psi = {}
    norm_sq = 0.0
    all_sites = list(range(N))

    for up_sites in combinations(all_sites, Nup):
        dn_sites = tuple(s for s in all_sites if s not in set(up_sites))
        amp = gutzwiller_amplitude(list(up_sites), list(dn_sites), f_mat)
        if np.abs(amp) > 1e-15:
            psi[(up_sites, dn_sites)] = amp
            norm_sq += np.abs(amp) ** 2

    norm = np.sqrt(norm_sq)
    return psi, norm


# ---------------------------------------------------------------------------
# Variational energy estimator (VMC skeleton)
# ---------------------------------------------------------------------------

def local_energy_heisenberg(config, f_mat, inv_f, J1, J2, Lx, Ly, idx_fn, nn1, nn4):
    """
    Compute the local energy E_loc = <x|H|Psi> / <x|Psi> for the
    J1-J2 Heisenberg model at a given spin configuration x = (up_sites, dn_sites).

    Uses the standard fast-update formula for det ratio:
      <x'|Psi> / <x|Psi> = det(f') / det(f) = [Sherman-Morrison]

    Args:
      config  : (up_sites, dn_sites) as lists
      f_mat   : full N x N pair wave-function matrix
      inv_f   : inverse of the current Nup x Ndn sub-matrix (maintained by caller)
      J1, J2  : coupling constants
      Lx, Ly  : lattice dimensions
      idx_fn  : site index function idx(x, y)
      nn1     : 1st-neighbor bond list  [(i, j, dx, dy), ...]
      nn4     : (unused here, but kept for interface consistency)

    Returns:
      e_loc   : complex local energy
    """
    up_sites, dn_sites = config
    up_set = set(up_sites)
    dn_set = set(dn_sites)
    N = Lx * Ly

    # Diagonal contribution: -J/4 for parallel, +J/4 for antiparallel on each bond
    # S_i . S_j = (1/2)(S+_i S-_j + S-_i S+_j) + S^z_i S^z_j
    # Diagonal: S^z_i S^z_j = +1/4 if same spin, -1/4 if opposite
    e_diag = 0.0
    all_bonds = [(J1, nn1)]
    # For J2 we need 2nd-neighbor bonds; build them from idx_fn
    nn2 = []
    sites = [(x, y) for x in range(Lx) for y in range(Ly)]
    for x, y in sites:
        i = idx_fn(x, y)
        for dx, dy in [(1, 1), (1, -1)]:
            j = idx_fn(x + dx, y + dy)
            if j > i:
                nn2.append((i, j, dx, dy))
    all_bonds.append((J2, nn2))

    for J, bonds in all_bonds:
        for (i, j, *_) in bonds:
            si_up = (i in up_set)
            sj_up = (j in up_set)
            if si_up == sj_up:
                e_diag += J * 0.25      # parallel
            else:
                e_diag -= J * 0.25      # antiparallel

    # Off-diagonal contribution: spin flips (S+_i S-_j + h.c.)
    e_offdiag = 0.0
    for J, bonds in all_bonds:
        for (i, j, *_) in bonds:
            si_up = (i in up_set)
            sj_up = (j in up_set)
            if si_up != sj_up:
                # Can flip: move up-spin from i (or j) to j (or i)
                # Det ratio via Sherman-Morrison
                if si_up and not sj_up:
                    # i is up, j is down -> flip: i becomes down, j becomes up
                    new_up = [s if s != i else j for s in up_sites]
                    new_dn = [s if s != j else i for s in dn_sites]
                else:
                    new_up = [s if s != j else i for s in up_sites]
                    new_dn = [s if s != i else j for s in dn_sites]

                sub_new = f_mat[np.ix_(new_up, new_dn)]
                sub_old = f_mat[np.ix_(up_sites, dn_sites)]
                ratio = np.linalg.det(sub_new) / (np.linalg.det(sub_old) + 1e-300)
                e_offdiag += J * 0.5 * ratio

    return e_diag + e_offdiag

# ---------------------------------------------------------------------------
# Grid Search
# ---------------------------------------------------------------------------

def plot_grid_search(Lx=4, Ly=4, J1=1.0, J2=0.5):
    import matplotlib.pyplot as plt
    import os
    import pickle
    
    t_vals = np.linspace(0.0, 0.5 , 20)
    d1_vals = np.linspace(1.0, 3.0, 20)
    d4_vals = np.linspace(0.0, 2.0, 20)
    d5_vals = np.linspace(0.0, 2.0, 20)
    
    def idx_fn(x, y): return (x % Lx) * Ly + (y % Ly)
    sites = [(x, y) for x in range(Lx) for y in range(Ly)]
    nn1 = []
    for x, y in sites:
        i = idx_fn(x, y)
        for dx, dy in [(1, 0), (0, 1)]:
            nn1.append((i, idx_fn(x + dx, y + dy), dx, dy))
            
    nn2 = []
    for x, y in sites:
        i = idx_fn(x, y)
        for dx, dy in [(1, 1), (1, -1)]:
            nn2.append((i, idx_fn(x + dx, y + dy), dx, dy))
            
    all_bonds = [(J1, nn1), (J2, nn2)]
    
    save_dir = "FreeFermions/plots/GridSearch"
    os.makedirs(save_dir, exist_ok=True)
    
    T_mesh, D1_mesh = np.meshgrid(t_vals, d1_vals)
    
    # Create a 4D array to store all energies
    all_energies_data = np.zeros((len(d4_vals), len(d5_vals), len(d1_vals), len(t_vals)))
    
    for id4, d4 in enumerate(d4_vals):
        for id5, d5 in enumerate(d5_vals):
            print(f"--- Grid search for Delta4={d4:.2f}, Delta5={d5:.2f} ---")
            energies = np.zeros_like(T_mesh)
            for i, d1 in enumerate(d1_vals):
                for j, t in enumerate(t_vals):
                    psi, norm = build_projected_state_exact(Lx, Ly, t, d1, d4, d5)
                    
                    if norm < 1e-10:
                        energies[i, j] = np.nan
                        continue
                        
                    energy = 0.0
                    norm_sq = 0.0
                    
                    for config, amp in psi.items():
                        up_sites, dn_sites = config
                        up_set = set(up_sites)
                        e_diag = 0.0
                        e_offdiag = 0.0
                        
                        for J, bonds in all_bonds:
                            for (u, v_site, *_) in bonds:
                                su_up = (u in up_set)
                                sv_up = (v_site in up_set)
                                if su_up == sv_up:
                                    e_diag += J * 0.25
                                else:
                                    e_diag -= J * 0.25
                                    if su_up and not sv_up:
                                        new_up = tuple(sorted([s for s in up_sites if s != u] + [v_site]))
                                        new_dn = tuple(sorted([s for s in dn_sites if s != v_site] + [u]))
                                    else:
                                        new_up = tuple(sorted([s for s in up_sites if s != v_site] + [u]))
                                        new_dn = tuple(sorted([s for s in dn_sites if s != u] + [v_site]))
                                        
                                    new_config = (new_up, new_dn)
                                    if new_config in psi:
                                        amp_new = psi[new_config]                                        
                                        # The off-diagonal term is +J/2 * (S+_i S-_j + S-_j S+_i)
                                        e_offdiag += J * 0.5 * amp_new / amp
                                        
                        prob = np.abs(amp)**2
                        energy += prob * (e_diag + e_offdiag.real)
                        norm_sq += prob
                        
                    energies[i, j] = (energy / norm_sq) / (Lx * Ly)
                    all_energies_data[id4, id5, i, j] = energies[i, j]

                    # Save all collected data incrementally
                    data_to_save = {
                        't_vals': t_vals,
                        'd1_vals': d1_vals,
                        'd4_vals': d4_vals,
                        'd5_vals': d5_vals,
                        'energies': all_energies_data
                    }
                    save_data_path = os.path.join(save_dir, "all_energies.pkl")
                    with open(save_data_path, 'wb') as f:
                        pickle.dump(data_to_save, f)

            """plt.figure(figsize=(8, 6))
            cmap = plt.get_cmap('viridis')
            im = plt.pcolormesh(T_mesh, D1_mesh, energies, cmap=cmap, shading='auto')
            plt.colorbar(im, label='Energy per site')
            plt.xlabel('Hopping $t$', fontsize=14)
            plt.ylabel('Pairing $\\Delta_1$', fontsize=14)
            plt.title(f'Energy Landscape ($\\Delta_4$={d4:.2f}, $\\Delta_5$={d5:.2f})', fontsize=14)
            
            if not np.all(np.isnan(energies)):
                min_idx = np.nanargmin(energies)
                min_i, min_j = np.unravel_index(min_idx, energies.shape)
                min_t = t_vals[min_j]
                min_d1 = d1_vals[min_i]
                min_e = energies[min_i, min_j]
                plt.scatter([min_t], [min_d1], color='red', marker='*', s=150, label=f'Min: E={min_e:.4f}')
                plt.legend()
            
            plt.tight_layout()
            save_path = os.path.join(save_dir, f"energy_grid_d4_{d4:.2f}_d5_{d5:.2f}.png")
            plt.savefig(save_path, dpi=300)
            plt.close()
            print(f"Saved plot to {save_path}")"""


# ---------------------------------------------------------------------------
# Main: example usage
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")

    # --- System parameters ---
    Lx, Ly = 4, 4
    N = Lx * Ly

    # --- Best variational parameters from Hu et al. PRB 2013 / Ferrari & Becca 2020
    # at J2/J1 = 0.5 (deep in the frustrated region)
    t      = 1.0    # s-wave hopping (sets energy scale)
    Delta1 = 0.9    # d_{x^2-y^2} pairing, 1st neighbors
    Delta4 = 0.1    # d_{x^2-y^2} pairing, 4th neighbors
    Delta5 = 0.05   # d_{xy} pairing, 5th neighbors

    print("=" * 60)
    print("Gutzwiller-projected BCS mean-field state")
    print(f"Lattice: {Lx}x{Ly}  (N={N} sites)")
    print(f"Parameters: t={t}, Delta1={Delta1}, Delta4={Delta4}, Delta5={Delta5}")
    print("=" * 60)

    # --- Build pair wave function ---
    f_mat, f_r = compute_pair_wavefunction(Lx, Ly, t, Delta1, Delta4, Delta5)
    print(f"\nPair wave function f(r) computed. Shape: {f_mat.shape}")
    print("f(r=0) =", f_r[0].real, "(real part, should be finite for BCS)")

    # --- Exact Gutzwiller projection (small system) ---
    print(f"\nBuilding exact Gutzwiller-projected state for {N} sites...")
    psi, norm = build_projected_state_exact(Lx, Ly, t, Delta1, Delta4, Delta5)
    print(f"Number of nonzero configurations: {len(psi)}")
    print(f"State norm: {norm:.6f}")

    # --- Sample a few amplitudes ---
    items = list(psi.items())[:5]
    print("\nSample amplitudes (up_sites, dn_sites) -> |amplitude|:")
    for (up, dn), amp in items:
        print(f"  up={up}, dn={dn} -> |psi|={np.abs(amp):.6e}")

    # --- Bogoliubov spectrum ---
    kpts = make_k_points(Lx, Ly)
    E, u, v = bcs_bogoliubov(kpts, t, Delta1, Delta4, Delta5)
    print(f"\nBogoliubov energies: min={E.min():.4f}, max={E.max():.4f}")
    print(f"Min gap (proxy for nodal structure): {E.min():.6f}")
    if E.min() < 1e-3:
        print("  -> Nodal (gapless) BCS state: Dirac spinon excitations (d-wave!)")
    else:
        print("  -> Gapped BCS state")

    # --- Local energy for a single configuration ---
    # Pick the configuration with the largest |amplitude| from the projected state
    best_config, best_amp = max(psi.items(), key=lambda kv: abs(kv[1]))
    best_up, best_dn = list(best_config[0]), list(best_config[1])

    _, _, idx_fn, nn1, nn4, nn5 = make_square_lattice(Lx, Ly)
    sub = f_mat[np.ix_(best_up, best_dn)]
    inv_sub = np.linalg.inv(sub)

    J1, J2 = 1.0, 0.5
    e_loc = local_energy_heisenberg(
        (best_up, best_dn), f_mat, inv_sub, J1, J2, Lx, Ly, idx_fn, nn1, nn4
    )
    print(f"\nLocal energy for largest-weight config at J2/J1={J2/J1}:")
    print(f"  up={best_up}, dn={best_dn}")
    print(f"  |amplitude| = {abs(best_amp):.6e}")
    print(f"  E_loc = {e_loc.real:.6f} (Re),  {e_loc.imag:.2e} (Im)")

    print("\nStarting grid search for energy landscape...")
    plot_grid_search(Lx=Lx, Ly=Ly, J1=1.0, J2=0.5)