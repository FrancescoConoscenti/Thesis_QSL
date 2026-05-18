"""
Staggered Flux + Néel (SF+N) Gutzwiller-projected mean-field state
for the J1-J2 Heisenberg model on the square lattice.

Reference: Dalla Piazza, PhD thesis (2014), section 2.4.6
  Eqs. 2.4.67–2.4.87

Physics summary
---------------
The SF+N mean-field Hamiltonian is  H_{SF+N} = H_SF + H_N  (eq. 2.4.67).

H_SF (eq. 2.4.68): staggered-flux hopping with flux parameter θ₀.
  On EVEN sites i: hop to i+x̂ with amplitude -½ e^{+iθ₀}, hop to i+ŷ with -½ e^{-iθ₀}
  On ODD  sites i: hop to i+x̂ with amplitude -½ e^{-iθ₀}, hop to i+ŷ with -½ e^{+iθ₀}
  (plus H.c. for all bonds)

H_N (eq. 2.4.69): staggered Néel field with amplitude h_N.
  H_N = -h_N Σ_σ σ (Σ_{i even} c†_{iσ}c_{iσ} - Σ_{i odd} c†_{iσ}c_{iσ})
  where σ ∈ {-1,+1} (↓ = -1, ↑ = +1).

The H_{SF+N} is diagonalised in the magnetic Brillouin zone (MBZ, |k|≤π) via
a 2×2 Bogoliubov-like transformation (eqs. 2.4.73–2.4.81).  The quasiparticle
dispersion is (eq. 2.4.80):

    ω_k = √(|Δ_k|² + h_N²)

with (eq. 2.4.72):

    Δ_k = ½(e^{+iθ₀} cos kx + e^{-iθ₀} cos ky)

The mean-field ground state is obtained by filling the lower band of each spin
species (eq. 2.4.85):

    |ψ_GS⟩ = ∏_{k∈MBZ} γ†_{k↑−} γ†_{k↓−} |0⟩

The variational state is the Gutzwiller-projected version (eq. 2.4.86):

    |GS(θ₀, h_N)⟩ = P_{D=0} |ψ_GS⟩

and the variational energy (eq. 2.4.87) is evaluated with VMC:

    E_GS(θ₀, h_N) = ⟨GS|H_Heisenberg|GS⟩ / ⟨GS|GS⟩

Key relationship to the pair wave function (eq. 2.4.81–2.4.82):
  The Gutzwiller-projected state has the same determinantal structure as the
  Becca-style ansatz: ⟨σ|GS⟩ = det[F(R_i↑ - R_j↓)] where F is the pair matrix
  built from the Bogoliubov coherence factors.  Unlike the BCS case, no pairing
  (Δ_ij=0 in the SF gauge) so the state is a pure Slater determinant — no
  anomalous propagator.  The pair matrix arises because the Bogoliubov
  transformation mixes up- and down-spinon operators from the enlarged (2-site)
  unit cell.

Grid search
-----------
Reproduces Figure 2.14 of the thesis: variational energy E_GS(θ₀, h_N) for
J1=1, J2=0 on a 4×4 lattice, scanned over
    θ₀ ∈ [0, π/2]   (≡ θ₀/π ∈ [0, 0.5])
    h_N ∈ [0, 0.15]
"""

import os
os.environ["JAX_ENABLE_X64"] = "1"

import argparse
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# ─────────────────────────────────────────────────────────────────────────────
# 1.  Lattice helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_lattice(Lx, Ly):
    """Return site index function and neighbour tables."""
    def idx(x, y):
        return (x % Lx) * Ly + (y % Ly)

    sites = [(x, y) for x in range(Lx) for y in range(Ly)]

    # Nearest-neighbour bonds stored once (i < j convention not enforced here;
    # we iterate over directed bonds for the hopping Hamiltonian)
    nn1 = []           # (i, j, dx, dy)  1st-neighbor directed bonds
    for x, y in sites:
        i = idx(x, y)
        for dx, dy in [(1, 0), (0, 1), (-1, 0), (0, -1)]:
            j = idx(x + dx, y + dy)
            nn1.append((i, j, dx, dy))

    return idx, sites, nn1


def is_even(x, y):
    """True if site (x,y) is on the 'even' sublattice: (ix+iy) % 2 == 0."""
    return (x + y) % 2 == 0


# ─────────────────────────────────────────────────────────────────────────────
# 2.  SF+N Hamiltonian matrix (single-particle, one spin species)
# ─────────────────────────────────────────────────────────────────────────────

def build_sfn_single_spin(Lx, Ly, theta0, h_N, sigma):
    """
    Build the N×N single-particle SF+N Hamiltonian matrix for spin σ ∈ {+1,-1}.

    H_SF contribution (eq. 2.4.68):
      For bond i→j = i+x̂:
        t_ij = -½ e^{+iθ₀}  if i is even
        t_ij = -½ e^{-iθ₀}  if i is odd
      For bond i→j = i+ŷ:
        t_ij = -½ e^{-iθ₀}  if i is even
        t_ij = -½ e^{+iθ₀}  if i is odd
      Plus H.c. (so t_{ji} = t*_{ij})

    H_N contribution (eq. 2.4.69):
      Diagonal: -h_N * σ * (+1 if i even, -1 if i odd)

    Parameters
    ----------
    sigma : +1 for spin-up (↑), -1 for spin-down (↓)
    """
    N = Lx * Ly
    H = np.zeros((N, N), dtype=complex)
    idx, sites, _ = make_lattice(Lx, Ly)

    # H_SF hopping
    for x, y in sites:
        i = idx(x, y)
        even = is_even(x, y)

        # hop along +x̂
        j = idx(x + 1, y)
        t = -0.5 * (np.exp(+1j * theta0) if even else np.exp(-1j * theta0))
        H[i, j] += t
        H[j, i] += np.conj(t)   # H.c.

        # hop along +ŷ
        j = idx(x, y + 1)
        t = -0.5 * (np.exp(-1j * theta0) if even else np.exp(+1j * theta0))
        H[i, j] += t
        H[j, i] += np.conj(t)   # H.c.

    # H_N Néel field
    for x, y in sites:
        i = idx(x, y)
        stagger = +1 if is_even(x, y) else -1
        H[i, i] += -h_N * sigma * stagger

    return H


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Mean-field ground state: single-particle eigenstates of H_{SF+N}
# ─────────────────────────────────────────────────────────────────────────────

def sfn_ground_state_slater(Lx, Ly, theta0, h_N):
    """
    Diagonalise H_SF+N for each spin and return the N×(N/2) matrices of
    occupied single-particle eigenvectors (the Slater determinant columns).

    At half-filling the lowest N/2 eigenstates are occupied for each spin.
    We pack them into matrices:
        U_up[i, alpha]  = <i|φ_alpha> for alpha = 0,...,N/2-1  (spin up)
        U_dn[i, alpha]  = same for spin down

    The unprojected mean-field ground state is:
        |ψ_GS> = det(U_up) * det(U_dn) * product of (γ†_{kσ-})|0>

    For the Gutzwiller projection we only need the N×(N/2) coefficient matrices.

    Note on spin: since H_N breaks spin degeneracy (σ appears explicitly),
    the two spin species have DIFFERENT single-particle Hamiltonians and
    hence different occupied subspaces.
    """
    N = Lx * Ly
    Nup = N // 2   # half-filling

    occupied = {}
    for sigma, label in [(+1, "up"), (-1, "dn")]:
        H = build_sfn_single_spin(Lx, Ly, theta0, h_N, sigma)
        evals, evecs = np.linalg.eigh(H)     # ascending order
        # Fill the Nup lowest bands
        occupied[label] = evecs[:, :Nup]     # shape (N, Nup), complex

    return occupied["up"], occupied["dn"]


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Pair (overlap) matrix for Gutzwiller amplitude
#
#  For a spin-↑ Slater determinant with columns U_up and spin-↓ with U_dn,
#  the overlap amplitude for a configuration |{R↑},{R↓}> is:
#
#    <{R↑,R↓}|ψ_GS> = det[U_up[R↑,:]] * det[U_dn[R↓,:]]
#
#  The Gutzwiller projector P_{D=0} keeps only configurations with exactly
#  one fermion per site (no empty, no doubly occupied).  Since we are at
#  half-filling and the two spin sectors are independent, we automatically
#  get N/2 up-spins on N/2 sites and N/2 down-spins on the remaining N/2
#  sites.  Hence:
#
#    <σ|GS> ∝ det[U_up[R↑,:]] * det[U_dn[R↓,:]]    (Gutzwiller projection)
#
#  where the constraint R↑ ∩ R↓ = ∅  is enforced by the projection.
# ─────────────────────────────────────────────────────────────────────────────

def gutzwiller_log_amplitude(up_sites, dn_sites, U_up, U_dn):
    """
    log|<σ|GS>| = log|det(U_up[up_sites, :])| + log|det(U_dn[dn_sites, :])|

    Returns (log_abs, sign) where sign is complex ∈ {+1, -1, +i, -i}.
    """
    sub_up = U_up[np.array(up_sites), :]     # (Nup, Nup)
    sub_dn = U_dn[np.array(dn_sites), :]     # (Ndn, Ndn)

    s_up, la_up = np.linalg.slogdet(sub_up)
    s_dn, la_dn = np.linalg.slogdet(sub_dn)

    return la_up + la_dn, s_up * s_dn


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Exact variational energy for small systems (4×4)
# ─────────────────────────────────────────────────────────────────────────────

def build_heisenberg_sparse(Lx, Ly, J1, J2):
    """
    Build the J1-J2 Heisenberg Hamiltonian as a sparse matrix in the S^z=0
    sector.  Returns (H_csr, configs, cfg_to_idx).

    H = J1 Σ_{<ij>} S_i·S_j + J2 Σ_{<<ij>>} S_i·S_j
    with S_i = (1/2) σ_i  so  S_i·S_j = (1/4) σ_i σ_j.

    Encoding: config = tuple of Nup site indices carrying ↑-spin (sorted).
    """
    from itertools import combinations
    from scipy.sparse import lil_matrix

    N = Lx * Ly
    idx_fn = lambda x, y: (x % Lx) * Ly + (y % Ly)
    all_sites = list(range(N))
    configs = list(combinations(all_sites, N // 2))
    cfg_to_idx = {c: i for i, c in enumerate(configs)}
    dim = len(configs)

    H = lil_matrix((dim, dim), dtype=float)

    def add_bond(J, i, j):
        for ci, cfg in enumerate(configs):
            up = set(cfg)
            si = +1 if i in up else -1
            sj = +1 if j in up else -1
            # Diagonal: J * (1/4) * si * sj
            H[ci, ci] += J * 0.25 * si * sj
            # Off-diagonal S+_i S-_j + h.c.: only if antiparallel
            if si != sj:
                if i in up:   # i=↑, j=↓ → flip i↓ j↑
                    new_up = tuple(sorted((up - {i}) | {j}))
                else:          # i=↓, j=↑ → flip i↑ j↓
                    new_up = tuple(sorted((up - {j}) | {i}))
                if new_up in cfg_to_idx:
                    H[ci, cfg_to_idx[new_up]] += J * 0.5

    # J1 nearest-neighbor bonds (each bond once)
    for x in range(Lx):
        for y in range(Ly):
            for dx, dy in [(1, 0), (0, 1)]:
                add_bond(J1, idx_fn(x, y), idx_fn(x + dx, y + dy))

    # J2 next-nearest-neighbor bonds (all 4 diagonals, deduplicated)
    seen = set()
    for x in range(Lx):
        for y in range(Ly):
            for dx, dy in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:
                bond = tuple(sorted((idx_fn(x, y), idx_fn(x + dx, y + dy))))
                if bond not in seen:
                    seen.add(bond)
                    add_bond(J2, bond[0], bond[1])

    return H.tocsr(), configs, cfg_to_idx


def variational_energy_exact(Lx, Ly, theta0, h_N, J1=1.0, J2=0.0):
    """
    Exact variational energy E_GS(θ₀, h_N) for the Gutzwiller-projected
    SF+N state on a small lattice.

    E = ⟨ψ|H_Heisenberg|ψ⟩ / ⟨ψ|ψ⟩
    where |ψ⟩ = P_{D=0}|ψ_GS(θ₀,h_N)⟩.

    Complexity: O(dim × N²) where dim = C(N, N/2) ~ 12870 for N=16.
    """
    N = Lx * Ly
    all_sites = list(range(N))

    # 1. Build the SF+N ground-state Slater matrices
    U_up, U_dn = sfn_ground_state_slater(Lx, Ly, theta0, h_N)

    # 2. Build H_Heisenberg
    H_sp, configs, _ = build_heisenberg_sparse(Lx, Ly, J1, J2)
    H_dense = H_sp.toarray()

    # 3. Evaluate the full state vector ψ[config] = det(U_up[R↑]) × det(U_dn[R↓])
    dim = len(configs)
    psi = np.zeros(dim, dtype=complex)
    for ci, cfg in enumerate(configs):
        up = list(cfg)
        dn = [s for s in all_sites if s not in set(cfg)]
        la, sign = gutzwiller_log_amplitude(up, dn, U_up, U_dn)
        psi[ci] = sign * np.exp(la)

    # Shift for numerical stability (log-sum trick not needed since all finite)
    # but large determinants can overflow; use slogdet already done above

    norm2 = np.real(np.dot(psi.conj(), psi))
    if norm2 < 1e-30:
        return np.nan

    E = np.real(np.dot(psi.conj(), H_dense @ psi)) / norm2
    return E / N    # per site


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Analytic MF dispersion (for cross-check and band plot)
# ─────────────────────────────────────────────────────────────────────────────

def sfn_dispersion(kx, ky, theta0, h_N):
    """
    Quasiparticle dispersion ω_k from eq. 2.4.80:
        Δ_k = ½(e^{+iθ₀} cos kx + e^{-iθ₀} cos ky)
        ω_k = √(|Δ_k|² + h_N²)

    k should be in the magnetic Brillouin zone.
    """
    Delta_k = 0.5 * (np.exp(+1j * theta0) * np.cos(kx)
                     + np.exp(-1j * theta0) * np.cos(ky))
    omega_k = np.sqrt(np.abs(Delta_k)**2 + h_N**2)
    return omega_k


# ─────────────────────────────────────────────────────────────────────────────
# 7.  Grid search: reproduce Figure 2.14
# ─────────────────────────────────────────────────────────────────────────────

def grid_search(Lx=4, Ly=4, J1=1.0, J2=0.0,
                n_theta=30, n_h=25,
                theta_max_frac=0.5,   # θ₀/π goes from 0 to theta_max_frac
                h_max=0.15,
                save_path=None):
    """
    Compute E_GS(θ₀, h_N) on a grid and reproduce Figure 2.14.

    Parameters
    ----------
    n_theta, n_h  : grid resolution
    theta_max_frac: maximum θ₀/π (0.5 covers the full range of Fig. 2.14)
    h_max         : maximum h_N
    save_path     : if given, save the figure to this path
    """
    theta_fracs = np.linspace(0, theta_max_frac, n_theta)   # θ₀/π
    h_vals = np.linspace(0, h_max, n_h)

    E_grid = np.zeros((n_theta, n_h))

    t0 = time.time()
    total = n_theta * n_h
    done = 0

    print(f"\nGrid search: {n_theta}×{n_h} = {total} points "
          f"on {Lx}×{Ly} lattice  (J1={J1}, J2={J2})")
    print(f"θ₀/π ∈ [0, {theta_max_frac}],  h_N ∈ [0, {h_max}]")
    print("-" * 60)

    for i, tf in enumerate(theta_fracs):
        theta0 = tf * np.pi
        for j, h in enumerate(h_vals):
            E_grid[i, j] = variational_energy_exact(
                Lx, Ly, theta0, h, J1=J1, J2=J2
            )
            done += 1
        elapsed = time.time() - t0
        eta = elapsed / done * (total - done)
        print(f"  θ₀/π={tf:.3f}  "
              f"E range=[{E_grid[i].min():.4f}, {E_grid[i].max():.4f}]  "
              f"({done}/{total}, ETA {eta:.0f}s)")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s")

    # ── Find optimum ──────────────────────────────────────────────────────────
    ij_min = np.unravel_index(np.nanargmin(E_grid), E_grid.shape)
    E_min = E_grid[ij_min]
    theta_opt = theta_fracs[ij_min[0]]
    h_opt = h_vals[ij_min[1]]
    print(f"\nOptimum:  E/N = {E_min:.6f}"
          f"  at  θ₀/π = {theta_opt:.4f},  h_N = {h_opt:.4f}")

    # ── Plot ─────────────────────────────────────────────────────────────────
    H_mesh, T_mesh = np.meshgrid(h_vals, theta_fracs)   # (n_theta, n_h)

    fig, ax = plt.subplots(figsize=(7, 6))

    # Use the same colormap style as Fig. 2.14 (blue=low, red=high)
    vmin = np.nanmin(E_grid)
    vmax = np.nanmax(E_grid)
    pcm = ax.pcolormesh(H_mesh, T_mesh, E_grid,
                        cmap="jet", shading="auto",
                        vmin=vmin, vmax=vmax)
    cbar = fig.colorbar(pcm, ax=ax)
    cbar.set_label(r"$E_{\rm GS}(\theta_0, h_{\rm N})$", fontsize=13)

    # Dashed contour at 0.5% above minimum (as in Fig. 2.14)
    threshold = E_min * (1 - 0.005) if E_min < 0 else E_min * (1 + 0.005)
    ax.contour(H_mesh, T_mesh, E_grid,
               levels=[threshold], colors="white",
               linestyles="--", linewidths=1.5)

    # Mark optimum
    ax.plot(h_opt, theta_opt, "w*", ms=12, label=f"min E/N={E_min:.4f}")
    ax.legend(fontsize=10, loc="upper right")

    ax.set_xlabel(r"$h_{\rm N}$", fontsize=14)
    ax.set_ylabel(r"$\theta_0 / \pi$", fontsize=14)
    ax.set_title(f"Variational energy  $P_{{D=0}}|\\psi_{{\\rm GS}}\\rangle$\n"
                 f"{Lx}×{Ly},  J1={J1}, J2={J2}", fontsize=13)

    plt.tight_layout()
    if save_path:
        if os.path.isdir(save_path):
            actual_save_path = os.path.join(save_path, f"SFN_grid_Lx{Lx}_Ly{Ly}_J1{J1}_J2{J2}.png")
        else:
            actual_save_path = save_path
        plt.savefig(actual_save_path, dpi=200)
        print(f"Saved figure to {actual_save_path}")
    plt.show()

    return theta_fracs, h_vals, E_grid


# ─────────────────────────────────────────────────────────────────────────────
# 8.  Band-structure visualisation (cross-check)
# ─────────────────────────────────────────────────────────────────────────────

def plot_bands(Lx, Ly, theta0, h_N, save_path=None):
    """Plot the SF+N single-particle bands (Fig. 2.11 style)."""
    N = Lx * Ly
    U_up, U_dn = sfn_ground_state_slater(Lx, Ly, theta0, h_N)
    H_up = build_sfn_single_spin(Lx, Ly, theta0, h_N, +1)
    evals_up = np.sort(np.linalg.eigvalsh(H_up))
    H_dn = build_sfn_single_spin(Lx, Ly, theta0, h_N, -1)
    evals_dn = np.sort(np.linalg.eigvalsh(H_dn))

    print(f"\nSF+N bands (θ₀={theta0/np.pi:.3f}π, h_N={h_N}):")
    print(f"  Spin-up  eigenvalues: {np.round(evals_up, 4)}")
    print(f"  Spin-dn  eigenvalues: {np.round(evals_dn, 4)}")
    print(f"  Fermi energy (spin up):  between {evals_up[N//2-1]:.4f} and {evals_up[N//2]:.4f}")

    # Analytic dispersion along Γ→M→X→Γ path
    path_k = []
    labels = []
    # Γ=(0,0) → M=(π,π)  (but in MBZ M is at (π/2,π/2))
    for t in np.linspace(0, 1, 30):
        path_k.append((t * np.pi / 2, t * np.pi / 2))
    labels += [(0, "Γ"), (29, "M")]
    # M=(π/2,π/2) → X=(π/2,0)
    for t in np.linspace(0, 1, 30)[1:]:
        path_k.append((np.pi / 2, (1 - t) * np.pi / 2))
    labels += [(58, "X")]
    # X=(π/2,0) → Γ=(0,0)
    for t in np.linspace(0, 1, 30)[1:]:
        path_k.append(((1 - t) * np.pi / 2, 0))
    labels += [(87, "Γ")]

    kpath = np.array(path_k)
    omega = sfn_dispersion(kpath[:, 0], kpath[:, 1], theta0, h_N)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(omega, "b-", lw=2, label=r"$\omega_k$ (lower band)")
    ax.plot(-omega, "r--", lw=2, label=r"$-\omega_k$ (upper band)")
    ax.axhline(0, color="k", lw=0.5, ls=":")
    for pos, lab in labels:
        ax.axvline(pos, color="gray", lw=0.5)
        ax.text(pos, ax.get_ylim()[0], lab, ha="center", fontsize=11)
    ax.set_ylabel(r"$E_{\bf k}^{\rm MF}$", fontsize=13)
    ax.set_title(f"SF+N bands  θ₀={theta0/np.pi:.2f}π, h_N={h_N}", fontsize=12)
    ax.legend(fontsize=10)
    plt.tight_layout()
    if save_path:
        if os.path.isdir(save_path):
            actual_save_path = os.path.join(save_path, f"SFN_bands_Lx{Lx}_Ly{Ly}_theta{theta0/np.pi:.2f}pi_hN{h_N:.2f}_SFN.png")
        else:
            actual_save_path = save_path
        plt.savefig(actual_save_path, dpi=150)
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# 9.  Quick sanity checks
# ─────────────────────────────────────────────────────────────────────────────

def sanity_checks(Lx=4, Ly=4):
    """Run a few quick sanity checks."""
    print("=" * 60)
    print("Sanity checks")
    print("=" * 60)

    # 1. At θ₀=π/4 and h_N=0: pure staggered flux, Dirac points
    #    The bands should be gapless (ω_k → 0 at k=(π/2,π/2))
    theta0_dirac = np.pi / 4
    Delta_dirac = sfn_dispersion(np.pi / 2, np.pi / 2, theta0_dirac, 0.0)
    print(f"\n1. θ₀=π/4, h_N=0: |Δ_k| at k=(π/2,π/2) = {abs(Delta_dirac):.6f}  (expect 0)")

    # 2. At θ₀=0: no flux → pure Néel hopping, Δ_k = ½(coskx + cosky)
    Delta_00 = sfn_dispersion(0.0, 0.0, 0.0, 0.0)
    Delta_pi0 = sfn_dispersion(np.pi / 2, 0.0, 0.0, 0.0)
    print(f"2. θ₀=0, k=(0,0): |Δ_k| = {abs(Delta_00):.4f}  (expect 1.0)")
    print(f"   θ₀=0, k=(π/2,0): |Δ_k| = {abs(Delta_pi0):.4f}  (expect 0.5)")

    # 3. Hermiticity of H_SF+N
    H = build_sfn_single_spin(Lx, Ly, np.pi / 6, 0.05, +1)
    print(f"\n3. H hermitian: max|H-H†| = {np.max(np.abs(H - H.conj().T)):.2e}  (expect 0)")

    # 4. Energy at a specific point to verify formula
    E = variational_energy_exact(Lx, Ly, np.pi / 8, 0.05, J1=1.0, J2=0.0)
    print(f"\n4. E/N(θ₀=π/8, h_N=0.05, J2=0) = {E:.6f}")
    print(f"   (No reference; checking it's finite and negative)")

    # 5. Symmetry: energy should be symmetric under θ₀ → π/2 - θ₀
    #    (SF Hamiltonian has this symmetry)
    E1 = variational_energy_exact(Lx, Ly, np.pi / 6, 0.05)
    E2 = variational_energy_exact(Lx, Ly, np.pi / 2 - np.pi / 6, 0.05)
    print(f"\n5. Symmetry θ₀ → π/2-θ₀:")
    print(f"   E(π/6, 0.05)     = {E1:.8f}")
    print(f"   E(π/3, 0.05)     = {E2:.8f}")
    print(f"   Difference       = {abs(E1 - E2):.2e}  (expect ~0)")

    print()


# ─────────────────────────────────────────────────────────────────────────────
# 10.  Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SF+N Gutzwiller state — grid search over (θ₀, h_N)"
    )
    parser.add_argument("--Lx",      type=int,   default=16)
    parser.add_argument("--Ly",      type=int,   default=16)
    parser.add_argument("--J1",      type=float, default=1.0)
    parser.add_argument("--J2",      type=float, default=0.0)
    parser.add_argument("--n_theta", type=int,   default=10,
                        help="Grid points along θ₀/π axis")
    parser.add_argument("--n_h",     type=int,   default=10,
                        help="Grid points along h_N axis")
    parser.add_argument("--h_max",   type=float, default=0.15)
    parser.add_argument("--save",    type=str,   default="/scratch/f/F.Conoscenti/Thesis_QSL/MFDallaPiazza",
                        help="Save figure to this path")
    parser.add_argument("--sanity",  action="store_true",
                        help="Run sanity checks only")
    parser.add_argument("--bands",   action="store_true",
                        help="Plot bands at optimal point")
    args = parser.parse_args()

    if args.sanity:
        sanity_checks(args.Lx, args.Ly)
    else:
        theta_fracs, h_vals, E_grid = grid_search(
            Lx=args.Lx, Ly=args.Ly,
            J1=args.J1, J2=args.J2,
            n_theta=args.n_theta,
            n_h=args.n_h,
            h_max=args.h_max,
            save_path=args.save,
        )
        if args.bands:
            ij_min = np.unravel_index(np.nanargmin(E_grid), E_grid.shape)
            theta_opt = theta_fracs[ij_min[0]] * np.pi
            h_opt = h_vals[ij_min[1]]
            plot_bands(args.Lx, args.Ly, theta_opt, h_opt)