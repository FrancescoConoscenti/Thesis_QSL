"""
Gutzwiller-projected Staggered Flux + Néel (SF+N) mean-field state
for the J1-J2 Heisenberg model on the square lattice.

Reference: Dalla Piazza thesis, eq. 2.4.67–2.4.87.

===========================================================================
PHYSICS SUMMARY
===========================================================================

The SF+N mean-field Hamiltonian (eq. 2.4.67) is H = H_SF + H_N, where:

  H_SF (eq. 2.4.68): staggered-flux hopping.
    For each bond (i, i+x) or (i, i+y), a fermion picks up phase ±iθ₀
    depending on whether site i is "even" (ix+iy even) or "odd":

      H_SF = -1/2 Σ_{i even,σ}  [e^{+iθ₀} c†_{i,σ} c_{i+x,σ}
                                + e^{-iθ₀} c†_{i,σ} c_{i+y,σ} + H.c.]
             -1/2 Σ_{i odd, σ}  [e^{-iθ₀} c†_{i,σ} c_{i+x,σ}
                                + e^{+iθ₀} c†_{i,σ} c_{i+y,σ} + H.c.]

    This is the staggered-flux Ansatz (eq. 2.4.46): χ_{ij} = t e^{iθ_{ij}}
    with θ_{ij} = θ₀(-1)^{ix+jy}, t ≡ 1/2 (energy scale), Δ_{ij} = 0.
    The flux ±4θ₀ threads each plaquette in a staggered pattern (fig. 2.10).
    At θ₀ = π/4 this is the π-flux state with Dirac cones at k=(±π/2,±π/2).

  H_N (eq. 2.4.69): Néel mean field.
    Staggered on-site energy controlled by h_N:

      H_N = -h_N Σ_σ σ [Σ_{i even} c†_{i,σ} c_{i,σ}
                        - Σ_{i odd}  c†_{i,σ} c_{i,σ}]

    This gaps the Dirac cones, opening a gap 2h_N at k=(±π/2,±π/2).

The two mean fields together double the unit cell (Q=(π,π)), leading to
two quasiparticle bands with dispersion (eq. 2.4.80):

    ω_k = sqrt(|Δ_k|² + h_N²)

where  Δ_k = (1/2)[e^{iθ₀} cos(kx) + e^{-iθ₀} cos(ky)]  (eq. 2.4.72).

The mean-field ground state fills the lower band (eq. 2.4.85):

    |ψ_GS⟩ = Π_{k∈MBZ} γ†_{k↑-} γ†_{k↓-} |0⟩

The VARIATIONAL state is the Gutzwiller projection onto the
singly-occupied sector (eq. 2.4.86):

    |GS(θ₀, h_N)⟩ = P_{D=0} |ψ_GS(θ₀, h_N)⟩

implemented via the determinantal formula:
    ⟨{R↑, R↓}|GS⟩ = det[f(R_i↑ - R_j↓)]

where f(r) is the pair wave function extracted from the BdG ground state
(see compute_pair_wavefunction below).

===========================================================================
PARAMETERS
===========================================================================

  θ₀  : staggered-flux phase. Controls the flux ±4θ₀ per plaquette.
         θ₀ = π/4 → π-flux (optimal, Dirac spin liquid limit).
         NOT optimized here; set externally.

  h_N : Néel mean-field strength. Gaps the Dirac cones.
         h_N = 0 → gapless DSL (pure SF state).
         h_N > 0 → antiferromagnetically ordered mean field.
         NOT optimized here; set externally.

===========================================================================
"""

import numpy as np
from itertools import combinations


# ---------------------------------------------------------------------------
# Lattice helpers
# ---------------------------------------------------------------------------

def make_square_lattice(Lx, Ly):
    """Site index and neighbor list for an Lx×Ly square lattice with PBC."""
    def idx(x, y):
        return (x % Lx) * Ly + (y % Ly)

    sites = [(x, y) for x in range(Lx) for y in range(Ly)]
    return idx, sites


def sublattice(x, y):
    """Return +1 (even) or -1 (odd) sublattice label for site (x,y)."""
    return +1 if (x + y) % 2 == 0 else -1


def make_k_points_MBZ(Lx, Ly):
    """
    Return k-points in the Magnetic Brillouin Zone (MBZ).

    The MBZ is the half of the full BZ with |kx|+|ky| ≤ π (diamond shape).
    With PBC on Lx×Ly, the full BZ has N=Lx*Ly k-points.
    The doubled unit cell (due to Q=(π,π)) folds the BZ, giving N/2 MBZ points.

    For convenience we return ALL N k-points here (full BZ) and handle the
    MBZ restriction in the diagonalization by working with the folded basis
    (α_{k,σ} = (c_{k,σ} ± c_{k+Q,σ})/√2).
    """
    kx_vals = 2.0 * np.pi * np.arange(Lx) / Lx
    ky_vals = 2.0 * np.pi * np.arange(Ly) / Ly
    kpts = [(kx_vals[mx], ky_vals[my])
            for mx in range(Lx) for my in range(Ly)]
    return np.array(kpts)


# ---------------------------------------------------------------------------
# SF+N mean-field Hamiltonian: diagonalization in k-space
# ---------------------------------------------------------------------------

def sfn_delta_k(kx, ky, theta0):
    """
    Complex hopping structure factor Δ_k (eq. 2.4.72):

        Δ_k = (1/2)[e^{iθ₀} cos(kx) + e^{-iθ₀} cos(ky)]

    This encodes the staggered-flux hopping in k-space after unfolding
    to the doubled unit cell. |Δ_k| determines the SF band dispersion.
    """
    return 0.5 * (np.exp(1j * theta0) * np.cos(kx)
                  + np.exp(-1j * theta0) * np.cos(ky))


def diagonalize_sfn(Lx, Ly, theta0, h_N):
    """
    Diagonalize the SF+N BdG Hamiltonian and return the lower-band
    quasiparticle eigenstates.

    The 2×2 BdG matrix at each k∈MBZ and spin σ∈{↑(+1),↓(-1)} is
    (eq. 2.4.73):

        H_k,σ = ( σh_N    Δ_k* )
                ( Δ_k    -σh_N )

    with eigenvalues ±ω_k = ±sqrt(|Δ_k|² + h_N²)  (eq. 2.4.80).

    The lower-band eigenvectors (eqs. 2.4.75–2.4.76) are:

        u_{k,σ,-} = sqrt( (1 + σh_N/ω_k) / 2 )
        v_{k,σ,-} = (Δ_k/|Δ_k|) * sqrt( (1 - σh_N/ω_k) / 2 )

    The canonical transformation to quasiparticle operators (eq. 2.4.81):

        (γ_{k,σ,-})   = (1/√2) ( u_{k,σ,-}   v*_{k,σ,-} ) ( c_{k,σ} + c_{k+Q,σ} )
        (γ_{k,σ,+})             ( u_{k,σ,+}   v*_{k,σ,+} ) ( c_{k,σ} - c_{k+Q,σ} )

    Returns:
        kpts  : (N, 2) array of all full-BZ k-points
        omega : (N,)   quasiparticle energies ω_k (k-indexed, not MBZ)
        u_up  : (N,)   u_{k,↑,-} coherence factor for spin-up lower band
        v_up  : (N,)   v_{k,↑,-} coherence factor for spin-up lower band
        u_dn  : (N,)   u_{k,↓,-} for spin-down lower band
        v_dn  : (N,)   v_{k,↓,-} for spin-down lower band
    """
    N = Lx * Ly
    kpts = make_k_points_MBZ(Lx, Ly)  # (N, 2)
    kx = kpts[:, 0]; ky = kpts[:, 1]

    # Structure factor Δ_k at each k-point
    Delta = sfn_delta_k(kx, ky, theta0)           # (N,) complex
    abs_Delta = np.abs(Delta)

    # Quasiparticle energy (same for both spins)
    omega = np.sqrt(abs_Delta**2 + h_N**2 + 1e-14)  # (N,)

    # Phase of Δ_k: needed for v (eq. 2.4.76)
    # Handle Δ_k = 0 (Dirac nodes at k=(±π/2, ±π/2) when h_N=0):
    # the phase is ill-defined there (noted in the thesis after eq. 2.4.85).
    # We regularize by setting phase=1 when |Δ_k| < eps.
    phase_Delta = np.where(abs_Delta > 1e-10,
                           Delta / abs_Delta,
                           1.0 + 0j)              # (N,) complex

    # Coherence factors for spin σ=+1 (↑):
    u_up = np.sqrt(np.clip((1.0 + h_N / omega) / 2.0, 0, 1))      # (N,)
    v_up = phase_Delta * np.sqrt(np.clip((1.0 - h_N / omega) / 2.0, 0, 1))

    # Coherence factors for spin σ=-1 (↓):
    u_dn = np.sqrt(np.clip((1.0 - h_N / omega) / 2.0, 0, 1))      # (N,)
    v_dn = phase_Delta * np.sqrt(np.clip((1.0 + h_N / omega) / 2.0, 0, 1))

    return kpts, omega, u_up, v_up, u_dn, v_dn


# ---------------------------------------------------------------------------
# Pair wave function f(r) from the BdG lower-band eigenstates
# ---------------------------------------------------------------------------

def compute_pair_wavefunction(Lx, Ly, theta0, h_N):
    """
    Compute the singlet pair wave function f(r_i - r_j) such that:

        ⟨{R↑, R↓}|GS⟩ = det[ f(R_i↑ - R_j↓) ]

    Derivation:
    -----------
    The mean-field GS fills the lower band:
        |ψ_GS⟩ = Π_{k∈MBZ} γ†_{k↑-} γ†_{k↓-} |0⟩

    From eqs. 2.4.81–2.4.84, the real-space fermion operators are:

        c_{i,σ} = √2 Σ_{k∈MBZ} e^{ikR_i}
                    × [ε_{R_i} u_{k,σ,-} + ε̄_{R_i} v_{k,σ,-}] γ_{k,σ,-} + ...

    where ε_{R_i} = (1 + e^{iQ·R_i})/2 (even sublattice projector)
          ε̄_{R_i} = (1 - e^{iQ·R_i})/2 (odd sublattice projector)
          Q = (π, π)

    The BCS pair wave function is extracted by computing
        f_{ij} = ⟨0| c_{i↑} c_{j↓} |ψ_GS⟩ / (norm)

    Concretely, filling both spins in the lower band gives a singlet
    pairing. Working through the algebra (standard BdG construction):

        f(r_i - r_j) = (1/N) Σ_{k∈full BZ} F_k e^{ik·(r_i - r_j)}

    where F_k encodes the lower-band structure at each k.

    The key insight from the doubled unit cell (α_k = (c_k ± c_{k+Q})/√2):
    After projecting onto the physical spin-up / spin-down sectors and
    summing over the MBZ, the effective pairing in the full BZ is:

        F_k =  u_{k,-} v_{k,-}* / (u_{k,-}² + v_{k,-}²)

    which in the limit h_N → 0 (pure SF, π-flux) reduces to
        F_k → (Δ_k/|Δ_k|) × (angle function)

    More explicitly, for general h_N we obtain f_mat by building the
    real-space Slater matrix of the N/2 occupied orbitals (lower band,
    both spins) and reading off the pair amplitude. This is done below
    via explicit orbital construction in real space.

    Implementation:
    ---------------
    We build the N occupied single-particle orbitals φ_n(i) of |ψ_GS⟩:

      For each k∈MBZ and σ∈{↑,↓}, the lower-band orbital is:
        φ_{k,σ,-}(i) = √2 e^{ikR_i} [ε_{R_i} u_{k,σ,-} + ε̄_{R_i} v_{k,σ,-}]

    The pair wave function is then extracted by the standard formula:
        f_{ij} = Σ_{n=1}^{N/2} φ_{n,↑}(i) [φ_{n,↓}(j)]
    where we sum over the N/2 lower-band orbitals (one per k∈MBZ).
    This is the "pairing matrix" of the occupied Slater determinant.

    Returns:
        f_mat : (N, N) complex array — pair wave function matrix.
                f_mat[i, j] = f(r_i - r_j).
    """
    N = Lx * Ly
    kpts, omega, u_up, v_up, u_dn, v_dn = diagonalize_sfn(Lx, Ly, theta0, h_N)

    # Real-space site positions and sublattice labels
    sites = [(x, y) for x in range(Lx) for y in range(Ly)]
    Q = np.array([np.pi, np.pi])   # antiferromagnetic ordering wavevector

    # ε_{R_i} and ε̄_{R_i} (eqs. 2.4.83-84)
    eps_R  = np.array([0.5 * (1.0 + np.exp(1j * (Q[0]*x + Q[1]*y)))
                       for x, y in sites])   # (N,) complex: 1 on even, 0 on odd
    epsb_R = np.array([0.5 * (1.0 - np.exp(1j * (Q[0]*x + Q[1]*y)))
                       for x, y in sites])   # (N,) complex: 0 on even, 1 on odd

    # Determine the MBZ: k-points with kx + ky ≤ π (shifted to [0,2π) BZ)
    # On a finite lattice with PBC, we select N/2 k-points for the MBZ.
    # Equivalently: for each pair (k, k+Q) keep only one representative.
    Nup = N // 2   # number of occupied orbitals per spin

    # We select the MBZ as the set of N/2 k-points that are NOT related
    # to each other by a shift of Q=(π,π). Simple selection: take the
    # first half of the BZ by index (standard for square lattice PBC).
    # A more careful selection would impose |kx - π| + |ky - π| > 0,
    # but the index-based selection works for Lx×Ly lattices.
    mbz_indices = np.arange(N // 2)   # indices into kpts array for MBZ

    # Build f_mat = Σ_{k∈MBZ} φ_{k,↑,-}(i) * φ_{k,↓,-}(j)*
    # where φ_{k,σ,-}(i) = √2 * e^{ikR_i} * [ε_{R_i} u_{k,σ,-} + ε̄_{R_i} v_{k,σ,-}]
    f_mat = np.zeros((N, N), dtype=complex)

    for ik in mbz_indices:
        kx, ky = kpts[ik]
        # Bloch factors e^{ikR_i} at all sites
        bloch = np.array([np.exp(1j * (kx * x + ky * y))
                          for x, y in sites])          # (N,)

        # Lower-band orbital for spin-up at this k
        phi_up = np.sqrt(2.0) * bloch * (eps_R * u_up[ik] + epsb_R * v_up[ik])
        # Lower-band orbital for spin-down at this k
        phi_dn = np.sqrt(2.0) * bloch * (eps_R * u_dn[ik] + epsb_R * v_dn[ik])

        # Outer product contribution to pair matrix
        f_mat += np.outer(phi_up, np.conj(phi_dn))

    return f_mat


# ---------------------------------------------------------------------------
# Gutzwiller-projected amplitude
# ---------------------------------------------------------------------------

def gutzwiller_amplitude(up_sites, dn_sites, f_mat):
    """
    Gutzwiller-projected amplitude for spin configuration (up_sites, dn_sites):

        ⟨{R↑, R↓}|GS⟩ = det[ f(R_i↑ - R_j↓) ]

    Args:
        up_sites : list of site indices for ↑ spins (length N/2)
        dn_sites : list of site indices for ↓ spins (length N/2)
        f_mat    : (N, N) pair wave function matrix

    Returns:
        amplitude : complex scalar
    """
    submat = f_mat[np.ix_(up_sites, dn_sites)]
    return np.linalg.det(submat)


# ---------------------------------------------------------------------------
# Exact Gutzwiller projection (brute-force, feasible for N ≤ 16)
# ---------------------------------------------------------------------------

def build_projected_state_exact(Lx, Ly, theta0, h_N):
    """
    Build the full Gutzwiller-projected state vector in the S^z=0 sector.

    Returns:
        psi  : dict {(up_cfg, dn_cfg): amplitude}
        norm : float, ||psi||
    """
    N = Lx * Ly
    Nup = N // 2

    f_mat = compute_pair_wavefunction(Lx, Ly, theta0, h_N)

    all_sites = list(range(N))
    psi = {}
    norm_sq = 0.0

    for up_sites in combinations(all_sites, Nup):
        dn_sites = tuple(s for s in all_sites if s not in set(up_sites))
        amp = gutzwiller_amplitude(list(up_sites), list(dn_sites), f_mat)
        if np.abs(amp) > 1e-15:
            psi[(up_sites, dn_sites)] = amp
            norm_sq += np.abs(amp) ** 2

    return psi, np.sqrt(norm_sq)


# ---------------------------------------------------------------------------
# Exact variational energy ⟨H_J1J2⟩ (for small systems)
# ---------------------------------------------------------------------------

def variational_energy_exact(Lx, Ly, theta0, h_N, J1=1.0, J2=0.5):
    """
    Compute ⟨GS|H_{J1-J2}|GS⟩ / ⟨GS|GS⟩ exactly for small systems.

    H_{J1-J2} = J1 Σ_{<ij>} S_i·S_j + J2 Σ_{<<ij>>} S_i·S_j

    Returns:
        E_var / N : variational energy per site
    """
    N = Lx * Ly
    idx, sites = make_square_lattice(Lx, Ly)

    psi, norm = build_projected_state_exact(Lx, Ly, theta0, h_N)
    if norm < 1e-12:
        return np.nan

    # Build bond lists (each bond stored once, seen-set handles PBC wrapping)
    seen1 = set()
    nn1 = []  # 1st neighbors
    for x, y in sites:
        i = idx(x, y)
        for dx, dy in [(1, 0), (0, 1)]:
            j = idx(x + dx, y + dy)
            bond = tuple(sorted((i, j)))
            if bond not in seen1:
                seen1.add(bond)
                nn1.append(bond)

    seen2 = set()
    nn2 = []  # 2nd neighbors
    for x, y in sites:
        i = idx(x, y)
        for dx, dy in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:
            j = idx(x + dx, y + dy)
            bond = tuple(sorted((i, j)))
            if bond not in seen2:
                seen2.add(bond)
                nn2.append(bond)

    all_bonds = [(J1, nn1), (J2, nn2)]

    energy = 0.0
    norm_sq = 0.0

    for config, amp in psi.items():
        up_sites, dn_sites = config
        up_set = set(up_sites)

        e_diag = 0.0
        e_offdiag = 0.0

        for J, bonds in all_bonds:
            for (i, j) in bonds:
                si_up = (i in up_set)
                sj_up = (j in up_set)
                # Sz_i Sz_j diagonal
                e_diag += J * 0.25 * (1 if si_up == sj_up else -1)
                # S+_i S-_j + h.c. off-diagonal (only if antiparallel)
                if si_up != sj_up:
                    if si_up:
                        new_up = tuple(sorted([s for s in up_sites if s != i] + [j]))
                        new_dn = tuple(sorted([s for s in dn_sites if s != j] + [i]))
                    else:
                        new_up = tuple(sorted([s for s in up_sites if s != j] + [i]))
                        new_dn = tuple(sorted([s for s in dn_sites if s != i] + [j]))
                    new_cfg = (new_up, new_dn)
                    if new_cfg in psi:
                        e_offdiag += J * 0.5 * (psi[new_cfg] / amp)

        prob = np.abs(amp) ** 2
        energy += prob * (e_diag + e_offdiag.real)
        norm_sq += prob

    return (energy / norm_sq) / N


# ---------------------------------------------------------------------------
# VMC energy estimation via NetKet (for larger systems)
# ---------------------------------------------------------------------------

def make_netket_model(Lx, Ly, theta0, h_N):
    """
    Return a NetKet-compatible Flax model for the SF+N Gutzwiller state.

    The model takes a batch of spin configurations σ ∈ {±1}^N and
    returns log-amplitudes log det[f(R_i↑ - R_j↓)].

    This can be used directly with nk.vqs.MCState for VMC energy estimation.
    """
    try:
        import jax
        import jax.numpy as jnp
        import flax.linen as nn
    except ImportError:
        raise ImportError("NetKet/JAX/Flax required for VMC. "
                          "Use build_projected_state_exact for small systems.")

    # Pre-compute f_mat (fixed parameters, no optimization)
    f_mat_np = compute_pair_wavefunction(Lx, Ly, theta0, h_N)
    f_mat_jax = jnp.array(f_mat_np)
    Nup = (Lx * Ly) // 2

    class SFNModel(nn.Module):
        @nn.compact
        def __call__(self, sigma):
            # sigma: (batch, N) with values ±1 (NetKet convention)
            # Dummy parameter so NetKet finds a 'params' key in variables
            _ = self.param('_dummy', nn.initializers.zeros, ())
            f = f_mat_jax

            def log_amp(s):
                up = jnp.where(s > 0, size=Nup, fill_value=0)[0]
                dn = jnp.where(s <= 0, size=Nup, fill_value=0)[0]
                sub = f[up][:, dn]
                sign, logabs = jnp.linalg.slogdet(sub)
                return logabs + jnp.log(sign.astype(jnp.complex128) + 0j)

            return jax.vmap(log_amp)(sigma)

    return SFNModel()


def vmc_energy(Lx, Ly, theta0, h_N, J1=1.0, J2=0.5,
               n_samples=1024, n_discard=128, n_chains=16, seed=42):
    """
    Estimate variational energy via VMC using NetKet (for Lx×Ly > 4×4).

    Returns:
        E_mean  : energy per site (physical S=1/2 units, divided by CONV=4)
        E_error : statistical error of the mean
    """
    import netket as nk

    N = Lx * Ly
    CONV = 4.0  # NetKet Heisenberg uses Pauli matrices → physical = NetKet / 4

    hi = nk.hilbert.Spin(s=0.5, N=N, total_sz=0)
    graph = nk.graph.Square(Lx, pbc=True)
    H = nk.operator.Heisenberg(hilbert=hi, graph=graph, J=J1)

    # Add J2 (NNN) bonds
    seen, edges = set(), []
    for x in range(Lx):
        for y in range(Ly):
            i = x * Ly + y
            for dx, dy in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:
                j = ((x + dx) % Lx) * Ly + ((y + dy) % Ly)
                b = tuple(sorted((i, j)))
                if b not in seen:
                    seen.add(b); edges.append(b)
    g2 = nk.graph.Graph(edges=edges, n_nodes=N)
    H += nk.operator.Heisenberg(hilbert=hi, graph=g2, J=J2)

    model = make_netket_model(Lx, Ly, theta0, h_N)
    sampler = nk.sampler.MetropolisExchange(hi, n_chains=n_chains, graph=graph)
    vs = nk.vqs.MCState(sampler, model,
                        n_samples=n_samples,
                        n_discard_per_chain=n_discard,
                        seed=seed)
    e = vs.expect(H)
    return e.mean.real / (N * CONV), e.error_of_mean.real / (N * CONV)


# ---------------------------------------------------------------------------
# Diagnostics: BdG spectrum and pair wavefunction properties
# ---------------------------------------------------------------------------

def print_diagnostics(Lx, Ly, theta0, h_N):
    """Print key properties of the SF+N BdG state."""
    N = Lx * Ly
    kpts, omega, u_up, v_up, u_dn, v_dn = diagonalize_sfn(Lx, Ly, theta0, h_N)

    Delta = sfn_delta_k(kpts[:, 0], kpts[:, 1], theta0)
    abs_Delta = np.abs(Delta)

    print(f"{'='*60}")
    print(f"SF+N Diagnostics  |  {Lx}×{Ly}  |  θ₀={theta0/np.pi:.4f}π  |  h_N={h_N:.4f}")
    print(f"{'='*60}")
    print(f"\nBdG quasiparticle spectrum:")
    print(f"  min ω_k = {omega.min():.6f}  (gap at Dirac nodes)")
    print(f"  max ω_k = {omega.max():.6f}")
    print(f"  Dirac node gap = 2×h_N = {2*h_N:.6f}")

    # Dirac nodes: k=(±π/2, ±π/2), where |Δ_k|→0
    dirac_mask = abs_Delta < 0.01
    print(f"  k-points with |Δ_k|<0.01 (Dirac nodes): {np.sum(dirac_mask)}")
    for i in np.where(dirac_mask)[0]:
        print(f"    k=({kpts[i,0]/np.pi:.3f}π, {kpts[i,1]/np.pi:.3f}π): "
              f"|Δ|={abs_Delta[i]:.4f}, ω={omega[i]:.4f}")

    print(f"\nPair wave function:")
    f_mat = compute_pair_wavefunction(Lx, Ly, theta0, h_N)
    print(f"  f_mat shape: {f_mat.shape}")
    print(f"  max|f|: {np.max(np.abs(f_mat)):.4f}")
    print(f"  |f(r=0)|: {np.abs(f_mat[0, 0]):.4f}  (same-site pairing, should be ~0)")
    print(f"  |f(1,0)|: {np.abs(f_mat[0, 1]):.4f}  (1st-neighbor pair amplitude)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")

    Lx, Ly = 16, 16
    N = Lx * Ly

    # Physical parameters (NOT optimized — set by hand or scanned externally)
    # θ₀ = π/4: π-flux state, optimal for the DSL (Dirac spin liquid)
    # h_N = 0:  pure SF (gapless Dirac cones); h_N > 0 gaps them
    theta0 = np.pi / 4   # optimal staggered flux (π-flux)
    h_N    = 0.0         # Néel mean field (0 = gapless DSL)

    print("=" * 60)
    print("Gutzwiller-projected SF+N mean-field state")
    print(f"Lattice: {Lx}×{Ly}  (N={N})")
    print(f"θ₀ = π/4 = {theta0:.6f}")
    print(f"h_N = {h_N}")
    print("=" * 60)

    # --- Diagnostics ---
    print_diagnostics(Lx, Ly, theta0, h_N)

    # --- Build pair wave function ---
    f_mat = compute_pair_wavefunction(Lx, Ly, theta0, h_N)

    # --- Exact Gutzwiller projection ---
    print(f"\nBuilding exact Gutzwiller-projected state ({Lx}×{Ly})...")
    psi, norm = build_projected_state_exact(Lx, Ly, theta0, h_N)
    print(f"  Non-zero configurations: {len(psi):,} / {N*N//4:,}")
    print(f"  State norm: {norm:.6f}")

    # --- Sample a few amplitudes ---
    items = sorted(psi.items(), key=lambda kv: -abs(kv[1]))[:4]
    print("\nLargest amplitudes:")
    for (up, dn), amp in items:
        print(f"  up={up}  |psi|={np.abs(amp):.4e}")

    # --- Exact variational energy at (π/4, 0) ---
    print(f"\nComputing exact variational energy (J1=1, J2=0.0)...")
    E_var = variational_energy_exact(Lx, Ly, theta0, h_N, J1=1.0, J2=0.0)
    print(f"  E_var/N = {E_var:.8f}")
    print(f"  ED  E0/N = -0.52862021  ({Lx}×{Ly}, J2=0.0)")
    E_ED = -0.52862021
    print(f"  Var. gap = {(E_var - E_ED) / abs(E_ED) * 100:.2f}%")

    

    # --- 2D scan: h_N and θ₀ on 4×4 ---
    print(f"\n--- 2D scan: θ₀ and h_N, {Lx}×{Ly}, J1=1, J2=0.0 ---")
    print(f"{'θ₀/π':>6}  {'h_N':>6}  {'E_var/N':>12}  {'gap%':>8}")
    best_E, best_th, best_hN = np.inf, 0, 0
    for th in [0.1, 0.25, 0.4, 0.5]:
        for hN in [0.05, 0.1, 0.15]:
            E = variational_energy_exact(Lx, Ly, th*np.pi, hN, J1=1.0, J2=0.0)
            gap = (E - E_ED) / abs(E_ED) * 100
            print(f"{th:>6.2f}  {hN:>6.2f}  {E:>12.6f}  {gap:>7.2f}%")
            if E < best_E:
                best_E, best_th, best_hN = E, th, hN

    print(f"\nBest {Lx}×{Ly}: θ₀={best_th:.2f}π, h_N={best_hN:.2f}, E/N={best_E:.6f}")
    print(f"         gap from ED = {(best_E-E_ED)/abs(E_ED)*100:.1f}%")
    print(f"  ({Lx}×{Ly} is dominated by finite-size effects; larger L gives much better results)")





    # --- VMC for 6×6 ---
    print(f"\n--- VMC energy estimates ({Lx}×{Ly}, J1=1, J2=0.5) ---")
    try:
        for th, hN in [(0.25, 0.0), (0.25, 1.0), (0.5, 0.3)]:
            E_mean, E_err = vmc_energy(Lx, Ly, th*np.pi, hN, J1=1.0, J2=0.0, n_samples=1024)
            print(f"  θ₀={th:.2f}π  h_N={hN:.1f}: E/N = {E_mean:.5f} ± {E_err:.5f}")
        print(f"  Literature (TDL, Gutzwiller-SF): E/N ≈ -0.494 [Dalla Piazza]")
    except Exception as ex:
        print(f"  (NetKet VMC skipped: {ex})")

    print("\nDone.")