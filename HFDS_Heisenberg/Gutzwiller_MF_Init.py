"""
init_orbitals_gmf — initialise the MF orbital matrix from the
Gutzwiller-projected SF+N (Staggered Flux + Néel) mean-field state.

Reference: Dalla Piazza, PhD thesis (2014), section 2.4.6, eqs 2.4.67–2.4.87.
Optimal parameters from Fig. 2.14: phi=0.10, h_N=0.055 J  (J2=0 Heisenberg).

Interface mirrors init_orbitals_mf (Fermi sea initialisation):
  Input:  L      — linear lattice size (N = L² sites)
          bounds — boundary condition string; 'APBC' uses anti-periodic BC
                   (recommended: avoids Dirac-node degeneracies at phi=pi/4)
          dtype  — jnp.float64 for real HFDS, jnp.complex128 for complex HFDS
                   (SF+N has complex phases so dtype=jnp.complex128 is correct;
                    with dtype=jnp.float64 only the real part is kept — not recommended)
          phi    — flux parameter in [0, 0.5] (theta0/pi in Dalla Piazza notation)
                   Default 0.10 — optimal from Fig. 2.14 for J2=0
          h_N    — Néel field amplitude in units of J
                   Default 0.055 — optimal from Fig. 2.14 for J2=0

  Output: mf array of shape (2N, N), same block structure as init_orbitals_mf:
            mf = [[U_up.T, 0  ],
                  [0,   U_dn.T]].T
          where U_up[:,k] is the k-th lowest eigenvector of H_SF+N(sigma=+1).
          Columns index MF orbitals (0..Nup-1 for up, Nup..N-1 for down).
          Rows index modes (0..N-1 for up-spin sites, N..2N-1 for dn-spin sites).
          Site ordering: site i = (x, y) with i = x*L + y, consistent with
          nk.graph.Grid([L, L]) and with init_orbitals_mf.

Physics:
  H_SF+N = H_SF + H_N  (eqs 2.4.68–2.4.69)

  H_SF — staggered-flux hopping with APBC:
    bulk x-bond  (i→j): t_ij = -exp((-1)^(x+y) · i·phi·π)
    bndry x-bond (wrap): t_ij = +exp(...)   ← sign flip for APBC
    bulk y-bond  (i→j): t_ij = -exp((-1)^(x+y+1) · i·phi·π)
    bndry y-bond (wrap): t_ij = +exp(...)   ← sign flip for APBC

  H_N — staggered Néel field:
    H[i,i] = -h_N · sigma · (+1 if (x+y) even, -1 if (x+y) odd)
    where sigma = +1 for spin-up, -1 for spin-down.
"""

import numpy as np
import jax.numpy as jnp


def init_orbitals_gmf(L, bounds, dtype, phi: float = 0.10, h_N: float = 0.055):
    """
    Return the SF+N Gutzwiller MF orbital matrix for HFDS initialisation.

    Parameters
    ----------
    L      : int    — linear lattice size (square lattice, N = L² sites)
    bounds : str    — 'APBC' (recommended) or 'PBC'
    dtype  :        — output dtype, e.g. jnp.complex128 or jnp.float64
    phi    : float  — flux parameter = theta0/pi ∈ [0, 0.5]  (default: 0.10)
    h_N    : float  — Néel field in units of J               (default: 0.055)

    Returns
    -------
    mf : jnp.ndarray of shape (2N, N) and the requested dtype
    """
    N   = L * L
    Nup = N // 2    # half-filling: N_up = N_dn = N/2

    # ── Site positions  (x*L + y, same as nk.graph.Grid([L,L])) ──────────────
    positions = [(x, y) for x in range(L) for y in range(L)]

    # ── Nearest-neighbour table  (forward bonds only: +x and +y) ─────────────
    def site(x, y):
        return (x % L) * L + (y % L)

    # ── Build single-particle Hamiltonian H_SF+N for one spin species ─────────
    def build_H(sigma: float) -> np.ndarray:
        """
        sigma = +1 for spin-up, -1 for spin-down.
        Returns the N×N complex Hamiltonian matrix.

        Hopping amplitude convention (matches gutzwiller.py / update_orbitals_apbc):
          Each undirected bond (i, j) contributes H[i,j] = t_ij and H[j,i] = t_ij*.
          t_ij = -exp(flux)   for bulk bonds   (APBC: sign unchanged)
          t_ij = +exp(flux)   for boundary bonds (APBC: sign flipped)

        Boundary detection (same logic as gutzwiller.py):
          x-direction: bond (i, ix) is BULK    if |i - ix| == L
                                    BOUNDARY   otherwise   (wrapping gives |i-ix| = L*(L-1))
          y-direction: bond (i, iy) is BULK    if |i - iy| == 1
                                    BOUNDARY   otherwise   (wrapping gives |i-iy| = L-1)
        """
        H = np.zeros((N, N), dtype=complex)

        for i, (x, y) in enumerate(positions):
            stagger = +1 if (x + y) % 2 == 0 else -1

            # Staggered flux phases (eq 2.4.68)
            flux_x = stagger * 1j * phi * np.pi
            flux_y = -stagger * 1j * phi * np.pi

            # x-bond: i → i+x̂  (forward only; H.c. added simultaneously)
            ix = site(x + 1, y)
            if bounds == 'APBC':
                bulk_x = (abs(i - ix) == L)
            else:  # PBC: all bonds treated the same (no sign flip)
                bulk_x = True
            t_x = -np.exp(flux_x) if bulk_x else +np.exp(flux_x)
            H[i, ix] += t_x
            H[ix, i] += t_x.conj()

            # y-bond: i → i+ŷ
            iy = site(x, y + 1)
            if bounds == 'APBC':
                bulk_y = (abs(i - iy) == 1)
            else:
                bulk_y = True
            t_y = -np.exp(flux_y) if bulk_y else +np.exp(flux_y)
            H[i, iy] += t_y
            H[iy, i] += t_y.conj()

            # Néel field (eq 2.4.69): diagonal, spin-dependent
            H[i, i] += -h_N * sigma * stagger

        return H

    # ── Diagonalise and fill lowest Nup eigenstates ───────────────────────────
    def occupied_orbitals(sigma: float) -> np.ndarray:
        """
        Returns U of shape (N, Nup): columns are the Nup lowest eigenvectors
        of H_SF+N(sigma), sorted by ascending eigenvalue.
        """
        H        = build_H(sigma)
        evals, evecs = np.linalg.eigh(H)    # eigh: correct for Hermitian H,
                                             # guarantees orthonormal evecs
                                             # and real, sorted eigenvalues
        return evecs[:, :Nup]               # (N, Nup)

    U_up = occupied_orbitals(+1)   # (N, Nup)
    U_dn = occupied_orbitals(-1)   # (N, Nup)

    # ── Assemble block matrix  (same convention as init_orbitals_mf) ──────────
    # upmatrix = U_up.T  shape (Nup, N):  upmatrix[orbital_k, site_i] = U_up[site_i, k]
    # dnmatrix = U_dn.T  shape (Nup, N)
    # block = [[upmatrix, 0      ],   shape (N, 2N)
    #          [0,        dnmatrix]]
    # .T -> shape (2N, N):
    #   rows 0..N-1   = up-spin site modes
    #   rows N..2N-1  = dn-spin site modes
    #   cols 0..Nup-1      = up-spin MF orbitals
    #   cols Nup..N-1      = dn-spin MF orbitals

    upmatrix = U_up.T                                        # (Nup, N)
    dnmatrix = U_dn.T                                        # (Nup, N)

    mf = np.block([
        [upmatrix,               np.zeros((Nup, N))],
        [np.zeros((Nup, N)),     dnmatrix          ]
    ]).T                                                     # (2N, N)

    return dtype(jnp.array(mf))
