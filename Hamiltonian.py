import netket as nk
import numpy as np
from itertools import product

def build_heisenberg_apbc(
    Lx: int,
    Ly: int,
    J1: float = 1.0,
    J2: float = 0.0,
    apbc_x: bool = True,
    apbc_y: bool = False,
) -> nk.operator.LocalOperator:
    """
    Build the J1-J2 Heisenberg Hamiltonian on an Lx×Ly square lattice
    with (anti-)periodic boundary conditions.

    APBC inserts a π-flux: the XX+YY exchange on boundary bonds crossing
    the anti-periodic direction gets a factor of -1.  The ZZ term is
    unaffected (diagonal, sign-insensitive to boundary twist).

    Args:
        Lx, Ly   : lattice dimensions
        J1       : nearest-neighbour exchange (>0 = AFM)
        J2       : next-nearest-neighbour exchange
        apbc_x   : anti-periodic in x-direction
        apbc_y   : anti-periodic in y-direction

    Returns:
        nk.operator.LocalOperator
    """
    N = Lx * Ly
    hi = nk.hilbert.Spin(s=0.5, N=N, total_sz=0)  # Spin-1/2 Hilbert space with zero total magnetization

    def site(x, y):
        return x % Lx + (y % Ly) * Lx

    # Pauli matrices (spin-1/2 operators: S = σ/2)
    Sp = np.array([[0, 1], [0, 0]], dtype=complex)   # S+
    Sm = np.array([[0, 0], [1, 0]], dtype=complex)   # S-
    Sz = np.array([[0.5, 0], [0, -0.5]], dtype=complex)

    # Heisenberg bond: J*(0.5*(S+_i S-_j + S-_i S+_j) + Sz_i Sz_j)
    # with an optional sign on the XX+YY part for APBC
    def heisenberg_bond(i, j, sign_xy: float = 1.0):
        """
        sign_xy = +1 for PBC bonds, -1 for APBC boundary bonds.
        Only the spin-flip part (S+S- + S-S+) changes sign.
        """
        op_list = []
        # ZZ term (sign-independent)
        op_list.append((np.kron(Sz, Sz), [i, j]))
        # XY terms (sign flips under APBC)
        op_list.append((sign_xy * 0.5 * np.kron(Sp, Sm), [i, j]))
        op_list.append((sign_xy * 0.5 * np.kron(Sm, Sp), [i, j]))
        return op_list

    H = nk.operator.LocalOperator(hi, dtype=complex)

    # ------------------------------------------------------------------ #
    # Nearest-neighbour bonds
    # ------------------------------------------------------------------ #
    for x, y in product(range(Lx), range(Ly)):
        i = site(x, y)

        # x-direction bond (i) -- (i+1)
        x_next = x + 1
        sign = -1.0 if (apbc_x and x == Lx - 1) else 1.0
        j = site(x_next, y)
        for op, sites in heisenberg_bond(i, j, sign_xy=sign):
            H += J1 * nk.operator.LocalOperator(hi, [op], [sites])

        # y-direction bond (i) -- (i+Lx)
        y_next = y + 1
        sign = -1.0 if (apbc_y and y == Ly - 1) else 1.0
        j = site(x, y_next)
        for op, sites in heisenberg_bond(i, j, sign_xy=sign):
            H += J1 * nk.operator.LocalOperator(hi, [op], [sites])

    # ------------------------------------------------------------------ #
    # Next-nearest-neighbour bonds (J2 term)
    # NNN bonds cross a boundary if BOTH x and y coordinates wrap.
    # ------------------------------------------------------------------ #
    if J2 != 0.0:
        for x, y in product(range(Lx), range(Ly)):
            i = site(x, y)
            for dx, dy in [(1, 1), (1, -1)]:
                x2, y2 = x + dx, y + dy
                # Boundary crossing in x and/or y
                cross_x = apbc_x and (x2 < 0 or x2 >= Lx)
                cross_y = apbc_y and (y2 < 0 or y2 >= Ly)
                # Each APBC crossing contributes a sign flip
                sign = (-1.0) ** (int(cross_x) + int(cross_y))
                j = site(x2, y2)
                for op, sites in heisenberg_bond(i, j, sign_xy=sign):
                    H += J2 * nk.operator.LocalOperator(hi, [op], [sites])

    return 4*H

def build_heisenberg_twisted(
    Lx: int,
    Ly: int,
    J1: float = 1.0,
    J2: float = 0.0,
    phi: float = 0.0,       # twist angle in units of pi; phi=1 reproduces APBC
    apbc_y: bool = False,
) -> nk.operator.LocalOperator:
    """
    Build the J1-J2 Heisenberg Hamiltonian on an Lx×Ly square lattice
    with a U(1) twist φ at the x-boundary and (anti-)periodic BC in y.

    The twist is implemented via the Peierls substitution on the spin-flip
    operators at the boundary bond (x = Lx-1) → (x = 0):

        S+_i S-_j  →  e^{+iφ} S+_i S-_j
        S-_i S+_j  →  e^{-iφ} S-_i S+_j

    Special cases:
        φ = 0   → PBC  (no twist)
        φ = 1   → APBC (π-flux, equivalent to apbc_x=True)

    The ZZ term is unaffected by the boundary twist.
    NNN bonds that cross the x-boundary pick up the same phase factor.
    NNN bonds crossing both x and y boundaries accumulate both phases.

    Args:
        Lx, Ly   : lattice dimensions
        J1       : nearest-neighbour exchange (>0 = AFM)
        J2       : next-nearest-neighbour exchange
        phi      : boundary twist angle in x-direction (radians)
        apbc_y   : anti-periodic (π-twist) in y-direction

    Returns:
        nk.operator.LocalOperator
    """
    N = Lx * Ly
    hi = nk.hilbert.Spin(s=0.5, N=N, total_sz=0)

    def site(x, y):
        return (x % Lx) + (y % Ly) * Lx

    Sp = np.array([[0, 1], [0, 0]], dtype=complex)
    Sm = np.array([[0, 0], [1, 0]], dtype=complex)
    Sz = np.array([[0.5, 0], [0, -0.5]], dtype=complex)

    def heisenberg_bond(i, j, phase: complex = 1.0):
        """
        phase = e^{iφ} for the S+_i S-_j term.
        The S-_i S+_j term gets the conjugate phase e^{-iφ}.
        ZZ term is always real and phase-independent.

        For PBC:   phase = +1
        For APBC:  phase = e^{iπ} = -1
        For twist: phase = e^{iφ}
        """
        op_list = []
        op_list.append((np.kron(Sz, Sz),                           [i, j]))
        op_list.append((0.5 * phase      * np.kron(Sp, Sm),        [i, j]))
        op_list.append((0.5 * np.conj(phase) * np.kron(Sm, Sp),   [i, j]))
        return op_list

    H = nk.operator.LocalOperator(hi, dtype=complex)

    phi_y = np.pi if apbc_y else 0.0   # y-boundary phase (0 or π)

    # ------------------------------------------------------------------ #
    # Nearest-neighbour bonds
    # ------------------------------------------------------------------ #
    for x, y in product(range(Lx), range(Ly)):
        i = site(x, y)

        # --- x-bond: (x,y) → (x+1, y) ---
        crosses_x = (x == Lx - 1)
        phase_x = np.exp(1j * phi * np.pi) if crosses_x else 1.0
        j = site(x + 1, y)
        for op, sites in heisenberg_bond(i, j, phase=phase_x):
            H += J1 * nk.operator.LocalOperator(hi, [op], [sites])

        # --- y-bond: (x,y) → (x, y+1) ---
        crosses_y = (y == Ly - 1)
        phase_y = np.exp(1j * phi_y) if crosses_y else 1.0
        j = site(x, y + 1)
        for op, sites in heisenberg_bond(i, j, phase=phase_y):
            H += J1 * nk.operator.LocalOperator(hi, [op], [sites])

    # ------------------------------------------------------------------ #
    # Next-nearest-neighbour bonds
    # ------------------------------------------------------------------ #
    if J2 != 0.0:
        for x, y in product(range(Lx), range(Ly)):
            i = site(x, y)
            for dx, dy in [(1, 1), (1, -1)]:
                x2, y2 = x + dx, y + dy

                # Accumulate phases from each boundary crossed
                phase = 1.0 + 0j
                if x2 < 0 or x2 >= Lx:        # crosses x-boundary
                    phase *= np.exp(1j * phi * np.pi)
                if y2 < 0 or y2 >= Ly:        # crosses y-boundary
                    phase *= np.exp(1j * phi_y)

                j = site(x2, y2)
                for op, sites in heisenberg_bond(i, j, phase=phase):
                    H += J2 * nk.operator.LocalOperator(hi, [op], [sites])

    return 4 * H


# ------------------------------------------------------------------ #
# Usage example: 4×4, J1=1, J2=0.5 (frustrated), APBC in x
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    Lx, Ly = 4, 4
    J2 = 0.5

    H_pbc  = build_heisenberg_apbc(Lx, Ly, J1=1.0, J2=J2, apbc_x=False, apbc_y=False)
    H_apbc = build_heisenberg_apbc(Lx, Ly, J1=1.0, J2=J2, apbc_x=True,  apbc_y=True)
    H_apbc_x = build_heisenberg_apbc(Lx, Ly, J1=1.0, J2=J2, apbc_x=True,  apbc_y=False)
    H_apbc_y = build_heisenberg_apbc(Lx, Ly, J1=1.0, J2=J2, apbc_x=False, apbc_y=True)

    #ha = nk.operator.Heisenberg(hilbert=hi, graph=lattice, J=[1.0, J2], sign_rule=[False, False]).to_jax_operator() # No Marshall sign rule"""

    # Quick exact diagonalisation check
    import scipy.sparse.linalg as spla

    E_pbc, gs_pbc = spla.eigsh(H_pbc.to_sparse(),  k=1, which="SA")
    E_apbc, gs_apbc = spla.eigsh(H_apbc.to_sparse(), k=1, which="SA")
    E_apbc_x, gs_apbc_x = spla.eigsh(H_apbc_x.to_sparse(), k=1, which="SA")
    E_apbc_y, gs_apbc_y = spla.eigsh(H_apbc_y.to_sparse(), k=1, which="SA")


    print(f"4×4 J1-J2 (J2={J2})")
    print(f"  E0/N  PBC  = {E_pbc[0]  / (4*Lx*Ly):.6f}")
    print(f"  E0/N  APBC = {E_apbc[0] / (4*Lx*Ly):.6f}")
    print(f"  E0/N  APBC_X = {E_apbc_x[0] / (4*Lx*Ly):.6f}")
    print(f"  E0/N  APBC_Y = {E_apbc_y[0] / (4*Lx*Ly):.6f}")
    
    print("overlap between PBC and APBC ground states:", np.abs(np.dot(gs_pbc.squeeze(1).conj(), gs_apbc.squeeze(1))))
    print("overlap between PBC and APBC_X ground states:", np.abs(np.dot(gs_pbc.squeeze(1).conj(), gs_apbc_x.squeeze(1))))
    print("overlap between APBC and APBC_X ground states:", np.abs(np.dot(gs_apbc.squeeze(1).conj(), gs_apbc_x.squeeze(1))))