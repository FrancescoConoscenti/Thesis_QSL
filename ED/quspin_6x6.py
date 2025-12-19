from quspin.operators import hamiltonian
from quspin.basis import spin_basis_general
import numpy as np
import os

L = 6
N = L**2
sites = np.arange(N).reshape((L, L))

# --- 1. Define Symmetry Operators ---

# Shift indices to the right (x) and up (y)
tx = np.roll(sites, -1, axis=1).flatten() 
ty = np.roll(sites, -1, axis=0).flatten()

# Rotation 90 degrees
rot = np.rot90(sites, k=1).flatten()

# Reflection about x-axis
px = sites[:, ::-1].flatten()

# --- 2. Create Bond Lists ---
J1_list = []
J2_list = []

for r in range(L):
    for c in range(L):
        s = sites[r, c]
        # J1: Right and Top neighbors (with PBC via % L)
        right = sites[r, (c + 1) % L]
        top   = sites[(r + 1) % L, c]
        J1_list.extend([[1.0, s, right], [1.0, s, top]])
        
        # J2: Top-Right and Top-Left diagonals
        tr_diag = sites[(r + 1) % L, (c + 1) % L]
        tl_diag = sites[(r + 1) % L, (c - 1) % L]
        J2_list.extend([[0.5, s, tr_diag], [0.5, s, tl_diag]])

# --- 3. Build Basis and Hamiltonian ---
# kxblock=0, kyblock=0 (Zero momentum)
# zblock=1 (Spin reflection / Parity even)
#basis_symm = spin_basis_general(N, m=0, S='1/2', kxblock=(tx, 0), kyblock=(ty, 0), rblock=(rot, 0), pblock=(px, 0))
basis = spin_basis_general(N, m=0, S='1/2')

# Heisenberg interaction: J(SxSx + SySy + SzSz) 
# In QuSpin "++" and "--" are needed for the XY part if not using built-in Heisenberg
static = [
    ["zz", J1_list], ["+-", [ [0.5*J[0], J[1], J[2]] for J in J1_list ]], ["-+", [ [0.5*J[0], J[1], J[2]] for J in J1_list ]],
    ["zz", J2_list], ["+-", [ [0.5*J[0], J[1], J[2]] for J in J2_list ]], ["-+", [ [0.5*J[0], J[1], J[2]] for J in J2_list ]]
]
# Note: Multiplying +- by 0.5 because S+S- + S-S+ = 2(SxSx + SySy)

H = hamiltonian(static, [], basis=basis, dtype=np.float64)
E, psi = H.eigsh(k=1, which='SA')
print(f"Ground state energy per site: {E[0]/N}")

output_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "energy_output_all_symmetry_6x6.txt")
with open(output_file, "w") as f:
    f.write(f"Ground state energy per site: {E[0]/N}\n")

psi_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "psi_output_6x6_all_symm.npy")
np.save(psi_file, psi)