import jax
import jax.numpy as jnp
import numpy as np

# --- 1. Mock Lattice Class ---
class MockLattice:
    def __init__(self, L):
        self.L = L
        self.positions = []
        # Create square lattice points
        for y in range(L):
            for x in range(L):
                self.positions.append((x, y))

# --- 2. The Corrected Hk Function ---
def Hk_corrected(sigmaz, phi, h, N_sites, positions, lattice, dtype):
    
    def determine_nns(graph):
        # Simplified NNS for the mock lattice structure
        nearest_neighbors = [{"x": [], "y": []} for _ in range(N_sites)]
        L = int(np.sqrt(N_sites))
        for i, (x, y) in enumerate(positions):
             # Neighbor Right (x+1)
             nx = (x + 1) % L
             idx_x = int(y * L + nx)
             nearest_neighbors[i]["x"].append(idx_x)

             # Neighbor Up (y+1)
             ny = (y + 1) % L
             idx_y = int(ny * L + x)
             nearest_neighbors[i]["y"].append(idx_y)
        return nearest_neighbors

    def hopping_x_conditions(inp):
        H, x, y, ix, i = inp
        flux = (-1)**(jnp.floor(x) + jnp.floor(y)) * 1j * phi * jnp.pi 
        H = H.at[i, ix].add(-0.5 * jnp.exp(flux)) 
        H = H.at[ix, i].add(-0.5 * jnp.exp(-flux))
        return H

    def hopping_y_conditions(inp):
        H, x, y, iy, i = inp
        # CORRECTED FLUX LOGIC: (-1 factor included, exp(flux) used)
        flux = -1 * (-1)**(jnp.floor(x) + jnp.floor(y)) * 1j * phi * jnp.pi
        H = H.at[i, iy].add(-0.5 * jnp.exp(flux)) 
        H = H.at[iy, i].add(-0.5 * jnp.exp(-flux))
        return H

    H = jnp.zeros([N_sites, N_sites], dtype=dtype)
    nearest_neighbors = determine_nns(lattice)

    for i, (x, y) in enumerate(positions):
        ix_list = nearest_neighbors[i]["x"]
        iy_list = nearest_neighbors[i]["y"]
        
        if len(ix_list) > 0:
            for ix in ix_list:
                H = hopping_x_conditions((H, x, y, ix, i))
        if len(iy_list) > 0:
            for iy in iy_list:
                H = hopping_y_conditions((H, x, y, iy, i))

        # Staggered Field (Diagonal)
        H = H.at[i, i].add(jnp.where((x + y) % 2 == 0, -sigmaz * h, sigmaz * h))

    return H

# --- 3. New Function: Evaluate Mean-Field Energy ---
def evaluate_mf_energy(phi, h, L, lattice):
    N_sites = L * L
    n_elecs = N_sites # Half filling total (N/2 up, N/2 down)
    n_occ_per_spin = n_elecs // 2
    dtype = jnp.complex128
    
    # --- Spin Up Sector (+h) ---
    H_up = Hk_corrected(1.0, phi, h, N_sites, lattice.positions, lattice, dtype)
    # Diagonalize
    evals_up, _ = jnp.linalg.eigh(H_up)
    # Sum lowest N/2 eigenvalues
    E_up = jnp.sum(evals_up[:n_occ_per_spin])
    
    # --- Spin Down Sector (-h) ---
    H_dn = Hk_corrected(-1.0, phi, h, N_sites, lattice.positions, lattice, dtype)
    # Diagonalize
    evals_dn, _ = jnp.linalg.eigh(H_dn)
    # Sum lowest N/2 eigenvalues
    E_dn = jnp.sum(evals_dn[:n_occ_per_spin])
    
    # Total Energy
    E_total = E_up + E_dn
    E_per_site = E_total / (N_sites*4)
    
    return E_per_site

# --- 4. Main Execution ---

# Setup
L = 4
N_sites = L * L
mock_lattice = MockLattice(L)
dtype = jnp.complex128

print("=== 1. Flux Verification ===")
phi_test = 0.1
H_matrix = Hk_corrected(1.0, phi_test, 0.0, N_sites, mock_lattice.positions, mock_lattice, dtype)
idx_00 = 0
idx_10 = 1
idx_11 = L + 1
idx_01 = L
t_1 = H_matrix[idx_10, idx_00] 
t_2 = H_matrix[idx_11, idx_10]
t_3 = H_matrix[idx_01, idx_11]
t_4 = H_matrix[idx_00, idx_01]
W = t_1 * t_2 * t_3 * t_4
measured_phase = jnp.angle(W)
print(f"Phi: {phi_test}")
print(f"Measured Phase / pi: {measured_phase / np.pi:.4f} pi")
if abs(measured_phase) > 1e-5:
    print("✅ Flux Check Passed")
else:
    print("❌ Flux Check Failed")

print("\n=== 2. Energy Verification ===")

# Case A: Free Fermions (Standard tight-binding limit)
# Expectation: ~ -1.62 per site total for full lattice (approx -0.81 per spin)
h_val = 0.0
phi_val = 0.0
E_0 = evaluate_mf_energy(phi_val, h_val, L, mock_lattice)
print(f"Parameters: h={h_val}, phi={phi_val}")
print(f"Mean-Field Energy per site: {E_0:.4f}")

# Case B: Staggered Flux State (Near the 'lobes' of interest)
# This energy should be HIGHER than Case A because this is the Mean-Field energy.
# (The lower physical energy only appears after Gutzwiller projection).
h_val = 0.08
phi_val = 0.12
E_sf = evaluate_mf_energy(phi_val, h_val, L, mock_lattice)
print(f"Parameters: h={h_val}, phi={phi_val}")
print(f"Mean-Field Energy per site: {E_sf:.4f}")

if E_sf > E_0:
    print("\nObservation: Staggered Flux MF energy is higher than Free Fermion energy.")
    print("This is correct! The Staggered Flux state is an excited state of the MF Hamiltonian.")
    print("It becomes the ground state only *after* projection in the t-J model.")