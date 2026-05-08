import netket as nk
import numpy as np
import matplotlib.pyplot as plt
import os
from netket.operator.spin import sigmax, sigmay, sigmaz
from scipy.optimize import curve_fit 
from itertools import product

def Corr_ij(vstate, hi, i, j):

    # Calculate operator S_i * S_j
    # Note: For spin-1/2, S = 0.5 * sigma. 
    # S*S = 0.25 * (sig_x*sig_x + ...)
    corr_ij = 0.25 * (sigmaz(hi, i)@sigmaz(hi, j) + sigmax(hi, i)@sigmax(hi, j) + sigmay(hi, i)@sigmay(hi, j))
            
    exp = vstate.expect(corr_ij)

    return exp


def Corr_r(vstate, lattice, L, hi):
    """
    Computes the spatially averaged correlation C(r) as a function of
    Euclidean distance r = |r_i - r_j|, considering PBC wrapping.

    Returns:
        corr_by_dist: dict {distance: mean_correlation}
        err_by_dist: dict {distance: error_of_mean_correlation}
    """
    N_tot = lattice.n_nodes
    pairs = [(i, j) for i in range(N_tot) for j in range(N_tot) if i != j]

    corr_stats = {(i, j): Corr_ij(vstate, hi, i, j) for i, j in pairs}

    # Group correlations by Euclidean distance
    corr_by_dist = {}
    err_by_dist = {}
    counts_by_dist = {}

    for i in range(N_tot):
        for j in range(N_tot):
            if i == j:
                continue
            r_vec = lattice.positions[i] - lattice.positions[j]
            # Wrap distance for PBC
            dx = np.abs(r_vec[0])
            dy = np.abs(r_vec[1])
            if dx > L / 2.0: dx = L - dx
            if dy > L / 2.0: dy = L - dy
            dist = float(np.round(np.linalg.norm([dx, dy]), decimals=6))

            val = corr_stats[(i, j)].mean.real
            err = corr_stats[(i, j)].error_of_mean

            if dist not in corr_by_dist:
                corr_by_dist[dist] = 0.0
                err_by_dist[dist] = 0.0
                counts_by_dist[dist] = 0

            corr_by_dist[dist] += val
            err_by_dist[dist] += err**2
            counts_by_dist[dist] += 1

    # Average over all pairs at each distance
    for dist in corr_by_dist:
        corr_by_dist[dist] /= counts_by_dist[dist]
        err_by_dist[dist] = np.sqrt(err_by_dist[dist]) / counts_by_dist[dist]

    return corr_by_dist, err_by_dist


def compute_correlations(vstate, lattice, L, hilbert, folder):
    """
    Returns the isotropic C(r) averaged over all pairs at each
    Euclidean distance, sorted by distance.
    """
    
    corr_by_dist, err_by_dist = Corr_r(vstate, lattice, L, hilbert)

    # Sort by distance
    corr_r = dict(sorted(corr_by_dist.items()))
    err_r = dict(sorted(err_by_dist.items()))

    for dist, val in corr_r.items():
        print(f"C({dist:.4f}) = {val:.6f} ± {err_r[dist]:.6f}")

    return corr_r, err_r


def compute_correlation_length(vstate, lattice, hilbert, L, folder):
    """
    Fits C(r) to extract the correlation length xi.
    """
    corr_r, err_r = compute_correlations(vstate, lattice, L, hilbert, folder)

    r_vals = np.array(list(corr_r.keys()), dtype=float)
    c_vals = np.array(list(corr_r.values()), dtype=float)  # signed
    err_vals = np.array(list(err_r.values()), dtype=float)

    # Exclude negligible signal and limit r to L/2 * sqrt(2)
    max_r = (L / 2.0) * np.sqrt(2)
    mask = (np.abs(c_vals) > 1e-10) & (r_vals <= max_r)
    r_fit = r_vals[mask]
    c_fit = c_vals[mask]  # signed
    err_fit = err_vals[mask]

    if len(r_fit) < 2:
        print("Not enough points to fit correlation length.")
        return None, None, None, r_fit, c_fit

    def plain_exp_decay(r, A, xi):
        return A * np.exp(-r / xi)

    def power_law_decay(r, a, n, b):
        return a * r**n + b

    c_fit_abs = np.abs(c_fit)

    p0_exp = [c_fit_abs[0], np.max(r_fit) / 4.0]
    
    popt_exp = None
    pcov_exp = None
    try:
        popt_exp, pcov_exp = curve_fit(plain_exp_decay, r_fit, c_fit_abs, p0=p0_exp, sigma=np.where(err_fit == 0, 1e-10, err_fit), absolute_sigma=True, maxfev=10000)
        A_exp, xi_exp = popt_exp
        xi_err = np.sqrt(np.abs(pcov_exp[1, 1]))
        print(f"[Exponential fit] A = {A_exp:.4f}, xi = {xi_exp:.4f} ± {xi_err:.4f}")
    except RuntimeError:
        print("Exponential fit failed.")
        xi_exp = None
        xi_err = None

    p0_pow = [c_fit_abs[0], -1.0, 0.0]
    popt_pow = None
    try:
        popt_pow, _ = curve_fit(power_law_decay, r_fit, c_fit_abs, p0=p0_pow, sigma=np.where(err_fit == 0, 1e-10, err_fit), absolute_sigma=True, maxfev=10000)
        a_pow, n_pow, b_pow = popt_pow
        print(f"[Power-law fit] a = {a_pow:.4f}, n = {n_pow:.4f}, b = {b_pow:.4f}")
    except RuntimeError:
        print("Power-law fit failed.")

    plot_corr_r(r_fit, c_fit, err_fit, popt_exp, popt_pow, folder)
    
    if popt_exp is not None:
        return xi_exp, xi_err, popt_exp, r_fit, c_fit
    else:
        return None, None, None, r_fit, c_fit

def plot_corr_r(r_fit, c_fit, err_fit, popt_exp, popt_pow, folder):
    if len(r_fit) > 0:
        r_plot = np.linspace(np.min(r_fit), np.max(r_fit), 100)
    else:
        r_plot = np.linspace(0.1, 1, 100)

    os.makedirs(f'{folder}/physical_obs', exist_ok=True)

    # --- Figure 1: Exponential Fit ---
    plt.figure(figsize=(6,4))
    plt.errorbar(r_fit, np.abs(c_fit), yerr=err_fit, fmt='o', label='Data $|C(r)|$', color='blue', capsize=3)
    
    if popt_exp is not None:
        A_fit, xi_fit = popt_exp
        c_plot_exp = A_fit * np.exp(-r_plot / xi_fit)
        plt.plot(r_plot, c_plot_exp, label=f'Exp Fit: A={A_fit:.2f}, $\\xi$={xi_fit:.2f}', color='red')
        
    plt.xlabel('Distance $r$')
    plt.ylabel('$|C(r)|$')
    plt.title('Spin-Spin Correlation Function $|C(r)|$')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(f'{folder}/physical_obs/Corr_decay_exp.png', bbox_inches='tight')
    plt.close()

    # --- Figure 2: Power-law Fit ---
    plt.figure(figsize=(6,4))
    plt.errorbar(r_fit, np.abs(c_fit), yerr=err_fit, fmt='o', label='Data $|C(r)|$', color='blue', capsize=3)

    if popt_pow is not None:
        a_pow, n_pow, b_pow = popt_pow
        c_plot_pow = a_pow * r_plot**n_pow + b_pow
        plt.plot(r_plot, c_plot_pow, label=f'Poly Fit: a={a_pow:.2f}, n={n_pow:.2f}, b={b_pow:.2f}', color='green', linestyle='--')
        
    plt.xlabel('Distance $r$')
    plt.ylabel('$|C(r)|$')
    plt.title('Spin-Spin Correlation Function $|C(r)|$ ')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(f'{folder}/physical_obs/Corr_decay_pow.png', bbox_inches='tight')
    plt.close()

    # --- Figure 3: Power-law Fit (Log-Log) ---
    plt.figure(figsize=(6,4))
    plt.errorbar(r_fit, np.abs(c_fit), yerr=err_fit, fmt='o', label='Data $|C(r)|$', color='blue', capsize=3)

    if popt_pow is not None:
        plt.plot(r_plot, c_plot_pow, label=f'Poly Fit: a={a_pow:.2f}, n={n_pow:.2f}, b={b_pow:.2f}', color='green', linestyle='--')
        
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Distance $r$')
    plt.ylabel('$|C(r)|$')
    plt.title('Spin-Spin Correlation Function $|C(r)|$')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(f'{folder}/physical_obs/Corr_decay_pow_loglog.png', bbox_inches='tight')
    plt.close()

def make_dimer_correlations( hilbert, Lx, Ly, direction="x"):
    """
    Build all dimer operators D_R^alpha = S_R . S_{R+alpha_hat}
    for a 2D square lattice with sites indexed row-major: site = i*Ly + j.
    
    Returns a list of LocalOperator objects, one per site R,
    and the corresponding list of origin-site indices.
    """
    assert direction in ("x", "y")

    # Pauli / spin-half operators in the local 2x2 basis
    Sp = nk.operator.spin.sigmap(hilbert, 0)   # placeholder – we'll use LocalOperator
    Sm = nk.operator.spin.sigmam(hilbert, 0)
    Sz = nk.operator.spin.sigmaz(hilbert, 0)

    def dot_op(site_a, site_b):
        """S_a . S_b = 0.5*(S+_a S-_b + S-_a S+_b) + Sz_a Sz_b"""
        Sp_a = nk.operator.spin.sigmap(hilbert, site_a)
        Sm_a = nk.operator.spin.sigmam(hilbert, site_a)
        Sz_a = nk.operator.spin.sigmaz(hilbert, site_a)
        Sp_b = nk.operator.spin.sigmap(hilbert, site_b)
        Sm_b = nk.operator.spin.sigmam(hilbert, site_b)
        Sz_b = nk.operator.spin.sigmaz(hilbert, site_b)
        return 0.5 * (Sp_a @ Sm_b + Sm_a @ Sp_b) + Sz_a @ Sz_b

    dimers = {}  # site_R -> LocalOperator for D_R^alpha
    for i, j in product(range(Lx), range(Ly)):
        R = i * Ly + j
        if direction == "x":
            # neighbour in x: (i, j+1) with PBC
            ni, nj = i, (j + 1) % Ly
        else:
            # neighbour in y: (i+1, j) with PBC
            ni, nj = (i + 1) % Lx, j
        R_neigh = ni * Ly + nj
        dimers[R] = dot_op(R, R_neigh)

    return dimers


def compute_dimer_correlations(vstate, L, folder, direction="x", origin=(0, 0)):
    """
    Compute the connected dimer-dimer correlator

        C(R) = <D_0^a D_R^a> - <D_0^a><D_R^a>

    for all R on the lattice, along `direction` bonds.

    Parameters
    ----------
    vstate    : nk.vqs.VariationalState
    L        : lattice dimension
    direction : "x" or "y"
    origin    : (i0, j0) site index of the reference dimer, default (0,0)

    Returns
    -------
    coords : np.ndarray, shape (N,2)  – (i,j) for each site R
    C      : np.ndarray, shape (N,)   – connected correlator values
    """
    hilbert = vstate.hilbert

    dimers = make_dimer_correlations(hilbert, L, L, direction)

    i0, j0 = origin
    R0 = i0 * L + j0
    D0 = dimers[R0]

    # <D_0>
    exp_D0 = vstate.expect(D0).mean.real

    coords = []
    C = []
    for i, j in product(range(L), range(L)):
        R = i * L + j
        DR = dimers[R]

        # <D_0 D_R>  via expect on the product operator
        D0_DR = D0 @ DR
        exp_D0_DR = vstate.expect(D0_DR).mean.real

        # <D_R>
        exp_DR = vstate.expect(DR).mean.real

        connected = exp_D0_DR - exp_D0 * exp_DR
        coords.append((i, j))
        C.append(connected)

    coords = np.array(coords)
    C = np.array(C)

    plot_dimer_correlations(coords, C, L, direction, folder)
    plot_dimer_correlation_decay(coords, C, L, direction, folder)

    return coords, C

def plot_dimer_correlations(coords, C, L, direction, folder):
    """
    Plot the dimer-dimer correlations C(R) on a 2D grid.

    Parameters
    ----------
    coords : np.ndarray, shape (N,2)  – (i,j) for each site R
    C      : np.ndarray, shape (N,)   – connected correlator values
    L      : lattice dimension
    direction : "x" or "y"
    folder : directory to save the plot
    """
    C_grid = C.reshape(L, L)

    plt.figure(figsize=(6,5))
    plt.imshow(C_grid, origin='lower', cmap='RdBu_r', vmin=-np.max(np.abs(C)), vmax=np.max(np.abs(C)))
    plt.colorbar(label='Connected Dimer-Dimer Correlation')
    plt.xlabel('j')
    plt.ylabel('i')
    plt.title(f'Dimer-Dimer Correlations along {direction}-bonds')
    plt.xticks(np.arange(L))
    plt.yticks(np.arange(L))
    plt.savefig(f'{folder}/physical_obs/Dimer_Corr_{direction}.png')
    plt.close()

def plot_dimer_correlation_decay(coords, C, L, direction, folder):
    """
    Plot the decay of the dimer-dimer correlation function with distance
    and fit it with a power law.
    """
    r_vals = []
    for i, j in coords:
        # Calculate Euclidean distance from origin (0,0) with PBC wrapping
        dx = np.abs(j)
        dy = np.abs(i)
        if dx > L / 2.0: dx = L - dx
        if dy > L / 2.0: dy = L - dy
        dist = np.sqrt(dx**2 + dy**2)
        r_vals.append(dist)
        
    r_vals = np.array(r_vals)
    C_abs = np.abs(C)

    # Group by distance to compute mean and error
    unique_r = np.unique(np.round(r_vals, 6))
    r_fit, c_fit, err_fit = [], [], []

    for r in unique_r:
        if r < 1e-5:  # Skip origin (r=0)
            continue
        mask = np.isclose(r_vals, r)
        vals = C_abs[mask]
        r_fit.append(r)
        c_fit.append(np.mean(vals))
        err_fit.append(np.std(vals) / np.sqrt(len(vals)) if len(vals) > 1 else 0.0)

    r_fit = np.array(r_fit)
    c_fit = np.array(c_fit)
    err_fit = np.array(err_fit)

    def power_law_decay(r, a, n, b):
        return a * r**n + b

    popt_pow = None
    try:
        p0 = [c_fit[0] if len(c_fit) > 0 else 1.0, -1.0, 0.0]
        sigma = np.where(err_fit == 0, 1e-10, err_fit)
        popt_pow, _ = curve_fit(power_law_decay, r_fit, c_fit, p0=p0, sigma=sigma, absolute_sigma=True, maxfev=10000)
        a_pow, n_pow, b_pow = popt_pow
        print(f"[Dimer Power-law fit ({direction})] a = {a_pow:.4f}, n = {n_pow:.4f}, b = {b_pow:.4f}")
    except Exception as e:
        print(f"Dimer Power-law fit failed: {e}")

    plt.figure(figsize=(6,4))
    plt.errorbar(r_fit, c_fit, yerr=err_fit, fmt='o', label='Data $|C(r)|$', color='blue', capsize=3)

    if popt_pow is not None:
        r_plot = np.linspace(np.min(r_fit), np.max(r_fit), 100)
        c_plot_pow = power_law_decay(r_plot, *popt_pow)
        plt.plot(r_plot, c_plot_pow, label=f'Poly Fit: n={n_pow:.2f}', color='green', linestyle='--')

    plt.xlabel('Distance $r$')
    plt.ylabel('Connected Dimer Correlation $|C(r)|$')
    plt.title(f'Dimer-Dimer Correlation Decay ({direction}-bonds)')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    
    os.makedirs(f'{folder}/physical_obs', exist_ok=True)
    plt.savefig(f'{folder}/physical_obs/Dimer_Corr_decay_pow_{direction}.png', bbox_inches='tight')
    plt.close()
    
    # --- Power-law Fit (Log-Log) ---
    plt.figure(figsize=(6,4))
    plt.errorbar(r_fit, c_fit, yerr=err_fit, fmt='o', label='Data $|C(r)|$', color='blue', capsize=3)

    if popt_pow is not None:
        plt.plot(r_plot, c_plot_pow, label=f'Poly Fit: n={n_pow:.2f}', color='green', linestyle='--')

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Distance $r$')
    plt.ylabel('Connected Dimer Correlation $|C(r)|$')
    plt.title(f'Dimer-Dimer Correlation Decay ({direction}-bonds)')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(f'{folder}/physical_obs/Dimer_Corr_decay_pow_loglog_{direction}.png', bbox_inches='tight')
    plt.close()

    return popt_pow


def Corr_Struct(lattice, vstate, L, folder, hi):
    N_tot = lattice.n_nodes
    corr_r = np.zeros((L, L))
    counts = np.zeros((L, L))

    # 1. Calculate Correlations
    for i in range(N_tot):
        for j in range(N_tot):
            r = lattice.positions[i] - lattice.positions[j]
            # Assumes lattice constant = 1.0
            r0, r1 = int(np.round(r[0])) % L , int(np.round(r[1])) % L 
            
            corr = Corr_ij(vstate, hi, i, j)

            corr_r[r0, r1] += corr.mean.real
            counts[r0, r1] += 1
            
    corr_r /= counts 
    
    # --- PLOTTING C(r) ---
    # Create a copy for plotting so we don't ruin the data for FFT
    corr_plot = corr_r.copy()
    corr_plot[0,0] = 0.0 # Set to 0 ONLY for visualization contrast
    
    plt.figure(figsize=(6,5))
    plt.imshow(corr_plot, origin='lower', cmap='viridis')
    plt.colorbar(label='C(r) (diag masked)')
    # ... (rest of plotting code) ...
    plt.savefig(f'{folder}/physical_obs/Corr.png')
    plt.close()

    # --- STRUCTURE FACTOR ---
    # Use the REAL corr_r (where C(0) is approx 0.75), do not zero it!
    S_q = np.fft.fft2(corr_r) 
    S_q_real = np.abs(S_q) # Take magnitude
    
    # Shift so q=(0,0) is in the center for plotting (optional but standard)
    # OR use your periodic filling method, which is fine.
    S_q_periodic = np.zeros((L+1, L+1))
    S_q_periodic[:L, :L] = S_q_real
    S_q_periodic[L, :] = S_q_periodic[0, :]    
    S_q_periodic[:, L] = S_q_periodic[:, 0]

    plt.figure(figsize=(6,5))
    plt.imshow(np.abs(S_q_periodic), origin='lower', cmap='viridis')
    plt.colorbar(label='|S(q)|')
    plt.xlabel('q_x')
    plt.ylabel('q_y')
    plt.title('Structure Factor S(q)')
    plt.xticks([0, 1/2*L, L], ['0', 'π', '2π'])
    plt.yticks([0, 1/2*L, L], ['0', 'π', '2π'])
    plt.savefig(f'{folder}/physical_obs/Struct.png')

    # Sharpness at (π,π)
    S_pi_pi = np.abs(S_q_periodic[L//2, L//2])
    
    # S(Q + delta q) - average over nearest neighbors
    S_neighbors = (np.abs(S_q_periodic[L//2+1, L//2]) + 
                   np.abs(S_q_periodic[L//2-1, L//2]) + 
                   np.abs(S_q_periodic[L//2, L//2+1]) + 
                   np.abs(S_q_periodic[L//2, L//2-1])) / 4.0

    R = 1 - S_neighbors/S_pi_pi
    
    return R

def Corr_Struct_Exact(lattice, ket_gs, L, J, folder, hi):

    N_tot = lattice.n_nodes

    corr_r = np.zeros((L, L))
    counts = np.zeros((L, L))
    
    # Ensure ket_gs is a numpy array
    ket_gs = np.array(ket_gs)

    for i in range(N_tot):
        for j in range(N_tot):
            r = lattice.positions[i] - lattice.positions[j]
            corr_ij = 0.25 * (sigmaz(hi, i) @ sigmaz(hi, j) + sigmax(hi, i) @ sigmax(hi, j) + sigmay(hi, i) @ sigmay(hi, j))
            
            # Convert operator to sparse matrix for exact computation
            op_sparse = corr_ij.to_sparse()
            # Compute expectation value <psi|O|psi>
            exp = np.vdot(ket_gs, op_sparse.dot(ket_gs))
            
            r0, r1 = int(r[0]) % L , int(r[1]) % L #PBC
            corr_r[r0, r1] += exp.real
            counts[r0, r1] += 1
    corr_r /= counts 
    #corr_r[0, 0] = 0  # set C(0) = 0


    plt.figure(figsize=(6,5))
    plt.imshow(corr_r, origin='lower', cmap='viridis')
    plt.colorbar(label='C(r)')
    plt.xlabel('dx')
    plt.ylabel('dy')
    plt.title('Exact Spin-Spin Correlation Function C(r) in 2D')
    plt.xticks(np.arange(L))  # integer ticks for x-axis
    plt.yticks(np.arange(L)) 
    plt.savefig(f'{folder}/Obs/J={J}/Corr_exact.png')
    plt.close()


    #Structure factor
    # Compute the 2D Fourier transform of corr_r
    S_q = np.fft.fft2(corr_r)
    S_q_periodic = np.zeros((L+1, L+1), dtype=S_q.dtype)
    S_q_periodic[:L, :L] = S_q  
    S_q_periodic[L, :] = S_q_periodic[0, :]    
    S_q_periodic[:, L] = S_q_periodic[:, 0]    

    plt.figure(figsize=(6,5))
    plt.imshow(np.abs(S_q_periodic), origin='lower', cmap='viridis')
    plt.colorbar(label='|S(q)|')
    plt.xlabel('q_x')
    plt.ylabel('q_y')
    plt.title('Exact Structure Factor S(q)')
    plt.xticks([0, 1/2*L, L], ['0', 'π', '2π'])
    plt.yticks([0, 1/2*L, L], ['0', 'π', '2π'])
    plt.savefig(f'{folder}/Obs/J={J}/Struct_exact.png')
    plt.close()