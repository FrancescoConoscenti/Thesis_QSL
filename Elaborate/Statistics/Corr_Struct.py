import netket as nk
import numpy as np
import matplotlib.pyplot as plt
import os
from netket.operator.spin import sigmax, sigmay, sigmaz
from scipy.optimize import curve_fit 

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
    """
    N_tot = lattice.n_nodes
    pairs = [(i, j) for i in range(N_tot) for j in range(N_tot) if i != j]

    corr_vals = {(i, j): Corr_ij(vstate, hi, i, j).mean.real for i, j in pairs}

    # Group correlations by Euclidean distance
    corr_by_dist = {}
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

            val = corr_vals[(i, j)]

            if dist not in corr_by_dist:
                corr_by_dist[dist] = 0.0
                counts_by_dist[dist] = 0

            corr_by_dist[dist] += val
            counts_by_dist[dist] += 1

    # Average over all pairs at each distance
    for dist in corr_by_dist:
        corr_by_dist[dist] /= counts_by_dist[dist]

    return corr_by_dist


def compute_correlations(vstate, lattice, L, hilbert, folder):
    """
    Returns the isotropic C(r) averaged over all pairs at each
    Euclidean distance, sorted by distance.
    """
    vstate.n_samples = 1024
    corr_by_dist = Corr_r(vstate, lattice, L, hilbert)

    # Sort by distance
    corr_r = dict(sorted(corr_by_dist.items()))

    for dist, val in corr_r.items():
        print(f"C({dist:.4f}) = {val:.6f}")

    return corr_r


def compute_correlation_length(vstate, lattice, hilbert, L, folder):
    """
    Fits C(r) to extract the correlation length xi.
    """
    corr_r = compute_correlations(vstate, lattice, L, hilbert, folder)

    r_vals = np.array(list(corr_r.keys()), dtype=float)
    c_vals = np.array(list(corr_r.values()), dtype=float)  # signed

    # Exclude negligible signal and limit r to L/2 * sqrt(2)
    max_r = (L / 2.0) * np.sqrt(2)
    mask = (np.abs(c_vals) > 1e-10) & (r_vals <= max_r)
    r_fit = r_vals[mask]
    c_fit = c_vals[mask]  # signed

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
        popt_exp, pcov_exp = curve_fit(plain_exp_decay, r_fit, c_fit_abs, p0=p0_exp, maxfev=10000)
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
        popt_pow, _ = curve_fit(power_law_decay, r_fit, c_fit_abs, p0=p0_pow, maxfev=10000)
        a_pow, n_pow, b_pow = popt_pow
        print(f"[Power-law fit] a = {a_pow:.4f}, n = {n_pow:.4f}, b = {b_pow:.4f}")
    except RuntimeError:
        print("Power-law fit failed.")

    plot_corr_r(r_fit, c_fit, popt_exp, popt_pow, folder)
    
    if popt_exp is not None:
        return xi_exp, xi_err, popt_exp, r_fit, c_fit
    else:
        return None, None, None, r_fit, c_fit

def plot_corr_r(r_fit, c_fit, popt_exp, popt_pow, folder):
    if len(r_fit) > 0:
        r_plot = np.linspace(np.min(r_fit), np.max(r_fit), 100)
    else:
        r_plot = np.linspace(0.1, 1, 100)

    plt.figure(figsize=(6,4))
    plt.scatter(r_fit, np.abs(c_fit), label='Data $|C(r)|$', color='blue')
    
    if popt_exp is not None:
        A_fit, xi_fit = popt_exp
        c_plot_exp = A_fit * np.exp(-r_plot / xi_fit)
        plt.plot(r_plot, c_plot_exp, label=f'Exp Fit: A={A_fit:.2f}, $\\xi$={xi_fit:.2f}', color='red')
        
    if popt_pow is not None:
        a_pow, n_pow, b_pow = popt_pow
        c_plot_pow = a_pow * r_plot**n_pow + b_pow
        plt.plot(r_plot, c_plot_pow, label=f'Poly Fit: a={a_pow:.2f}, n={n_pow:.2f}, b={b_pow:.2f}', color='green', linestyle='--')
        
    #plt.figure(figsize=(8, 6), dpi=80)
    plt.xlabel('Distance $r$')
    plt.ylabel('$|C(r)|$')
    plt.title('Absolute Spin-Spin Correlation Function $|C(r)|$')
    plt.legend()
    plt.grid()
    os.makedirs(f'{folder}/physical_obs', exist_ok=True)
    plt.tight_layout()
    plt.savefig(f'{folder}/physical_obs/Corr_decay.png', bbox_inches='tight')
    plt.close()


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