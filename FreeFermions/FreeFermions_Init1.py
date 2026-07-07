import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.optimize import curve_fit
import os

import numpy as np
import matplotlib.pyplot as plt

def get_spectral_states(L, phi_range):
    n_sites = L * L
    gs_energies = []
    excited_energies = []

    def idx(x, y): return (x % L) * L + (y % L)

    for phi in phi_range:
        h_matrix = np.zeros((n_sites, n_sites), dtype=complex)
        for x in range(L):
            for y in range(L):
                # Vertical hopping
                h_matrix[idx(x, y), idx(x, y+1)] = -1.0
                h_matrix[idx(x, y+1), idx(x, y)] = -1.0
                
                # Horizontal hopping with pi-flux (-1)^y
                t_horiz = -1.0 * ((-1)**y)
                
                # Boundary twist phi
                if x == L - 1:
                    h_matrix[idx(x, y), idx(0, y)] = t_horiz * np.exp(1j * phi)
                    h_matrix[idx(0, y), idx(x, y)] = np.conj(t_horiz * np.exp(1j * phi))
                else:
                    h_matrix[idx(x, y), idx(x+1, y)] = t_horiz
                    h_matrix[idx(x+1, y), idx(x, y)] = t_horiz

        eigenvalues = np.sort(np.linalg.eigvalsh(h_matrix))
        
        # Ground State (GS): Sum of lower N/2 states
        e_gs = np.sum(eigenvalues[:n_sites // 2])
        
        # First Excited State: Move particle from highest occupied (N/2 - 1)
        # to lowest unoccupied (N/2)
        # E_exc = E_gs - epsilon_occ + epsilon_unocc
        e_exc = e_gs - eigenvalues[n_sites // 2 - 1] + eigenvalues[n_sites // 2]
        
        gs_energies.append(e_gs / n_sites)
        excited_energies.append(e_exc / n_sites)

    return np.array(gs_energies), np.array(excited_energies)

def plot_pi_flux_dispersion():
    # Create a dense grid of k-points in the Brillouin Zone
    k = np.linspace(-np.pi, np.pi, 100)
    kx, ky = np.meshgrid(k, k)

    # Dispersion for pi-flux square lattice (2-site unit cell)
    # epsilon(k) = +/- 2 * sqrt( cos(kx)^2 + cos(ky)^2 )
    # This specific form depends on the gauge, but hits 0 at (+/-pi/2, +/-pi/2)
    energy_plus = 2 * np.sqrt(np.cos(kx)**2 + np.cos(ky)**2)
    energy_minus = -energy_plus

    # --- Plotting ---
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot Upper and Lower Bands
    surf1 = ax.plot_surface(kx, ky, energy_plus, cmap=cm.coolwarm, alpha=0.8, antialiased=True)
    surf2 = ax.plot_surface(kx, ky, energy_minus, cmap=cm.coolwarm, alpha=0.8, antialiased=True)

    # Mark the Dirac Points
    dirac_points = [ (np.pi/2, np.pi/2), (-np.pi/2, -np.pi/2), (np.pi/2, -np.pi/2), (-np.pi/2, np.pi/2) ]
    for px, py in dirac_points:
        ax.scatter([px], [py], [0], color='black', s=50, zorder=10)

    ax.set_title(r"Dispersion Relation of $\pi$-flux Square Lattice (Dirac Cones)")
    ax.set_xlabel(r"$k_x$")
    ax.set_ylabel(r"$k_y$")
    ax.set_zlabel(r"Energy $\epsilon(k)$")
    ax.view_init(elev=20, azim=45)

    plt.savefig("FreeFermions/plots/Dispersions/pi_flux_dispersion.png", dpi=300)
    
import pathlib as Path

def plot_dirac_cuts(L, phi):
    # 1. Create the continuous 3D surface
    k = np.linspace(-np.pi, np.pi, 100)
    kx_surf, ky_surf = np.meshgrid(k, k)
    energy_surf = 2 * np.sqrt(np.cos(kx_surf)**2 + np.cos(ky_surf)**2)

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the Dirac Cones (Upper and Lower)
    ax.plot_surface(kx_surf, ky_surf, energy_surf, cmap=cm.Blues, alpha=0.3, antialiased=True)
    ax.plot_surface(kx_surf, ky_surf, -energy_surf, cmap=cm.Blues, alpha=0.3, antialiased=True)

    # 2. Calculate and plot the discrete lines (cuts) for a given phi
    nx_values = np.arange(L)
    ky_continuous = np.linspace(-np.pi, np.pi, 200)

    for nx in nx_values:
        kx_val = (2 * np.pi * nx + phi) / L
        # Normalize kx to be within [-pi, pi] for the plot
        kx_plot = ((kx_val + np.pi) % (2 * np.pi)) - np.pi

        # Calculate energy along this cut
        e_line = 2 * np.sqrt(np.cos(kx_plot)**2 + np.cos(ky_continuous)**2)
        
        # CHANGED: Commented out the upper dispersion lines, keeping only the lower ones
        # ax.plot([kx_plot]*len(ky_continuous), ky_continuous, e_line, color='red', lw=2, alpha=0.8)
        ax.plot([kx_plot]*len(ky_continuous), ky_continuous, -e_line, color='red', lw=2, alpha=0.8)

    # 3. Mark the discrete points (kx, ky) actually sampled by the LxL lattice
    for nx in range(L):
        for ny in range(L):
            kx_pt = (( (2 * np.pi * nx + phi) / L + np.pi) % (2 * np.pi)) - np.pi
            ky_pt = (( (2 * np.pi * ny) / L + np.pi) % (2 * np.pi)) - np.pi
            e_pt = 2 * np.sqrt(np.cos(kx_pt)**2 + np.cos(ky_pt)**2)
            
            ax.scatter([kx_pt], [ky_pt], [-e_pt], color='black', s=20)

    ax.set_xlabel(r"$k_y$")
    ax.set_ylabel(r"$k_x$")
    ax.grid(False)
    
    # CHANGED: Label the z-axis Energy epsilon(k)
    ax.set_zlabel(r"Energy $\epsilon(k)$")
    
    # CHANGED: Set explicit ticks for x and y axes to just -pi, 0, pi
    ax.set_xticks([-np.pi, 0, np.pi])
    ax.set_xticklabels([r'$-\pi$', r'$0$', r'$\pi$'])
    ax.set_yticks([-np.pi, 0, np.pi])
    ax.set_yticklabels([r'$-\pi$', r'$0$', r'$\pi$'])
    ax.set_zticklabels([ r'$0$'])
    ax.set_zticks([0])
    
    ax.view_init(elev=25, azim=45)
    
    # Ensured directory exists before saving
    save_path = f"FreeFermions/plots/Dispersions/dirac_cuts_L{L}_phi{phi/np.pi:.1f}.png"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.show()

def get_0_flux_spectral_flow(L, phi_range):
    n_sites = L * L
    gs_energies = []
    excited_energies = []
    def idx(x, y): return (x % L) * L + (y % L)

    for phi in phi_range:
        h_matrix = np.zeros((n_sites, n_sites), dtype=complex)
        for x in range(L):
            for y in range(L):
                # Uniform Hopping (No staggered signs)
                h_matrix[idx(x, y), idx(x, y+1)] = -1.0
                h_matrix[idx(x, y+1), idx(x, y)] = -1.0
                
                # Boundary twist phi
                t_horiz = -1.0
                if x == L - 1:
                    h_matrix[idx(x, y), idx(0, y)] = t_horiz * np.exp(1j * phi)
                    h_matrix[idx(0, y), idx(x, y)] = np.conj(t_horiz * np.exp(1j * phi))
                else:
                    h_matrix[idx(x, y), idx(x+1, y)] = t_horiz
                    h_matrix[idx(x+1, y), idx(x, y)] = t_horiz

        eigenvalues = np.sort(np.linalg.eigvalsh(h_matrix))
        e_gs = np.sum(eigenvalues[:n_sites // 2])
        e_exc = e_gs - eigenvalues[n_sites // 2 - 1] + eigenvalues[n_sites // 2]
        gs_energies.append(e_gs / n_sites)
        excited_energies.append(e_exc / n_sites)

    return np.array(gs_energies), np.array(excited_energies)



def plot_fermi_surface_dispersion():
    k = np.linspace(-np.pi, np.pi, 100)
    kx, ky = np.meshgrid(k, k)

    # Standard Tight-Binding Dispersion (0-Flux)
    # epsilon(k) = -2t * (cos(kx) + cos(ky))
    energy_surf = -2 * (np.cos(kx) + np.cos(ky))

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the Single Band
    surf = ax.plot_surface(kx, ky, energy_surf, cmap=cm.viridis, alpha=0.8)
    
    # Draw the Fermi Level at 0 (Half-filling)
    ax.contour(kx, ky, energy_surf, levels=[0], colors='red', linewidths=3)

    ax.set_title("0-Flux Dispersion: Standard Fermi Surface (Red Diamond)")
    ax.set_xlabel(r"$k_x$")
    ax.set_ylabel(r"$k_y$")
    ax.set_zlabel(r"Energy $\epsilon(k)$")
    plt.savefig("FreeFermions/plots/Dispersions/0_flux_fermi_surface_dispersion.png", dpi=300)
    plt.show()

def plot_0_flux_cuts(L, phi):
    # 1. Create the continuous 3D surface (Standard cosine band)
    k = np.linspace(-np.pi, np.pi, 100)
    kx_surf, ky_surf = np.meshgrid(k, k)
    energy_surf = -2 * (np.cos(kx_surf) + np.cos(ky_surf))

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the Cosine Band
    ax.plot_surface(kx_surf, ky_surf, energy_surf, cmap=cm.viridis, alpha=0.3, antialiased=True)
    
    # Highlight the Fermi Surface (0-energy contour)
    ax.contour(kx_surf, ky_surf, energy_surf, levels=[0], colors='red', linewidths=3, offset=0)

    # 2. Calculate and plot the discrete lines (cuts) for a given phi
    nx_values = np.arange(L)
    ky_continuous = np.linspace(-np.pi, np.pi, 200)

    for nx in nx_values:
        kx_val = (2 * np.pi * nx + phi) / L
        kx_plot = ((kx_val + np.pi) % (2 * np.pi)) - np.pi
        
        # Calculate energy along this cut
        e_line = -2 * (np.cos(kx_plot) + np.cos(ky_continuous))
        
        # Draw the lines on the surface
        ax.plot([kx_plot]*len(ky_continuous), ky_continuous, e_line, color='darkorange', lw=2, alpha=0.8)

    # 3. Mark the discrete points (kx, ky) actually sampled
    for nx in range(L):
        for ny in range(L):
            kx_pt = (( (2 * np.pi * nx + phi) / L + np.pi) % (2 * np.pi)) - np.pi
            ky_pt = (( (2 * np.pi * ny) / L + np.pi) % (2 * np.pi)) - np.pi
            e_pt = -2 * (np.cos(kx_pt) + np.cos(ky_pt))
            ax.scatter([kx_pt], [ky_pt], [e_pt], color='black', s=20)

    ax.set_title(f"0-Flux Fermi Surface Sampling (L={L}, $\phi$={phi/np.pi:.2f}$\pi$)")
    ax.set_xlabel(r"$k_x$")
    ax.set_ylabel(r"$k_y$")
    ax.view_init(elev=25, azim=45)
    plt.savefig(f"FreeFermions/plots/Dispersions/0_flux_cuts_L{L}_phi{phi/np.pi:.1f}.png", dpi=300)
    

def plot_side_by_side(L_list, phis):
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    fig3, ax3 = plt.subplots(figsize=(8, 6))
    x_phi = phis / np.pi
    
    # Colors for different L sizes
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray']
    
    for i, L in enumerate(L_list):
        color = colors[i % len(colors)]
        
        # --- 0-Flux (Standard Fermi Sea) ---
        gs0, exc0 = get_0_flux_spectral_flow(L, phis)
        ax1.plot(x_phi, gs0, label=f"GS L={L}", color=color, lw=2)
        ax1.plot(x_phi, exc0, color=color, linestyle='--', alpha=0.5) # Excited state
        
        # --- Pi-Flux (Dirac Spin Liquid) ---
        gs_pi, exc_pi = get_spectral_states(L, phis)
        if L % 4 == 0:
            ax2.plot(x_phi, gs_pi, label=f"GS L={L}", color=color, lw=2)
            #ax2.plot(x_phi, exc_pi, color=color, linestyle='--', alpha=0.5) # Excited state
        elif L % 4 == 2:
            ax3.plot(x_phi, gs_pi, label=f"GS L={L}", color=color, lw=2)
            #ax3.plot(x_phi, exc_pi, color=color, linestyle='--', alpha=0.5) # Excited state

    # Formatting 0-Flux Plot
    ax1.set_title("0-Flux: Uniform Fermi Sea", fontsize=14)
    ax1.set_xlabel(r"Twist Angle $\phi$ ($\pi$ units)")
    ax1.set_ylabel("Energy per site (t units)")
    ax1.grid(True, linestyle='--', alpha=0.6)
    if len(ax1.lines) > 0:
        ax1.legend(title="Lattice Size")
    fig1.tight_layout()
    fig1.savefig("FreeFermions/plots/Spectral_flow/0_flux_spectral_flow.png", dpi=300)

    # Formatting Pi-Flux Plot (L = 4, 8, 12...)
    ax2.set_title(r"$\pi$-Flux: Dirac Spin Liquid ", fontsize=14)
    ax2.set_xlabel(r"Twist Angle $\phi$ ($\pi$ units)")
    ax2.set_ylabel("Energy per site (t units)")
    ax2.grid(True, linestyle='--', alpha=0.6)
    if len(ax2.lines) > 0:
        ax2.legend(title="Lattice Size")
    fig2.tight_layout()
    fig2.savefig("FreeFermions/plots/Spectral_flow/pi_flux_spectral_flow_L4n.png", dpi=300)

    # Formatting Pi-Flux Plot (L = 6, 10, 14...)
    ax3.set_title(r"$\pi$-Flux: Dirac Spin Liquid", fontsize=14)
    ax3.set_xlabel(r"Twist Angle $\phi$ ($\pi$ units)")
    ax3.set_ylabel("Energy per site (t units)")
    ax3.grid(True, linestyle='--', alpha=0.6)
    if len(ax3.lines) > 0:
        ax3.legend(title="Lattice Size")
    fig3.tight_layout()
    fig3.savefig("FreeFermions/plots/Spectral_flow/pi_flux_spectral_flow_L4n_plus_2.png", dpi=300)
    


def get_mf_correlation_length(L, phi, flux_type='pi'):
    n_sites = L * L
    def idx(x, y): return (x % L) * L + (y % L)
    
    # 1. Build Hamiltonian
    h_matrix = np.zeros((n_sites, n_sites), dtype=complex)
    for x in range(L):
        for y in range(L):
            # Vertical
            h_matrix[idx(x, y), idx(x, y+1)] = -1.0
            h_matrix[idx(x, y+1), idx(x, y)] = -1.0
            # Horizontal
            t_horiz = -1.0 * ((-1)**y if flux_type == 'pi' else 1.0)
            if x == L - 1:
                h_matrix[idx(x, y), idx(0, y)] = t_horiz * np.exp(1j * phi)
                h_matrix[idx(0, y), idx(x, y)] = np.conj(t_horiz * np.exp(1j * phi))
            else:
                h_matrix[idx(x, y), idx(x+1, y)] = t_horiz
                h_matrix[idx(x+1, y), idx(x, y)] = t_horiz

    # 2. Diagonalize and get Correlation Matrix P
    vals, vecs = np.linalg.eigh(h_matrix)
    # Occupation: fill the lower half
    occ_vecs = vecs[:, :n_sites // 2]
    P = occ_vecs @ occ_vecs.conj().T
    
    # 3. Measure correlations along x-axis (y=0)
    r_vals = np.arange(1, L)
    corrs = []
    origin = idx(0, 0)
    for r in r_vals:
        # We take the absolute square to mimic spin correlations
        val = np.abs(P[origin, idx(r, 0)])**2
        corrs.append(val)
    
    # 4. Periodic Fit with Robustness
    def periodic_model(r, A, xi):
        # Using a very small epsilon to prevent division by zero if xi is tiny
        return A * (np.exp(-r/xi) + np.exp(-(L-r)/xi))

    # Lower the initial guess if you expect very short correlations
    initial_guess = [corrs[0], 1.0]
    
    # Change the 0.1 to something much smaller, like 1e-6, or even 0
    try:
        popt, _ = curve_fit(
            periodic_model, 
            r_vals, 
            corrs, 
            p0=initial_guess, 
            # Lower bound for xi is now 1e-6 instead of 0.1
            bounds=((0, 1e-6), (np.inf, L * 5)), 
            maxfev=10000 # Increased for better convergence at small xi
        )
        return popt[1]
    except RuntimeError:
        # If it still fails, it's usually because xi is effectively infinite
        # (the system is gapless). You can return a large value or NaN.
        return np.nan


def plot_1_over_Correlation_lengths(L_values, phi_range):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Convert phi_range to units of pi for the x-axis
    x_phi = phi_range 

    for L in L_values:
        # Calculate 1/xi. Note: if xi is np.nan or very large, 1/xi correctly approaches 0.
        xi_0_raw = np.array([get_mf_correlation_length(L, p, '0') for p in phi_range])
        xi_pi_raw = np.array([get_mf_correlation_length(L, p, 'pi') for p in phi_range])
        
        # Handle potential NaNs or division by zero gracefully
        inv_xi_0 = 1.0 / xi_0_raw
        inv_xi_pi = 1.0 / xi_pi_raw

        ax1.plot(x_phi, inv_xi_0, marker='o', markersize=4, label=f'L={L}')
        ax2.plot(x_phi, inv_xi_pi, marker='o', markersize=4, label=f'L={L}')

    # Formatting Ax1 (0-Flux)
    ax1.set_title(r"0-Flux: Inverse Correlation Length $1/\xi$")
    ax1.set_ylabel(r"$1/\xi$ (Inverse lattice units)")
    ax1.set_xlabel(r"Twist $\phi$ ($\pi$ units)")
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend()

    # Formatting Ax2 (Pi-Flux)
    ax2.set_title(r"$\pi$-Flux: Inverse Correlation Length $1/\xi$")
    ax2.set_ylabel(r"$1/\xi$ (Inverse lattice units)")
    ax2.set_xlabel(r"Twist $\phi$ ($\pi$ units)")
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend()

    plt.tight_layout()
    # Updated filename to reflect the change
    plt.savefig("FreeFermions/plots/Correlation_lengths/inv_correlation_length_comparison.png", dpi=300)
    

def get_second_moment_inv_xi(L, phi, flux_type='pi'):
    n_sites = L * L
    def idx(x, y): return (x % L) * L + (y % L)
    
    # 1. Build Hamiltonian
    h_matrix = np.zeros((n_sites, n_sites), dtype=complex)
    for x in range(L):
        for y in range(L):
            # Vertical hopping
            h_matrix[idx(x, y), idx(x, y+1)] = -1.0
            h_matrix[idx(x, y+1), idx(x, y)] = -1.0
            
            # Horizontal hopping with flux
            t_horiz = -1.0 * ((-1)**y if flux_type == 'pi' else 1.0)
            if x == L - 1:
                h_matrix[idx(x, y), idx(0, y)] = t_horiz * np.exp(1j * phi)
                h_matrix[idx(0, y), idx(x, y)] = np.conj(t_horiz * np.exp(1j * phi))
            else:
                h_matrix[idx(x, y), idx(x+1, y)] = t_horiz
                h_matrix[idx(x+1, y), idx(x, y)] = t_horiz

    # 2. Get Correlation Matrix P (Ground State)
    vals, vecs = np.linalg.eigh(h_matrix)
    occ_vecs = vecs[:, :n_sites // 2]
    P = occ_vecs @ occ_vecs.conj().T
    
    # 3. Calculate Structure Factor S(q)
    # We use Q = (pi, pi) for the Antiferromagnetic peak
    # and k_min = (2*pi/L, 0) for the nearest neighbor momentum
    Q = np.array([np.pi, np.pi])
    k_min = np.array([2 * np.pi / L, 0])
    
    sq_peak = 0j
    sq_neighbor = 0j
    
    # Use site (0,0) as reference and sum over all r = (dx, dy)
    for dx in range(L):
        for dy in range(L):
            # For spin systems, correlation is roughly |P_ij|^2
            # We add the (-1)^(dx+dy) to account for the AF staggering
            corr = ((-1)**(dx+dy)) * np.abs(P[idx(0,0), idx(dx,dy)])**2
            
            r_vec = np.array([dx, dy])
            sq_peak += corr * np.exp(1j * np.dot(Q, r_vec))
            sq_neighbor += corr * np.exp(1j * np.dot(Q + k_min, r_vec))
    
    # 4. Apply Second Moment Formula
    sq_peak = np.abs(sq_peak)
    sq_neighbor = np.abs(sq_neighbor)
    
    # Formula: xi = 1/(2*sin(pi/L)) * sqrt(S(Q)/S(Q+k) - 1)
    ratio = sq_peak / sq_neighbor
    
    # If ratio < 1 (no peak), the system is extremely gapped
    if ratio <= 1.0:
        return 10.0 # Return a high value for 1/xi
    
    xi_2nd = (1.0 / (2.0 * np.sin(np.pi / L))) * np.sqrt(ratio - 1.0)
    
    return 1.0 / xi_2nd

def plot_second_moment_comparison(L_values, phi_range):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    x_phi = phi_range / np.pi

    for L in L_values:
        print(f"Calculating for L={L}...")
        inv_xi_0 = [get_second_moment_inv_xi(L, p, '0') for p in phi_range]
        inv_xi_pi = [get_second_moment_inv_xi(L, p, 'pi') for p in phi_range]
        
        ax1.plot(x_phi, inv_xi_0, marker='o', markersize=4, label=f'L={L}')
        ax2.plot(x_phi, inv_xi_pi, marker='o', markersize=4, label=f'L={L}')

    # Formatting
    ax1.set_title(r"0-Flux: $1/\xi_{2nd}$ (Fermi Surface)", fontsize=14)
    ax1.set_ylabel(r"Inverse Correlation Length $1/\xi_{2nd}$")
    ax1.set_xlabel(r"Twist Angle $\phi$ ($\pi$ units)")
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend()

    ax2.set_title(r"$\pi$-Flux: $1/\xi_{2nd}$ (Dirac Cones)", fontsize=14)
    ax2.set_xlabel(r"Twist Angle $\phi$ ($\pi$ units)")
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend()

    plt.tight_layout()
    plt.savefig("FreeFermions/plots/Correlation_lengths/inv_xi_2nd_moment_comparison.png", dpi=300)
    

def plot_structure_factor(L, phi, flux_type='pi'):
    n_sites = L * L
    def idx(x, y): return (x % L) * L + (y % L)
    
    # 1. Build Hamiltonian
    h_matrix = np.zeros((n_sites, n_sites), dtype=complex)
    for x in range(L):
        for y in range(L):
            h_matrix[idx(x, y), idx(x, y+1)] = -1.0
            h_matrix[idx(x, y+1), idx(x, y)] = -1.0
            t_horiz = -1.0 * ((-1)**y if flux_type == 'pi' else 1.0)
            if x == L - 1:
                h_matrix[idx(x, y), idx(0, y)] = t_horiz * np.exp(1j * phi)
                h_matrix[idx(0, y), idx(x, y)] = np.conj(t_horiz * np.exp(1j * phi))
            else:
                h_matrix[idx(x, y), idx(x+1, y)] = t_horiz
                h_matrix[idx(x+1, y), idx(x, y)] = t_horiz

    # 2. Get Correlation Matrix P (Ground State)
    vals, vecs = np.linalg.eigh(h_matrix)
    occ_vecs = vecs[:, :n_sites // 2]
    P = occ_vecs @ occ_vecs.conj().T
    
    # 3. Calculate Structure Factor S(q) via 2D FFT
    corr_r = np.zeros((L, L))
    for dx in range(L):
        for dy in range(L):
            # True physical spin-spin correlation for free fermions via Wick's theorem:
            # C(r) = 3/4 * delta_{r,0} - 3/2 * |P_{0,r}|^2
            P_sq = np.abs(P[idx(0, 0), idx(dx, dy)])**2
            if dx == 0 and dy == 0:
                corr_r[dx, dy] = 0.75 - 1.5 * P_sq
            else:
                corr_r[dx, dy] = -1.5 * P_sq
            
    S_q = np.abs(np.fft.fft2(corr_r))
    
    # Wrap around for a fully periodic plot in momentum space
    S_q_periodic = np.zeros((L+1, L+1))
    S_q_periodic[:L, :L] = S_q
    S_q_periodic[L, :L] = S_q[0, :]
    S_q_periodic[:L, L] = S_q[:, 0]
    S_q_periodic[L, L] = S_q[0, 0]
    
    # 4. Plot
    fig, ax = plt.subplots(figsize=(6, 5))
    c = ax.imshow(S_q_periodic, origin='lower', cmap='viridis', extent=[0, 2*np.pi, 0, 2*np.pi])
    fig.colorbar(c, label='|S(q)|')
    ax.set_xlabel(r'$q_x$')
    ax.set_ylabel(r'$q_y$')
    ax.set_title(f'Structure Factor S(q) [{flux_type}-flux, L={L}, $\phi$={phi/np.pi:.2f}$\pi$]')
    ax.set_xticks([0, np.pi, 2*np.pi], ['0', r'$\pi$', r'$2\pi$'])
    ax.set_yticks([0, np.pi, 2*np.pi], ['0', r'$\pi$', r'$2\pi$'])
    fig.tight_layout()
    fig.savefig(f"FreeFermions/plots/Correlation_lengths/structure_factor_L{L}_{flux_type}flux.png", dpi=300)

def plot_spin_stiffness(L_list):
    """
    Calculates and plots the spin stiffness (helicity modulus) for different system sizes.
    The stiffness is computed as the curvature of the ground state energy per site
    with respect to a boundary twist angle phi, evaluated at phi=0.
    rho_s = d^2(E_gs/N) / d(phi^2) | at phi=0
    """
    stiffness_0_flux = []
    stiffness_pi_flux = []

    # A small range of phi around 0 to perform the quadratic fit
    phi_fit_range = np.linspace(-0.05, 0.05, 21) # Small range for accurate curvature at phi=0

    print("Calculating spin stiffness...")
    for L in L_list:
        print(f"  L = {L}")
        # --- 0-Flux Case ---
        gs_energies_0, _ = get_0_flux_spectral_flow(L, phi_fit_range)
        # Fit E(phi) = a*phi^2 + b*phi + c. The stiffness is 2*a.
        coeffs_0 = np.polyfit(phi_fit_range, gs_energies_0, 2)
        stiffness_0_flux.append(2 * coeffs_0[0])

        # --- Pi-Flux Case ---
        gs_energies_pi, _ = get_spectral_states(L, phi_fit_range)
        coeffs_pi = np.polyfit(phi_fit_range, gs_energies_pi, 2)
        stiffness_pi_flux.append(2 * coeffs_pi[0])

    # --- Plotting ---
    fig, ax = plt.subplots(figsize=(8, 6))
    
    inv_L = [1/L for L in L_list]

    ax.plot(inv_L, stiffness_0_flux, marker='o', linestyle='-', label='0-Flux (Fermi Sea)')
    ax.plot(inv_L, stiffness_pi_flux, marker='s', linestyle='--', label=r'$\pi$-Flux (Dirac Liquid)')

    ax.axhline(0, color='grey', lw=1, linestyle=':')
    ax.set_title("Spin Stiffness vs. Inverse System Size", fontsize=14)
    ax.set_xlabel(r"$1/L$")
    ax.set_ylabel(r"Spin Stiffness $\rho_s$")
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend()
    fig.tight_layout()
    fig.savefig("FreeFermions/plots/Stiffness/spin_stiffness.png", dpi=300)
    print("Spin stiffness plot saved to FreeFermions/plots/Stiffness/spin_stiffness.png")

def plot_energy_gap_vs_L_fit(L_list, target_phi, flux_type='pi'):
    """
    Calculates the energy gap E(0)-E(phi) for different system sizes L,
    plots it against 1/L, and fits it with a power law.
    """
    print(f"\nCalculating absolute energy gap for phi={target_phi/np.pi:.2f}pi, flux={flux_type}")
    
    inv_L_vals = []
    gaps = []

    for L in L_list:
        if flux_type == 'pi':
            gs_energies, _ = get_spectral_states(L, [0, target_phi])
        else: # '0' flux
            gs_energies, _ = get_0_flux_spectral_flow(L, [0, target_phi])
        
        e0 = gs_energies[0]
        e_phi = gs_energies[1]
        gap = np.abs(e0 - e_phi)
        
        inv_L_vals.append(1/L)
        gaps.append(gap)
        print(f"  L={L}: E(0)={e0:.6f}, E(phi)={e_phi:.6f}, Absolute Gap={gap:.6f}")

    inv_L_arr = np.array(inv_L_vals)
    gaps_arr = np.array(gaps)

    # --- Fitting ---
    def fit_func(x, a, n, b):
        return a * x**n + b

    def fit_func_exp(x, a, c, b):
        x_safe = np.where(x == 0, 1e-10, x)
        return a * np.exp(-1.0 / (x_safe * c)) + b

    popt = None
    popt_exp = None
    err_pl = float('inf')
    err_exp = float('inf')
    inv_L_fine = np.linspace(0, max(inv_L_arr) * 1.1, 100)
    
    if len(inv_L_arr) >= 3:
        try:
            popt, _ = curve_fit(fit_func, inv_L_arr, gaps_arr, p0=[gaps_arr[0], 2.0, 0.0], maxfev=5000)
            a, n_fit, b = popt
            fit_label = f'Power-law $y = a(1/L)^n + b$ ($n={n_fit:.3f}$)'
            gap_fit = fit_func(inv_L_fine, *popt)
            err_pl = np.sum((gaps_arr - fit_func(inv_L_arr, *popt))**2)
        except Exception as e:
            print(f"Power-law fit failed: {e}")
            
        try:
            popt_exp, _ = curve_fit(fit_func_exp, inv_L_arr, gaps_arr, p0=[gaps_arr[0], 5.0, 0.0], 
                                    bounds=([-np.inf, 1e-5, -np.inf], [np.inf, np.inf, np.inf]), maxfev=5000)
            a_exp, c_exp, b_exp = popt_exp
            fit_label_exp = f'Exponential $y = a e^{{-L/\\xi}} + b$ ($\\xi={c_exp:.3f}$)'
            gap_fit_exp = fit_func_exp(inv_L_fine, *popt_exp)
            err_exp = np.sum((gaps_arr - fit_func_exp(inv_L_arr, *popt_exp))**2)
        except Exception as e:
            print(f"Exponential fit failed: {e}")

    use_pl = False
    use_exp = False
    if popt is not None and popt_exp is not None:
        if err_pl <= err_exp:
            use_pl = True
        else:
            use_exp = True
    elif popt is not None:
        use_pl = True
    elif popt_exp is not None:
        use_exp = True

    # --- Plotting ---
    plt.figure(figsize=(9, 6))
    plt.plot(inv_L_arr, gaps_arr, 'o', color='tab:red', markersize=8, label='Data')

    if use_pl:
        plt.plot(inv_L_fine, gap_fit, '--', color='black', label=fit_label)
        plt.plot(0, popt[2], '*', color='blue', markersize=12, markeredgecolor='black', label=f'PL Extrapolated: {popt[2]:.4f}')
    elif use_exp:
        plt.plot(inv_L_fine, gap_fit_exp, ':', color='tab:green', linewidth=2, label=fit_label_exp)
        plt.plot(0, popt_exp[2], 'X', color='tab:green', markersize=10, markeredgecolor='black', label=f'Exp Extrapolated: {popt_exp[2]:.4f}')

    plt.xlabel('1/L', fontsize=14)
    plt.ylabel(f'Absolute Energy Gap per site $|E(0) - E(\phi)|$', fontsize=14)
    plt.title(f'Free Fermion Energy Gap vs 1/L ({flux_type}-flux, $\phi$={target_phi/np.pi:.2f}$\pi$)', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    plt.xlim(left=0)
    plt.tight_layout()
    
    save_dir = "FreeFermions/plots/Gap_scaling"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"gap_vs_invL_{flux_type}_phi{target_phi/np.pi:.1f}.png")
    plt.savefig(save_path, dpi=300)
    print(f"Plot saved to {save_path}")
    plt.show()

if __name__ == "__main__":

    L = 6 
    
    # --- Twisting phi for pi-flux and 0-flux-------------------------------------------------------
    phis = np.linspace(0, 2*np.pi, 20)
    #gspi, excpi = get_spectral_states(L, phis)
    #gs0, exc0 = get_0_flux_spectral_flow(L, phis)

    # --- plot 0,pi for Ls twist phi-------------------------------------------------------
    phis = np.linspace(0, 2*np.pi, 100)
    L_values = [4, 6,8, 10] 
    #plot_side_by_side(L_values, phis)


    # ------------pi flux dispersion kx ky----------------------------------------------------------------------
    plot_pi_flux_dispersion()
    plot_dirac_cuts(L=6, phi=0.5*np.pi)

    #------------0 flux dispersion kx ky----------------------------------------------------------------------
    #plot_fermi_surface_dispersion()
    #plot_0_flux_cuts(L=8, phi=0.0)


    #--- Correllation length ------------------------
    L = 8
    phi = 0
    #xi_0 = get_mf_correlation_length(L, phi, '0')
    #xi_pi = get_mf_correlation_length(L, phi, 'pi')

    L_values = [4,6,8,10]
    #phi_range = np.linspace(0, 2*np.pi, 100)
    #plot_1_over_Correlation_lengths(L_values, phi_range)

    L_values = [4,6,8,10]
    #phis = np.linspace(0, 2*np.pi, 30)
    #plot_second_moment_comparison(L_values, phis)    

    #--- Structure factor S(q) ------------------------
    L = 10
    phi = np.pi
    #plot_structure_factor(L, phi, '0')
    #plot_structure_factor(L, phi, 'pi')

    #--- Spin Stiffness -----------------------------
    os.makedirs("FreeFermions/plots/Stiffness", exist_ok=True)
    L_stiffness = [4, 6, 8, 10, 12, 14, 16, 20, 24, 50, 100 ]
    #plot_spin_stiffness(L_stiffness)

    #--- Gap scaling vs 1/L ---
    L_gap_scaling = [6,10, 14, 18, 22]
    target_phi_gap = 1 * np.pi
    #plot_energy_gap_vs_L_fit(L_gap_scaling, target_phi_gap, flux_type='pi')
    #plot_energy_gap_vs_L_fit(L_gap_scaling, target_phi_gap, flux_type='0')

    