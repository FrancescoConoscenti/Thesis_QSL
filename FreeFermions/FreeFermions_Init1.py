import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.optimize import curve_fit

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

    plt.figure(figsize=(8, 6))
    plt.plot(phis/np.pi, np.array(gs_energies), label="GS (0-Flux)", color='blue')
    plt.plot(phis/np.pi, np.array(excited_energies), label="Excited (0-Flux)", color='red', linestyle='--')
    plt.title(f"0-Flux Fermi Sea (L={L})")
    plt.ylabel("Energy per site (t units)")
    plt.xlabel(r"Twist Angle $\phi$ ($\pi$ units)")
    plt.legend()
    plt.savefig(f"FreeFermions/plots/Spectral_flow/0_flux_spectral_flow_L{L}.png", dpi=300)
    plt.show()

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
    plt.show()

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
    # We plot lines along ky for each discrete kx
    nx_values = np.arange(L)
    ky_continuous = np.linspace(-np.pi, np.pi, 200)

    for nx in nx_values:
        kx_val = (2 * np.pi * nx + phi) / L
        # Normalize kx to be within [-pi, pi] for the plot
        kx_plot = ((kx_val + np.pi) % (2 * np.pi)) - np.pi

        # Calculate energy along this cut
        e_line = 2 * np.sqrt(np.cos(kx_plot)**2 + np.cos(ky_continuous)**2)
        
        # Draw the lines on the cones
        ax.plot([kx_plot]*len(ky_continuous), ky_continuous, e_line, color='red', lw=2, alpha=0.8)
        ax.plot([kx_plot]*len(ky_continuous), ky_continuous, -e_line, color='red', lw=2, alpha=0.8)

    # 3. Mark the discrete points (kx, ky) actually sampled by the LxL lattice
    for nx in range(L):
        for ny in range(L):
            kx_pt = (( (2 * np.pi * nx + phi) / L + np.pi) % (2 * np.pi)) - np.pi
            ky_pt = (( (2 * np.pi * ny) / L + np.pi) % (2 * np.pi)) - np.pi
            e_pt = 2 * np.sqrt(np.cos(kx_pt)**2 + np.cos(ky_pt)**2)
            
            ax.scatter([kx_pt], [ky_pt], [e_pt], color='black', s=20)
            ax.scatter([kx_pt], [ky_pt], [-e_pt], color='black', s=20)

    ax.set_title(f"Dirac Cone Sampling (L={L}, $\phi$={phi/np.pi:.2f}$\pi$)")
    ax.set_xlabel(r"$k_x$")
    ax.set_ylabel(r"$k_y$")
    ax.view_init(elev=25, azim=45)
    plt.savefig(f"FreeFermions/plots/Dispersions/dirac_cuts_L{L}_phi{phi/np.pi:.1f}.png", dpi=300)
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

    
    plt.figure(figsize=(8, 6))
    plt.plot(phis/np.pi, np.array(gs_energies), label="Ground State (GS)", color='blue', lw=2)
    plt.plot(phis/np.pi, np.array(excited_energies), label="First Excited State", color='red', linestyle='--', lw=2)
    plt.title(f"pi-flux Fermi Sea (L={L})")
    plt.xlabel(r"Twist Angle $\phi$ ($\pi$ units)")
    plt.ylabel("Energy per site (t units)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(f"FreeFermions/plots/Spectral_flow/0_flux_spectral_flow_L{L}.png", dpi=300)
    plt.show()

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
    plt.show()

def plot_side_by_side(L_list, phis):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), sharey=False)
    x_phi = phis / np.pi
    
    # Colors for different L sizes
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray']
    
    for i, L in enumerate(L_list):
        color = colors[i % len(colors)]
        
        # --- Left Plot: 0-Flux (Standard Fermi Sea) ---
        gs0, exc0 = get_0_flux_spectral_flow(L, phis)
        ax1.plot(x_phi, gs0, label=f"GS L={L}", color=color, lw=2)
        ax1.plot(x_phi, exc0, color=color, linestyle='--', alpha=0.5) # Excited state
        
        # --- Right Plot: Pi-Flux (Dirac Spin Liquid) ---
        gs_pi, exc_pi = get_spectral_states(L, phis)
        ax2.plot(x_phi, gs_pi, label=f"GS L={L}", color=color, lw=2)
        ax2.plot(x_phi, exc_pi, color=color, linestyle='--', alpha=0.5) # Excited state

    # Formatting Left Plot
    ax1.set_title("0-Flux: Uniform Fermi Sea", fontsize=14)
    ax1.set_xlabel(r"Twist Angle $\phi$ ($\pi$ units)")
    ax1.set_ylabel("Energy per site (t units)")
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend(title="Lattice Size")

    # Formatting Right Plot
    ax2.set_title(r"$\pi$-Flux: Dirac Spin Liquid", fontsize=14)
    ax2.set_xlabel(r"Twist Angle $\phi$ ($\pi$ units)")
    ax2.set_ylabel("Energy per site (t units)")
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend(title="Lattice Size")

    plt.tight_layout()
    plt.savefig("FreeFermions/plots/Spectral_flow/side_by_side_spectral_flow.png", dpi=300)
    plt.show()


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
    plt.show()

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
    plt.show()


if __name__ == "__main__":

    L = 6 
    
    # --- Twisting phi for pi-flux and 0-flux-------------------------------------------------------
    phis = np.linspace(0, 2*np.pi, 20)
    #gspi, excpi = get_spectral_states(L, phis)
    #gs0, exc0 = get_0_flux_spectral_flow(L, phis)

    # --- plot 0,pi for Ls twist phi-------------------------------------------------------
    phis = np.linspace(0, 2*np.pi, 100)
    L_values = [8, 10, 12 ,14, 16] 
    #plot_side_by_side(L_values, phis)


    # ------------pi flux dispersion kx ky----------------------------------------------------------------------
    #plot_pi_flux_dispersion()
    #plot_dirac_cuts(L=10, phi=3.14)

    #------------0 flux dispersion kx ky----------------------------------------------------------------------
    #plot_fermi_surface_dispersion()
    #plot_0_flux_cuts(L=8, phi=0.0)


    #--- Correllation length ------------------------
    L = 8
    phi = 0
    xi_0 = get_mf_correlation_length(L, phi, '0')
    xi_pi = get_mf_correlation_length(L, phi, 'pi')

    L_values = [4,6,8,10]
    phi_range = np.linspace(0, 2*np.pi, 100)
    plot_1_over_Correlation_lengths(L_values, phi_range)

    L_values = [4,6,8,10]
    phis = np.linspace(0, 2*np.pi, 30)
    plot_second_moment_comparison(L_values, phis)    
