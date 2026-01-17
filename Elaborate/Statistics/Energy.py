import matplotlib.pyplot as plt
import numpy as np
import os

def plot_energy_log_abs(energy_per_site, folder):
    plt.figure()
    plt.plot(np.abs(energy_per_site))
    plt.title("Absolute Energy per site vs Iterations")
    plt.xlabel("Iterations")
    plt.ylabel("|Energy per site|")
    plt.yscale("log")
    plt.savefig(f'{folder}/Energy_abs_log.png')
    plt.close()

def plot_energy_diff_log(energy_per_site, E_exact, folder):
    plt.figure()
    diff = np.abs(energy_per_site - E_exact)
    plt.plot(diff)
    plt.title("Energy Error vs Iterations")
    plt.xlabel("Iterations")
    plt.ylabel("|E - E_exact|")
    plt.yscale("log")
    plt.tight_layout()
    plt.savefig(f'{folder}/Energy_error_log.png')
    plt.close()

def Energy(log, L, folder, E_exact=None):
    # Divide by 4 to convert from Pauli matrix units (sigma) to Spin-1/2 units (S).
    # S = sigma/2 => S*S = 1/4 * sigma*sigma.
    energy_per_site = log.data["Energy"]["Mean"].real / (L * L * 4)
    E_vs = energy_per_site[-1]
    print("Last value: ", energy_per_site[-1])

    plt.figure()
    plt.plot(energy_per_site)
    plt.title("Energy per site vs Iterations")
    plt.xlabel("Iterations")
    plt.ylabel("Energy per site")
    plt.savefig(f'{folder}/Energy.png')
    plt.close()

    plot_energy_log_abs(energy_per_site, folder)

    if E_exact is not None:
        plot_energy_diff_log(energy_per_site, E_exact, folder)

    return E_vs

def get_initial_energy(log, L):
    """Computes the energy of the initial (pre-optimization) variational state."""
    # Normalize by number of sites (L*L) and factor of 4 for Pauli -> Spin conversion
    energy_per_site = log.data["Energy"]["Mean"].real / (L * L * 4)
    initial_energy_val = energy_per_site[0] # Normalize by number of sites
    print("Initial energy: ", initial_energy_val)
    return initial_energy_val 

def Exact_gs_en_6x6(J2):
    
    file_path = f"/scratch/f/F.Conoscenti/Thesis_QSL/ED/energy_J2_{J2}.txt"
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            line = f.readline()
            return float(line.split(":")[-1].strip())
    else:
        print(f"Exact energy file not found: {file_path}")
        return None
