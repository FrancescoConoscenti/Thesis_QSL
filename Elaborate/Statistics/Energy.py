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

def plot_energy_diff_log(energy_per_site, E_exact, E_last, folder):

    if E_exact is not None:
        plt.figure()
        diff = np.abs(energy_per_site - E_exact)
        plt.plot(diff)
        plt.title("Energy Error vs Iterations")
        plt.xlabel("Iterations")
        plt.ylabel("|E - E_exact|")
        plt.yscale("log")
        plt.tight_layout()
        plt.savefig(f'{folder}/Energy_plot/Energy_error_log.png')
        plt.close()

    if E_last is not None:
        plt.figure()
        diff_last = np.abs(E_last - E_last)
        plt.plot(diff_last)
        plt.title("Final Energy Error vs Iterations")
        plt.xlabel("Iterations")
        plt.ylabel("|E_final - E_last|")
        plt.yscale("log")
        plt.tight_layout()
        plt.savefig(f'{folder}/Energy_plot/Final_Energy_error_log.png')
        plt.close()
  
def Energy(log, L):
    # Divide by 4 to convert from Pauli matrix units (sigma) to Spin-1/2 units (S).
    # S = sigma/2 => S*S = 1/4 * sigma*sigma.
    energy_per_iterations = log.data["Energy"]["Mean"].real / (L * L * 4)
    E_vs = energy_per_iterations[-1]
    #print("Last value: ", energy_per_iterations[-1])
    return E_vs, energy_per_iterations

def plot_energy(folder, energy_per_iterations, E_exact=None, E_last=None):

    plt.figure()
    plt.plot(energy_per_iterations)
    plt.title("Energy per site vs Iterations")
    plt.xlabel("Iterations")
    plt.ylabel("Energy per site")
    plt.savefig(f'{folder}/Energy_plot/Energy.png')
    plt.close()

    if E_exact is not None:
        plot_energy_diff_log(energy_per_iterations, E_exact, E_last, folder)


def get_initial_energy(log, L):
    """Computes the energy of the initial (pre-optimization) variational state."""
    # Normalize by number of sites (L*L) and factor of 4 for Pauli -> Spin conversion
    energy_per_site = log.data["Energy"]["Mean"].real / (L * L * 4)
    initial_energy_val = energy_per_site[0] # Normalize by number of sites
    print("Initial energy: ", initial_energy_val)
    return initial_energy_val 

def Exact_gs_en_6x6(J2):
    
    file_path = f"/scratch/f/F.Conoscenti/Thesis_QSL/ED/energy_J2_{J2}.txt"
    if not os.path.exists(file_path):
        file_path = f"/cluster/home/fconoscenti/Thesis_QSL/ED/energy_J2_{J2}.txt"

    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            line = f.readline()
            return float(line.split(":")[-1].strip())
    else:
        print(f"Exact energy file not found: {file_path}")
        return None
