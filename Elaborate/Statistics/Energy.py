import matplotlib.pyplot as plt

def Energy(log, L, folder):
    energy_per_site = log.data["Energy"]["Mean"].real / (L * L * 4)
    E_vs = energy_per_site[-1]
    print("Last value: ", energy_per_site[-1])

    plt.plot(energy_per_site)
    plt.title("Energy per site vs Iterations")
    plt.xlabel("Iterations")
    plt.ylabel("Energy per site")
    plt.savefig(f'{folder}/Energy.png')
    plt.show()

    return E_vs

def get_initial_energy(log, L):
    """Computes the energy of the initial (pre-optimization) variational state."""
    energy_per_site = log.data["Energy"]["Mean"].real / (L * L * 4)
    initial_energy_val = energy_per_site[0] # Normalize by number of sites
    print("Initial energy: ", initial_energy_val)
    return initial_energy_val 