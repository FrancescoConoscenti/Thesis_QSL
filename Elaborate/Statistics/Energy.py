import matplotlib.pyplot as plt

def Energy(log, L, folder):
    # Divide by 4 to convert from Pauli matrix units (sigma) to Spin-1/2 units (S).
    # S = sigma/2 => S*S = 1/4 * sigma*sigma.
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
    # Normalize by number of sites (L*L) and factor of 4 for Pauli -> Spin conversion
    energy_per_site = log.data["Energy"]["Mean"].real / (L * L * 4)
    initial_energy_val = energy_per_site[0] # Normalize by number of sites
    print("Initial energy: ", initial_energy_val)
    return initial_energy_val 

def Exact_gs_en_6x6(J2):
    if J2==0.0:
        return -0.678872
    if J2==0.2:
        return -0.599046
    if J2==0.4:
        return -0.529745
    if J2==0.45:
        return -0.515655
    if J2==0.5:
        return -0.503810
    if J2==0.55:
        return -0.495178
    if J2==0.6:
        return -0.493239
    if J2==0.7:
        return -0.530001
    if J2==0.8:
        return -0.586487
    if J2==1.0:
        return -0.714360