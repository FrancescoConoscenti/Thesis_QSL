import matplotlib.pyplot as plt

def Energy(log, L, folder):
    energy_per_site = log.data["Energy"]["Mean"].real / (L * L * 4)
    E_vs = energy_per_site[-1]
    print("Last value: ", energy_per_site[-1])

    plt.plot(energy_per_site)
    plt.title("Energy per site vs Ierations")
    plt.xlabel("Iterations")
    plt.ylabel("Energy per site")
    plt.savefig(f'{folder}/Energy.png')
    plt.show()

    return E_vs