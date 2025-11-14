import matplotlib.pyplot as plt
import os
import numpy as np

def plot_DMRG_energies(energies, bond_dims, sweeps, model_params):
    """
    Plots DMRG energy convergence and saves the figure.
    """
    Lx = model_params.get('Lx', 'N/A')
    J2 = model_params.get('J2', 'N/A')

    # --- Plot convergence with bond dimension annotations ---
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(sweeps, energies, 'o-', label='Energy')

    # Annotate each point with the bond dimension
    for i, E in enumerate(energies):
        ax.annotate(f'χ={bond_dims[i]}', (sweeps[i], E), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=8)

    ax.set_xlabel('Sweep')
    ax.set_ylabel('Energy per Site')
    ax.set_title(f'DMRG Energy Convergence (L={Lx} J2={J2})')
    ax.grid(True)
    
    model_name = f"DMRG_L{Lx}.png"
    output_dir = f"DMRG/plot/{model_name}/J={J2}"
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "Energies")
    fig.savefig(save_path, dpi=300)
    print(f"✅ DMRG plot saved to {save_path}")
    plt.show()

    print(f"Final ground state energy: {energies[-1]:.6f}")

def plot_correlation_function(corr_r, model_params):
    """Plots the real-space spin-spin correlation function C(r)."""
    Lx = model_params.get('Lx', 'N/A')
    Ly = model_params.get('Ly', 'N/A')
    J2 = model_params.get('J2', 'N/A')

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(corr_r.real, origin='lower', cmap='viridis')
    fig.colorbar(im, ax=ax, label="C(r)")
    ax.set_xlabel("dx", fontsize=12)
    ax.set_ylabel("dy", fontsize=12)
    ax.set_title(f"Spin-Spin Correlation C(r) (Lx={Lx}, Ly={Ly}, J2={J2})", fontsize=14)
    ax.set_xticks(np.arange(Lx)) # Ensure ticks are correct for Lx
    ax.set_yticks(np.arange(Ly)) # Ensure ticks are correct for Ly
    plt.tight_layout()

    filename = f"DMRG_L{Lx}.png"
    output_dir = f"DMRG/plot/{filename}/J={J2}"
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "Correlation")
    fig.savefig(save_path, dpi=300)
    plt.close(fig)

def plot_structure_factor(S_q, model_params):
    """Plots the static spin structure factor S(q)."""
    Lx = model_params.get('Lx', 'N/A')
    Ly = model_params.get('Ly', 'N/A')
    J2 = model_params.get('J2', 'N/A')

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(np.abs(S_q), origin='lower', cmap='viridis')
    fig.colorbar(im, ax=ax, label="|S(q)|")
    ax.set_xlabel(r"$q_x$", fontsize=12)
    ax.set_ylabel(r"$q_y$", fontsize=12)
    ax.set_title(f"Structure Factor S(q) (Lx={Lx}, Ly={Ly}, J2={J2})", fontsize=14)
    ax.set_xticks([0, Lx // 2, Lx - 1])
    ax.set_xticklabels(['0', r'$\pi$', r'$2\pi$'])
    ax.set_yticks([0, Ly // 2, Ly - 1])
    ax.set_yticklabels(['0', r'$\pi$', r'$2\pi$'])
    plt.tight_layout()

    filename = f"DMRG_L{Lx}.png"
    output_dir = f"DMRG/plot/{filename}/J={J2}"
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "Structure_Factor")
    fig.savefig(save_path, dpi=300)
    plt.close(fig)