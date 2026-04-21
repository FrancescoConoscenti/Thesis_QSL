import os
import re
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import jax
import jax.numpy as jnp
import flax

if "/scratch/f/F.Conoscenti/Thesis_QSL" not in sys.path:
    sys.path.append("/scratch/f/F.Conoscenti/Thesis_QSL")
import netket as nk
from Observables import parse_model_path, setup_system, setup_model, load_vstate, save_variables

def plot_energy_vs_phi(base_dir, target_J=0.5):
    phi_values = []
    energies = []
    energy_errors = []

    base_path = Path(base_dir)
    if not base_path.exists():
        print(f"Error: Directory {base_dir} does not exist.")
        return

    print(f"Scanning {base_dir} for models with varying phi...")

    for model_dir in base_path.iterdir():
        if not model_dir.is_dir():
            continue
        
        # Extract phi value from folder name (e.g., ..._phi0.0_...)
        match = re.search(r'phi([\d\.]+)', model_dir.name)
        if match:
            phi = float(match.group(1))
        else:
            continue

        # Find the specific J folder
        target_j_folder = None
        for d in model_dir.iterdir():
            if d.is_dir() and (d.name.startswith("J=") or d.name.startswith("J2=")):
                try:
                    part = d.name.split('=')[1]
                    val_str = part.split('_')[0] if '_' in part else part
                    if abs(float(val_str) - target_J) < 1e-5:
                        target_j_folder = d
                        break
                except ValueError:
                    continue
        
        if not target_j_folder:
            print(f"Warning: Could not find J={target_J} folder in {model_dir.name}")
            continue

        # Read energies from seeds
        seed_energies = []
        for seed_dir in target_j_folder.iterdir():
            if seed_dir.is_dir() and "seed" in seed_dir.name:
                E_final = None
                var_path = seed_dir / "variables.pkl"
                
                # Attempt to load from variables.pkl
                if var_path.exists():
                    try:
                        with open(var_path, "rb") as f:
                            data = pickle.load(f)
                            if 'E_vs_final' in data:
                                E_final = float(np.real(data['E_vs_final']))
                    except Exception as e:
                        print(f"Error reading {var_path}: {e}")
                
                # Fallback to output.txt
                out_path = seed_dir / "output.txt"
                if E_final is None and out_path.exists():
                    try:
                        with open(out_path, "r") as f:
                            content = f.read()
                        matches = re.findall(r"Final Energy from VMC:\s*([-\d\.e]+)", content)
                        if matches:
                            E_final = float(matches[-1])
                    except Exception as e:
                        print(f"Error reading {out_path}: {e}")
                
                if E_final is not None:
                    seed_energies.append(E_final)

        if seed_energies:
            phi_values.append(phi)
            means_e = np.mean(seed_energies)
            std_e = np.std(seed_energies) / np.sqrt(len(seed_energies)) if len(seed_energies) > 1 else 0.0
            energies.append(means_e)
            energy_errors.append(std_e)
            print(f"Found phi = {phi}, E = {means_e:.6f} ± {std_e:.6f} ({len(seed_energies)} seeds)")

    if not phi_values:
        print("No data found to plot.")
        return

    # Sort by phi for a clean line plot
    sort_idx = np.argsort(phi_values)
    phi_values = np.array(phi_values)[sort_idx]
    energies = np.array(energies)[sort_idx]
    energy_errors = np.array(energy_errors)[sort_idx]

    # Plotting
    plt.figure(figsize=(8, 6))
    plt.errorbar(phi_values, energies, yerr=energy_errors, fmt='-o', color='tab:blue',
                 capsize=5, markersize=8, markeredgecolor='black', linewidth=2, label="VMC Energy")
    
    plt.xlabel(r'Twist angle $\phi$ (units of $\pi$)', fontsize=14)
    plt.ylabel('Final VMC Energy per site', fontsize=14)
    plt.title(f'Energy vs Boundary Twist $\phi$ (J={target_J})', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    
    # Save the plot
    save_dir = Path("/scratch/f/F.Conoscenti/Thesis_QSL/Elaborate/plot/Degeneracy")
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"Energy_vs_phi_J={target_J}.png"
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"\n✅ Plot saved successfully to {save_path}")
    plt.show()



########################################################################################################



def plot_adiabatic_energy_vs_phi(base_directory, target_J):
    phi_values = []
    energies = []
    energy_errors = []

    base_path = Path(base_directory)
    if not base_path.exists():
        print(f"Error: Directory {base_directory} does not exist.")
        return
        
    match_L = re.search(r'(\d+)x\d+', str(base_path))
    L = int(match_L.group(1)) if match_L else "Unknown"

    print(f"Scanning {base_directory} for models with varying phi...")

    for model_dir in base_path.iterdir():
        if not model_dir.is_dir():
            continue
        
        # Extract phi value from folder name (e.g., ..._phi0.0_...)
        match = re.search(r'phi([\d\.]+)', model_dir.name)
        if match:
            phi = float(match.group(1))
        else:
            continue

        # Find the specific J folder
        target_j_folder = None
        for d in model_dir.iterdir():
            if d.is_dir() and (d.name.startswith("J=") or d.name.startswith("J2=")):
                try:
                    part = d.name.split('=')[1]
                    val_str = part.split('_')[0] if '_' in part else part
                    if abs(float(val_str) - target_J) < 1e-5:
                        target_j_folder = d
                        break
                except ValueError:
                    continue
        
        if not target_j_folder:
            print(f"Warning: Could not find J={target_J} folder in {model_dir.name}")
            continue

        # Read energies from seeds
        seed_energies = []
        for seed_dir in target_j_folder.iterdir():
            if seed_dir.is_dir() and "seed" in seed_dir.name:
                E_final = None
                var_path = seed_dir / "variables.pkl"
                
                # Attempt to load from variables.pkl
                if var_path.exists():
                    try:
                        with open(var_path, "rb") as f:
                            data = pickle.load(f)
                            if 'E_vs_final' in data:
                                E_final = float(np.real(data['E_vs_final']))
                    except Exception as e:
                        print(f"Error reading {var_path}: {e}")
                
                # Fallback to output.txt
                out_path = seed_dir / "output.txt"
                if E_final is None and out_path.exists():
                    try:
                        with open(out_path, "r") as f:
                            content = f.read()
                        matches = re.findall(r"Final Energy from VMC:\s*([-\d\.e]+)", content)
                        if not matches:
                            matches = re.findall(r"Final Energy per site:\s*([-\d\.e]+)", content)
                        if matches:
                            E_final = float(matches[-1])
                    except Exception as e:
                        print(f"Error reading {out_path}: {e}")
                
                if E_final is not None:
                    seed_energies.append(E_final)

        if seed_energies:
            phi_values.append(phi)
            means_e = np.mean(seed_energies)
            std_e = np.std(seed_energies) / np.sqrt(len(seed_energies)) if len(seed_energies) > 1 else 0.0
            energies.append(means_e)
            energy_errors.append(std_e)
            print(f"Found phi = {phi}, Adiabatic E = {means_e:.6f} ± {std_e:.6f} ({len(seed_energies)} seeds)")

    if not phi_values:
        print("No data found to plot.")
        return
    # Sort by phi for a clean line plot
    sort_idx = np.argsort(phi_values)
    phi_values = np.array(phi_values)[sort_idx]
    energies = np.array(energies)[sort_idx]
    energy_errors = np.array(energy_errors)[sort_idx]      
    # Plotting
    plt.figure(figsize=(8, 6))
    plt.errorbar(phi_values, energies, yerr=energy_errors, fmt='-o', color='tab:orange',
                 capsize=5, markersize=8, markeredgecolor='black', linewidth=2, label="Adiabatic Energy")       
    plt.xlabel(r'Twist angle $\phi$ (units of $\pi$)', fontsize=14)
    plt.ylabel('Final Adiabatic Energy per site', fontsize=14)
    plt.title(f'Adiabatic Energy vs Boundary Twist $\phi$ L = {L} (J={target_J})', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    # Save the plot
    plt.savefig(f"Elaborate/plot/Degeneracy/adiabatic_energy_vs_phi_J{target_J}_L{L}.png", dpi=300, bbox_inches='tight')
    print(f"\n✅ Plot saved successfully to Elaborate/plot/Degeneracy/adiabatic_energy_vs_phi_J{target_J}_L{L}.png")
    plt.show()  

def plot_adiabatic_energy_vs_L(directories, target_J):
    data_by_phi = {}

    print(f"Scanning directories for models with varying phi to plot vs 1/L...")

    for base_dir in directories:
        base_path = Path(base_dir)
        if not base_path.exists():
            print(f"Warning: Directory {base_dir} does not exist.")
            continue
        
        # Extract L from path (e.g. "4x4", "6x6")
        match_L = re.search(r'(\d+)x\d+', str(base_path))
        if not match_L:
            print(f"Warning: Could not determine L from path {base_dir}")
            continue
        L = int(match_L.group(1))

        for model_dir in base_path.iterdir():
            if not model_dir.is_dir():
                continue
            
            # Extract phi value from folder name
            match = re.search(r'phi([\d\.]+)', model_dir.name)
            if match:
                phi = float(match.group(1))
            else:
                continue

            # Find the specific J folder
            target_j_folder = None
            for d in model_dir.iterdir():
                if d.is_dir() and (d.name.startswith("J=") or d.name.startswith("J2=")):
                    try:
                        part = d.name.split('=')[1]
                        val_str = part.split('_')[0] if '_' in part else part
                        if abs(float(val_str) - target_J) < 1e-5:
                            target_j_folder = d
                            break
                    except ValueError:
                        continue
            
            if not target_j_folder:
                continue

            # Read energies from seeds
            seed_energies = []
            for seed_dir in target_j_folder.iterdir():
                if seed_dir.is_dir() and "seed" in seed_dir.name:
                    E_final = None
                    var_path = seed_dir / "variables.pkl"
                    
                    if var_path.exists():
                        try:
                            with open(var_path, "rb") as f:
                                data = pickle.load(f)
                                if 'E_vs_final' in data:
                                    E_final = float(np.real(data['E_vs_final']))
                        except Exception as e:
                            pass
                    
                    out_path = seed_dir / "output.txt"
                    if E_final is None and out_path.exists():
                        try:
                            with open(out_path, "r") as f:
                                content = f.read()
                            matches = re.findall(r"Final Energy from VMC:\s*([-\d\.e]+)", content)
                            if not matches:
                                matches = re.findall(r"Final Energy per site:\s*([-\d\.e]+)", content)
                            if matches:
                                E_final = float(matches[-1])
                        except Exception as e:
                            pass
                    
                    if E_final is not None:
                        seed_energies.append(E_final)

            if seed_energies:
                mean_e = np.mean(seed_energies)
                std_e = np.std(seed_energies) / np.sqrt(len(seed_energies)) if len(seed_energies) > 1 else 0.0
                
                if phi not in data_by_phi:
                    data_by_phi[phi] = {'L': [], 'E': [], 'err': []}
                
                data_by_phi[phi]['L'].append(L)
                data_by_phi[phi]['E'].append(mean_e)
                data_by_phi[phi]['err'].append(std_e)
                print(f"Found L = {L}, phi = {phi}, E = {mean_e:.6f} ± {std_e:.6f} ({len(seed_energies)} seeds)")

    if not data_by_phi:
        print("No data found to plot.")
        return

    plt.figure(figsize=(10, 6))
    
    phis_sorted = sorted(data_by_phi.keys())
    cmap = plt.cm.viridis
    # Add small offset to vmax if there's only 1 phi value to avoid division by zero in Normalize
    norm = plt.Normalize(vmin=min(phis_sorted), vmax=max(phis_sorted) if max(phis_sorted) > min(phis_sorted) else min(phis_sorted) + 0.1)

    for i, phi in enumerate(phis_sorted):
        L_vals = np.array(data_by_phi[phi]['L'])
        inv_L = 1.0 / L_vals
        
        E_vals = np.array(data_by_phi[phi]['E'])
        err_vals = np.array(data_by_phi[phi]['err'])

        # Sort by 1/L for clean continuous lines
        sort_idx = np.argsort(inv_L)
        inv_L = inv_L[sort_idx]
        E_vals = E_vals[sort_idx]
        err_vals = err_vals[sort_idx]

        plt.errorbar(inv_L, E_vals, yerr=err_vals, fmt='-o', color=cmap(norm(phi)),
                     capsize=5, markersize=8, markeredgecolor='black', linewidth=2)

    plt.xlabel('1/L', fontsize=14)
    plt.ylabel('Final Adiabatic Energy per site', fontsize=14)
    plt.title(f'Adiabatic Energy vs 1/L for varying $\phi$ (J={target_J})', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=plt.gca())
    cbar.set_label(r'Twist Angle $\phi$ (units of $\pi$)', fontsize=12)
    
    # Save the plot
    save_dir = Path("/scratch/f/F.Conoscenti/Thesis_QSL/Elaborate/plot/Degeneracy")
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"Adiabatic_Energy_vs_invL_J={target_J}.png"
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"\n✅ Plot saved successfully to {save_path}")
    plt.show()

@jax.jit
def fast_overlap_kernel(log_psi_x, log_phi_x, log_psi_y, log_phi_y):
    """
    Computes the fidelity using the log-amplitudes.
    x ~ samples from psi
    y ~ samples from phi
    """
    ratio_psi = jnp.mean(jnp.exp(log_phi_x - log_psi_x))
    ratio_phi = jnp.mean(jnp.exp(log_psi_y - log_phi_y))
    return jnp.abs(ratio_psi * ratio_phi)

def calculate_overlap_vmap(vstate_psi, vstate_phi):
    s_psi = vstate_psi.sample().reshape(-1, vstate_psi.hilbert.size)
    s_phi = vstate_phi.sample().reshape(-1, vstate_phi.hilbert.size)

    log_psi_x = vstate_psi.log_value(s_psi)
    log_phi_x = vstate_phi.log_value(s_psi)
    
    log_psi_y = vstate_psi.log_value(s_phi)
    log_phi_y = vstate_phi.log_value(s_phi)

    return fast_overlap_kernel(log_psi_x, log_phi_x, log_psi_y, log_phi_y)


def compute_adiabatic_fidelity(folder, vstate, params):
    j_dir = os.path.dirname(folder) 
    model_dir = os.path.dirname(j_dir)
    base_dir = os.path.dirname(model_dir)
    model_name = os.path.basename(model_dir)
    
    match_phi = re.search(r"_phi([\d\.]+)_", model_name)
    if not match_phi:
        return None
        
    current_phi = float(match_phi.group(1))
    prefix = model_name[:match_phi.start(1)]
    suffix = model_name[match_phi.end(1):]
    
    phi_values = []
    model_dirs = {}
    
    # Find other model directories with matching hyperparameters varying only phi
    for d in os.listdir(base_dir):
        if d.startswith(prefix) and d.endswith(suffix):
            try:
                phi_val = float(d[len(prefix):-len(suffix)])
                phi_values.append(phi_val)
                model_dirs[phi_val] = os.path.join(base_dir, d)
            except ValueError:
                pass
                
    phi_values.sort()
    
    if len(phi_values) <= 1:
        return None
        
    current_idx = phi_values.index(current_phi)
    
    # Process if we have a previous smaller phi adiabatic step evaluated 
    if current_idx > 0:
        prev_phi = phi_values[current_idx - 1]
        prev_model_dir = model_dirs[prev_phi]
        prev_folder = os.path.join(prev_model_dir, os.path.basename(j_dir), os.path.basename(folder))
        
        if os.path.exists(prev_folder):
            print(f"Calculating adiabatic fidelity with adjacent phi={prev_phi}...")
            
            prev_params = params.copy()
            prev_params['phi'] = prev_phi
            
            lattice, hilbert, hamiltonian = setup_system(prev_params['L'], prev_params['J2'], prev_params)
            model = setup_model(prev_params, hilbert, prev_params['L'])
            sampler = nk.sampler.MetropolisExchange(
                hilbert=hilbert,
                graph=lattice,
                d_max=2,
                n_chains=1024,
                sweep_size=lattice.n_nodes,
            )
            
            vstate_prev = load_vstate(prev_folder, sampler, model)
            if vstate_prev is not None:
                vstate_prev.n_samples = 1024
                vstate.n_samples = 1024
                
                fidelity = float(calculate_overlap_vmap(vstate_prev, vstate))
                print(f"Fidelity with adjacent phi={prev_phi}: {fidelity}")
                return {'adiabatic_fidelity_prev_phi': prev_phi, 'adiabatic_fidelity': fidelity}
            
    return None



def plot_adiabatic_fidelity_vs_phi(base_directory, target_J):   
    phi_values = []
    fidelities = []
    fidelity_errors = []

    base_path = Path(base_directory)
    if not base_path.exists():
        print(f"Error: Directory {base_directory} does not exist.")
        return

    print(f"Scanning {base_directory} for models with varying phi...")

    for model_dir in base_path.iterdir():
        if not model_dir.is_dir():
            continue
        
        # Extract phi value from folder name (e.g., ..._phi0.0_...)
        match = re.search(r'phi([\d\.]+)', model_dir.name)
        if match:
            phi = float(match.group(1))
        else:
            continue

        # Find the specific J folder
        target_j_folder = None
        for d in model_dir.iterdir():
            if d.is_dir() and (d.name.startswith("J=") or d.name.startswith("J2=")):
                try:
                    part = d.name.split('=')[1]
                    val_str = part.split('_')[0] if '_' in part else part
                    if abs(float(val_str) - target_J) < 1e-5:
                        target_j_folder = d
                        break
                except ValueError:
                    continue
        
        if not target_j_folder:
            print(f"Warning: Could not find J={target_J} folder in {model_dir.name}")
            continue

        # Read fidelities from seeds
        seed_fidelities = []
        for seed_dir in target_j_folder.iterdir():
            if seed_dir.is_dir() and "seed" in seed_dir.name:
                fidelity_final = None
                var_path = seed_dir / "variables.pkl"
                
                # Attempt to load from variables.pkl
                if var_path.exists():
                    try:
                        with open(var_path, "rb") as f:
                            data = pickle.load(f)
                            if 'adiabatic_fidelity' in data:
                                fidelity_final = float(np.real(data['adiabatic_fidelity']))
                    except Exception as e:
                        print(f"Error reading {var_path}: {e}")
                
                # Fallback to output.txt
                out_path = seed_dir / "output.txt"
                if fidelity_final is None and out_path.exists():
                    try:
                        with open(out_path, "r") as f:
                            content = f.read()
                        matches = re.findall(r"Fidelity with adjacent phi=[\d\.]+:\s*([-\d\.e]+)", content)
                        if matches:
                            fidelity_final = float(matches[-1])
                    except Exception as e:
                        print(f"Error reading {out_path}: {e}")
                
                if fidelity_final is None:
                    print(f"Computing adiabatic fidelity on the fly for {seed_dir}...")
                    try:
                        folder = str(seed_dir)
                        params = parse_model_path(folder)
                        
                        lattice, hilbert, hamiltonian = setup_system(params['L'], params['J2'], params)
                        model = setup_model(params, hilbert, params['L'])
                        sampler = nk.sampler.MetropolisExchange(
                            hilbert=hilbert,
                            graph=lattice,
                            d_max=2,
                            n_chains=1024,
                            sweep_size=lattice.n_nodes,
                        )
                        vstate = load_vstate(folder, sampler, model)
                        
                        if vstate is not None:
                            # Overlap Adiabatic states
                            adiabatic_vars = compute_adiabatic_fidelity(folder, vstate, params)
                            if adiabatic_vars is not None:
                                fidelity_final = float(np.real(adiabatic_vars['adiabatic_fidelity']))
                                variables = {}
                                if var_path.exists():
                                    with open(var_path, "rb") as f:
                                        variables = pickle.load(f)
                                variables.update(adiabatic_vars)
                                save_variables(folder, variables)
                    except Exception as e:
                        print(f"Failed to compute adiabatic fidelity for {seed_dir}: {e}")

                if fidelity_final is not None:
                    seed_fidelities.append(fidelity_final)
        if seed_fidelities:
            phi_values.append(phi)
            means_fid = np.mean(seed_fidelities)
            std_fid = np.std(seed_fidelities) / np.sqrt(len(seed_fidelities)) if len(seed_fidelities) > 1 else 0.0
            fidelities.append(means_fid)
            fidelity_errors.append(std_fid)
            print(f"Found phi = {phi}, Adiabatic Fidelity = {means_fid:.6f} ± {std_fid:.6f} ({len(seed_fidelities)} seeds)")

    if not phi_values:
        print("No data found to plot.")
        return
    # Sort by phi for a clean line plot
    sort_idx = np.argsort(phi_values)
    phi_values = np.array(phi_values)[sort_idx]
    fidelities = np.array(fidelities)[sort_idx]
    fidelity_errors = np.array(fidelity_errors)[sort_idx]
    # Plotting
    plt.figure(figsize=(8, 6))
    plt.errorbar(phi_values, fidelities, yerr=fidelity_errors, fmt='-o', color='tab:green',
                 capsize=5, markersize=8, markeredgecolor='black', linewidth=2, label="Adiabatic Fidelity")
    plt.xlabel(r'Twist angle $\phi$ (units of $\pi$)', fontsize=14)
    plt.ylabel('Final Adiabatic Fidelity', fontsize=14)
    plt.title(f'Adiabatic Fidelity vs Boundary Twist $\phi$ (J={target_J})', fontsize=16)       
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    # Save the plot
    plt.savefig(f"adiabatic_fidelity_vs_phi_J{target_J}.png", dpi=300, bbox_inches='tight')
    print(f"\n✅ Plot saved successfully to adiabatic_fidelity_vs_phi_J{target_J}.png")
    plt.show()

def plot_adiabatic_corrlength_vs_phi(base_dir, target_J):
    phi_values = []
    energies = []
    energy_errors = []

    base_path = Path(base_dir)
    if not base_path.exists():
        print(f"Error: Directory {base_dir} does not exist.")
        return
 
    print(f"Scanning {base_dir} for models with varying phi...")

    for model_dir in base_path.iterdir():
        if not model_dir.is_dir():
            continue
        
        # Extract phi value from folder name (e.g., ..._phi0.0_...)
        match = re.search(r'phi([\d\.]+)', model_dir.name)
        if match:
            phi = float(match.group(1))
        else:
            continue

        # Find the specific J folder
        target_j_folder = None
        for d in model_dir.iterdir():
            if d.is_dir() and (d.name.startswith("J=") or d.name.startswith("J2=")):
                try:
                    part = d.name.split('=')[1]
                    val_str = part.split('_')[0] if '_' in part else part
                    if abs(float(val_str) - target_J) < 1e-5:
                        target_j_folder = d
                        break
                except ValueError:
                    continue
        
        if not target_j_folder:
            print(f"Warning: Could not find J={target_J} folder in {model_dir.name}")
            continue

        # Read energies from seeds
        seed_energies = []
        for seed_dir in target_j_folder.iterdir():
            if seed_dir.is_dir() and "seed" in seed_dir.name:
                E_final = None
                var_path = seed_dir / "variables.pkl"
                
                # Attempt to load from variables.pkl
                if var_path.exists():
                    try:
                        with open(var_path, "rb") as f:
                            data = pickle.load(f)
                            if 'correlation_length' in data:
                                E_final = float(np.real(data['correlation_length']))
                    except Exception as e:
                        print(f"Error reading {var_path}: {e}")
                
                # Fallback to output.txt
                out_path = seed_dir / "output.txt"
                if E_final is None and out_path.exists():
                    try:
                        with open(out_path, "r") as f:
                            content = f.read()
                        matches = re.findall(r"Correlation Length:\s*([-\d\.e]+)", content)
                        if matches:
                            E_final = float(matches[-1])
                    except Exception as e:
                        print(f"Error reading {out_path}: {e}")
                
                if E_final is not None:
                    seed_energies.append(E_final)

        if seed_energies:
            phi_values.append(phi)
            means_e = np.mean(seed_energies)
            std_e = np.std(seed_energies) / np.sqrt(len(seed_energies)) if len(seed_energies) > 1 else 0.0
            energies.append(means_e)
            energy_errors.append(std_e)
            print(f"Found phi = {phi}, E = {means_e:.6f} ± {std_e:.6f} ({len(seed_energies)} seeds)")

    if not phi_values:
        print("No data found to plot.")
        return

    # Sort by phi for a clean line plot
    sort_idx = np.argsort(phi_values)
    phi_values = np.array(phi_values)[sort_idx]
    energies = np.array(energies)[sort_idx]
    energy_errors = np.array(energy_errors)[sort_idx]

    # Plotting
    plt.figure(figsize=(8, 6))
    plt.errorbar(phi_values, energies, yerr=energy_errors, fmt='-o', color='tab:blue',
                 capsize=5, markersize=8, markeredgecolor='black', linewidth=2, label="VMC Energy")
    
    plt.xlabel(r'Twist angle $\phi$ (units of $\pi$)', fontsize=14)
    plt.ylabel('Final Correlation Length', fontsize=14)
    plt.title(f'Correlation Length vs Boundary Twist $\phi$ (J={target_J})', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    
    # Save the plot
    save_dir = Path("/scratch/f/F.Conoscenti/Thesis_QSL/Elaborate/plot/Degeneracy")
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"Correlation_Length_vs_phi_J={target_J}.png"
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"\n✅ Plot saved successfully to {save_path}")
    plt.show()

    

if __name__ == "__main__":
    target_J = 0.5 
    # PHI
    #base_directory = "/scratch/f/F.Conoscenti/Thesis_QSL/HFDS_Heisenberg/plot/4x4/PHI"
    #plot_energy_vs_phi(base_directory, target_J)

    #phi
    base_directory = "/scratch/f/F.Conoscenti/Thesis_QSL/HFDS_Heisenberg/plot/4x4/phi"
    plot_adiabatic_energy_vs_phi(base_directory, target_J)
    #plot_adiabatic_fidelity_vs_phi(base_directory, target_J)
    
    base_directory = "/scratch/f/F.Conoscenti/Thesis_QSL/HFDS_Heisenberg/plot/6x6/phi"
    plot_adiabatic_energy_vs_phi(base_directory, target_J)
    #plot_adiabatic_fidelity_vs_phi(base_directory, target_J)
    
    base_directory = "/scratch/f/F.Conoscenti/Thesis_QSL/HFDS_Heisenberg/plot/8x8/phi"
    plot_adiabatic_energy_vs_phi(base_directory, target_J)
    #plot_adiabatic_fidelity_vs_phi(base_directory, target_J)
    


    # Plot against 1/L for multiple sizes:
    l_directories = [
         "/scratch/f/F.Conoscenti/Thesis_QSL/HFDS_Heisenberg/plot/4x4/phi",
         "/scratch/f/F.Conoscenti/Thesis_QSL/HFDS_Heisenberg/plot/6x6/phi",
         "/scratch/f/F.Conoscenti/Thesis_QSL/HFDS_Heisenberg/plot/8x8/phi"
    ]
    plot_adiabatic_energy_vs_L(l_directories, target_J)