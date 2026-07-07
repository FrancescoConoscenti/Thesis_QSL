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
    match_L = re.search(r'(\d+)x\d+', str(base_path))
    L = int(match_L.group(1)) if match_L else "Unknown"

    if L == 4:
        print("Using provided manual data for 4x4 lattice...")
        phi_values = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
        energies = [-0.5287, -0.5211, -0.5048, -0.4926, -0.4862, -0.4858, -0.4858, -0.4938, -0.5059, -0.5215, -0.5285]
        energy_errors = [0.0] * len(energies)
        dirs_to_scan = []
    else:
        if not base_path.exists():
            print(f"Error: Directory {base_dir} does not exist.")
            return
        dirs_to_scan = list(base_path.iterdir())

    print(f"Scanning {base_dir} for models with varying phi...")

    for model_dir in dirs_to_scan:
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
    
    plt.xlabel(r'Twist angle $\phi$ (units of $\pi$)')
    plt.ylabel('Final VMC Energy per site')
    plt.title(f'Energy vs Boundary Twist $\phi$ (J={target_J})')
    plt.grid(True, linestyle='--', alpha=0.7)
    
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
    match_L = re.search(r'(\d+)x\d+', str(base_path))
    L = int(match_L.group(1)) if match_L else "Unknown"

    if L == 4:
        print("Using provided manual data for 4x4 lattice...")
        phi_values = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
        energies = [-0.5287, -0.5211, -0.5048, -0.4926, -0.4862, -0.4858, -0.4858, -0.4938, -0.5059, -0.5215, -0.5285]
        energy_errors = [0.0] * len(energies)
        dirs_to_scan = []
    else:
        if not base_path.exists():
            print(f"Error: Directory {base_directory} does not exist.")
            return
        dirs_to_scan = list(base_path.iterdir())

    print(f"Scanning {base_directory} for models with varying phi...")

    for model_dir in dirs_to_scan:
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
    plt.xlabel(r'Twist angle $\phi$ (units of $\pi$)')
    plt.ylabel('Adiabatic Energy per site')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    # Save the plot
    plt.savefig(f"Elaborate/plot/Degeneracy/adiabatic_energy_vs_phi_J{target_J}_L{L}.png", dpi=300, bbox_inches='tight')
    print(f"\n✅ Plot saved successfully to Elaborate/plot/Degeneracy/adiabatic_energy_vs_phi_J{target_J}_L{L}.png")
    plt.show()  

def plot_adiabatic_energy_vs_L(directories, target_J):
    data_by_phi = {}

    print(f"Scanning directories for models with varying phi to plot vs 1/L...")

    for base_dir in directories:
        base_path = Path(base_dir)
        
        # Extract L from path (e.g. "4x4", "6x6")
        match_L = re.search(r'(\d+)x\d+', str(base_path))
        if not match_L:
            print(f"Warning: Could not determine L from path {base_dir}")
            continue
        L = int(match_L.group(1))

        if L == 4:
            manual_phis = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
            manual_energies = [-0.5287, -0.5211, -0.5048, -0.4926, -0.4862, -0.4858, -0.4858, -0.4938, -0.5059, -0.5215, -0.5285]
            for p, e in zip(manual_phis, manual_energies):
                if p not in data_by_phi:
                    data_by_phi[p] = {'L': [], 'E': [], 'err': []}
                data_by_phi[p]['L'].append(4)
                data_by_phi[p]['E'].append(e)
                data_by_phi[p]['err'].append(0.0)
            print(f"Using provided manual data for L=4")
            continue

        if not base_path.exists():
            print(f"Warning: Directory {base_dir} does not exist.")
            continue

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

    # Add missing data points using symmetry E(phi) = E(2.0 - phi)
    existing_phis = list(data_by_phi.keys())
    for phi in existing_phis:
        sym_phi = round(2.0 - phi, 5)
        if 0.0 <= sym_phi <= 2.0:
            sym_phi_key = next((p for p in data_by_phi.keys() if abs(p - sym_phi) < 1e-5), sym_phi)
            
            if sym_phi_key not in data_by_phi:
                data_by_phi[sym_phi_key] = {'L': [], 'E': [], 'err': []}
            
            for i, L_val in enumerate(data_by_phi[phi]['L']):
                if L_val not in data_by_phi[sym_phi_key]['L']:
                    data_by_phi[sym_phi_key]['L'].append(L_val)
                    data_by_phi[sym_phi_key]['E'].append(data_by_phi[phi]['E'][i])
                    data_by_phi[sym_phi_key]['err'].append(data_by_phi[phi]['err'][i])

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

        # Convert from energy per site to total energy
        E_vals = E_vals * (L_vals ** 2)
        err_vals = err_vals * (L_vals ** 2)

        # Sort by 1/L for clean continuous lines
        sort_idx = np.argsort(inv_L)
        inv_L = inv_L[sort_idx]
        E_vals = E_vals[sort_idx]
        err_vals = err_vals[sort_idx]

        plt.errorbar(inv_L, E_vals, yerr=err_vals, fmt='-o', color=cmap(norm(phi)),
                     capsize=5, markersize=8, markeredgecolor='black', linewidth=2)

    plt.xlabel('1/L')
    plt.ylabel('Final Adiabatic Total Energy')
    plt.title(f'Adiabatic Total Energy vs 1/L for varying $\phi$ (J={target_J})')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=plt.gca())
    cbar.set_label(r'Twist Angle $\phi$ (units of $\pi$)', )
    
    # Save the plot
    save_dir = Path("/scratch/f/F.Conoscenti/Thesis_QSL/Elaborate/plot/Degeneracy")
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"Adiabatic_Total_Energy_vs_invL_J={target_J}.png"
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"\n✅ Plot saved successfully to {save_path}")
    plt.show()

def plot_adiabatic_energy_vs_phi_all_L(directories, target_J):
    data_by_key = {}

    print(f"Scanning directories for models with varying phi to plot vs phi for all L...")

    for base_dir in directories:
        base_path = Path(base_dir)

        # Distinguish the data source folder (e.g. "phi_sz1" vs "phi_new") so
        # they get separate labels and are never connected to each other.
        source = "sz1" if "phi_sz1" in base_path.name else "default"

        # Extract L from path (e.g. "4x4", "6x6")
        match_L = re.search(r'(\d+)x\d+', str(base_path))
        if not match_L:
            print(f"Warning: Could not determine L from path {base_dir}")
            continue
        L = int(match_L.group(1))
        key = (L, source)

        if L == 4 and source == "default":
            manual_phis = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
            manual_energies = [-0.5287, -0.5211, -0.5048, -0.4926, -0.4862, -0.4858, -0.4858, -0.4938, -0.5059, -0.5215, -0.5285]
            if key not in data_by_key:
                data_by_key[key] = {'phi': [], 'E': [], 'err': []}
            for p, e in zip(manual_phis, manual_energies):
                data_by_key[key]['phi'].append(p)
                data_by_key[key]['E'].append(e)
                data_by_key[key]['err'].append(0.0)
            print(f"Using provided manual data for L=4")
            continue

        if not base_path.exists():
            print(f"Warning: Directory {base_dir} does not exist.")
            continue

        if key not in data_by_key:
            data_by_key[key] = {'phi': [], 'E': [], 'err': []}

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

                data_by_key[key]['phi'].append(phi)
                data_by_key[key]['E'].append(mean_e)
                data_by_key[key]['err'].append(std_e)
                print(f"Found L = {L}, phi = {phi}, E = {mean_e:.6f} ± {std_e:.6f} ({len(seed_energies)} seeds)")

    if not data_by_key:
        print("No data found to plot.")
        return

    # Add missing data points using symmetry E(phi) = E(2.0 - phi), within each (L, source) group
    for key in data_by_key:
        existing_phis = list(data_by_key[key]['phi'])
        existing_Es = list(data_by_key[key]['E'])
        existing_errs = list(data_by_key[key]['err'])

        for i, phi in enumerate(existing_phis):
            sym_phi = round(2.0 - phi, 5)
            if 0.0 <= sym_phi <= 2.0:
                if not any(abs(p - sym_phi) < 1e-5 for p in data_by_key[key]['phi']):
                    data_by_key[key]['phi'].append(sym_phi)
                    data_by_key[key]['E'].append(existing_Es[i])
                    data_by_key[key]['err'].append(existing_errs[i])

    plt.figure(figsize=(10, 6))

    L_sorted = sorted(set(L for L, _ in data_by_key.keys()))
    colors = dict(zip(L_sorted, plt.cm.viridis(np.linspace(0, 0.9, len(L_sorted)))))

    # Style per source so phi_sz1 datapoints are visually distinct and never
    # connected to phi_new (or other) datapoints of the same L.
    source_style = {
        "default": {"fmt": "-o", "label_suffix": ""},
        "sz1": {"fmt": "--s", "label_suffix": " (Sz=1)"},
    }

    for key in sorted(data_by_key.keys()):
        L, source = key
        phi_vals = np.array(data_by_key[key]['phi'])
        E_vals = np.array(data_by_key[key]['E'])
        err_vals = np.array(data_by_key[key]['err'])

        # Sort by phi
        sort_idx = np.argsort(phi_vals)
        phi_vals = phi_vals[sort_idx]
        E_vals = E_vals[sort_idx]
        err_vals = err_vals[sort_idx]

        style = source_style.get(source, source_style["default"])
        plt.errorbar(phi_vals, E_vals, yerr=err_vals, fmt=style["fmt"], color=colors[L],
                     capsize=5, markersize=8, markeredgecolor='black', linewidth=2,
                     label=f"L={L}{style['label_suffix']}")

    plt.xlabel(r'Twist angle $\phi$ (units of $\pi$)')
    plt.ylabel('Adiabatic Energy per site')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Save the plot
    save_dir = Path("/scratch/f/F.Conoscenti/Thesis_QSL/Elaborate/plot/Degeneracy")
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"Adiabatic_Energy_vs_phi_all_L_J={target_J}_sz1.png"
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"\n✅ Plot saved successfully to {save_path}")
    plt.show()

def plot_energy_gap_vs_L_fit(directories, target_J, target_phi, degree=2):
    print(f"\nCalculating absolute energy gap per site |E(0) - E({target_phi})| vs 1/L and fitting...")
    energy_data = {}
    energy_err = {}

    for base_dir in directories:
        base_path = Path(base_dir)
        match_L = re.search(r'(\d+)x\d+', str(base_path))
        if not match_L:
            continue
        L = int(match_L.group(1))

        if L not in energy_data:
            energy_data[L] = {}
            energy_err[L] = {}

        # Pre-fill manually available L=4 and L=8 points 
    
        if not base_path.exists():
            continue

        # Scrape dynamically
        for model_dir in base_path.iterdir():
            if not model_dir.is_dir(): continue
            match = re.search(r'phi([\d\.]+)', model_dir.name)
            if not match: continue
            phi = float(match.group(1))

            # We only need data for 0.0 and our target_phi to calculate the gap
            if abs(phi) > 1e-5 and abs(phi - target_phi) > 1e-5:
                continue

            target_j_folder = None
            for d in model_dir.iterdir():
                if d.is_dir() and (d.name.startswith("J=") or d.name.startswith("J2=")):
                    try:
                        part = d.name.split('=')[1]
                        val_str = part.split('_')[0] if '_' in part else part
                        if abs(float(val_str) - target_J) < 1e-5:
                            target_j_folder = d
                            break
                    except ValueError: pass
            
            if not target_j_folder: continue

            seed_energies = []
            for seed_dir in target_j_folder.iterdir():
                if seed_dir.is_dir() and "seed" in seed_dir.name:
                    E_final = None
                    var_path = seed_dir / "variables.pkl"
                    if var_path.exists():
                        try:
                            with open(var_path, "rb") as f:
                                data = pickle.load(f)
                                if 'E_vs_final' in data: E_final = float(np.real(data['E_vs_final']))
                        except Exception: pass
                    out_path = seed_dir / "output.txt"
                    if E_final is None and out_path.exists():
                        try:
                            with open(out_path, "r") as f: content = f.read()
                            matches = re.findall(r"Final Energy from VMC:\s*([-\d\.e]+)", content)
                            if not matches: matches = re.findall(r"Final Energy per site:\s*([-\d\.e]+)", content)
                            if matches: E_final = float(matches[-1])
                        except Exception: pass
                    if E_final is not None: seed_energies.append(E_final)

            if seed_energies:
                mean_e = np.mean(seed_energies)
                std_e = np.std(seed_energies) / np.sqrt(len(seed_energies)) if len(seed_energies) > 1 else 0.0
                energy_data[L][phi] = mean_e
                energy_err[L][phi] = std_e

    L_list, gaps, gap_errs = [], [], []
    for L in sorted(energy_data.keys()):
        phi_0_key = next((k for k in energy_data[L].keys() if abs(k) < 1e-5), None)
        phi_t_key = next((k for k in energy_data[L].keys() if abs(k - target_phi) < 1e-5), None)

        if phi_0_key is not None and phi_t_key is not None:
            e0, e_phi = energy_data[L][phi_0_key], energy_data[L][phi_t_key]
            err0, err_phi = energy_err[L][phi_0_key], energy_err[L][phi_t_key]
            
            gap = np.abs(e0 - e_phi)
            err = np.sqrt(err0**2 + err_phi**2)
            L_list.append(L); gaps.append(gap); gap_errs.append(err)
            print(f"L={L}: E(0)={e0:.6f}, E({target_phi})={e_phi:.6f} => Absolute Gap per site={gap:.6f} ± {err:.6f}")

    if len(L_list) < 2:
        print(f"Not enough data points to compute and fit energy gap for phi={target_phi}")
        return

    L_arr = np.array(L_list)
    inv_L = 1.0 / L_arr
    gaps_arr, gap_errs_arr = np.array(gaps), np.array(gap_errs)

    plt.figure(figsize=(9, 6))
    plt.errorbar(inv_L, gaps_arr, yerr=gap_errs_arr, fmt='o', color='tab:red',
                 capsize=5, markersize=8, markeredgecolor='black', label='Gap Data')

    from scipy.optimize import curve_fit

    def fit_func(x, a, n, b):
        return a * x**n + b

    def fit_func_cubic(x, a, b):
        return a * x**3

    if len(inv_L) >= 3:
        try:
            p0 = [gaps_arr[0], 1.0, 0.0]
            popt, pcov = curve_fit(fit_func, inv_L, gaps_arr, p0=p0, maxfev=10000)
            a, n_fit, b = popt
            inv_L_fit = np.linspace(0, max(inv_L)*1.2, 100)
            plt.plot(inv_L_fit, fit_func(inv_L_fit, *popt), '--', color='black', label=f'Fit $y = ax^n + b$ ($n={n_fit:.3f}$)')
        except Exception as e:
            print(f"Curve fit failed: {e}")
            actual_degree = min(degree, len(inv_L) - 1)
            if actual_degree > 0:
                coeffs = np.polyfit(inv_L, gaps_arr, actual_degree)
                p = np.poly1d(coeffs)
                inv_L_fit = np.linspace(0, max(inv_L)*1.2, 100)
                plt.plot(inv_L_fit, p(inv_L_fit), '--', color='black', label=f'Fit (deg {actual_degree})')
    else:
        actual_degree = min(degree, len(inv_L) - 1)
        if actual_degree > 0:
            coeffs = np.polyfit(inv_L, gaps_arr, actual_degree)
            p = np.poly1d(coeffs)
            inv_L_fit = np.linspace(0, max(inv_L)*1.2, 100)
            plt.plot(inv_L_fit, p(inv_L_fit), '--', color='black', label=f'Fit (deg {actual_degree})')

    if len(inv_L) >= 2:
        try:
            p0_cub = [gaps_arr[0], 0.0]
            popt_cub, _ = curve_fit(fit_func_cubic, inv_L, gaps_arr, p0=p0_cub, maxfev=10000)
            inv_L_fit = np.linspace(0, max(inv_L)*1.2, 100)
            plt.plot(inv_L_fit, fit_func_cubic(inv_L_fit, *popt_cub), ':', color='tab:blue', linewidth=2, label=r'Fit $y = ax^3 + b$')
        except Exception as e:
            print(f"Cubic curve fit failed: {e}")

    plt.xlabel('1/L')
    plt.ylabel(f'Absolute Energy Gap per site $|E(0) - E({target_phi})|$')
    plt.title(f'Absolute Energy Gap per site vs 1/L for $\\phi$={target_phi} (J={target_J})')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.xlim(left=0)
    plt.tight_layout()
    
    save_dir = Path("/scratch/f/F.Conoscenti/Thesis_QSL/Elaborate/plot/Degeneracy")
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"Energy_Gap_per_site_Fit_vs_invL_phi={target_phi}_J={target_J}.png"
    
    plt.savefig(save_path, dpi=300)
    print(f"✅ Plot saved successfully to {save_path}\n")
    plt.show()

    # ---------------------------------------------------------
    # Plot Total Energy Gap vs 1/L
    # ---------------------------------------------------------
    total_gaps_arr = gaps_arr * (L_arr ** 2)
    total_gap_errs_arr = gap_errs_arr * (L_arr ** 2)

    plt.figure(figsize=(9, 6))
    plt.errorbar(inv_L, total_gaps_arr, yerr=total_gap_errs_arr, fmt='s', color='tab:purple',
                 capsize=5, markersize=8, markeredgecolor='black', label='Total Gap Data')

    actual_degree = min(degree, len(inv_L) - 1)
    if actual_degree > 0:
        coeffs = np.polyfit(inv_L, total_gaps_arr, actual_degree)
        p_poly = np.poly1d(coeffs)
        inv_L_fit = np.linspace(0, max(inv_L)*1.2, 100)
        plt.plot(inv_L_fit, p_poly(inv_L_fit), '--', color='black', label=f'Polynomial Fit (deg {actual_degree})')

    plt.xlabel('1/L')
    plt.ylabel(f'Absolute Total Energy Gap $|E_{{tot}}(0) - E_{{tot}}({target_phi})|$')
    plt.title(f'Absolute Total Energy Gap vs 1/L for $\\phi$={target_phi} (J={target_J})')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.xlim(left=0)
    plt.tight_layout()
    
    save_path_total = save_dir / f"Total_Energy_Gap_Fit_vs_invL_phi={target_phi}_J={target_J}.png"
    plt.savefig(save_path_total, dpi=300)
    print(f"✅ Plot saved successfully to {save_path_total}\n")
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
    plt.xlabel(r'Twist angle $\phi$ (units of $\pi$)')
    plt.ylabel('Final Adiabatic Fidelity')
    plt.title(f'Adiabatic Fidelity vs Boundary Twist $\phi$ (J={target_J})')       
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
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
    
    plt.xlabel(r'Twist angle $\phi$ (units of $\pi$)')
    plt.ylabel('Final Correlation Length')
    plt.title(f'Correlation Length vs Boundary Twist $\phi$ (J={target_J})')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
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
    base_directory = "/scratch/f/F.Conoscenti/Thesis_QSL/HFDS_Heisenberg/plot/4x4/phiJ0"
    #plot_adiabatic_energy_vs_phi(base_directory, target_J)
    #plot_adiabatic_fidelity_vs_phi(base_directory, target_J)
    
    base_directory = "/scratch/f/F.Conoscenti/Thesis_QSL/HFDS_Heisenberg/plot/6x6/phi"
    #plot_adiabatic_energy_vs_phi(base_directory, target_J)
    #plot_adiabatic_fidelity_vs_phi(base_directory, target_J)
    
    base_directory = "/scratch/f/F.Conoscenti/Thesis_QSL/HFDS_Heisenberg/plot/8x8/phi1"
    #plot_adiabatic_energy_vs_phi(base_directory, target_J)
    #plot_adiabatic_fidelity_vs_phi(base_directory, target_J)
    


    # Plot against 1/L for multiple sizes:
    l_directories = [
         "/scratch/f/F.Conoscenti/Thesis_QSL/HFDS_Heisenberg/plot/4x4/phi_new",
         "/scratch/f/F.Conoscenti/Thesis_QSL/HFDS_Heisenberg/plot/6x6/phi_new",
         "/scratch/f/F.Conoscenti/Thesis_QSL/HFDS_Heisenberg/plot/8x8/phi_new",
        "/scratch/f/F.Conoscenti/Thesis_QSL/HFDS_Heisenberg/plot/10x10/phi_new",
        "/scratch/f/F.Conoscenti/Thesis_QSL/HFDS_Heisenberg/plot/4x4/phi_sz1",
                 "/scratch/f/F.Conoscenti/Thesis_QSL/HFDS_Heisenberg/plot/6x6/phi_sz1",
         "/scratch/f/F.Conoscenti/Thesis_QSL/HFDS_Heisenberg/plot/8x8/phi_sz1",
         "/scratch/f/F.Conoscenti/Thesis_QSL/HFDS_Heisenberg/plot/10x10/phi_sz1",



    ]
    #plot_adiabatic_energy_vs_L(l_directories, target_J)
    plot_adiabatic_energy_vs_phi_all_L(l_directories, target_J)
    
    # Plot and fit energy gap for twist phi=1.0 (or any other desired target phi)
    #plot_energy_gap_vs_L_fit(l_directories, target_J, target_phi=0.6, degree=2)
