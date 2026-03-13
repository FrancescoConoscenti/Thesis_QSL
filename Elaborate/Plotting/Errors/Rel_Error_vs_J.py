import re
import os
import sys
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib.patches import Patch
import pickle
from pathlib import Path
import numpy as np
import matplotlib.ticker as mticker

def get_available_js(model_folder):
    model_path = Path(model_folder)
    if not model_path.exists():
        return []
    js = set()
    
    for d in model_path.iterdir():
        if d.is_dir() and (d.name.startswith("J=") or d.name.startswith("J2=")):
            try:
                part = d.name.split('=')[1]
                if '_' in part:
                    val_str = part.split('_')[0]
                else:
                    val_str = part
                js.add(float(val_str))
            except ValueError:
                continue
    return sorted(list(js))

def get_rel_error_from_seeds(model_folder, j_val):
    errors = []
    
    model_path = Path(model_folder)
    if not model_path.exists():
        return errors

    target_j_folder = None
    # Robustly find J folder
    for d in model_path.iterdir():
        if d.is_dir() and (d.name.startswith("J=") or d.name.startswith("J2=")):
            try:
                part = d.name.split('=')[1]
                if '_' in part:
                    val_str = part.split('_')[0]
                else:
                    val_str = part
                
                if abs(float(val_str) - j_val) < 1e-5:
                    target_j_folder = d
                    break
            except ValueError:
                continue
    
    if target_j_folder is None:
        return errors

    for seed_dir in target_j_folder.iterdir():
        if seed_dir.is_dir() and seed_dir.name.startswith("seed_"):
            pkl_path = seed_dir / "variables.pkl"
            if not pkl_path.exists():
                pkl_path = seed_dir / "variables"
            
            if pkl_path.exists():
                try:
                    with open(pkl_path, "rb") as f:
                        data = pickle.load(f)
                        val = None
                        if 'rel_err_E' in data:
                            val = data['rel_err_E']
              
                        if val is not None:
                            if isinstance(val, (list, np.ndarray)):
                                errors.append(float(np.real(val[-1])))
                            else:
                                errors.append(float(np.real(val)))
                except Exception as e:
                    print(f"Error reading {pkl_path}: {e}")
    
    return errors

def get_param_count(model_folder):
    model_path = Path(model_folder)
    if not model_path.exists():
        return None
    
    for d in model_path.iterdir():
        if d.is_dir() and (d.name.startswith("J=") or d.name.startswith("J2=")):
            for seed_dir in d.iterdir():
                if seed_dir.is_dir() and seed_dir.name.startswith("seed_"):
                    pkl_path = seed_dir / "variables.pkl"
                    if not pkl_path.exists():
                        pkl_path = seed_dir / "variables"
                    
                    if pkl_path.exists():
                        try:
                            with open(pkl_path, "rb") as f:
                                data = pickle.load(f)
                                if 'count_params' in data:
                                    return data['count_params']
                                if 'params' in data:
                                    return data['params']
                        except Exception:
                            pass
    return None

def get_energy_from_seeds(model_folder, j_val):
    energies = []
    
    model_path = Path(model_folder)
    if not model_path.exists():
        return energies

    target_j_folder = None
    # Robustly find J folder
    for d in model_path.iterdir():
        if d.is_dir() and (d.name.startswith("J=") or d.name.startswith("J2=")):
            try:
                part = d.name.split('=')[1]
                if '_' in part:
                    val_str = part.split('_')[0]
                else:
                    val_str = part
                
                if abs(float(val_str) - j_val) < 1e-5:
                    target_j_folder = d
                    break
            except ValueError:
                continue
    
    if target_j_folder is None:
        return energies

    for seed_dir in target_j_folder.iterdir():
        if seed_dir.is_dir() and seed_dir.name.startswith("seed_"):
            pkl_path = seed_dir / "variables.pkl"
            if not pkl_path.exists():
                pkl_path = seed_dir / "variables"
            
            if pkl_path.exists():
                try:
                    with open(pkl_path, "rb") as f:
                        data = pickle.load(f)
                        val = None
                        if 'E_vs_final' in data:
                            val = data['E_vs_final']
              
                        if val is not None:
                            if isinstance(val, (list, np.ndarray)):
                                energies.append(float(np.real(val[-1])))
                            else:
                                energies.append(float(np.real(val)))
                except Exception as e:
                    print(f"Error reading {pkl_path}: {e}")
    
    return energies

def get_energy_diff_from_seeds(model_folder, j_val):
    diffs = []
    
    model_path = Path(model_folder)
    if not model_path.exists():
        return diffs

    target_j_folder = None
    # Robustly find J folder
    for d in model_path.iterdir():
        if d.is_dir() and (d.name.startswith("J=") or d.name.startswith("J2=")):
            try:
                part = d.name.split('=')[1]
                if '_' in part:
                    val_str = part.split('_')[0]
                else:
                    val_str = part
                
                if abs(float(val_str) - j_val) < 1e-5:
                    target_j_folder = d
                    break
            except ValueError:
                continue
    
    if target_j_folder is None:
        return diffs

    for seed_dir in target_j_folder.iterdir():
        if seed_dir.is_dir() and seed_dir.name.startswith("seed_"):
            pkl_path = seed_dir / "variables.pkl"
            if not pkl_path.exists():
                pkl_path = seed_dir / "variables"
            
            if pkl_path.exists():
                try:
                    with open(pkl_path, "rb") as f:
                        data = pickle.load(f)
                        rel_err = None
                        E_val = None
                        E_exact = None
                        if 'rel_err_E' in data:
                            rel_err = data['rel_err_E']
                        if 'E_vs_final' in data:
                            E_val = data['E_vs_final']
                        if 'E_exact' in data:
                            E_exact = data['E_exact']
              
                        if rel_err is not None and E_val is not None:
                            if isinstance(rel_err, (list, np.ndarray)): rel_err = float(np.real(rel_err[-1]))
                            else: rel_err = float(np.real(rel_err))
                        if E_val is not None:
                            if isinstance(E_val, (list, np.ndarray)): E_val = float(np.real(E_val[-1]))
                            else: E_val = float(np.real(E_val))
                            
                            if rel_err is not None:
                                if isinstance(rel_err, (list, np.ndarray)): rel_err = float(np.real(rel_err[-1]))
                                else: rel_err = float(np.real(rel_err))
                                
                                # diff = E - E_exact. Derived from rel_err = (E - E_exact)/|E_exact|
                                # Assuming E_exact < 0: diff = - E * rel_err / (1 - rel_err)
                                diff = - E_val * rel_err / (1 - rel_err)
                                diffs.append(diff)
                            elif E_exact is not None:
                                if isinstance(E_exact, (list, np.ndarray)): E_exact = float(np.real(E_exact[-1]))
                                else: E_exact = float(np.real(E_exact))
                                diff = E_val - E_exact
                                diffs.append(diff)
                except Exception as e:
                    print(f"Error reading {pkl_path}: {e}")
    
    return diffs

def get_fidelity_from_seeds(model_folder, j_val):
    fidelities = []
    
    model_path = Path(model_folder)
    if not model_path.exists():
        return fidelities

    target_j_folder = None
    # Robustly find J folder
    for d in model_path.iterdir():
        if d.is_dir() and (d.name.startswith("J=") or d.name.startswith("J2=")):
            try:
                part = d.name.split('=')[1]
                if '_' in part:
                    val_str = part.split('_')[0]
                else:
                    val_str = part
                
                if abs(float(val_str) - j_val) < 1e-5:
                    target_j_folder = d
                    break
            except ValueError:
                continue
    
    if target_j_folder is None:
        return fidelities

    for seed_dir in target_j_folder.iterdir():
        if seed_dir.is_dir() and seed_dir.name.startswith("seed_"):
            pkl_path = seed_dir / "variables.pkl"
            if not pkl_path.exists():
                pkl_path = seed_dir / "variables"
            
            if pkl_path.exists():
                try:
                    with open(pkl_path, "rb") as f:
                        data = pickle.load(f)
                        val = None
                        if 'fidelity' in data:
                            val = data['fidelity']
              
                        if val is not None:
                            if isinstance(val, (list, np.ndarray)):
                                fidelities.append(float(np.real(val[-1])))
                            else:
                                fidelities.append(float(np.real(val)))
                except Exception as e:
                    print(f"Error reading {pkl_path}: {e}")
    
    return fidelities

def plot_energy_vs_js(model_paths, save_name="Energy_vs_J2.png"):
    plt.figure(figsize=(10, 6))
    
    hfds_energies = {}
    vit_energies = {}
    hfds_params = None
    vit_params = None

    lattice_size = ""
    if len(model_paths) > 0:
        path_str = str(model_paths[0])
        if "10x10" in path_str:
            lattice_size = "10x10"
        elif "8x8" in path_str:
            lattice_size = "8x8"
        elif "6x6" in path_str:
            lattice_size = "6x6"
        elif "4x4" in path_str:
            lattice_size = "4x4"

    for model_path in model_paths:
        if not os.path.exists(model_path):
            model_path = model_path.replace("/cluster/home/fconoscenti/Thesis_QSL", "/scratch/f/F.Conoscenti/Thesis_QSL")
            if not os.path.exists(model_path):
                print(f"Path not found: {model_path}")
                continue
        
        js = get_available_js(model_path)
        energies = {}
        
        for j in js:
            vals = get_energy_from_seeds(model_path, j)
            if vals:
                energies[j] = np.min(vals)
        
        n_params = get_param_count(model_path)
        
        if "HFDS" in str(model_path):
            hfds_energies = energies
            hfds_params = n_params
        elif "ViT" in str(model_path):
            vit_energies = energies
            vit_params = n_params

    common_js = sorted(list(set(hfds_energies.keys()) & set(vit_energies.keys())))
    
    if common_js:
        diffs = [hfds_energies[j] - vit_energies[j] for j in common_js]
        plt.scatter(common_js, diffs, marker='o', color='tab:blue', label='$E_{HFDS} - E_{ViT}$')
        p_info = []
        if hfds_params: p_info.append(f"HFDS: {hfds_params} params")
        if vit_params: p_info.append(f"ViT: {vit_params} params")
        if p_info:
            plt.plot([], [], linestyle="None", label="\n".join(p_info))
        plt.axhline(0, color='black', linestyle='--')

    plt.xlabel("$J_2$", fontsize=12)
    plt.ylabel("$E_{HFDS} - E_{ViT}$", fontsize=12)
    title = "Energy Difference ($E_{HFDS} - E_{ViT}$) vs $J_2$"
    if lattice_size:
        title += f" ({lattice_size})"
    plt.title(title, fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='best')
    
    formatter = mticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True) 
    formatter.set_powerlimits((-1, 1))
    plt.gca().yaxis.set_major_formatter(formatter)
    plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(0.05))
    
    if lattice_size:
        save_name = save_name.replace(".png", f"_{lattice_size}.png")
    
    save_path = f"/scratch/f/F.Conoscenti/Thesis_QSL/Elaborate/plot/Errors/{save_name}"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {save_path}")
    plt.show()

def plot_energy_diff_vs_js(model_paths, values_HFDS=None, values_ViT=None, save_name="Energy_Diff_vs_J2.png"):
    plt.figure(figsize=(10, 6))
    
    markers = ['o', 's', '^', 'v', 'D', 'p', '*', 'X']
    colors = plt.cm.tab10(np.linspace(0, 1, len(model_paths)))

    lattice_size = ""
    if len(model_paths) > 0:
        path_str = str(model_paths[0])
        if "10x10" in path_str:
            lattice_size = "10x10"
        elif "8x8" in path_str:
            lattice_size = "8x8"
        elif "6x6" in path_str:
            lattice_size = "6x6"
        elif "4x4" in path_str:
            lattice_size = "4x4"

    gs_10x10_path = "/scratch/f/F.Conoscenti/Thesis_QSL/Elaborate/Plotting/Errors/gs10x10.pkl"
    gs_10x10 = {}
    if os.path.exists(gs_10x10_path):
        with open(gs_10x10_path, 'rb') as f:
            gs_10x10 = pickle.load(f)

    for i, model_path in enumerate(model_paths):
        if not os.path.exists(model_path):
            model_path = model_path.replace("/cluster/home/fconoscenti/Thesis_QSL", "/scratch/f/F.Conoscenti/Thesis_QSL")
            if not os.path.exists(model_path):
                print(f"Path not found: {model_path}")
                continue
            
        js = get_available_js(model_path)
        mean_diffs = []
        std_diffs = []
        
        valid_js = []
        
        is_10x10 = "10x10" in str(model_path)

        for j in js:
            found_energies = get_energy_from_seeds(model_path, j)
            print(f"Model: {Path(model_path).name}, J={j}, Found Energies: {found_energies}")

            exact_E = None

            if "HFDS" in str(model_path) and values_HFDS is not None:
                for k, v in values_HFDS.items():
                    if abs(float(k) - j) < 1e-5:
                        exact_E = v
                        break
            elif "ViT" in str(model_path) and values_ViT is not None:
                for k, v in values_ViT.items():
                    if abs(float(k) - j) < 1e-5:
                        exact_E = v
                        break
            
            if exact_E is None and is_10x10:
                for k, v in gs_10x10.items():
                    if abs(float(k) - j) < 1e-5:
                        exact_E = v
                        break
            
            if exact_E is not None:
                vals = get_energy_from_seeds(model_path, j)
                if vals:
                    current_exact = exact_E
                    if abs(np.mean(vals)) > 5.0 and abs(current_exact) < 2.0:
                        if "4x4" in str(model_path): N=16
                        elif "6x6" in str(model_path): N=36
                        elif "8x8" in str(model_path): N=64
                        elif "10x10" in str(model_path): N=100
                        else: N=1
                        current_exact *= N
                    diffs = [v - current_exact for v in vals]
                    mean_diffs.append(np.mean(diffs))
                    if len(diffs) > 1:
                        std_diffs.append(np.std(diffs) / np.sqrt(len(diffs)))
                    else:
                        std_diffs.append(0.0)
                    valid_js.append(j)
            else:
                vals = get_energy_diff_from_seeds(model_path, j)
                if vals:
                    mean_diffs.append(np.mean(vals))
                    if len(vals) > 1:
                        std_diffs.append(np.std(vals) / np.sqrt(len(vals)))
                    else:
                        std_diffs.append(0.0)
                    valid_js.append(j)
        
        # --- Manual Data Injection ---
        # Add missing points manually here: (J_value, Mean_Diff, Std_Error)
        if "HFDS" in str(model_path) and "6x6" in str(model_path):
            manual_points = [
                (0.4, 0.000252892, 0.0000000041475633),
                (0.7, 0.001064545, 0.0),
            ]
            for mj, mdiff, mstd in manual_points:
                # Optional: Only add if not already present
                if mj not in valid_js:
                    valid_js.append(mj)
                    mean_diffs.append(mdiff)
                    std_diffs.append(mstd)

        elif "ViT" in str(model_path) and "6x6" in str(model_path):
            manual_points = [
                (0.4, 0.000251323, 0),  # Example point for ViT at J=0.4
            ]
            for mj, mdiff, mstd in manual_points:
                if mj not in valid_js:
                    valid_js.append(mj)
                    mean_diffs.append(mdiff)
                    std_diffs.append(mstd)
        # -----------------------------
        
        if valid_js:
            # Sort data by J to ensure lines are plotted correctly
            sorted_data = sorted(zip(valid_js, mean_diffs, std_diffs))
            valid_js, mean_diffs, std_diffs = zip(*sorted_data)

            n_params = get_param_count(model_path)
            if "ViT" in str(model_path):
                base_label = "ViT"
            elif "HFDS" in str(model_path):
                base_label = "HFDS"
            else:
                base_label = Path(model_path).name
            
            if n_params is not None:
                label = f"{base_label} ({n_params} params)"
            else:
                label = base_label

            plt.errorbar(valid_js, mean_diffs, yerr=std_diffs, label=label, 
                         marker=markers[i % len(markers)], color=colors[i % len(colors)], capsize=5, linestyle='-', alpha=0.8)

    plt.xlabel("$J_2$", fontsize=12)
    plt.ylabel("Energy Difference ($E - E_{exact}$)", fontsize=12)
    
    title = "Energy Difference vs $J_2$"
    if lattice_size:
        title += f" ({lattice_size})"
    plt.title(title, fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='best')
    plt.yscale("log")
    
    if lattice_size:
        save_name = save_name.replace(".png", f"_{lattice_size}.png")
    
    save_path = f"/scratch/f/F.Conoscenti/Thesis_QSL/Elaborate/plot/Errors/{save_name}"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {save_path}")
    plt.show()

def plot_rel_error_vs_js(model_paths, save_name="Rel_Error_vs_J2.png"):
    plt.figure(figsize=(10, 6))
    
    markers = ['o', 's', '^', 'v', 'D', 'p', '*', 'X']
    colors = plt.cm.tab10(np.linspace(0, 1, len(model_paths)))

    for i, model_path in enumerate(model_paths):
        if not os.path.exists(model_path):
            model_path = model_path.replace("/cluster/home/fconoscenti/Thesis_QSL", "/scratch/f/F.Conoscenti/Thesis_QSL")
            if not os.path.exists(model_path):
                print(f"Path not found: {model_path}")
                continue
            
        js = get_available_js(model_path)
        mean_errors = []
        std_errors = []
        
        valid_js = []
        
        for j in js:
            vals = get_rel_error_from_seeds(model_path, j)
            if vals:
                mean_errors.append(np.mean(vals))
                if len(vals) > 1:
                    std_errors.append(np.std(vals) / np.sqrt(len(vals)))
                else:
                    std_errors.append(0.0)
                valid_js.append(j)
        
        if valid_js:
            n_params = get_param_count(model_path)
            if "ViT" in str(model_path):
                base_label = "ViT"
            elif "HFDS" in str(model_path):
                base_label = "HFDS"
            else:
                base_label = Path(model_path).name
            
            if n_params is not None:
                label = f"{base_label} ({n_params} params)"
            else:
                label = base_label

            plt.errorbar(valid_js, mean_errors, yerr=std_errors, label=label, 
                         marker=markers[i % len(markers)], color=colors[i % len(colors)], capsize=5, linestyle='-', alpha=0.8)

    plt.xlabel("$J_2$", fontsize=12)
    plt.ylabel("Relative Error", fontsize=12)
    plt.title("Relative Error vs $J_2$", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='best')
    plt.yscale("log")
    
    save_path = f"/scratch/f/F.Conoscenti/Thesis_QSL/Elaborate/plot/Errors/{save_name}"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {save_path}")
    plt.show()

def plot_fidelity_vs_js(model_paths, save_name="Infidelity_vs_J2.png"):
    plt.figure(figsize=(10, 6))
    
    markers = ['o', 's', '^', 'v', 'D', 'p', '*', 'X']
    colors = plt.cm.Reds(np.linspace(0.5, 1.0, len(model_paths)))

    lattice_size = ""
    if len(model_paths) > 0:
        path_str = str(model_paths[0])
        if "10x10" in path_str:
            lattice_size = "10x10"
        elif "8x8" in path_str:
            lattice_size = "8x8"
        elif "6x6" in path_str:
            lattice_size = "6x6"
        elif "4x4" in path_str:
            lattice_size = "4x4"

    for i, model_path in enumerate(model_paths):
        if not os.path.exists(model_path):
            model_path = model_path.replace("/cluster/home/fconoscenti/Thesis_QSL", "/scratch/f/F.Conoscenti/Thesis_QSL")
            if not os.path.exists(model_path):
                print(f"Path not found: {model_path}")
                continue
            
        js = get_available_js(model_path)
        mean_fidelities = []
        std_fidelities = []
        
        valid_js = []
        
        for j in js:
            vals = get_fidelity_from_seeds(model_path, j)
            if vals:
                mean_fidelities.append(1 - np.mean(vals))
                if len(vals) > 1:
                    std_fidelities.append(np.std(vals) / np.sqrt(len(vals)))
                else:
                    std_fidelities.append(0.0)
                valid_js.append(j)
        
        if valid_js:
            n_params = get_param_count(model_path)
            if "ViT" in str(model_path):
                base_label = "ViT"
            elif "HFDS" in str(model_path):
                base_label = "HFDS"
            else:
                base_label = Path(model_path).name
            
            if n_params is not None:
                label = f"{base_label} ({n_params} params)"
            else:
                label = base_label

            plt.errorbar(valid_js, mean_fidelities, yerr=std_fidelities, label=label, 
                         marker=markers[i % len(markers)], color=colors[i], capsize=5, linestyle='-', alpha=0.8)

    plt.xlabel("$J_2$", fontsize=12)
    plt.ylabel("1 - Fidelity", fontsize=12)
    plt.yscale("log")
    
    title = "1 - Fidelity vs $J_2$"
    if lattice_size:
        title += f" ({lattice_size})"
    plt.title(title, fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='best')
    
    if lattice_size:
        save_name = save_name.replace(".png", f"_{lattice_size}.png")
    
    save_path = f"/scratch/f/F.Conoscenti/Thesis_QSL/Elaborate/plot/Errors/{save_name}"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {save_path}")
    plt.show()

if __name__ == "__main__":

    values_HFDS = None

    values_ViT = None
    
    """models = [
        "/scratch/f/F.Conoscenti/Thesis_QSL/HFDS_Heisenberg/plot/6x6/layers1_hidd6_feat128_sample1024_lr0.02_iter2000_parityTrue_rotTrue_InitFermi_typecomplex",
        "/scratch/f/F.Conoscenti/Thesis_QSL/ViT_Heisenberg/plot/6x6/layers3_d40_heads8_patch2_sample1024_lr0.0075_iter3000_parityTrue_rotTrue_latest_model"
        #"/scratch/f/F.Conoscenti/Thesis_QSL/ViT_Heisenberg/plot/6x6/layers2_d60_heads10_patch2_sample4096_lr0.0075_iter3000_parityTrue_rotTrue_latest_model"
        ]"""
    

    """models=[
        "/scratch/f/F.Conoscenti/Thesis_QSL/HFDS_Heisenberg/plot/4x4/layers1_hidd4_feat64_sample1024_lr0.02_iter1000_parityTrue_rotTrue_InitFermi_typecomplex",
        "/scratch/f/F.Conoscenti/Thesis_QSL/ViT_Heisenberg/plot/4x4/layers2_d16_heads4_patch2_sample1024_lr0.0075_iter4000_parityTrue_rotTrue_latest_model"
            ]"""
    
    """models=[
        "/scratch/f/F.Conoscenti/Thesis_QSL/HFDS_Heisenberg/plot/8x8/layers1_hidd8_feat64_sample4096_lr0.02_iter2000_parityTrue_rotTrue_InitFermi_typecomplex_8",
        "/scratch/f/F.Conoscenti/Thesis_QSL/ViT_Heisenberg/plot/8x8/layers3_d40_heads8_patch2_sample1024_lr0.0075_iter4000_parityTrue_rotTrue_latest_model"
    ]"""

    models=["/scratch/f/F.Conoscenti/Thesis_QSL/HFDS_Heisenberg/plot/10x10/layers1_hidd8_feat32_sample4096_lr0.02_iter2000_parityTrue_rotTrue_InitFermi_typecomplex_10",
            "/scratch/f/F.Conoscenti/Thesis_QSL/ViT_Heisenberg/plot/10x10/layers2_d60_heads10_patch2_sample4096_lr0.0075_iter4000_parityTrue_rotTrue_latest_model"]
    
    
    #plot_rel_error_vs_js(models)
    #plot_energy_vs_js(models)
    plot_energy_diff_vs_js(models, values_HFDS, values_ViT)
    #plot_fidelity_vs_js(models)


    
    folder = "/scratch/f/F.Conoscenti/Thesis_QSL/Elaborate/Plotting/Errors"
    variables = {}
    variables.update({
        "0.4": -0.52371,
        "0.5": -0.4976921,
        "0.55": -0.485434,
        "0.6": -0.47604
    })
    with open(os.path.join(folder, "gs10x10.pkl"), 'wb') as f:
        pickle.dump(variables, f)
    
