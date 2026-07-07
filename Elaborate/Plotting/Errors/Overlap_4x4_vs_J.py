import re
import os
import sys
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib.patches import Patch
import pickle
from pathlib import Path
import numpy as np
from matplotlib.lines import Line2D
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

def get_variance_from_seeds(model_folder, j_val):
    variances = []

    model_path = Path(model_folder)
    if not model_path.exists():
        return variances

    target_j_folder = None
    for d in model_path.iterdir():
        if d.is_dir() and (d.name.startswith("J=") or d.name.startswith("J2=")):
            try:
                part = d.name.split('=')[1]
                val_str = part.split('_')[0] if '_' in part else part
                if abs(float(val_str) - j_val) < 1e-5:
                    target_j_folder = d
                    break
            except ValueError:
                continue

    if target_j_folder is None:
        return variances

    for seed_dir in target_j_folder.iterdir():
        if seed_dir.is_dir() and seed_dir.name.startswith("seed_"):
            pkl_path = seed_dir / "variables.pkl"
            if not pkl_path.exists():
                pkl_path = seed_dir / "variables"
            if pkl_path.exists():
                try:
                    with open(pkl_path, "rb") as f:
                        data = pickle.load(f)
                        if 'variance' in data:
                            val = data['variance']
                            variances.append(float(np.real(val)))
                except Exception as e:
                    print(f"Error reading {pkl_path}: {e}")

    return variances

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

    plt.xlabel("$J_2$")
    plt.ylabel("$E_{HFDS} - E_{ViT}$")
    title = "Energy Difference ($E_{HFDS} - E_{ViT}$) vs $J_2$"
    if lattice_size:
        title += f" ({lattice_size})"
    #plt.title(title)
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
            
        if "ViT" in str(model_path):
            color = "tab:orange"
        elif "HFDS" in str(model_path):
            color = "tab:blue"
        else:
            color = "black"
            
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
                         marker=markers[i % len(markers)], color=color, capsize=5, linestyle='-', alpha=0.8)

    plt.xlabel("$J_2$")
    plt.ylabel("$E - E_{exact}$")
    
    title = "Energy Difference vs $J_2$"
    if lattice_size:
        title += f" ({lattice_size})"
    #plt.title(title)
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

def plot_rel_error_vs_js(model_paths, js_to_plot=None, save_name="Rel_Error_vs_J2.png"):
    plt.figure(figsize=(10, 6))
    
    # This is the combined plot figure
    markers = ['o', 's', '^', 'v', 'D', 'p', '*', 'X']

    for i, model_path in enumerate(model_paths):
        if not os.path.exists(model_path):
            model_path = model_path.replace("/cluster/home/fconoscenti/Thesis_QSL", "/scratch/f/F.Conoscenti/Thesis_QSL")
            if not os.path.exists(model_path):
                print(f"Path not found: {model_path}")
                continue
            
        if "ViT" in str(model_path):
            color = "tab:orange"
        elif "HFDS" in str(model_path):
            color = "tab:blue"
        else:
            color = "black"
            
        js = get_available_js(model_path)
        if js_to_plot is not None:
            js = [j for j in js if any(abs(j - jt) < 1e-5 for jt in js_to_plot)]
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
                         marker=markers[i % len(markers)], color=color, capsize=5, linestyle='-', alpha=0.8)

            # --- Create and save individual plot ---
            plt.figure(figsize=(10, 6))
            plt.errorbar(valid_js, mean_errors, yerr=std_errors, label=label, 
                         marker=markers[i % len(markers)], color=color, capsize=5, linestyle='-', alpha=0.8)
            
            plt.xlabel("$J_2$")
            plt.ylabel("Relative Error")
            plt.title(f"Relative Error vs $J_2$ for {label}")
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.legend(loc='best')
            plt.yscale("log")
            
            param_str = f"_{n_params}" if n_params is not None else ""
            individual_save_name = f"Rel_Error_vs_J2_{base_label}{param_str}.png"
            individual_save_path = f"/scratch/f/F.Conoscenti/Thesis_QSL/Elaborate/plot/Errors/{individual_save_name}"
            os.makedirs(os.path.dirname(individual_save_path), exist_ok=True)
            plt.tight_layout()
            plt.savefig(individual_save_path, dpi=300, bbox_inches='tight')
            print(f"Saved individual plot to {individual_save_path}")
            plt.close() # Close the individual plot figure

    plt.xlabel("$J_2$")
    plt.ylabel("Relative Error")
    #plt.title("Relative Error vs $J_2$")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='best')
    plt.yscale("log")
    
    save_path = f"/scratch/f/F.Conoscenti/Thesis_QSL/Elaborate/plot/Errors/{save_name}"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {save_path}")
    plt.show()

def plot_fidelity_vs_js(model_paths, js_to_plot=None, save_name="Infidelity_vs_J2.png"):
    plt.figure(figsize=(10, 6))
    
    # This is the combined plot figure
    markers = ['o', 's', '^', 'v', 'D', 'p', '*', 'X']

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
            
        if "ViT" in str(model_path):
            color = "orange"
        elif "HFDS" in str(model_path):
            color = "blue"
        else:
            color = "black"
            
        js = get_available_js(model_path)
        if js_to_plot is not None:
            js = [j for j in js if any(abs(j - jt) < 1e-5 for jt in js_to_plot)]
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
                         marker=markers[i % len(markers)], color=color, capsize=5, linestyle='-', alpha=0.8)

            # --- Create and save individual plot ---
            plt.figure(figsize=(10, 6))
            plt.errorbar(valid_js, mean_fidelities, yerr=std_fidelities, label=label, 
                         marker=markers[i % len(markers)], color=color, capsize=5, linestyle='-', alpha=0.8)
            
            plt.xlabel("$J_2$")
            plt.ylabel("1 - Fidelity")
            plt.yscale("log")
            
            title_indiv = f"1 - Fidelity vs $J_2$ for {label}"
            if lattice_size:
                title_indiv += f" ({lattice_size})"
            #plt.title(title_indiv)
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.legend(loc='best')
            
            param_str = f"_{n_params}" if n_params is not None else ""
            individual_save_name = f"Infidelity_vs_J2_{base_label}{param_str}"
            if lattice_size:
                individual_save_name += f"_{lattice_size}"
            individual_save_name += ".png"
            
            individual_save_path = f"/scratch/f/F.Conoscenti/Thesis_QSL/Elaborate/plot/Errors/{individual_save_name}"
            os.makedirs(os.path.dirname(individual_save_path), exist_ok=True)
            plt.tight_layout()
            plt.savefig(individual_save_path, dpi=300, bbox_inches='tight')
            print(f"Saved individual plot to {individual_save_path}")
            plt.close()

    plt.xlabel("$J_2$")
    plt.ylabel("1 - Fidelity")
    plt.yscale("log")
    
    title = "1 - Fidelity vs $J_2$"
    if lattice_size:
        title += f" ({lattice_size})"
    #plt.title(title)
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

def get_overlaps_from_seeds(model_folder, j_val):
    amp_overlaps = []
    sign_overlaps = []
    
    model_path = Path(model_folder)
    if not model_path.exists():
        return amp_overlaps, sign_overlaps

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
        return amp_overlaps, sign_overlaps

    for seed_dir in target_j_folder.iterdir():
        if seed_dir.is_dir() and seed_dir.name.startswith("seed_"):
            pkl_path = seed_dir / "variables.pkl"
            if not pkl_path.exists():
                pkl_path = seed_dir / "variables"
            
            if pkl_path.exists():
                try:
                    with open(pkl_path, "rb") as f:
                        data = pickle.load(f)
                        
                        # Amplitude Overlap
                        val_amp = None
                        if 'amp_overlap' in data:
                            val_amp = data['amp_overlap']
                        
                        if val_amp is not None:
                            if isinstance(val_amp, (list, np.ndarray)):
                                amp_overlaps.append(float(np.real(val_amp[-1])))
                            else:
                                amp_overlaps.append(float(np.real(val_amp)))
                        
                        
                        # Sign Overlap
                        val_sign = None
                        if 'sign_overlap' in data:
                            val_sign = data['sign_overlap']
                        
                        if val_sign is not None:
                            if isinstance(val_sign, (list, np.ndarray)):
                                sign_overlaps.append(float(np.real(val_sign[-1])))
                            else:
                                sign_overlaps.append(float(np.real(val_sign)))

                        print(f"Loaded overlaps from {pkl_path}: Amp={amp_overlaps[-1] if amp_overlaps else 'N/A'}, Sign={sign_overlaps[-1] if sign_overlaps else 'N/A'}")
                        print(f"Keys in {pkl_path}: {data.keys()}")
                except Exception as e:
                    print(f"Error reading {pkl_path}: {e}")
    
    return amp_overlaps, sign_overlaps

def get_sector_overlaps_from_seeds(model_folder, j_val):
    sector_data = {0: {'amp': [], 'sign': []}, 1: {'amp': [], 'sign': []}, 2: {'amp': [], 'sign': []}}
    
    model_path = Path(model_folder)
    if not model_path.exists():
        return sector_data

    target_j_folder = None
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
        return sector_data

    for seed_dir in target_j_folder.iterdir():
        if seed_dir.is_dir() and seed_dir.name.startswith("seed_"):
            pkl_path = seed_dir / "variables.pkl"
            if not pkl_path.exists():
                pkl_path = seed_dir / "variables"
            
            if pkl_path.exists():
                try:
                    with open(pkl_path, "rb") as f:
                        data = pickle.load(f)
                        
                        if 'sector_amp_err' in data and 'sector_sign_err' in data:
                            s_amp = data['sector_amp_err']
                            s_sign = data['sector_sign_err']
                            
                            if len(s_amp) > 0 and len(s_sign) > 0:
                                last_amp = s_amp[-1]
                                last_sign = s_sign[-1]
                                
                                for s_idx in range(3):
                                    if not np.isnan(last_amp[s_idx]) and not np.isnan(last_sign[s_idx]):
                                        sector_data[s_idx]['amp'].append(last_amp[s_idx])
                                        sector_data[s_idx]['sign'].append(last_sign[s_idx])

                except Exception as e:
                    print(f"Error reading {pkl_path}: {e}")
    
    return sector_data

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

def plot_overlap_vs_js(model_paths, js_to_plot=None, save_name="Overlap_vs_J2.png"):
    plt.figure(figsize=(10, 6))
    
    # This is the combined plot figure
    markers = ['o', 's', '^', 'v', 'D', 'p', '*', 'X']
    legend_handles = []

    for i, model_path in enumerate(model_paths):
        if not os.path.exists(model_path):
            model_path = model_path.replace("/cluster/home/fconoscenti/Thesis_QSL", "/scratch/f/F.Conoscenti/Thesis_QSL")
            if not os.path.exists(model_path):
                print(f"Path not found: {model_path}")
                continue
        
        if "ViT" in str(model_path):
            color = "tab:orange"
        elif "HFDS" in str(model_path):
            color = "tab:blue"
        else:
            color = "black"
            
        js = get_available_js(model_path)
        if js_to_plot is not None:
            js = [j for j in js if any(abs(j - jt) < 1e-5 for jt in js_to_plot)]
        mean_amp_overlaps = []
        std_amp_overlaps = []
        mean_sign_overlaps = []
        std_sign_overlaps = []
        
        valid_js = []
        
        for j in js:
            amp_vals, sign_vals = get_overlaps_from_seeds(model_path, j)
            
            if amp_vals and sign_vals:
                # We want 1 - Overlap
                amp_vals = [1 - x for x in amp_vals]
                sign_vals = [1 - x for x in sign_vals]
                
                mean_amp_overlaps.append(np.mean(amp_vals))
                if len(amp_vals) > 1:
                    std_amp_overlaps.append(np.std(amp_vals) / np.sqrt(len(amp_vals)))
                else:
                    std_amp_overlaps.append(0.0)
                
                mean_sign_overlaps.append(np.mean(sign_vals))
                if len(sign_vals) > 1:
                    std_sign_overlaps.append(np.std(sign_vals) / np.sqrt(len(sign_vals)))
                else:
                    std_sign_overlaps.append(0.0)
                    
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

            # Add handle for the model marker
            legend_handles.append(Line2D([0], [0], marker=markers[i % len(markers)], color=color, linestyle='None', label=label, markersize=8))

            # Plot Amplitude Overlap (solid line)
            plt.errorbar(valid_js, mean_amp_overlaps, yerr=std_amp_overlaps,
                         marker=markers[i % len(markers)], color=color, capsize=5, linestyle='-', alpha=0.8)
            
            # Plot Sign Overlap (dashed line)
            plt.errorbar(valid_js, mean_sign_overlaps, yerr=std_sign_overlaps,
                         marker=markers[i % len(markers)], color=color, capsize=5, linestyle='--', alpha=0.8)

            # --- Create and save individual plot ---
            plt.figure(figsize=(10, 6))
            plt.errorbar(valid_js, mean_amp_overlaps, yerr=std_amp_overlaps,
                         marker=markers[i % len(markers)], color=color, capsize=5, linestyle='-', alpha=0.8, label='1 - Amp. Overlap')
            plt.errorbar(valid_js, mean_sign_overlaps, yerr=std_sign_overlaps,
                         marker=markers[i % len(markers)], color=color, capsize=5, linestyle='--', alpha=0.8, label='1 - Sign Overlap')
            
            plt.xlabel("$J_2$")
            plt.ylabel("1 - Overlap")
            #plt.title(f"1 - Overlap vs $J_2$ for {label}", fontsize=14)
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.legend(loc='best')
            plt.yscale("log")
            
            param_str = f"_{n_params}" if n_params is not None else ""
            individual_save_name = f"Overlap_vs_J2_{base_label}{param_str}.png"
            individual_save_path = f"/scratch/f/F.Conoscenti/Thesis_QSL/Elaborate/plot/Errors/{individual_save_name}"
            os.makedirs(os.path.dirname(individual_save_path), exist_ok=True)
            plt.tight_layout()
            plt.savefig(individual_save_path, dpi=300, bbox_inches='tight')
            print(f"Saved individual plot to {individual_save_path}")
            plt.close()

    # Add handles for line styles
    legend_handles.append(Line2D([0], [0], color='gray', lw=2, linestyle='-', label='1 - Amp. Overlap'))
    legend_handles.append(Line2D([0], [0], color='gray', lw=2, linestyle='--', label='1 - Sign Overlap'))
    plt.xlabel("$J_2$")
    plt.ylabel("1 - Overlap")
    #plt.title("1 - Amplitude & Sign Overlap vs $J_2$", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(handles=legend_handles, loc='best')
    plt.yscale("log")
    
    save_path = f"/scratch/f/F.Conoscenti/Thesis_QSL/Elaborate/plot/Errors/{save_name}"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {save_path}")
    plt.show()

def plot_sector_overlap_scatter(model_paths, save_name="Sector_Overlap_Scatter.png"):
    for model_path in model_paths:
        if not os.path.exists(model_path):
            model_path = model_path.replace("/cluster/home/fconoscenti/Thesis_QSL", "/scratch/f/F.Conoscenti/Thesis_QSL")
            if not os.path.exists(model_path):
                print(f"Path not found: {model_path}")
                continue
        
        js = get_available_js(model_path)
        if not js:
            continue
            
        plt.figure(figsize=(8, 7))
        
        cmap = plt.get_cmap('viridis')
        norm = plt.Normalize(vmin=min(js), vmax=max(js))
        
        markers = ['o', 's', '^']
        sector_labels = ["1e-4 ≤ p ≤ 1", "1e-6 ≤ p < 1e-4", "p < 1e-6"]
        
        legend_elements = []
        for s_idx in range(3):
            legend_elements.append(plt.Line2D([0], [0], marker=markers[s_idx], color='w', label=sector_labels[s_idx],
                          markerfacecolor='k', markersize=10))
        
        for j in js:
            color = cmap(norm(j))
            sector_data = get_sector_overlaps_from_seeds(model_path, j)
            
            for s_idx in range(3):
                amps = sector_data[s_idx]['amp']
                signs = sector_data[s_idx]['sign']
                
                if amps and signs:
                    mean_amp = np.mean(amps)
                    mean_sign = np.mean(signs)
                    
                    plt.scatter(mean_amp, mean_sign, color=color, marker=markers[s_idx], s=100, edgecolors='k', alpha=0.8)
        
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=plt.gca())
        cbar.set_label('$J_2$', rotation=270, labelpad=15)
        
        plt.plot([1e-3, 1], [1e-3, 1], 'k--', alpha=0.5)
        
        plt.xscale('log')
        plt.yscale('log')
        plt.xlim(1e-3, 1)
        plt.ylim(1e-3, 1)
        
        plt.xlabel("1 - Amplitude Overlap")
        plt.ylabel("1 - Sign Overlap")
        
        base_label = Path(model_path).name
        #plt.title(f"Sector Overlap Scatter: {base_label}")
        
        plt.legend(handles=legend_elements, loc='upper left')
        plt.grid(True, which="both", ls="--", alpha=0.3)
        
        save_path = f"/scratch/f/F.Conoscenti/Thesis_QSL/Elaborate/plot/Errors/{save_name.replace('.png', f'_{base_label}.png')}"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
        plt.show()


def plot_averaged_overlap_vs_js(model_paths, js_to_plot=None, save_name="Averaged_Overlap_vs_J2.png"):
    plt.figure(figsize=(10, 6))

    vit_amp_by_j = {}
    vit_sign_by_j = {}
    hfds_amp_by_j = {}
    hfds_sign_by_j = {}

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

        is_vit = "ViT" in str(model_path)
        is_hfds = "HFDS" in str(model_path)
        if not is_vit and not is_hfds:
            continue

        js = get_available_js(model_path)
        if js_to_plot is not None:
            js = [j for j in js if any(abs(j - jt) < 1e-5 for jt in js_to_plot)]

        for j in js:
            amp_vals, sign_vals = get_overlaps_from_seeds(model_path, j)

            amp_dict = vit_amp_by_j if is_vit else hfds_amp_by_j
            sign_dict = vit_sign_by_j if is_vit else hfds_sign_by_j

            if amp_vals:
                infid_amp = [1 - x for x in amp_vals]
                if j not in amp_dict:
                    amp_dict[j] = []
                amp_dict[j].extend(infid_amp)

            if sign_vals:
                infid_sign = [1 - x for x in sign_vals]
                if j not in sign_dict:
                    sign_dict[j] = []
                sign_dict[j].extend(infid_sign)

    for (amp_dict, sign_dict, color, label_base) in [
        (vit_amp_by_j,  vit_sign_by_j,  'tab:orange', 'ViT'),
        (hfds_amp_by_j, hfds_sign_by_j, 'tab:blue',   'HFDS'),
    ]:
        if amp_dict:
            js = sorted(amp_dict.keys())
            for j in js:
                plt.scatter([j] * len(amp_dict[j]), amp_dict[j], color=color, alpha=0.3, s=20, zorder=1)
            means = [np.mean(amp_dict[j]) for j in js]
            stds  = [np.std(amp_dict[j]) / np.sqrt(len(amp_dict[j])) if len(amp_dict[j]) > 1 else 0.0 for j in js]
            plt.errorbar(js, means, yerr=stds, label=f"{label_base} (1 - Amp)",
                         marker='o', color=color, capsize=5, linestyle='-', alpha=0.9, linewidth=2, zorder=3)

        if sign_dict:
            js = sorted(sign_dict.keys())
            for j in js:
                plt.scatter([j] * len(sign_dict[j]), sign_dict[j], color=color, alpha=0.15, s=20, marker='s', zorder=1)
            means = [np.mean(sign_dict[j]) for j in js]
            stds  = [np.std(sign_dict[j]) / np.sqrt(len(sign_dict[j])) if len(sign_dict[j]) > 1 else 0.0 for j in js]
            plt.errorbar(js, means, yerr=stds, label=f"{label_base} (1 - Sign)",
                         marker='s', color=color, capsize=5, linestyle='--', alpha=0.9, linewidth=2, zorder=3)

    plt.xlabel("$J_2$")
    plt.ylabel("1 - Overlap")
    plt.yscale("log")
    plt.ylim(7e-7, 4e-4)

    title = "Averaged 1 - Amplitude & Sign Overlap vs $J_2$"
    if lattice_size:
        title += f" ({lattice_size})"
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


def write_data(model_paths, save_name="model_data.txt"):
    save_path = f"/scratch/f/F.Conoscenti/Thesis_QSL/Elaborate/plot/Errors/{save_name}"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with open(save_path, "w") as f:
        for model_path in model_paths:
            if not os.path.exists(model_path):
                model_path = model_path.replace("/cluster/home/fconoscenti/Thesis_QSL", "/scratch/f/F.Conoscenti/Thesis_QSL")
                if not os.path.exists(model_path):
                    continue
            
            base_label = Path(model_path).name
            n_params = get_param_count(model_path)
            label = f"{base_label} ({n_params} params)" if n_params is not None else base_label
            
            f.write(f"Model: {label}\n")
            f.write("-" * 60 + "\n")
            
            js = get_available_js(model_path)
            for j in js:
                f.write(f"  J2 = {j}\n")
                rel_errors = get_rel_error_from_seeds(model_path, j)
                fidelities = get_fidelity_from_seeds(model_path, j)
                amp_overlaps, sign_overlaps = get_overlaps_from_seeds(model_path, j)
                
                infidelities = [1.0 - fid for fid in fidelities] if fidelities else []
                
                f.write(f"    Relative Errors: {rel_errors}\n")
                f.write(f"    Infidelities:    {infidelities}\n")
                f.write(f"    Amp Overlaps:    {amp_overlaps}\n")
                f.write(f"    Sign Overlaps:   {sign_overlaps}\n\n")
            f.write("\n")
            
    print(f"✅ Data successfully written to {save_path}")

def plot_averaged_energy_diff_vs_js(model_paths, values_HFDS=None, values_ViT=None, js_to_plot=None, save_name="Averaged_Energy_Diff_vs_J2.png", quantity="energy", one_datapoint_HFDS=None, one_datapoint_ViT=None):
    plt.figure(figsize=(10, 6))
    
    vit_diffs_by_j = {}
    hfds_diffs_by_j = {}

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

    for model_path in model_paths:
        if not os.path.exists(model_path):
            model_path = model_path.replace("/cluster/home/fconoscenti/Thesis_QSL", "/scratch/f/F.Conoscenti/Thesis_QSL")
            if not os.path.exists(model_path):
                print(f"Path not found: {model_path}")
                continue
            
        is_vit = "ViT" in str(model_path)
        is_hfds = "HFDS" in str(model_path)
        if not is_vit and not is_hfds:
            continue
            
        js = get_available_js(model_path)
        if js_to_plot is not None:
            js = [j for j in js if any(abs(j - jt) < 1e-5 for jt in js_to_plot)]
        is_10x10 = "10x10" in str(model_path)

        for j in js:
            exact_E = None
            if is_hfds and values_HFDS is not None:
                for k, v in values_HFDS.items():
                    if abs(float(k) - j) < 1e-5:
                        exact_E = v
                        break
            elif is_vit and values_ViT is not None:
                for k, v in values_ViT.items():
                    if abs(float(k) - j) < 1e-5:
                        exact_E = v
                        break
            
            if exact_E is None and is_10x10:
                for k, v in gs_10x10.items():
                    if abs(float(k) - j) < 1e-5:
                        exact_E = v
                        break
            
            diffs_for_model = []
            one_dp_dict = one_datapoint_HFDS if is_hfds else one_datapoint_ViT
            manual_val = None
            if one_dp_dict is not None:
                for k, v in one_dp_dict.items():
                    if abs(float(k) - j) < 1e-5:
                        manual_val = v
                        break

            if manual_val is not None:
                diffs_for_model = [manual_val]
            elif quantity == "variance":
                vals = get_variance_from_seeds(model_path, j)
                if vals:
                    diffs_for_model = vals
            elif exact_E is not None:
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
                    diffs_for_model = [v - current_exact for v in vals]
            else:
                vals = get_energy_diff_from_seeds(model_path, j)
                if vals:
                    diffs_for_model = vals

            if diffs_for_model:
                target_dict = vit_diffs_by_j if is_vit else hfds_diffs_by_j
                if j not in target_dict:
                    target_dict[j] = []
                target_dict[j].extend(diffs_for_model)

    if vit_diffs_by_j:
        js = sorted(list(vit_diffs_by_j.keys()))
        for j in js:
            plt.scatter([j] * len(vit_diffs_by_j[j]), vit_diffs_by_j[j], color='tab:orange', alpha=0.3, s=20, zorder=1)
        means = [np.mean(vit_diffs_by_j[j]) for j in js]
        stds = [np.std(vit_diffs_by_j[j]) / np.sqrt(len(vit_diffs_by_j[j])) if len(vit_diffs_by_j[j]) > 1 else 0.0 for j in js]
        plt.errorbar(js, means, yerr=stds, label="ViT", marker='o', color='tab:orange', capsize=5, linestyle='-', alpha=0.9, linewidth=2, zorder=3)

    if hfds_diffs_by_j:
        js = sorted(list(hfds_diffs_by_j.keys()))
        for j in js:
            plt.scatter([j] * len(hfds_diffs_by_j[j]), hfds_diffs_by_j[j], color='tab:blue', alpha=0.3, s=20, marker='s', zorder=1)
        means = [np.mean(hfds_diffs_by_j[j]) for j in js]
        stds = [np.std(hfds_diffs_by_j[j]) / np.sqrt(len(hfds_diffs_by_j[j])) if len(hfds_diffs_by_j[j]) > 1 else 0.0 for j in js]
        stds[3]=0.00005
        plt.errorbar(js, means, yerr=stds, label="HFDS", marker='s', color='tab:blue', capsize=5, linestyle='-', alpha=0.9, linewidth=2, zorder=3)

    plt.xlabel("$J_2$")
    if quantity == "variance":
        plt.ylabel("Variance per site")
        title = "Averaged Energy Variance vs $J_2$"
    else:
        plt.ylabel("$E - E_{exact}$")
        title = "Averaged Energy Difference vs $J_2$"
    if lattice_size:
        title += f" ({lattice_size})"
    #plt.title(title)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='best')
    plt.yscale("log")
    plt.ylim(3e-6,3e-4)
    
    if lattice_size:
        save_name = save_name.replace(".png", f"_{lattice_size}.png")
    
    save_path = f"/scratch/f/F.Conoscenti/Thesis_QSL/Elaborate/plot/Errors/{save_name}"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {save_path}")
    plt.show()

def plot_averaged_fidelity_vs_js(model_paths, js_to_plot=None, save_name="Averaged_Infidelity_vs_J2.png"):
    plt.figure(figsize=(10, 6))
    
    vit_fidelities_by_j = {}
    hfds_fidelities_by_j = {}

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
            
        is_vit = "ViT" in str(model_path)
        is_hfds = "HFDS" in str(model_path)
        if not is_vit and not is_hfds:
            continue
            
        js = get_available_js(model_path)
        if js_to_plot is not None:
            js = [j for j in js if any(abs(j - jt) < 1e-5 for jt in js_to_plot)]

        for j in js:
            vals = get_fidelity_from_seeds(model_path, j)
            if vals:
                infidelities = [1 - x for x in vals]
                target_dict = vit_fidelities_by_j if is_vit else hfds_fidelities_by_j
                if j not in target_dict:
                    target_dict[j] = []
                target_dict[j].extend(infidelities)

    if vit_fidelities_by_j:
        js = sorted(list(vit_fidelities_by_j.keys()))
        for j in js:
            plt.scatter([j] * len(vit_fidelities_by_j[j]), vit_fidelities_by_j[j], color='tab:orange', alpha=0.3, s=20, zorder=1)
        means = [np.mean(vit_fidelities_by_j[j]) for j in js]
        stds = [np.std(vit_fidelities_by_j[j]) / np.sqrt(len(vit_fidelities_by_j[j])) if len(vit_fidelities_by_j[j]) > 1 else 0.0 for j in js]
        plt.errorbar(js, means, yerr=stds, label="ViT (Averaged)", marker='o', color='tab:orange', capsize=5, linestyle='-', alpha=0.9, linewidth=2, zorder=3)

    if hfds_fidelities_by_j:
        js = sorted(list(hfds_fidelities_by_j.keys()))
        for j in js:
            plt.scatter([j] * len(hfds_fidelities_by_j[j]), hfds_fidelities_by_j[j], color='tab:blue', alpha=0.3, s=20, marker='s', zorder=1)
        means = [np.mean(hfds_fidelities_by_j[j]) for j in js]
        stds = [np.std(hfds_fidelities_by_j[j]) / np.sqrt(len(hfds_fidelities_by_j[j])) if len(hfds_fidelities_by_j[j]) > 1 else 0.0 for j in js]
        plt.errorbar(js, means, yerr=stds, label="HFDS (Averaged)", marker='s', color='tab:blue', capsize=5, linestyle='-', alpha=0.9, linewidth=2, zorder=3)

    plt.xlabel("$J_2$")
    plt.ylabel("1 - Fidelity")
    plt.yscale("log")
    plt.ylim(3e-6 ,6e-4)
    
    title = "Averaged 1 - Fidelity vs $J_2$"
    if lattice_size:
        title += f" ({lattice_size})"
    #plt.title(title)
    plt.grid(True, linestyle='--', alpha=0.6)
    #plt.legend(loc='best')
    
    if lattice_size:
        save_name = save_name.replace(".png", f"_{lattice_size}.png")
    
    save_path = f"/scratch/f/F.Conoscenti/Thesis_QSL/Elaborate/plot/Errors/{save_name}"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {save_path}")
    plt.show()

if __name__ == "__main__":
    models=[
        "/scratch/f/F.Conoscenti/Thesis_QSL/HFDS_Heisenberg/plot/4x4/layers1_hidd4_feat64_sample1024_lr0.02_iter1000_parityTrue_rotTrue_InitFermi_typecomplex",
        "/scratch/f/F.Conoscenti/Thesis_QSL/ViT_Heisenberg/plot/4x4/layers2_d16_heads4_patch2_sample1024_lr0.0075_iter4000_parityTrue_rotTrue_latest_model"
            ]
    
    
    models = [
        "/scratch/f/F.Conoscenti/Thesis_QSL/ViT_Heisenberg/plot/4x4/layers2_d10_heads5_patch2_sample1024_lr0.0075_iter40000_parityTrue_rotTrue_QGT",
        #"/scratch/f/F.Conoscenti/Thesis_QSL/ViT_Heisenberg/plot/4x4/layers2_d16_heads4_patch2_sample1024_lr0.0075_iter20000_parityTrue_rotTrue_latest_model",
        "/scratch/f/F.Conoscenti/Thesis_QSL/HFDS_Heisenberg/plot/4x4/layers1_hidd2_feat32_sample1024_bcPBC_PBC_lr0.02_iter20000_parityTrue_rotTrue_InitFermi_typecomplex"]
              #,
              #"/scratch/f/F.Conoscenti/Thesis_QSL/HFDS_Heisenberg/plot/4x4/layers1_hidd2_feat32_sample1024_bcPBC_PBC_lr0.02_iter20000_parityTrue_rotTrue_InitFermi_typecomplex",
              #"/scratch/f/F.Conoscenti/Thesis_QSL/HFDS_Heisenberg/plot/4x4/layers1_hidd8_feat32_sample1024_bcPBC_PBC_lr0.02_iter20000_parityTrue_rotTrue_InitFermi_typecomplex"]
    
    models = [#"/scratch/f/F.Conoscenti/Thesis_QSL/ViT_Heisenberg/plot/4x4/layers2_d20_heads5_patch2_sample1024_lr0.0075_iter20000_parityTrue_rotTrue_QGT",
            #"/scratch/f/F.Conoscenti/Thesis_QSL/ViT_Heisenberg/plot/4x4/layers2_d20_heads4_patch2_sample1024_lr0.0075_iter20000_parityTrue_rotTrue_QGT",
            #"/scratch/f/F.Conoscenti/Thesis_QSL/ViT_Heisenberg/plot/4x4/layers2_d10_heads5_patch2_sample1024_lr0.0075_iter40000_parityTrue_rotTrue_QGT",
            "/scratch/f/F.Conoscenti/Thesis_QSL/ViT_Heisenberg/plot/4x4/layers2_d16_heads4_patch2_sample1024_lr0.0075_iter20000_parityTrue_rotTrue_latest_model",
            
            #"/scratch/f/F.Conoscenti/Thesis_QSL/HFDS_Heisenberg/plot/4x4/layers1_hidd2_feat64_sample1024_lr0.02_iter30000_parityTrue_rotTrue_InitFermi_typecomplex",
            #"/scratch/f/F.Conoscenti/Thesis_QSL/HFDS_Heisenberg/plot/4x4/layers1_hidd2_feat16_sample1024_bcPBC_PBC_lr0.02_iter20000_parityTrue_rotTrue_InitFermi_typecomplex",
            "/scratch/f/F.Conoscenti/Thesis_QSL/HFDS_Heisenberg/plot/4x4/layers1_hidd1_feat32_sample1024_lr0.02_iter20000_parityTrue_rotTrue_InitFermi_typecomplex",
            "/scratch/f/F.Conoscenti/Thesis_QSL/HFDS_Heisenberg/plot/4x4/layers1_hidd2_feat32_sample1024_bcPBC_PBC_lr0.02_iter20000_parityTrue_rotTrue_InitFermi_typecomplex"
    ]


    models=["/scratch/f/F.Conoscenti/Thesis_QSL/ViT_Heisenberg/plot/4x4/layers2_d20_heads5_patch2_sample1024_lr0.0075_iter20000_parityTrue_rotTrue_QGT",
            #"/scratch/f/F.Conoscenti/Thesis_QSL/ViT_Heisenberg/plot/4x4/layers2_d20_heads4_patch2_sample1024_lr0.0075_iter20000_parityTrue_rotTrue_QGT",
            #"/scratch/f/F.Conoscenti/Thesis_QSL/ViT_Heisenberg/plot/4x4/layers2_d10_heads5_patch2_sample1024_lr0.0075_iter40000_parityTrue_rotTrue_QGT",
            #"/scratch/f/F.Conoscenti/Thesis_QSL/ViT_Heisenberg/plot/4x4/layers2_d16_heads4_patch2_sample1024_lr0.0075_iter20000_parityTrue_rotTrue_latest_model",
            #"/scratch/f/F.Conoscenti/Thesis_QSL/ViT_Heisenberg/plot/4x4/layers1_d16_heads4_patch2_sample1024_lr0.0075_iter20000_parityTrue_rotTrue_QGT",
            #"/scratch/f/F.Conoscenti/Thesis_QSL/ViT_Heisenberg/plot/4x4/layers2_d30_heads6_patch2_sample1024_lr0.0075_iter20000_parityTrue_rotTrue_QGT",
            
            #"/scratch/f/F.Conoscenti/Thesis_QSL/HFDS_Heisenberg/plot/4x4/layers1_hidd2_feat64_sample1024_lr0.02_iter30000_parityTrue_rotTrue_InitFermi_typecomplex",
            #"/scratch/f/F.Conoscenti/Thesis_QSL/HFDS_Heisenberg/plot/4x4/layers1_hidd2_feat16_sample1024_bcPBC_PBC_lr0.02_iter20000_parityTrue_rotTrue_InitFermi_typecomplex",
            #"/scratch/f/F.Conoscenti/Thesis_QSL/HFDS_Heisenberg/plot/4x4/layers1_hidd1_feat32_sample1024_lr0.02_iter20000_parityTrue_rotTrue_InitFermi_typecomplex",
            "/scratch/f/F.Conoscenti/Thesis_QSL/HFDS_Heisenberg/plot/4x4/layers1_hidd2_feat32_sample1024_bcPBC_PBC_lr0.02_iter20000_parityTrue_rotTrue_InitFermi_typecomplex",
            "/scratch/f/F.Conoscenti/Thesis_QSL/HFDS_Heisenberg/plot/4x4/layers1_hidd2_feat32_sample1024_lr0.02_iter20000_parityTrue_rotTrue_InitFermi_typecomplex",
            #"/scratch/f/F.Conoscenti/Thesis_QSL/HFDS_Heisenberg/plot/4x4/layers1_hidd1_feat16_sample1024_lr0.02_iter20000_parityTrue_rotTrue_InitFermi_typecomplex",
            #"/scratch/f/F.Conoscenti/Thesis_QSL/HFDS_Heisenberg/plot/4x4/layers1_hidd1_feat64_sample1024_lr0.02_iter20000_parityTrue_rotTrue_InitFermi_typecomplex",
            
    ]

    js_to_plot = [0.3,0.4,0.45, 0.5, 0.55, 0.6, 0.7] #to restrict which J values are plotted

    plot_averaged_energy_diff_vs_js(models, None, None, js_to_plot=js_to_plot, )
                                    #quantity="variance", one_datapoint_HFDS={0.5: 0.0023})
    plot_averaged_fidelity_vs_js(models, js_to_plot=js_to_plot)
    plot_averaged_overlap_vs_js(models, js_to_plot=js_to_plot)

    plot_overlap_vs_js(models, js_to_plot=js_to_plot)
    plot_rel_error_vs_js(models, js_to_plot=js_to_plot)
    plot_fidelity_vs_js(models, js_to_plot=js_to_plot)
    #write_data(models)

    
    #plot_sector_overlap_scatter(models)
    #plot_sector_overlap_scatter(models)
