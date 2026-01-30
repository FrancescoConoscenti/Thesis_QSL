import re
import os
import sys
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib.patches import Patch
import pickle
from pathlib import Path
import numpy as np

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

if __name__ == "__main__":
    models = [
        "/scratch/f/F.Conoscenti/Thesis_QSL/ViT_Heisenberg/plot/6x6/layers3_d40_heads8_patch2_sample1024_lr0.0075_iter3000_parityTrue_rotTrue_latest_model",
        "/scratch/f/F.Conoscenti/Thesis_QSL/ViT_Heisenberg/plot/6x6/layers2_d16_heads4_patch2_sample1024_lr0.0075_iter4000_parityTrue_rotTrue_latest_model",
        "/scratch/f/F.Conoscenti/Thesis_QSL/HFDS_Heisenberg/plot/6x6/layers1_hidd6_feat128_sample1024_lr0.02_iter2000_parityTrue_rotTrue_InitFermi_typecomplex"
    ]
    
    plot_rel_error_vs_js(models)
