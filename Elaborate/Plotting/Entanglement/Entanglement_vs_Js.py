import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import re

def get_available_js(model_folder):
    js = set()
    model_path = Path(model_folder)
    if not model_path.exists():
        return []
    
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

def get_entanglement_from_seeds(model_folder, j_val):
    s2_values = []
    s2_errors = []
    
    model_path = Path(model_folder)
    if not model_path.exists():
        return s2_values, s2_errors

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
        return s2_values, s2_errors

    for seed_dir in target_j_folder.iterdir():
        if seed_dir.is_dir() and seed_dir.name.startswith("seed_"):
            pkl_path = seed_dir / "variables.pkl"
            if not pkl_path.exists():
                pkl_path = seed_dir / "variables"
            
            if pkl_path.exists():
                try:
                    with open(pkl_path, "rb") as f:
                        data = pickle.load(f)
                        if 's2' in data:
                            s2_values.append(data['s2'])
                            if 's2_error' in data:
                                s2_errors.append(data['s2_error'])
                            else:
                                s2_errors.append(0.0)
                except Exception as e:
                    print(f"Error reading {pkl_path}: {e}")
    
    return s2_values, s2_errors

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
                        except Exception:
                            pass
    return None

def plot_entanglement_vs_js(model_paths, save_name="Entanglement_vs_J2.png"):
    plt.figure(figsize=(10, 6))
    
    markers = ['o', 's', '^', 'v', 'D', 'p', '*', 'X']
    colors = plt.cm.tab10(np.linspace(0, 1, len(model_paths)))

    filename_suffix = ""
    first_info_captured = False

    for i, model_path in enumerate(model_paths):
        if not os.path.exists(model_path):
            model_path = model_path.replace("/cluster/home/fconoscenti/Thesis_QSL", "/scratch/f/F.Conoscenti/Thesis_QSL")
            if not os.path.exists(model_path):
                print(f"Path not found: {model_path}")
                continue
            
        js = get_available_js(model_path)
        mean_s2 = []
        std_s2 = []
        
        valid_js = []
        
        for j in js:
            vals, errs = get_entanglement_from_seeds(model_path, j)
            if vals:
                mean_s2.append(np.mean(vals))
                if len(vals) > 1:
                    std_s2.append(np.std(vals) / np.sqrt(len(vals)))
                else:
                    std_s2.append(errs[0] if errs else 0.0)
                valid_js.append(j)
        
        if valid_js:
            n_params = get_param_count(model_path)
            if "ViT" in str(model_path):
                base_label = "ViT"
            elif "HFDS" in str(model_path):
                base_label = "HFDS"
            else:
                base_label = Path(model_path).name
            
            if not first_info_captured:
                m_type = "Model"
                if "ViT" in str(model_path): m_type = "ViT"
                elif "HFDS" in str(model_path): m_type = "HFDS"
                filename_suffix = f"_{m_type}"
                if n_params is not None:
                    filename_suffix += f"_{n_params}params"
                first_info_captured = True

            if n_params is not None:
                label = f"{base_label} ({n_params} params)"
            else:
                label = base_label

            plt.errorbar(valid_js, mean_s2, yerr=std_s2, label=label, 
                         marker=markers[i % len(markers)], color=colors[i % len(colors)], capsize=5, linestyle='-', alpha=0.8)

    plt.xlabel("$J_2$", fontsize=12)
    plt.ylabel("Renyi Entropy $S_2$", fontsize=12)
    plt.title("Entanglement Entropy $S_2$ vs $J_2$", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='best')
    
    if save_name.endswith(".png"):
        final_save_name = save_name[:-4] + filename_suffix + ".png"
    else:
        final_save_name = save_name + filename_suffix

    save_path = f"/scratch/f/F.Conoscenti/Thesis_QSL/Elaborate/plot/Entanglement/{final_save_name}"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {save_path}")
    plt.show()

if __name__ == "__main__":
    models = [
        "/scratch/f/F.Conoscenti/Thesis_QSL/ViT_Heisenberg/plot/6x6/layers2_d16_heads4_patch2_sample1024_lr0.0075_iter4000_parityTrue_rotTrue_latest_model",
        "/scratch/f/F.Conoscenti/Thesis_QSL/HFDS_Heisenberg/plot/6x6/layers1_hidd6_feat128_sample1024_lr0.02_iter2000_parityTrue_rotTrue_InitFermi_typecomplex"
    ]
    
    plot_entanglement_vs_js(models)