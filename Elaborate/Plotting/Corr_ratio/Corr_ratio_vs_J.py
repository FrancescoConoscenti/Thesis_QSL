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

def get_corr_ratio_from_seeds(model_folder, j_val):
    r_values = []
    
    model_path = Path(model_folder)
    if not model_path.exists():
        return r_values

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
        return r_values

    for seed_dir in target_j_folder.iterdir():
        if seed_dir.is_dir() and seed_dir.name.startswith("seed_"):
            pkl_path = seed_dir / "variables.pkl"
            if not pkl_path.exists():
                pkl_path = seed_dir / "variables"
            
            if pkl_path.exists():
                try:
                    with open(pkl_path, "rb") as f:
                        data = pickle.load(f)
                        if 'R' in data:
                            r_values.append(data['R'])
                except Exception as e:
                    print(f"Error reading {pkl_path}: {e}")
    
    return r_values

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

def plot_corr_ratio_vs_js(model_paths, save_name="Corr_Ratio_vs_J2.png"):
    plt.figure(figsize=(10, 6))
    
    markers = ['o', 's', '^', 'v', 'D', 'p', '*', 'X']
    
    vit_paths = [p for p in model_paths if "ViT" in str(p)]
    hfds_paths = [p for p in model_paths if "HFDS" in str(p)]
    
    vit_colors = plt.cm.Reds(np.linspace(0.5, 1.0, len(vit_paths))) if vit_paths else []
    hfds_colors = plt.cm.Blues(np.linspace(0.5, 1.0, len(hfds_paths))) if hfds_paths else []

    filename_suffix = ""
    first_info_captured = False

    for i, model_path in enumerate(model_paths):
        if not os.path.exists(model_path):
            model_path = model_path.replace("/cluster/home/fconoscenti/Thesis_QSL", "/scratch/f/F.Conoscenti/Thesis_QSL")
            if not os.path.exists(model_path):
                print(f"Path not found: {model_path}")
                continue
        
        if "ViT" in str(model_paths[i]):
            color = vit_colors[vit_paths.index(model_paths[i])]
        elif "HFDS" in str(model_paths[i]):
            color = hfds_colors[hfds_paths.index(model_paths[i])]
        else:
            color = 'black'
            
        js = get_available_js(model_path)
        mean_r = []
        std_r = []
        
        valid_js = []
        
        for j in js:
            vals = get_corr_ratio_from_seeds(model_path, j)
            if vals:
                mean_r.append(np.mean(vals))
                if len(vals) > 1:
                    std_r.append(np.std(vals) / np.sqrt(len(vals)))
                else:
                    std_r.append(0.0)
                valid_js.append(j)
        
        if valid_js:
            n_params = get_param_count(model_path)
            
            size_match = re.search(r"(\d+x\d+)", str(model_path))
            lattice_size = size_match.group(1) if size_match else ""

            if "ViT" in str(model_path):
                base_label = f"ViT {lattice_size}"
            elif "HFDS" in str(model_path):
                base_label = f"HFDS {lattice_size}"
            else:
                base_label = Path(model_path).name
                if lattice_size and lattice_size not in base_label:
                    base_label += f" {lattice_size}"
            
            base_label = base_label.strip()
            
            if not first_info_captured:
                m_type = "Model"
                if "ViT" in str(model_path): m_type = "ViT"
                elif "HFDS" in str(model_path): m_type = "HFDS"
                filename_suffix = f"_{m_type}"
                if n_params is not None:
                    filename_suffix += f"_{n_params}params"
                first_info_captured = True

            label = base_label

            plt.errorbar(valid_js, mean_r, yerr=std_r, label=label, 
                         marker=markers[i % len(markers)], color=color, capsize=5, linestyle='-', alpha=0.8)

    plt.xlabel("$J_2$", fontsize=12)
    plt.ylabel("Correlation Ratio $R$", fontsize=12)
    plt.title("Correlation Ratio $R$ vs $J_2$", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='best')
    
    if save_name.endswith(".png"):
        final_save_name = save_name[:-4] + filename_suffix + ".png"
    else:
        final_save_name = save_name + filename_suffix

    save_path = f"/scratch/f/F.Conoscenti/Thesis_QSL/Elaborate/plot/Correlation/{final_save_name}"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {save_path}")
    plt.show()

if __name__ == "__main__":
    models = [
        "/scratch/f/F.Conoscenti/Thesis_QSL/HFDS_Heisenberg/plot/6x6/layers1_hidd6_feat128_sample1024_lr0.02_iter2000_parityTrue_rotTrue_InitFermi_typecomplex",
        "/scratch/f/F.Conoscenti/Thesis_QSL/HFDS_Heisenberg/plot/4x4/layers1_hidd4_feat64_sample1024_lr0.02_iter1000_parityTrue_rotTrue_InitFermi_typecomplex",
        "/scratch/f/F.Conoscenti/Thesis_QSL/ViT_Heisenberg/plot/4x4/layers2_d16_heads4_patch2_sample1024_lr0.0075_iter4000_parityTrue_rotTrue_latest_model",
        "/scratch/f/F.Conoscenti/Thesis_QSL/ViT_Heisenberg/plot/6x6/layers3_d40_heads8_patch2_sample1024_lr0.0075_iter3000_parityTrue_rotTrue_latest_model",
        "/scratch/f/F.Conoscenti/Thesis_QSL/ViT_Heisenberg/plot/8x8/layers3_d40_heads8_patch2_sample1024_lr0.0075_iter4000_parityTrue_rotTrue_latest_model"
    ]
    
    plot_corr_ratio_vs_js(models)
