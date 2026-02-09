import re
import os
import matplotlib.pyplot as plt
import pickle
from pathlib import Path
import numpy as np


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

# --- Helper to read Marshall Sign from seeds ---
def get_marshall_signs_from_seeds(model_folder, j_val):
    signs = []
    model_path = Path(model_folder)
    if not model_path.exists():
        return signs

    target_j_folder = None
    # Robustly find J folder
    for d in model_path.iterdir():
        if d.is_dir() and (d.name.startswith("J=") or d.name.startswith("J2=")):
            try:
                # Extract number after = and before any _
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
        return signs

    for seed_dir in target_j_folder.iterdir():
        if seed_dir.is_dir() and seed_dir.name.startswith("seed_"):
            pkl_path = seed_dir / "variables.pkl"
            if not pkl_path.exists():
                pkl_path = seed_dir / "variables"
            
            if pkl_path.exists():
                try:
                    with open(pkl_path, "rb") as f:
                        data = pickle.load(f)
                        # Prefer MCMC sign if available, else look for others
                        if 'sign_vstate_MCMC' in data:
                            val = data['sign_vstate_MCMC']
                            signs.append(np.real(val))
                        elif 'sign_vstate' in data:
                             val = data['sign_vstate']
                             if isinstance(val, (list, np.ndarray)):
                                 signs.append(np.real(val[-1]))
                             else:
                                 signs.append(np.real(val))
                except Exception as e:
                    print(f"Error reading {pkl_path}: {e}")
    return signs

#main
if __name__ == "__main__":

    models_HFDS = ["/scratch/f/F.Conoscenti/Thesis_QSL/HFDS_Heisenberg/plot/6x6/layers1_hidd6_feat128_sample1024_lr0.02_iter2000_parityTrue_rotTrue_InitFermi_typecomplex",
                   "/scratch/f/F.Conoscenti/Thesis_QSL/HFDS_Heisenberg/plot/4x4/layers1_hidd6_feat128_sample1024_lr0.02_iter500_parityTrue_rotTrue_InitFermi_typecomplex",
                   "/scratch/f/F.Conoscenti/Thesis_QSL/HFDS_Heisenberg/plot/8x8/layers1_hidd8_feat64_sample4096_lr0.02_iter2000_parityTrue_rotTrue_InitFermi_typecomplex_8",
                   "/scratch/f/F.Conoscenti/Thesis_QSL/HFDS_Heisenberg/plot/10x10/layers1_hidd8_feat32_sample4096_lr0.02_iter2000_parityTrue_rotTrue_InitFermi_typecomplex_10"
                ]
    models_ViT = [
        "/scratch/f/F.Conoscenti/Thesis_QSL/ViT_Heisenberg/plot/6x6/layers3_d40_heads8_patch2_sample1024_lr0.0075_iter3000_parityTrue_rotTrue_latest_model",
        "/scratch/f/F.Conoscenti/Thesis_QSL/ViT_Heisenberg/plot/8x8/layers3_d40_heads8_patch2_sample1024_lr0.0075_iter4000_parityTrue_rotTrue_latest_model",
        "/scratch/f/F.Conoscenti/Thesis_QSL/ViT_Heisenberg/plot/4x4/layers2_d16_heads4_patch2_sample1024_lr0.0075_iter4000_parityTrue_rotTrue_latest_model"
        ]

    # --- Plot setup ---
    fig, ax = plt.subplots(figsize=(8, 5))

    # --- Style setup ---
    lattice_markers = {"4x4": "o", "6x6": "s", "8x8": "D", "10x10": "^"}

    # --- Plot HFDS models ---
    for i, model_path in enumerate(models_HFDS):
        if model_path and os.path.exists(model_path):
            print(f"Processing HFDS: {model_path}")
            
            if "4x4" in model_path:
                label_name = "HFDS 4x4"
                color = "cornflowerblue"
                marker = lattice_markers["4x4"]
            elif "6x6" in model_path:
                label_name = "HFDS 6x6"
                color = "navy"
                marker = lattice_markers["6x6"]
            elif "8x8" in model_path:
                label_name = "HFDS 8x8"
                color = "blue"
                marker = lattice_markers["8x8"]
            elif "10x10" in model_path:
                label_name = "HFDS 10x10"
                color = "navy"
                marker = lattice_markers["10x10"]
            else:
                label_name = f"HFDS {i+1}"
                color = "tab:blue"
                marker = "v"
            
            js = []
            means = []
            
            for J_val in get_available_js(model_path):
                signs = get_marshall_signs_from_seeds(model_path, J_val)
                if signs:
                    js.append(J_val)
                    means.append(np.mean(np.abs(signs)))
            
            if js:
                ax.plot(js, means, color=color, marker=marker, markersize=8, alpha=0.7, label=label_name)

    # --- Plot ViT models ---
    for i, model_path in enumerate(models_ViT):
        if model_path and os.path.exists(model_path):
            print(f"Processing ViT: {model_path}")
            
            if "4x4" in model_path:
                label_name = "ViT 4x4"
                color = "sandybrown"
                marker = lattice_markers["4x4"]
            elif "6x6" in model_path:
                label_name = "ViT 6x6"
                color = "chocolate"
                marker = lattice_markers["6x6"]
            elif "8x8" in model_path:
                label_name = "ViT 8x8"
                color = "brown"
                marker = lattice_markers["8x8"]
            elif "10x10" in model_path:
                label_name = "ViT 10x10"
                color = "chocolate"
                marker = lattice_markers["10x10"]
            else:
                label_name = f"ViT {i+1}"
                color = "tab:orange"
                marker = "v"
            
            js = []
            means = []
            
            for J_val in get_available_js(model_path):
                signs = get_marshall_signs_from_seeds(model_path, J_val)
                if signs:
                    js.append(J_val)
                    means.append(np.mean(np.abs(signs)))
            
            if js:
                ax.plot(js, means, color=color, marker=marker, markersize=8, alpha=0.7, label=label_name)

    # --- Labels, title, legend ---
    ax.set_xlabel("$J_2$", fontsize=12)
    ax.set_ylabel("Marshall Sign", fontsize=12)
    ax.set_title("Marshall Sign vs $J_2$")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc='best', fontsize=10)

    fig.tight_layout()
    save_path = "Elaborate/plot/Sign/MarshallSign_vs_J2_ViT_HFDS.png"
    if not os.path.exists(os.path.dirname(save_path)):
        save_path = os.path.join("/scratch/f/F.Conoscenti/Thesis_QSL", save_path)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Plot saved to {save_path}")
    plt.show()
    