import re
import os
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib.patches import Patch
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

    model_HFDS = None
    model_ViT = "/scratch/f/F.Conoscenti/Thesis_QSL/ViT_Heisenberg/plot/6x6/layers2_d24_heads4_patch2_sample1024_lr0.0075_iter500_parityTrue_rotTrue_latest_model"

    # --- Plot setup ---
    fig, ax = plt.subplots(figsize=(8, 5))

    # --- Style setup ---
    marker_ViT = "o"
    marker_HFDS = "^"
    color_ViT = "tab:orange"
    color_HFDS = "tab:blue"

    # --- Plot HFDS model ---
    if model_HFDS and os.path.exists(model_HFDS):
        print(f"Processing HFDS: {model_HFDS}")
        first_plot = True
        for J_val in get_available_js(model_HFDS):
            signs = get_marshall_signs_from_seeds(model_HFDS, J_val)
            if signs:
                ax.scatter([J_val]*len(signs), signs, color=color_HFDS, marker=marker_HFDS, s=80, alpha=0.7, edgecolors='none', label="HFDS" if first_plot else "")
                first_plot = False

    # --- Plot ViT model ---
    if model_ViT and os.path.exists(model_ViT):
        print(f"Processing ViT: {model_ViT}")
        first_plot = True
        for J_val in get_available_js(model_ViT):
            signs = get_marshall_signs_from_seeds(model_ViT, J_val)
            if signs:
                ax.scatter([J_val]*len(signs), signs, color=color_ViT, marker=marker_ViT, s=80, alpha=0.7, edgecolors='none', label="ViT" if first_plot else "")
                first_plot = False

    # --- Labels, title, legend ---
    ax.set_xlabel("$J_2$", fontsize=12)
    ax.set_ylabel("Marshall Sign", fontsize=12)
    ax.set_title("Marshall Sign vs $J_2$")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(frameon=False)

    fig.tight_layout()
    save_path = "Elaborate/plot/MarshallSign_vs_J2_ViT_HFDS.png"
    if not os.path.exists(os.path.dirname(save_path)):
        save_path = os.path.join("/scratch/f/F.Conoscenti/Thesis_QSL", save_path)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Plot saved to {save_path}")
    plt.show()
    