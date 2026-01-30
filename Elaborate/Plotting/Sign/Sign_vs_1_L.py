import os
import re
import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def extract_L(path_str):
    """Extracts L from path string assuming format like '4x4', '6x6'."""
    match = re.search(r"(\d+)x(\d+)", str(path_str))
    if match:
        return int(match.group(1))
    return None

def get_marshall_signs_for_J(model_folder, target_J):
    """
    Finds the subfolder for target_J in model_folder, reads variables.pkl from seeds,
    and returns a list of Marshall sign values.
    """
    signs = []
    model_path = Path(model_folder)
    if not model_path.exists():
        # Try replacing cluster path with scratch if needed
        if "/cluster/home/" in str(model_path):
             model_path = Path(str(model_path).replace("/cluster/home/fconoscenti", "/scratch/f/F.Conoscenti"))
        
        if not model_path.exists():
            print(f"Path not found: {model_path}")
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
                
                if abs(float(val_str) - target_J) < 1e-5:
                    target_j_folder = d
                    break
            except ValueError:
                continue
    
    if target_j_folder is None:
        print(f"J={target_J} folder not found in {model_path}")
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
                        # Prefer MCMC sign if available
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

def plot_sign_vs_inv_L(datasets, target_J, save_name="MarshallSign_vs_1_L"):
    plt.figure(figsize=(8, 6))
    
    for label, models, color, marker in datasets:
        inv_Ls = []
        means = []
        errors = []
        
        for model_path in models:
            L = extract_L(model_path)
            if L is None:
                print(f"Could not extract L from {model_path}")
                continue
            
            signs = get_marshall_signs_for_J(model_path, target_J)
            if signs:
                inv_Ls.append(1.0/L)
                means.append(np.mean(np.abs(signs)))
                # Standard error of the mean across seeds
                if len(signs) > 1:
                    errors.append(np.std(np.abs(signs)) / np.sqrt(len(signs)))
                else:
                    errors.append(0.0)
            else:
                print(f"No signs found for {model_path} at J={target_J}")

        if inv_Ls:
            # Sort by 1/L
            sorted_indices = np.argsort(inv_Ls)
            inv_Ls = np.array(inv_Ls)[sorted_indices]
            means = np.array(means)[sorted_indices]
            errors = np.array(errors)[sorted_indices]
            
            plt.errorbar(inv_Ls, means, yerr=errors, label=label, 
                         color=color, marker=marker, capsize=5, linestyle='-', alpha=0.8)

    plt.xlabel("$1/L$", fontsize=14)
    plt.ylabel("Mean Marshall Sign", fontsize=14)
    plt.title(f"Marshall Sign vs $1/L$ at $J_2={target_J}$", fontsize=16)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend(fontsize=12)
    plt.xlim(left=0)

    save_dir = Path("/scratch/f/F.Conoscenti/Thesis_QSL/Elaborate/plot/Sign")
    save_dir.mkdir(parents=True, exist_ok=True)
    full_save_path = save_dir / (save_name+"J="+str(target_J)+".png")
    
    plt.savefig(full_save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {full_save_path}")
    plt.show()

if __name__ == "__main__":
    
    target_J = 0.6
    
    # --- Define your datasets here ---
    # Add paths for different system sizes (4x4, 6x6, etc.) for each model type
    
    vit_models = [
        "/scratch/f/F.Conoscenti/Thesis_QSL/ViT_Heisenberg/plot/6x6/layers3_d40_heads8_patch2_sample1024_lr0.0075_iter3000_parityTrue_rotTrue_latest_model",
        # 4x4
        "/scratch/f/F.Conoscenti/Thesis_QSL/ViT_Heisenberg/plot/4x4/layers2_d16_heads4_patch2_sample1024_lr0.0075_iter4000_parityTrue_rotTrue_latest_model",
        # 6x6
        "/scratch/f/F.Conoscenti/Thesis_QSL/ViT_Heisenberg/plot/6x6/layers2_d16_heads4_patch2_sample1024_lr0.0075_iter4000_parityTrue_rotTrue_latest_model"
    ]
    
    hfds_models = [
        # 4x4
        "/scratch/f/F.Conoscenti/Thesis_QSL/HFDS_Heisenberg/plot/4x4/layers1_hidd6_feat128_sample1024_lr0.02_iter500_parityTrue_rotTrue_InitFermi_typecomplex",
        # 6x6
        "/scratch/f/F.Conoscenti/Thesis_QSL/HFDS_Heisenberg/plot/6x6/layers1_hidd6_feat128_sample1024_lr0.02_iter2000_parityTrue_rotTrue_InitFermi_typecomplex"
    ]

    datasets = [
        ("ViT", vit_models, "tab:orange", "o"),
        ("HFDS", hfds_models, "tab:blue", "s")
    ]
    
    plot_sign_vs_inv_L(datasets, target_J)
