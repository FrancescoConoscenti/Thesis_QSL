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

def plot_overlap_vs_js(model_paths, save_name="Overlap_vs_J2.png"):
    plt.figure(figsize=(10, 6))
    
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

            # Plot Amplitude Overlap (solid line)
            plt.errorbar(valid_js, mean_amp_overlaps, yerr=std_amp_overlaps, label=f"{label} (1-Amp)", 
                         marker=markers[i % len(markers)], color=color, capsize=5, linestyle='-', alpha=0.8)
            
            # Plot Sign Overlap (dashed line)
            plt.errorbar(valid_js, mean_sign_overlaps, yerr=std_sign_overlaps, label=f"{label} (1-Sign)", 
                         marker=markers[i % len(markers)], color=color, capsize=5, linestyle='--', alpha=0.8)

    plt.xlabel("$J_2$", fontsize=12)
    plt.ylabel("1 - Overlap", fontsize=12)
    plt.title("1 - Amplitude & Sign Overlap vs $J_2$", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='best')
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
        plt.title(f"Sector Overlap Scatter: {base_label}")
        
        plt.legend(handles=legend_elements, loc='upper left')
        plt.grid(True, which="both", ls="--", alpha=0.3)
        
        save_path = f"/scratch/f/F.Conoscenti/Thesis_QSL/Elaborate/plot/Errors/{save_name.replace('.png', f'_{base_label}.png')}"
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
    
    plot_overlap_vs_js(models)
    plot_sector_overlap_scatter(models)
