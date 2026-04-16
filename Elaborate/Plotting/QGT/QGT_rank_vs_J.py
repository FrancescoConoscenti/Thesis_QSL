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

def get_qgt_rank_from_seeds(model_folder, j_val):
    ranks = []
    
    model_path = Path(model_folder)
    if not model_path.exists():
        return ranks

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
        return ranks

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
                        if 'mean_rest_norm' in data:
                            val = data['mean_rest_norm']
              
                        if val is not None:
                            if isinstance(val, (list, np.ndarray)):
                                ranks.append(float(np.real(val[-1])))
                            else:
                                ranks.append(float(np.real(val)))
                except Exception as e:
                    print(f"Error reading {pkl_path}: {e}")
    
    return ranks

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

def plot_qgt_rank_vs_js(model_paths, save_name="QGT_Rank_vs_J2.png"):
    plt.figure(figsize=(10, 6))
    
    markers = ['o', 's', '^', 'v', 'D', 'p', '*', 'X']
    colors = plt.cm.tab10(np.linspace(0, 1, len(model_paths)))
    
    all_data = []
    all_js_set = set()
    all_mean_ranks = []

    for i, model_path in enumerate(model_paths):
        if not os.path.exists(model_path):
            model_path = model_path.replace("/cluster/home/fconoscenti/Thesis_QSL", "/scratch/f/F.Conoscenti/Thesis_QSL")
            if not os.path.exists(model_path):
                print(f"Path not found: {model_path}")
                continue
            
        js = get_available_js(model_path)
        mean_ranks = []
        std_ranks = []
        
        valid_js = []
        
        for j in js:
            vals = get_qgt_rank_from_seeds(model_path, j)
            if vals:
                mean_val = np.mean(vals)
                mean_ranks.append(mean_val)
                all_mean_ranks.append(mean_val)
                if len(vals) > 1:
                    std_ranks.append(np.std(vals) / np.sqrt(len(vals)))
                else:
                    std_ranks.append(0.0)
                valid_js.append(j)
                all_js_set.add(j)
        
        if valid_js:
            all_data.append({
                'model_path': model_path,
                'valid_js': valid_js,
                'mean_ranks': mean_ranks,
                'std_ranks': std_ranks,
                'color': colors[i % len(colors)]
            })

    sorted_all_js = sorted(list(all_js_set))
    j_map = {j: idx for idx, j in enumerate(sorted_all_js)}

    total_width = 0.8
    n_models = len(all_data)
    if n_models > 0:
        bar_width = total_width / n_models
    else:
        bar_width = total_width

    for i, data in enumerate(all_data):
        model_path = data['model_path']
        valid_js = data['valid_js']
        mean_ranks = data['mean_ranks']
        std_ranks = data['std_ranks']
        color = data['color']

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

        indices = [j_map[j] for j in valid_js]
        offset = (i - (n_models - 1) / 2) * bar_width
        
        plt.bar(np.array(indices) + offset, mean_ranks, yerr=std_ranks, width=bar_width, label=label, 
                     color=color, capsize=5, alpha=0.8, edgecolor='none')


    lattice_dim = ""
    if model_paths:
        match = re.search(r'/(\d+x\d+)/', str(model_paths[0]))
        if match:
            lattice_dim = match.group(1)

    if lattice_dim:
        base, ext = os.path.splitext(save_name)
        save_name = f"{base}_{lattice_dim}{ext}"


    plt.xlabel("$J_2$", fontsize=12)
    plt.ylabel("QGT Rank", fontsize=12)
    plt.title(f"QGT Rank vs $J_2$ {lattice_dim}", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='best')
    
    plt.xticks(range(len(sorted_all_js)), sorted_all_js)

    if all_mean_ranks:
        min_val = min(all_mean_ranks)
        max_val = max(all_mean_ranks)
        margin = (max_val - min_val) * 0.2 if max_val > min_val else min_val * 0.1
        if margin == 0: margin = 1
        bottom = max(0, min_val - margin)
        plt.ylim(bottom=bottom)


    save_path = f"/scratch/f/F.Conoscenti/Thesis_QSL/Elaborate/plot/QGT/{save_name}"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {save_path}")
    plt.show()

def get_qgt_condition_number_from_seeds(model_folder, j_val):
    cond_nums = []
    
    model_path = Path(model_folder)
    if not model_path.exists():
        return cond_nums

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
        return cond_nums

    for seed_dir in target_j_folder.iterdir():
        if seed_dir.is_dir() and seed_dir.name.startswith("seed_"):
            pkl_path = seed_dir / "variables.pkl"
            if not pkl_path.exists():
                pkl_path = seed_dir / "variables"
            
            if pkl_path.exists():
                try:
                    with open(pkl_path, "rb") as f:
                        data = pickle.load(f)
                        
                        # Try to get eigenvalues
                        eigs_dict = data.get('eigenvalues_S')
                        if eigs_dict is None:
                            eigs_dict = data.get('eigenvalues')
                        
                        if eigs_dict is not None and isinstance(eigs_dict, dict):
                            seed_conds = []
                            for key, val in eigs_dict.items():
                                if val is None or len(val) == 0:
                                    continue
                                
                                # Ensure numpy array and sort descending
                                eigs = np.sort(np.abs(np.array(val)))[::-1]
                                
                                if len(eigs) > 0 and eigs[0] > 0:
                                    # Calculate rank based on 1e-16 threshold relative to max
                                    max_eig = eigs[0]
                                    norm_eigs = eigs / max_eig
                                    rank = np.sum(norm_eigs > 1e-14)
                                    
                                    if rank > 0:
                                        min_relevant = eigs[rank - 20]
                                        c = max_eig / min_relevant
                                        seed_conds.append(c)
                            
                            if seed_conds:
                                cond_nums.append(np.mean(seed_conds))
                                
                except Exception as e:
                    print(f"Error reading {pkl_path}: {e}")
    
    return cond_nums

def plot_qgt_condition_number_vs_js(model_paths, save_name="QGT_Condition_Number_vs_J2.png"):
    plt.figure(figsize=(10, 6))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(model_paths)))
    
    all_data = []
    all_js_set = set()

    for i, model_path in enumerate(model_paths):
        if not os.path.exists(model_path):
            model_path = model_path.replace("/cluster/home/fconoscenti/Thesis_QSL", "/scratch/f/F.Conoscenti/Thesis_QSL")
            if not os.path.exists(model_path):
                print(f"Path not found: {model_path}")
                continue
            
        js = get_available_js(model_path)
        mean_conds = []
        std_conds = []
        
        valid_js = []
        
        for j in js:
            vals = get_qgt_condition_number_from_seeds(model_path, j)
            if vals:
                mean_val = np.mean(vals)
                mean_conds.append(mean_val)
                if len(vals) > 1:
                    std_conds.append(np.std(vals) / np.sqrt(len(vals)))
                else:
                    std_conds.append(0.0)
                valid_js.append(j)
                all_js_set.add(j)
        
        if valid_js:
            all_data.append({
                'model_path': model_path,
                'valid_js': valid_js,
                'mean_conds': mean_conds,
                'std_conds': std_conds,
                'color': colors[i % len(colors)]
            })

    sorted_all_js = sorted(list(all_js_set))
    j_map = {j: idx for idx, j in enumerate(sorted_all_js)}

    total_width = 0.8
    n_models = len(all_data)
    if n_models > 0:
        bar_width = total_width / n_models
    else:
        bar_width = total_width

    for i, data in enumerate(all_data):
        model_path = data['model_path']
        valid_js = data['valid_js']
        mean_conds = data['mean_conds']
        std_conds = data['std_conds']
        color = data['color']

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

        indices = [j_map[j] for j in valid_js]
        offset = (i - (n_models - 1) / 2) * bar_width
        
        plt.bar(np.array(indices) + offset, mean_conds, yerr=std_conds, width=bar_width, label=label, 
                     color=color, capsize=5, alpha=0.8, edgecolor='none')

        lattice_dim = ""
    if model_paths:
        match = re.search(r'/(\d+x\d+)/', str(model_paths[0]))
        if match:
            lattice_dim = match.group(1)

    if lattice_dim:
        base, ext = os.path.splitext(save_name)
        save_name = f"{base}_{lattice_dim}{ext}"


    plt.xlabel("$J_2$", fontsize=12)
    plt.ylabel("QGT Condition Number", fontsize=12)
    plt.title(f"QGT Condition Number vs $J_2$ {lattice_dim}", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='best')
    plt.yscale('log')
    
    plt.xticks(range(len(sorted_all_js)), sorted_all_js)
    


    save_path = f"/scratch/f/F.Conoscenti/Thesis_QSL/Elaborate/plot/QGT/{save_name}"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {save_path}")
    plt.show()

def get_qgt_condition_number_vs_iteration(model_folder, j_val, single_seed=False):
    model_path = Path(model_folder)
    if not model_path.exists():
        return [], [], []

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
        return [], [], []

    iter_data = {}

    seed_dirs = sorted([d for d in target_j_folder.iterdir() if d.is_dir() and d.name.startswith("seed_")])
    for seed_dir in seed_dirs:
        pkl_path = seed_dir / "variables.pkl"
        if not pkl_path.exists():
            pkl_path = seed_dir / "variables"
        
        if pkl_path.exists():
            try:
                with open(pkl_path, "rb") as f:
                    data = pickle.load(f)
                    
                    eigs_dict = data.get('eigenvalues_S')
                    if eigs_dict is None:
                        eigs_dict = data.get('eigenvalues')
                    
                    if eigs_dict is not None and isinstance(eigs_dict, dict):
                        for key, val in eigs_dict.items():
                            if val is None or len(val) == 0:
                                continue
                            
                            if isinstance(key, str) and key.startswith("iter_"):
                                try:
                                    iter_idx = int(key.split('_')[1])
                                except ValueError:
                                    continue
                            else:
                                continue

                            eigs = np.sort(np.abs(np.array(val)))[::-1]
                            if len(eigs) > 0 and eigs[0] > 0:
                                max_eig = eigs[0]
                                norm_eigs = eigs / max_eig
                                rank = np.sum(norm_eigs > 1e-16)
                                
                                if rank > 0:
                                    min_relevant = eigs[rank - 1]
                                    cond_num = max_eig / min_relevant
                                    
                                    if iter_idx not in iter_data:
                                        iter_data[iter_idx] = []
                                    iter_data[iter_idx].append(cond_num)
                            
            except Exception as e:
                print(f"Error reading {pkl_path}: {e}")
        
        if single_seed and iter_data:
            break

    if not iter_data:
        return [], [], []

    sorted_indices = sorted(iter_data.keys())
    
    save_every = 1.0
    total_iter = None
    match = re.search(r'_iter(\d+)', str(model_folder))
    if match:
        total_iter = int(match.group(1))
    
    if total_iter is not None and len(sorted_indices) > 1:
        max_idx = sorted_indices[-1]
        if max_idx > 0:
            save_every = total_iter / max_idx
            
    sorted_iterations = [idx * save_every for idx in sorted_indices]
    
    mean_conds = []
    std_conds = []

    for idx in sorted_indices:
        vals = iter_data[idx]
        mean_conds.append(np.mean(vals))
        if len(vals) > 1:
            std_conds.append(np.std(vals) / np.sqrt(len(vals)))
        else:
            std_conds.append(0.0)
            
    return sorted_iterations, mean_conds, std_conds

def plot_qgt_condition_number_vs_iteration(model_paths, save_name="QGT_Condition_Number_vs_Iteration.png"):
    groups = {"ViT": [], "HFDS": []}
    for mp in model_paths:
        if "ViT" in str(mp):
            groups["ViT"].append(mp)
        elif "HFDS" in str(mp):
            groups["HFDS"].append(mp)
        else:
            if "Other" not in groups: groups["Other"] = []
            groups["Other"].append(mp)

    for group_name, group_models in groups.items():
        if not group_models:
            continue

        plt.figure(figsize=(10, 6))
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(group_models)))
        linestyles = ['-', '--', '-.', ':']
        
        for i, model_path in enumerate(group_models):
            if not os.path.exists(model_path):
                model_path = model_path.replace("/cluster/home/fconoscenti/Thesis_QSL", "/scratch/f/F.Conoscenti/Thesis_QSL")
                if not os.path.exists(model_path):
                    print(f"Path not found: {model_path}")
                    continue
            
            js = get_available_js(model_path)
            
            n_params = get_param_count(model_path)
            if "ViT" in str(model_path):
                base_label = "ViT"
            elif "HFDS" in str(model_path):
                base_label = "HFDS"
            else:
                base_label = Path(model_path).name
            
            if n_params is not None:
                model_label = f"{base_label} ({n_params} params)"
            else:
                model_label = base_label
                
            color = colors[i % len(colors)]
            
            for k, j in enumerate(js):
                iters, means, stds = get_qgt_condition_number_vs_iteration(model_path, j, single_seed=True)
                if iters:
                    ls = linestyles[k % len(linestyles)]
                    label = f"{model_label}, J={j}"
                    
                    plt.plot(iters, means, label=label, color=color, linestyle=ls, alpha=0.8)

        plt.xlabel("Iterations", fontsize=12)
        plt.ylabel("QGT Condition Number", fontsize=12)
        plt.title(f"QGT Condition Number vs Iteration ({group_name})", fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend(loc='best', fontsize='small')
        plt.yscale('log')
        
        lattice_dim = ""
        if group_models:
            match = re.search(r'/(\d+x\d+)/', str(group_models[0]))
            if match:
                lattice_dim = match.group(1)

        current_save_name = save_name
        base, ext = os.path.splitext(current_save_name)
        if lattice_dim:
            base = f"{base}_{lattice_dim}"
        
        final_save_name = f"{base}_{group_name}{ext}"

        save_path = f"/scratch/f/F.Conoscenti/Thesis_QSL/Elaborate/plot/QGT/{final_save_name}"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
        plt.show()

if __name__ == "__main__":
    """
    #4x4
    models = [
        "/scratch/f/F.Conoscenti/Thesis_QSL/HFDS_Heisenberg/plot/4x4/layers1_hidd4_feat64_sample1024_lr0.02_iter1000_parityTrue_rotTrue_InitFermi_typecomplex",
        "/scratch/f/F.Conoscenti/Thesis_QSL/ViT_Heisenberg/plot/4x4/layers2_d16_heads4_patch2_sample1024_lr0.0075_iter4000_parityTrue_rotTrue_latest_model"    
        ]


    """
    """#6x6
    models = [
        "/scratch/f/F.Conoscenti/Thesis_QSL/HFDS_Heisenberg/plot/6x6/layers1_hidd4_feat64_sample4096_lr0.02_iter500_parityTrue_rotTrue_InitFermi_typecomplex",
        "/scratch/f/F.Conoscenti/Thesis_QSL/ViT_Heisenberg/plot/6x6/layers2_d24_heads6_patch2_sample4096_lr0.0075_iter1000_parityTrue_rotTrue_latest_model"

    ]"""

    #8x8
    models = [
        "/scratch/f/F.Conoscenti/Thesis_QSL/HFDS_Heisenberg/plot/8x8/QGT/layers1_hidd4_feat32_sample8192_bcPBC_PBC_phi0.0_lr0.02_iter200_parityTrue_rotTrue_InitFermi_typecomplex_phi"
    ]


    plot_qgt_rank_vs_js(models)

    plot_qgt_condition_number_vs_js(models)

    #plot_qgt_condition_number_vs_iteration(models)