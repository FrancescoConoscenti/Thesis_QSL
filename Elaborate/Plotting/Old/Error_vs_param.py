import re
import os
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from pathlib import Path
import pickle
import numpy as np

# --- Model Paths ---
model_HFDS1 = "/scratch/f/F.Conoscenti/Thesis_QSL/HFDS_Heisenberg/plot/4x4/layers1_hidd2_feat16_sample1024_bcPBC_PBC_lr0.02_iter20000_parityTrue_rotTrue_InitFermi_typecomplex"
model_HFDS1 = "/scratch/f/F.Conoscenti/Thesis_QSL/HFDS_Heisenberg/plot/4x4/layers1_hidd1_feat16_sample1024_lr0.02_iter500_parityTrue_rotTrue_InitFermi_typecomplex"
model_HFDS2 = "/scratch/f/F.Conoscenti/Thesis_QSL/HFDS_Heisenberg/plot/4x4/layers1_hidd2_feat32_sample1024_bcPBC_PBC_lr0.02_iter20000_parityTrue_rotTrue_InitFermi_typecomplex"
model_HFDS3 = "/scratch/f/F.Conoscenti/Thesis_QSL/HFDS_Heisenberg/plot/4x4/layers1_hidd8_feat32_sample1024_bcPBC_PBC_phi0.0_lr0.02_iter20000_parityTrue_rotTrue_InitFermi_typecomplex_phi"
model_HFDS3 = "/scratch/f/F.Conoscenti/Thesis_QSL/HFDS_Heisenberg/plot/4x4/layers1_hidd4_feat64_sample1024_lr0.02_iter1000_parityTrue_rotTrue_InitFermi_typecomplex"

model_ViT1 = "/scratch/f/F.Conoscenti/Thesis_QSL/ViT_Heisenberg/plot/4x4/layers1_d16_heads4_patch2_sample1024_lr0.0075_iter2000_parityTrue_rotTrue_latest_model"
model_ViT2 = "/scratch/f/F.Conoscenti/Thesis_QSL/ViT_Heisenberg/plot/4x4/layers2_d8_heads4_patch2_sample1024_lr0.0075_iter2000_parityTrue_rotTrue_latest_model"
model_ViT3 = "/scratch/f/F.Conoscenti/Thesis_QSL/ViT_Heisenberg/plot/4x4/layers2_d16_heads4_patch2_sample1024_lr0.0075_iter4000_parityTrue_rotTrue_latest_model"


#models_HFDS = [model_HFDS1,  model_HFDS3, model_HFDS4, model_HFDS5]#, model_HFDS6, model_HFDS7, model_HFDS8, model_HFDS9]
models_ViT = [model_ViT1,model_ViT2,model_ViT3]
models_HFDS =[model_HFDS1,  model_HFDS2, model_HFDS3]
#models_HFDS =[model_HFDS_ferm1, model_HFDS_ferm2, model_HFDS_ferm3, model_HFDS_ferm4 , model_HFDS_ferm5 , model_HFDS_ferm6]

# --- Markers for J2 values ---
j_values = [0.5]
color_HFDS = "tab:blue"
color_ViT = "tab:orange"
markers = ["o", "s", "D", "^", "v"]

# --- Functions ---
def read_num_params(model_folder):
    model_path = Path(model_folder)
    if not model_path.exists():
        return None
    for j_folder in model_path.iterdir():
        if j_folder.is_dir() and (j_folder.name.startswith("J=") or j_folder.name.startswith("J2=")):
            for seed_folder in j_folder.iterdir():
                if seed_folder.is_dir() and seed_folder.name.startswith("seed_"):
                    pkl_path = seed_folder / "variables.pkl"
                    if not pkl_path.exists():
                        pkl_path = seed_folder / "variables"
                    if pkl_path.exists():
                        try:
                            with open(pkl_path, "rb") as f:
                                data = pickle.load(f)
                                if 'count_params' in data: return data['count_params']
                                if 'params' in data: return data['params']
                        except Exception:
                            pass
                    file_path = seed_folder / "output.txt"
                    if file_path.exists():
                        with open(file_path, "r") as f:
                            for line in f:
                                match = re.search(r"(?i).*params\s*[:=]\s*(\d+)", line)
                                if match: return int(match.group(1))
    return None

def read_relative_error(model_folder, target_j):
    model_path = Path(model_folder)
    if not model_path.exists():
        return None
    for j_folder in model_path.iterdir():
        if j_folder.is_dir() and (j_folder.name.startswith("J=") or j_folder.name.startswith("J2=")):
            try:
                part = j_folder.name.split('=')[1]
                val_str = part.split('_')[0]
                if abs(float(val_str) - target_j) < 1e-5:
                    errs = []
                    for seed_folder in j_folder.iterdir():
                        if seed_folder.is_dir() and seed_folder.name.startswith("seed_"):
                            pkl_path = seed_folder / "variables.pkl"
                            if not pkl_path.exists():
                                pkl_path = seed_folder / "variables"
                            if pkl_path.exists():
                                try:
                                    with open(pkl_path, "rb") as f:
                                        data = pickle.load(f)
                                        if "rel_err_E" in data and data["rel_err_E"] is not None:
                                            val = data["rel_err_E"]
                                            if isinstance(val, (list, np.ndarray)): errs.append(float(np.real(val[-1])))
                                            else: errs.append(float(np.real(val)))
                                        elif "E_vs_final" in data and "E_exact" in data and data["E_exact"] is not None:
                                            e_val = data["E_vs_final"]
                                            e_exact = data["E_exact"]
                                            if isinstance(e_val, (list, np.ndarray)): e_val = float(np.real(e_val[-1]))
                                            else: e_val = float(np.real(e_val))
                                            if isinstance(e_exact, (list, np.ndarray)): e_exact = float(np.real(e_exact[-1]))
                                            else: e_exact = float(np.real(e_exact))
                                            errs.append(abs((e_val - e_exact) / e_exact))
                                except Exception:
                                    pass
                    if errs:
                        return np.mean(errs)
            except ValueError:
                continue
    return None

# --- Plot setup ---
fig, ax = plt.subplots(figsize=(8,6))

for i, model in enumerate(models_HFDS):
    num_params = read_num_params(model)
    if num_params is None:
        print(f"⚠️ Missing parameter count for {model}")
        continue

    for j, target_j in enumerate(j_values):
        rel_error = read_relative_error(model, target_j)
        if rel_error is None:
            print(f"⚠️ Missing Relative Error for {model}, J={target_j}")
            continue
        marker = markers[j % len(markers)]
        ax.scatter(num_params, rel_error, color=color_HFDS, marker=marker, s=100, linewidth=0.8, alpha=0.8,
                   label=f"HFDS J2={target_j}" if i == 0 else "")

# ==========================
#   Plot ViT models
# ==========================

for i, model in enumerate(models_ViT):
    num_params = read_num_params(model)
    if num_params is None:
        print(f"⚠️ Missing parameter count for {model}")
        continue
    
    for j, target_j in enumerate(j_values):
        rel_error = read_relative_error(model, target_j)
        if rel_error is None:
            print(f"⚠️ Missing Relative Error for {model}, J={target_j}")
            continue
        marker = markers[j % len(markers)]
        ax.scatter(num_params, rel_error, color=color_ViT, marker=marker, s=100, linewidth=0.8, alpha=0.8,
                   label=f"ViT J2={target_j}" if i == 0 else "")

# --- Axis and styling ---
ax.set_xlabel("Number of Parameters", fontsize=12)
ax.set_ylabel("Relative Error", fontsize=12)
ax.set_xscale("log")
ax.set_yscale("log")
ax.grid(True, linestyle="--", alpha=0.3)
plt.title("Relative Error vs Number of Parameters ViT", fontsize=13)

# --- Legend cleanup ---
handles, labels = ax.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax.legend(by_label.values(), by_label.keys(), title="Models", fontsize=9, frameon=False)

fig.tight_layout()
os.makedirs("Elaborate/plot", exist_ok=True)
plt.savefig("Elaborate/plot/RelError_vs_Params.png", dpi=300, bbox_inches="tight")
plt.show()
