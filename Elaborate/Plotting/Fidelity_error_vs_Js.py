import re
import os
import pickle
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.cm import get_cmap
import matplotlib.lines as mlines

# --- Model Paths ---
model_ViT1 = "/scratch/f/F.Conoscenti/Thesis_QSL/ViT_Heisenberg/plot/4x4/layers2_d16_heads4_patch2_sample1024_lr0.0075_iter500_parityTrue_rotTrue_latest_model"
model_ViT2 = "/scratch/f/F.Conoscenti/Thesis_QSL/ViT_Heisenberg/plot/4x4/layers1_d16_heads4_patch2_sample1024_lr0.0075_iter500_parityTrue_rotTrue_latest_model"
model_ViT3 = "/scratch/f/F.Conoscenti/Thesis_QSL/ViT_Heisenberg/plot/4x4/layers2_d8_heads4_patch2_sample1024_lr0.0075_iter200_parityTrue_rotTrue_latest_model"

model_HFDS1 = "/scratch/f/F.Conoscenti/Thesis_QSL/HFDS_Heisenberg/plot/4x4/layers1_hidd6_feat128_sample1024_lr0.02_iter500_parityTrue_rotTrue_InitFermi_typecomplex"

models_HFDS = [model_HFDS1]
models_ViT = [model_ViT1, model_ViT2, model_ViT3]

models = models_HFDS + models_ViT

# --- Config ---
keywords = ["Fidelity <vstate|exact>", "Relative error"]

# --- Color gradients ---
n_vit = sum(1 for m in models if "ViT" in m)
n_hfds = sum(1 for m in models if "HFDS" in m)

vit_cmap = get_cmap("Blues")
hfds_cmap = get_cmap("Oranges")

colors = []
vit_count = 0
hfds_count = 0

for model in models:
    if "ViT" in model:
        val = 0.5 + 0.4 * (vit_count / (n_vit - 1)) if n_vit > 1 else 0.7
        colors.append(vit_cmap(val))
        vit_count += 1
    elif "HFDS" in model:
        val = 0.4 + 0.4 * (hfds_count / (n_hfds - 1)) if n_hfds > 1 else 0.6
        colors.append(hfds_cmap(val))
        hfds_count += 1
    else:
        colors.append("black")

# --- Marker shapes per model ---
markers = ["o", "s", "D", "^", "v", "P", "X"]

# --- Function to read parameter count ---
def read_num_params(model_folder):
    """Search for output.txt in any seed folder within J subfolders to find parameter count."""
    model_path = Path(model_folder)
    if not model_path.exists():
        return None
    
    # Iterate over J folders to find one with a seed folder containing output.txt
    for j_folder in model_path.iterdir():
        if j_folder.is_dir() and (j_folder.name.startswith("J=") or j_folder.name.startswith("J2=")):
            for seed_folder in j_folder.iterdir():
                if seed_folder.is_dir() and seed_folder.name.startswith("seed_"):
                    file_path = seed_folder / "output.txt"
                    if file_path.exists():
                        with open(file_path, "r") as f:
                            for line in f:
                                match = re.search(r"(?i).*params\s*[:=]\s*(\d+)", line)
                                if match:
                                    return int(match.group(1))
    return None  # if not found

# --- Plot setup ---
fig_fid, ax_fid = plt.subplots(figsize=(7, 5))
fig_err, ax_err = plt.subplots(figsize=(7, 5))

# --- Main Loop ---
model_symbols = []  # store symbol handles for legend
for i, model_name in enumerate(models):
    values = {key: [] for key in keywords}
    num_params = read_num_params(model_name)
    param_label = f"{num_params} params" if num_params is not None else "Unknown params"
    marker = markers[i % len(markers)]

    # Extract data
    model_path = Path(model_name)
    if not model_path.exists():
        print(f"⚠️ Model path not found: {model_name}")
        continue

    # Automatically detect J folders
    j_folders = sorted(
        [d for d in model_path.iterdir() if d.is_dir() and (d.name.startswith("J=") or d.name.startswith("J2="))],
        key=lambda p: float(p.name.split('=')[1].split('_')[0])
    )

    for j_folder in j_folders:
        try:
            J_value = float(j_folder.name.split('=')[1].split('_')[0])
        except ValueError:
            continue

        avg_file = j_folder / "variables_average.pkl"
        if not avg_file.exists():
            avg_file = j_folder / "variables_average"
        
        if not avg_file.exists():
            print(f"⚠️ Missing average file in: {j_folder}")
            continue

        try:
            with open(avg_file, "rb") as f:
                data = pickle.load(f)
            
            if "fidelity_mean" in data:
                fid_mean = data["fidelity_mean"][-1]
                fid_var = data.get("fidelity_var")
                fid_std = fid_var[-1]**0.5 if fid_var is not None else 0
                values["Fidelity <vstate|exact>"].append((J_value, fid_mean, fid_std))
            if "rel_err_E_mean" in data:
                err_mean = data["rel_err_E_mean"][-1]
                err_var = data.get("rel_err_E_var")
                err_std = err_var[-1]**0.5 if err_var is not None else 0
                values["Relative error"].append((J_value, err_mean, err_std))
        except Exception as e:
            print(f"Error reading {avg_file}: {e}")

    # --- Sort and unpack ---
    data_fid = sorted(values[keywords[0]], key=lambda x: x[0])
    data_err = sorted(values[keywords[1]], key=lambda x: x[0])
    J_fid, Y_fid, E_fid = zip(*data_fid) if data_fid else ([], [], [])
    J_err, Y_err, E_err = zip(*data_err) if data_err else ([], [], [])


    model_type = "ViT" if "ViT" in model_name else ("HFDS" if "HFDS" in model_name else "Model")
    # --- Plot Fidelity (blue) ---
    if J_fid:
        ax_fid.errorbar(
            J_fid, Y_fid, yerr=E_fid,
            marker=marker, linestyle="-", color=colors[i],
            linewidth=2, markersize=6, capsize=3,
        )

    # --- Plot Relative Error (red) ---
    if J_err:
        ax_err.errorbar(
            J_err, Y_err, yerr=E_err,
            marker=marker, linestyle="-", color=colors[i],
            linewidth=2, markersize=6, capsize=3,
            label=f"{model_type} {i+1} ({param_label})"
        )

# --- Axis styling ---
ax_fid.set_xlabel("J₂")
ax_fid.set_ylabel("Fidelity <vstate|exact>")
ax_fid.grid(True, linestyle="--", alpha=0.4)
ax_fid.set_title("Fidelity vs J₂", fontsize=12)
ax_fid.legend(loc="best", fontsize=8, frameon=False)

ax_err.set_xlabel("J₂")
ax_err.set_ylabel("Relative error")
ax_err.grid(True, linestyle="--", alpha=0.4)
ax_err.set_title("Relative Error vs J₂", fontsize=12)
ax_err.legend(loc="best", fontsize=8, frameon=False)

fig_fid.tight_layout()
fig_fid.savefig("Elaborate/plot/Fidelity_vs_J_ViT.png", dpi=300, bbox_inches="tight")
print("Plot saved as 'Elaborate/plot/Fidelity_vs_J_ViT.png'")

fig_err.tight_layout()
fig_err.savefig("Elaborate/plot/Error_vs_J_ViT.png", dpi=300, bbox_inches="tight")
print("Plot saved as 'Elaborate/plot/Error_vs_J_ViT.png'")

plt.show()
