import re
import os
import pickle
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.cm import get_cmap
import matplotlib.lines as mlines

# --- Model Paths ---
#model_HFDS1 = 
model_ViT1 = "/scratch/f/F.Conoscenti/Thesis_QSL/ViT_Heisenberg/plot/4x4/layers2_d16_heads4_patch2_sample1024_lr0.0075_iter500_parityTrue_rotTrue_latest_model"
model_ViT2 = "/scratch/f/F.Conoscenti/Thesis_QSL/ViT_Heisenberg/plot/4x4/layers1_d16_heads4_patch2_sample1024_lr0.0075_iter500_parityTrue_rotTrue_latest_model"
model_ViT3 = "/scratch/f/F.Conoscenti/Thesis_QSL/ViT_Heisenberg/plot/4x4/layers2_d8_heads4_patch2_sample1024_lr0.0075_iter200_parityTrue_rotTrue_latest_model"
#models = [model_HFDS1]
models = [model_ViT1, model_ViT2, model_ViT3]

# --- Config ---
keywords = ["Fidelity <vstate|exact>", "Relative error"]

# --- Color gradients ---
blue_cmap = get_cmap("Blues")
red_cmap = get_cmap("Reds")
n_models = len(models)
blue_colors = [blue_cmap(0.4 + 0.5 * i / (n_models - 1)) for i in range(n_models)]
red_colors  = [red_cmap(0.4 + 0.5 * i / (n_models - 1)) for i in range(n_models)]

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
fig, ax1 = plt.subplots(figsize=(7, 5))
ax2 = ax1.twinx()

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


    # --- Plot Fidelity (blue) ---
    if J_fid:
        ax1.errorbar(
            J_fid, Y_fid, yerr=E_fid,
            marker=marker, linestyle="-", color=blue_colors[i],
            linewidth=2, markersize=6, capsize=3

        )

    # --- Plot Relative Error (red) ---
    if J_err:
        ax2.errorbar(
            J_err, Y_err, yerr=E_err,
            marker=marker, linestyle="-", color=red_colors[i],
            linewidth=2, markersize=6, capsize=3
        )

    # --- Add only symbol (no line) for legend ---
    model_symbols.append(
        mlines.Line2D(
            [], [], color="black", marker=marker, linestyle="None",
            markersize=6, label=f"ViT {i+1} ({param_label})"
        )
    )

# --- Axis styling ---
ax1.set_xlabel("J₂")
ax1.set_ylabel("Fidelity <vstate|exact>", color="navy")
ax2.set_ylabel("Relative error", color="darkred")
ax1.tick_params(axis="y", labelcolor="navy")
ax2.tick_params(axis="y", labelcolor="darkred")
ax1.grid(True, linestyle="--", alpha=0.4)
plt.title("Error and Fidelity vs J₂ for Multiple ViT Models", fontsize=12)

# --- Legend: Blue line, Red dashed line, then symbols ---
fidelity_line = mlines.Line2D([], [], color="blue", linestyle="-", label="Fidelity")
error_line = mlines.Line2D([], [], color="red", linestyle="-", label="Rel Error")

# Combine
legend_handles = [fidelity_line, error_line] + model_symbols
ax1.legend(
    handles=legend_handles,
    loc="center left",
    fontsize=8,
    frameon=False
)

fig.tight_layout()
plt.savefig("Elaborate/plot/Error_Fidelity_vs_J_ViT.png", dpi=300, bbox_inches="tight")
print("Plot saved as 'Elaborate/plot/Error_Fidelity_vs_J_ViT.png'")
plt.show()
