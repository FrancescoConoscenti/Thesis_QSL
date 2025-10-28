import re
import os
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import matplotlib.lines as mlines

# --- Model Paths ---
model_HFDS1 = f"/scratch/f/F.Conoscenti/Thesis_QSL/HFDS_Heisenberg/plot/J1J2/spin/layers1_hidd2_feat8_sample1024_lr0.02_iter700_symmTrue_Hannah"
model_HFDS2 = f"/scratch/f/F.Conoscenti/Thesis_QSL/HFDS_Heisenberg/plot/J1J2/spin/layers1_hidd4_feat16_sample1024_lr0.01_iter600_symmTrue_Hannah"
model_HFDS3 = f"/scratch/f/F.Conoscenti/Thesis_QSL/HFDS_Heisenberg/plot/J1J2/spin/layers1_hidd4_feat16_sample1024_lr0.02_iter700_symmTrue_Hannah"
model_HFDS4 = f"/scratch/f/F.Conoscenti/Thesis_QSL/HFDS_Heisenberg/plot/J1J2/spin/layers1_hidd1_feat4_sample1024_lr0.02_iter700_symmTrue_Hannah"


model_ViT1 = "/scratch/f/F.Conoscenti/Thesis_QSL/ViT_Heisenberg/plot/layers2_d16_heads4_patch2_sample1024_lr0.01_iter2000_symmTrue"
model_ViT2 = "/scratch/f/F.Conoscenti/Thesis_QSL/ViT_Heisenberg/plot/layers1_d16_heads4_patch2_sample1024_lr0.01_iter2000_symmTrue"
#model_ViT3 = "/scratch/f/F.Conoscenti/Thesis_QSL/ViT_Heisenberg/plot/layers1_d8_heads1_patch2_sample1024_lr0.001_iter1000_symmTrue"
#model_ViT4 = "/scratch/f/F.Conoscenti/Thesis_QSL/ViT_Heisenberg/plot/layers2_d16_heads2_patch2_sample1024_lr0.0075_iter400_symmTrue"

#models = [model_HFDS1, model_HFDS2]#, model_HFDS3, model_HFDS4]
models = [model_ViT1, model_ViT2]#, model_ViT3]

# --- Config ---
keywords = ["Fidelity <vstate|exact>", "Relative error"]
#lattice_names = ["J2=0.0_L=4", "J2=0.2_L=4", "J2=0.5_L=4", "J2=0.7_L=4", "J2=1.0_L=4"]
lattice_names = ["J=0.0_L=4", "J=0.2_L=4", "J=0.5_L=4", "J=0.7_L=4", "J=1.0_L=4"]

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
    """Search any lattice folder's output.txt for the total parameter count."""
    for lattice_name in lattice_names:
        file_path = f"{model_folder}/{lattice_name}/output.txt"
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                for line in f:
                    match = re.search(r"(?i).*number of parameters\s*[:=]\s*(\d+)", line)
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
    for key in keywords:
        for lattice_name in lattice_names:
            J_value = float(lattice_name.split("_")[0].replace("J=", ""))
            folder = f"{model_name}/{lattice_name}"
            file_path = f"{folder}/output.txt"

            if not os.path.exists(file_path):
                print(f"⚠️ Missing file: {file_path}")
                continue

            with open(file_path, "r") as f:
                for line in f:
                    if key == "Fidelity <vstate|exact>":
                        pattern = fr"{re.escape(key)}\s*=\s*\[([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)[^]]*\]"
                    else:
                        pattern = fr"{re.escape(key)}\s*=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
                    match = re.search(pattern, line)
                    if match:
                        try:
                            val = float(match.group(1))
                            values[key].append((J_value, val))
                        except Exception as e:
                            print(f"Error converting '{match.group(1)}' to float: {e}")

    # --- Sort and unpack ---
    data_fid = sorted(values[keywords[0]], key=lambda x: x[0])
    data_err = sorted(values[keywords[1]], key=lambda x: x[0])
    J_fid, Y_fid = zip(*data_fid) if data_fid else ([], [])
    J_err, Y_err = zip(*data_err) if data_err else ([], [])

    # --- Plot Fidelity (blue) ---
    if J_fid:
        ax1.plot(
            J_fid, Y_fid,
            marker=marker, linestyle="-", color=blue_colors[i],
            linewidth=2, markersize=6
        )

    # --- Plot Relative Error (red) ---
    if J_err:
        ax2.plot(
            J_err, Y_err,
            marker=marker, linestyle="-", color=red_colors[i],
            linewidth=2, markersize=6
        )

    # --- Add only symbol (no line) for legend ---
    model_symbols.append(
        mlines.Line2D(
            [], [], color="black", marker=marker, linestyle="None",
            markersize=6, label=f"ViT {i+1} " #({param_label})"
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
plt.show()
