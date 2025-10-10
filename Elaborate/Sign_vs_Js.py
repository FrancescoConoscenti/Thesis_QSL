import re
import os
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib.patches import Patch

# --- Model Paths ---
import os, re
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# --- Model groups ---
models_HFDS = [
    "/scratch/f/F.Conoscenti/Thesis_QSL/HFDS_Heisenberg/plot/J1J2/spin/layers1_hidd2_feat32_sample1024_lr0.02_iter600_symmTrue_Hannah",
    "/scratch/f/F.Conoscenti/Thesis_QSL/HFDS_Heisenberg/plot/J1J2/spin/layers1_hidd2_feat64_sample1024_lr0.02_iter600_symmTrue_Hannah",
    "/scratch/f/F.Conoscenti/Thesis_QSL/HFDS_Heisenberg/plot/J1J2/spin/layers1_hidd4_feat32_sample1024_lr0.02_iter800_symmTrue_Hannah",
    "/scratch/f/F.Conoscenti/Thesis_QSL/HFDS_Heisenberg/plot/J1J2/spin/layers1_hidd4_feat64_sample1024_lr0.02_iter600_symmTrue_Hannah",
]

models_ViT = [
    "/scratch/f/F.Conoscenti/Thesis_QSL/ViT_Heisenberg/plot/layers1_d16_heads4_patch2_sample1024_lr0.005_iter1000_symmTrue",
    "/scratch/f/F.Conoscenti/Thesis_QSL/ViT_Heisenberg/plot/layers1_d16_heads4_patch2_sample1024_lr0.01_iter2000_symmTrue",
    "/scratch/f/F.Conoscenti/Thesis_QSL/ViT_Heisenberg/plot/layers2_d16_heads4_patch2_sample1024_lr0.01_iter2000_symmTrue",
    "/scratch/f/F.Conoscenti/Thesis_QSL/ViT_Heisenberg/plot/layers2_d16_heads2_patch2_sample1024_lr0.0075_iter2000_symmTrue",
]

# --- Common J values ---
j_values = [0.0, 0.2, 0.5, 0.7, 1.0]

# --- Helper to read Marshall Sign ---
def read_marshall_sign(model_folder, j_val, lattice_suffix="_L=4"):
    """Tries both J= and J2= folder names."""
    folder_variants = [f"J={j_val}{lattice_suffix}", f"J2={j_val}{lattice_suffix}"]
    for folder_name in folder_variants:
        file_path = f"{model_folder}/{folder_name}/output.txt"
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                for line in f:
                    match = re.search(r"Marshall Sign\s*=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", line)
                    if match:
                        return float(match.group(1))
    return None

# --- Plot setup ---
fig, ax = plt.subplots(figsize=(8, 5))

# --- Style setup ---
marker_ViT = "o"
marker_HFDS = "^"
color_ViT = "tab:orange"
color_HFDS = "tab:blue"

# --- Plot HFDS models ---
for model in models_HFDS:
    for J_val in j_values:
        sign_val = read_marshall_sign(model, J_val)
        if sign_val is not None:
            ax.scatter(J_val, sign_val, color=color_HFDS, marker=marker_HFDS, s=80, alpha=0.7, edgecolors='none')

# --- Plot ViT models ---
for model in models_ViT:
    for J_val in j_values:
        sign_val = read_marshall_sign(model, J_val)
        if sign_val is not None:
            ax.scatter(J_val, sign_val, color=color_ViT, marker=marker_ViT, s=80, alpha=0.7, edgecolors='none')

# --- Labels, title, legend ---
ax.set_xlabel("$J_2$", fontsize=12)
ax.set_ylabel("Marshall Sign", fontsize=12)
ax.set_title("Marshall Sign vs $J_2$ for ViT and HFDS Models")
ax.grid(True, linestyle="--", alpha=0.4)

legend_elements = [
    Patch(facecolor=color_ViT, label="ViT Models"),
    Patch(facecolor=color_HFDS, label="HFDS Models"),
]
ax.legend(handles=legend_elements, frameon=False)

fig.tight_layout()
plt.savefig("Elaborate/plot/MarshallSign_vs_J2_ViT_HFDS.png", dpi=300, bbox_inches="tight")
plt.show()
