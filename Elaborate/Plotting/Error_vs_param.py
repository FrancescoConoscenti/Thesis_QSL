import re
import os
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap

# --- Model Paths ---
model_HFDS1 = "/scratch/f/F.Conoscenti/Thesis_QSL/HFDS_Heisenberg/plot/J1J2/spin/layers1_hidd2_feat8_sample1024_lr0.02_iter700_symmTrue_Hannah"
model_HFDS2 = "/scratch/f/F.Conoscenti/Thesis_QSL/HFDS_Heisenberg/plot/J1J2/spin/layers1_hidd4_feat16_sample1024_lr0.01_iter600_symmTrue_Hannah"
model_HFDS3 = "/scratch/f/F.Conoscenti/Thesis_QSL/HFDS_Heisenberg/plot/J1J2/spin/layers1_hidd4_feat16_sample1024_lr0.02_iter700_symmTrue_Hannah"
model_HFDS4 = "/scratch/f/F.Conoscenti/Thesis_QSL/HFDS_Heisenberg/plot/J1J2/spin/layers1_hidd1_feat4_sample1024_lr0.02_iter700_symmTrue_Hannah"
model_HFDS5 = "/scratch/f/F.Conoscenti/Thesis_QSL/HFDS_Heisenberg/plot/J1J2/spin/layers1_hidd4_feat32_sample2048_lr0.02_iter700_symmTrue_Hannah"
model_HFDS6 = "/scratch/f/F.Conoscenti/Thesis_QSL/HFDS_Heisenberg/plot/J1J2/spin/layers1_hidd4_feat64_sample1024_lr0.01_iter700_symmTrue_Hannah"
model_HFDS7 = "/scratch/f/F.Conoscenti/Thesis_QSL/HFDS_Heisenberg/plot/J1J2/spin/layers1_hidd8_feat32_sample1024_lr0.02_iter500_symmTrue_Hannah"
model_HFDS8 = "/scratch/f/F.Conoscenti/Thesis_QSL/HFDS_Heisenberg/plot/J1J2/spin/layers1_hidd16_feat32_sample1024_lr0.02_iter500_symmTrue_Hannah"
model_HFDS9 = "/scratch/f/F.Conoscenti/Thesis_QSL/HFDS_Heisenberg/plot/J1J2/spin/layers1_hidd20_feat32_sample1024_lr0.02_iter500_symmTrue_Hannah"

model_ViT1 = "/scratch/f/F.Conoscenti/Thesis_QSL/ViT_Heisenberg/plot/layers1_d16_heads4_patch2_sample1024_lr0.01_iter2000_symmTrue"
model_ViT2 = "/scratch/f/F.Conoscenti/Thesis_QSL/ViT_Heisenberg/plot/layers1_d8_heads1_patch2_sample1024_lr0.001_iter1000_symmTrue"
model_ViT3 = "/scratch/f/F.Conoscenti/Thesis_QSL/ViT_Heisenberg/plot/layers2_d16_heads4_patch2_sample1024_lr0.01_iter2000_symmTrue"
model_ViT4 = "/scratch/f/F.Conoscenti/Thesis_QSL/ViT_Heisenberg/plot/layers1_d16_heads2_patch2_sample1024_lr0.005_iter2000_symmTrue"
model_ViT5 = "/scratch/f/F.Conoscenti/Thesis_QSL/ViT_Heisenberg/plot/layers2_d16_heads2_patch2_sample1024_lr0.0075_iter2000_symmTrue"

model_HFDS_hidd1 = "/scratch/f/F.Conoscenti/Thesis_QSL/HFDS_Heisenberg/plot/J1J2/spin/layers1_hidd2_feat8_sample1024_lr0.02_iter700_symmTrue_Hannah"
model_HFDS_hidd2 = "/scratch/f/F.Conoscenti/Thesis_QSL/HFDS_Heisenberg/plot/J1J2/spin/layers1_hidd2_feat16_sample1024_lr0.02_iter600_symmTrue_Hannah"
model_HFDS_hidd3 = "/scratch/f/F.Conoscenti/Thesis_QSL/HFDS_Heisenberg/plot/J1J2/spin/layers1_hidd2_feat32_sample1024_lr0.02_iter600_symmTrue_Hannah"
model_HFDS_hidd4 = "/scratch/f/F.Conoscenti/Thesis_QSL/HFDS_Heisenberg/plot/J1J2/spin/layers1_hidd2_feat64_sample1024_lr0.02_iter600_symmTrue_Hannah"

model_HFDS_ferm1 = "/scratch/f/F.Conoscenti/Thesis_QSL/HFDS_Heisenberg/plot/J1J2/spin/layers1_hidd1_feat32_sample1024_lr0.02_iter800_symmTrue_Hannah"
model_HFDS_ferm2 = "/scratch/f/F.Conoscenti/Thesis_QSL/HFDS_Heisenberg/plot/J1J2/spin/layers1_hidd2_feat32_sample1024_lr0.02_iter600_symmTrue_Hannah"
model_HFDS_ferm3 = "/scratch/f/F.Conoscenti/Thesis_QSL/HFDS_Heisenberg/plot/J1J2/spin/layers1_hidd4_feat32_sample2048_lr0.02_iter700_symmTrue_Hannah"
model_HFDS_ferm4 = "/scratch/f/F.Conoscenti/Thesis_QSL/HFDS_Heisenberg/plot/J1J2/spin/layers1_hidd8_feat32_sample1024_lr0.02_iter500_symmTrue_Hannah"
model_HFDS_ferm5 = "/scratch/f/F.Conoscenti/Thesis_QSL/HFDS_Heisenberg/plot/J1J2/spin/layers1_hidd16_feat32_sample1024_lr0.02_iter500_symmTrue_Hannah"
model_HFDS_ferm6 = "/scratch/f/F.Conoscenti/Thesis_QSL/HFDS_Heisenberg/plot/J1J2/spin/layers1_hidd20_feat32_sample1024_lr0.02_iter500_symmTrue_Hannah"

#models_HFDS = [model_HFDS1,  model_HFDS3, model_HFDS4, model_HFDS5]#, model_HFDS6, model_HFDS7, model_HFDS8, model_HFDS9]
models_ViT = [model_ViT1,model_ViT2,model_ViT3]
models_HFDS =[model_HFDS_hidd1, model_HFDS_hidd2, model_HFDS_hidd3, model_HFDS_hidd4]
#models_HFDS =[model_HFDS_ferm1, model_HFDS_ferm2, model_HFDS_ferm3, model_HFDS_ferm4 , model_HFDS_ferm5 , model_HFDS_ferm6]


#lattice_names = ["J2=0.0_L=4", "J2=0.2_L=4", "J2=0.5_L=4", "J2=0.7_L=4", "J2=1.0_L=4"]
lattice_names_ViT = [ "J=0.2_L=4", "J=0.5_L=4", "J=0.7_L=4"]
lattice_names_HFDS = [ "J2=0.2_L=4", "J2=0.5_L=4", "J2=0.7_L=4"]

# --- Markers for J2 values ---
j_values_HFDS = [float(name.split("_")[0].replace("J2=", "")) for name in lattice_names_HFDS]
j_values_ViT = [float(name.split("_")[0].replace("J=", "")) for name in lattice_names_ViT]
cmap = get_cmap("tab10")
color_HFDS = cmap(0)  # blue
color_ViT = cmap(1)   # orange
markers = ["o", "s", "D", "^", "v"]

# --- Functions ---
def read_num_params(model_folder, lattice_names):
    for lattice_name in lattice_names:
        file_path = f"{model_folder}/{lattice_name}/output.txt"
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                for line in f:
                    match = re.search(r"Total number of parameters:\s*(\d+)", line)
                    if match:
                        return int(match.group(1))
                    else:
                        match = re.search(r"Number of parameters\s*=\s*(\d+)", line)
                        if match:
                            return int(match.group(1))
    return None


def read_relative_error(model_folder, lattice_name):
    file_path = f"{model_folder}/{lattice_name}/output.txt"
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            for line in f:
                match = re.search(r"Relative error\s*=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", line)
                if match:
                    return float(match.group(1))
    return None

# --- Plot setup ---
fig, ax = plt.subplots(figsize=(8,6))

"""for i, model in enumerate(models_HFDS):
    num_params = read_num_params(model, lattice_names_HFDS)
    if num_params is None:
        print(f"⚠️ Missing parameter count for {model}")
        continue

    for j, lattice_name in enumerate(lattice_names_HFDS):
        rel_error = read_relative_error(model, lattice_name)
        if rel_error is None:
            print(f"⚠️ Missing Relative Error for {model}, {lattice_name}")
            continue
        marker = markers[j % len(markers)]
        ax.scatter(num_params, rel_error, color=color_HFDS, marker=marker, s=100,
                   label=f"HFDS J2={j_values_HFDS[j]}" if i == 0 else "")"""

# ==========================
#   Plot ViT models
# ==========================

for i, model in enumerate(models_ViT):
    num_params = read_num_params(model, lattice_names_ViT)
    if num_params is None:
        print(f"⚠️ Missing parameter count for {model}")
        continue
    
    for j, lattice_name in enumerate(lattice_names_ViT):
        rel_error = read_relative_error(model, lattice_name)
        if rel_error is None:
            print(f"⚠️ Missing Relative Error for {model}, {lattice_name}")
            continue
        marker = markers[j % len(markers)]
        ax.scatter(num_params, rel_error, color=color_ViT, marker=marker, s=100, linewidth=0.8,
                   label=f"ViT J={j_values_ViT[j]}" if i == 0 else "")

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
plt.savefig("Elaborate/plot/RelError_vs_Params.png", dpi=300, bbox_inches="tight")
plt.show()
