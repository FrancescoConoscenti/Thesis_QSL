import re
import matplotlib.pyplot as plt


ViT_lessparam_nosymm = f"/scratch/f/F.Conoscenti/Thesis_QSL/ViT_Heisenberg/plot/layers1_d8_heads1_patch2_sample1024_lr0.0075_iter100_symmFalse"
ViT_lessparam_symm = f"/scratch/f/F.Conoscenti/Thesis_QSL/ViT_Heisenberg/plot/layers1_d8_heads2_patch2_sample1024_lr0.0075_iter100_symmTrue"
ViT_moreparam_symm = f"/scratch/f/F.Conoscenti/Thesis_QSL/ViT_Heisenberg/plot/layers1_d16_heads2_patch2_sample1024_lr0.0075_iter100_symmTrue"

HFDS_lessparam_nosymm = f"/scratch/f/F.Conoscenti/Thesis_QSL/ViT_Heisenberg/plot/layers1_d8_heads1_patch2_sample1024_lr0.0075_iter100_symmFalse"
HFDS_lessparam_symm = f"/scratch/f/F.Conoscenti/Thesis_QSL/ViT_Heisenberg/plot/layers1_d8_heads2_patch2_sample1024_lr0.0075_iter100_symmTrue"
HFDS_moreparam_symm = f"/scratch/f/F.Conoscenti/Thesis_QSL/ViT_Heisenberg/plot/layers1_d16_heads2_patch2_sample1024_lr0.0075_iter100_symmTrue"

J = 0.5

models = {
    "964 param, no_symm": (ViT_lessparam_nosymm, HFDS_lessparam_nosymm),
    "968 param, symm": (ViT_lessparam_symm, HFDS_lessparam_symm),
    "3464 param, symm":    (ViT_moreparam_symm, HFDS_moreparam_symm),
}

# Collect values
values = {name: {"ViT": None, "HFDS": None} for name in models}

for label, (vit_model, hfds_model) in models.items():
    for model_name, tag in [(vit_model, "ViT"), (hfds_model, "HFDS")]:
        folder = f'{model_name}/J={J}_L=4'
        with open(f"{folder}/output.txt", "r") as f:
            for line in f:
                pattern = r"Last value\s*:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
                match = re.search(pattern, line)
                if match:
                    val = float(match.group(1))
                    values[label][tag] = val
                    break  # stop after first match

# --- Plotting ---
fig, ax = plt.subplots()

x_labels = list(models.keys())
x_pos = range(len(x_labels))

for i, label in enumerate(x_labels):
    vit_val = values[label]["ViT"]
    hfds_val = values[label]["HFDS"]

    ax.scatter(i, vit_val, color="tab:blue", marker="o", label="ViT" if i == 0 else "")
    ax.scatter(i, hfds_val, color="tab:red", marker="s", label="HFDS" if i == 0 else "")

ax.set_xticks(x_pos)
ax.set_xticklabels(x_labels, rotation=20)
ax.set_ylabel("Last value")
ax.set_title(f"Comparison at J={J} (sample1024_lr0.0075_iter100)")
ax.legend()
ax.grid(True, linestyle="--", alpha=0.6)

plt.tight_layout()
plt.savefig("Elaborate/plot/ViT_vs_HFDS_comparison.png", dpi=300, bbox_inches="tight")
plt.show()
