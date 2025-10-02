import pickle
import re
import matplotlib.pyplot as plt


model_ViT = f"/scratch/f/F.Conoscenti/Thesis_QSL/ViT_Heisenberg/plot/layers1_d16_heads1_patch2_sample2048_lr0.0075_iter200_symmFalse"
model_HFDS = f"/scratch/f/F.Conoscenti/Thesis_QSL/HFDS_Heisenberg/plot/spin/layers1_hidd1_feat1_sample20_lr0.02_iter2_symmTrue"

keywords = ["Staggered Magnetization", "Striped Magnetization"]
lattice_names = ["J=0.0_L=4", "J=0.2_L=4", "J=0.5_L=4", "J=0.7_L=4", "J=1.0_L=4"]
values = {key: [] for key in keywords}
models = [model_ViT]#, model_HFDS]

for model_name in models:
    for key in keywords:
        for lattice_name in lattice_names:
            # Extract the J value from the lattice_name
            J_value = float(lattice_name.split("_")[0].replace("J=", ""))
            folder = f'{model_name}/{lattice_name}'
            with open(f"{folder}/output.txt", "r") as f:
                for line in f:
                    pattern = fr"{key}\s*=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
                    match = re.search(pattern, line)
                    if match:
                        val = float(match.group(1))
                        values[key].append((J_value, val))

    for key in values:
        values[key] = [(j_val, abs(v_val)) for j_val, v_val in values[key]]

    # --- Plot with 2 y-axes ---
    fig, ax1 = plt.subplots()

    # First keyword → left axis
    key1 = keywords[0]
    data1 = sorted(values[key1], key=lambda x: x[0])
    J1 = [j for j, _ in data1]
    Y1 = [v for _, v in data1]

    ax1.plot(J1, Y1, "o-", color="tab:blue", label=key1)
    ax1.set_xlabel("J")
    ax1.set_ylabel(key1, color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    # Second keyword → right axis
    if len(keywords) > 1:
        key2 = keywords[1]
        data2 = sorted(values[key2], key=lambda x: x[0])
        J2 = [j for j, _ in data2]
        Y2 = [v for _, v in data2]

        ax2 = ax1.twinx()  # create a second y-axis
        ax2.plot(J2, Y2, "s-", color="tab:red", label=key2)
        ax2.set_ylabel(key2, color="tab:red")
        ax2.tick_params(axis="y", labelcolor="tab:red")

    # Title and grid
    plt.title("AF & Stripe order vs J")
    fig.tight_layout()

    # Save and show
    plt.savefig("Elaborate/plot/AF_&_Stripe_vs_J_dual_axis.png", dpi=300, bbox_inches="tight")
    plt.show()




