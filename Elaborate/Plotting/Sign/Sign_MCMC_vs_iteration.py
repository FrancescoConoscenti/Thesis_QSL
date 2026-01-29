import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import re

def get_sign_history(folder):
    """
    Loads sign_MCMC_history and sign_MCMC_variance_history from variables.pkl in the given folder.
    """
    variables_path = folder / "variables.pkl"
    if not variables_path.exists():
        variables_path = folder / "variables"
    
    if variables_path.exists():
        try:
            with open(variables_path, "rb") as f:
                data = pickle.load(f)
                if "sign_MCMC_history" in data:
                    param_count = data.get("count_params", data.get("params", None))
                    return data["sign_MCMC_history"], data.get("sign_MCMC_variance_history", None), param_count
        except Exception as e:
            print(f"Error reading {variables_path}: {e}")
    return None, None, None

def plot_sign_vs_iteration(model_paths, save_name="Sign_MCMC_vs_iteration.png"):
    plt.figure(figsize=(10, 6))
    
    has_data = False
    
    filename_suffix = ""
    first_info_captured = False

    for model_path in model_paths:
        path = Path(model_path)
        if not path.exists():
            # Try replacing cluster path with scratch path if needed
            path_str = str(path).replace("/cluster/home/fconoscenti/Thesis_QSL", "/scratch/f/F.Conoscenti/Thesis_QSL")
            path = Path(path_str)
            if not path.exists():
                print(f"Path not found: {model_path}")
                continue

        # Find all seed folders recursively
        seed_folders = []
        if path.name.startswith("seed_"):
            seed_folders.append(path)
        else:
            for root, dirs, files in os.walk(path):
                for d in dirs:
                    if d.startswith("seed_"):
                        seed_folders.append(Path(root) / d)
        
        if not seed_folders:
            print(f"No seed folders found in {path}")
            continue

        for seed_folder in seed_folders:
            sign_hist, sign_var_hist, param_count = get_sign_history(seed_folder)
            
            if sign_hist is not None and len(sign_hist) > 0:
                has_data = True
                # Try to determine iterations from model files if possible
                models_dir = seed_folder / "models"
                iterations = []
                if models_dir.exists():
                    files = [f for f in os.listdir(models_dir) if f.endswith(".mpack")]
                    # Filter files that match the pattern
                    files = [f for f in files if re.search(r"model_(\d+)", f)]
                    if len(files) == len(sign_hist):
                         files.sort(key=lambda x: int(re.search(r"model_(\d+)", x).group(1)))
                         iterations = [int(re.search(r"model_(\d+)", x).group(1)) for x in files]
                
                if not iterations:
                    iterations = range(len(sign_hist))

                # Construct a label
                model_type = "ViT" if "ViT" in str(seed_folder) else ("HFDS" if "HFDS" in str(seed_folder) else "Model")
                
                j_val = ""
                for part in seed_folder.parts:
                    if part.startswith("J=") or part.startswith("J2="):
                        j_val = part
                        break
                
                if not first_info_captured:
                    filename_suffix = f"_{model_type}"
                    if param_count:
                        filename_suffix += f"_params{param_count}"
                    if j_val:
                        filename_suffix += f"_{j_val.replace('=', '_')}"
                    first_info_captured = True

                param_str = f"{param_count} params" if param_count is not None else "Unknown params"
                label = f"{model_type}, {j_val}, {param_str}"
                
                # Calculate error bars if variance is available
                yerr = None
                if sign_var_hist is not None:
                    try:
                        # Assuming n_samples = 1024 as per Observables.py
                        yerr = np.sqrt(np.array(sign_var_hist) / 1024)
                    except:
                        yerr = None

                plt.errorbar(iterations, np.real(sign_hist), yerr=yerr, label=label, marker='o', capsize=3, alpha=0.7)

    if has_data:
        plt.xlabel("Model Iteration Index")
        plt.ylabel("Mean Marshall Sign")
        plt.title("Marshall Sign History (MCMC)")
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        if save_name.endswith(".png"):
            final_save_name = save_name[:-4] + filename_suffix + ".png"
        else:
            final_save_name = save_name + filename_suffix

        save_path = Path("/scratch/f/F.Conoscenti/Thesis_QSL/Elaborate/plot/Sign/Sign_plot") / final_save_name
        os.makedirs(save_path.parent, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
        plt.show()
    else:
        print("No sign_MCMC_history data found to plot.")

if __name__ == "__main__":
    # Example usage
    models = [
        "/scratch/f/F.Conoscenti/Thesis_QSL/ViT_Heisenberg/plot/6x6/layers2_d24_heads4_patch2_sample1024_lr0.0075_iter500_parityTrue_rotTrue_latest_model"
    ]
    plot_sign_vs_iteration(models)
