import os
import re
import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import curve_fit

def extract_L(path_str):
    """Extracts L from path string assuming format like '4x4', '6x6'."""
    match = re.search(r"(\d+)x(\d+)", str(path_str))
    if match:
        return int(match.group(1))
    return None

def get_s2_for_J(model_folder, target_J):
    """
    Finds the subfolder for target_J in model_folder, reads variables.pkl from seeds,
    and returns a list of S2 values.
    """
    s2_values = []
    s2_errors = []
    
    model_path = Path(model_folder)
    if not model_path.exists():
        # Try replacing cluster path with scratch if needed
        if "/cluster/home/" in str(model_path):
             model_path = Path(str(model_path).replace("/cluster/home/fconoscenti", "/scratch/f/F.Conoscenti"))
        
        if not model_path.exists():
            print(f"Path not found: {model_path}")
            return s2_values, s2_errors

    target_j_folder = None
    # Robustly find J folder
    for d in model_path.iterdir():
        if d.is_dir() and (d.name.startswith("J=") or d.name.startswith("J2=")):
            try:
                # Extract number after = and before any _
                part = d.name.split('=')[1]
                if '_' in part:
                    val_str = part.split('_')[0]
                else:
                    val_str = part
                
                if abs(float(val_str) - target_J) < 1e-5:
                    target_j_folder = d
                    break
            except ValueError:
                continue
    
    if target_j_folder is None:
        # Check if the model_folder itself is the J folder
        if model_path.name.startswith("J=") or model_path.name.startswith("J2="):
             try:
                part = model_path.name.split('=')[1]
                if '_' in part:
                    val_str = part.split('_')[0]
                else:
                    val_str = part
                if abs(float(val_str) - target_J) < 1e-5:
                    target_j_folder = model_path
             except:
                 pass

    if target_j_folder is None:
        return s2_values, s2_errors

    # Check for variables_average first (aggregated statistics)
    avg_file = target_j_folder / "variables_average.pkl"
    if not avg_file.exists():
        avg_file = target_j_folder / "variables_average"
    
    if avg_file.exists():
        try:
            with open(avg_file, "rb") as f:
                data = pickle.load(f)
                if 's2_mean' in data:
                    # Assuming s2_mean is a list over iterations, take the last one
                    val = data['s2_mean']
                    if isinstance(val, (list, np.ndarray)):
                        s2_values.append(val[-1])
                    else:
                        s2_values.append(val)
                    
                    if 's2_var' in data:
                        var = data['s2_var']
                        if isinstance(var, (list, np.ndarray)):
                            s2_errors.append(np.sqrt(var[-1]))
                        else:
                            s2_errors.append(np.sqrt(var))
                    return s2_values, s2_errors
        except Exception as e:
            print(f"Error reading {avg_file}: {e}")

    # Fallback to seeds
    for seed_dir in target_j_folder.iterdir():
        if seed_dir.is_dir() and seed_dir.name.startswith("seed_"):
            pkl_path = seed_dir / "variables.pkl"
            if not pkl_path.exists():
                pkl_path = seed_dir / "variables"
            
            if pkl_path.exists():
                try:
                    with open(pkl_path, "rb") as f:
                        data = pickle.load(f)
                        if 's2' in data:
                            val = data['s2']
                            s2_values.append(val)
                            if 's2_error' in data:
                                s2_errors.append(data['s2_error'])
                            else:
                                s2_errors.append(0.0)
                except Exception as e:
                    print(f"Error reading {pkl_path}: {e}")
    
    return s2_values, s2_errors

def plot_entanglement_vs_L(models, target_J, save_name="Entanglement_vs_L"):
    save_dir = Path("/scratch/f/F.Conoscenti/Thesis_QSL/Elaborate/plot/Entanglement")
    save_dir.mkdir(parents=True, exist_ok=True)
    
    Ls = []
    means = []
    errors = []
    
    print(f"Extracting data for J={target_J}...")
    
    for model_path in models:
        L = extract_L(model_path)
        if L is None:
            print(f"Could not extract L from {model_path}")
            continue
        
        vals, errs = get_s2_for_J(model_path, target_J)
        
        if vals:
            # If we have multiple seeds/values, take mean and propagate error
            mean_val = np.mean(vals)
            if len(vals) > 1:
                err_val = np.std(vals) / np.sqrt(len(vals))
            elif len(errs) > 0:
                err_val = errs[0]
            else:
                err_val = 0.0
            
            # Un-normalize the entropy (assuming half-system partition)
            # S2_stored = S2_real / max_ent
            max_ent = (L*L/2) * np.log(2)
            mean_val *= max_ent
            err_val *= max_ent
            
            Ls.append(L)
            means.append(mean_val)
            errors.append(err_val)
            print(f"  L={L}: S2={mean_val:.4f} ± {err_val:.4f}")
        else:
            print(f"  No S2 data found for {model_path}")

    if not Ls:
        print("No data found.")
        return

    # Sort by L
    sorted_indices = np.argsort(Ls)
    Ls = np.array(Ls)[sorted_indices]
    means = np.array(means)[sorted_indices]
    errors = np.array(errors)[sorted_indices]

    # Linear fit: S2 = alpha * L + beta
    def linear_model(x, alpha, beta):
        return alpha * x + beta
    
    # Perform weighted fit
    sigma = errors if np.any(errors > 0) else None
    if sigma is not None:
        sigma[sigma == 0] = 1e-10 # Avoid zero division
        absolute_sigma = True
    else:
        absolute_sigma = False

    popt, pcov = curve_fit(linear_model, Ls, means, sigma=sigma, absolute_sigma=absolute_sigma)
    alpha, beta = popt
    alpha_err, beta_err = np.sqrt(np.diag(pcov))
    
    # TEE is negative intercept (assuming Area Law S = a*L - gamma)
    gamma_topo = -beta
    gamma_topo_err = beta_err
    
    print("\nFIT RESULTS: S2 = α × L + β")
    print(f"α (slope)        = {alpha:.4f} ± {alpha_err:.4f}")
    print(f"β (intercept)    = {beta:.4f} ± {beta_err:.4f}")
    print(f"γ_topo = -β      = {gamma_topo:.4f} ± {gamma_topo_err:.4f}")
    
    model_type = ""
    if len(models) > 0:
        if "ViT" in str(models[0]):
            model_type = "ViT "
        elif "HFDS" in str(models[0]):
            model_type = "HFDS "

    # Plotting
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot data
    ax.errorbar(Ls, means, yerr=errors, fmt='o', markersize=8, capsize=5, 
                label='Computed S₂', color='steelblue')
    
    # Plot fit
    L_fit = np.linspace(Ls.min() - 0.5, Ls.max() + 0.5, 100)
    S2_fit = linear_model(L_fit, alpha, beta)
    ax.plot(L_fit, S2_fit, '--', label=f'Fit: $S_2 = {alpha:.3f}L {beta:+.3f}$', color='coral', linewidth=2)
    
    ax.set_xlabel('System Size L', fontsize=12)
    ax.set_ylabel('Rényi-2 Entropy $S_2$', fontsize=12)
    ax.set_title(f'{model_type}Entanglement Entropy Scaling (J={target_J})', fontsize=14)
    
    # Add text box with gamma_topo (TEE)
    textstr = f'$\gamma_{{topo}} = {gamma_topo:.3f} \pm {gamma_topo_err:.3f}$'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=props)
    
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Save
    filename_model_part = f"_{model_type.strip()}" if model_type.strip() else ""
    full_save_path = save_dir / f"{save_name}{filename_model_part}_J={target_J}.png"
    plt.savefig(full_save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {full_save_path}")
    plt.show()

if __name__ == "__main__":
    # Example usage
    target_J = 0.5
    
    models_hfds = [
        "/scratch/f/F.Conoscenti/Thesis_QSL/HFDS_Heisenberg/plot/4x4/layers1_hidd6_feat128_sample1024_lr0.02_iter500_parityTrue_rotTrue_InitFermi_typecomplex",
        "/scratch/f/F.Conoscenti/Thesis_QSL/HFDS_Heisenberg/plot/6x6/layers1_hidd6_feat128_sample1024_lr0.02_iter2000_parityTrue_rotTrue_InitFermi_typecomplex",
        "/scratch/f/F.Conoscenti/Thesis_QSL/HFDS_Heisenberg/plot/8x8/layers1_hidd8_feat64_sample4096_lr0.02_iter2000_parityTrue_rotTrue_InitFermi_typecomplex_8",
        #"/scratch/f/F.Conoscenti/Thesis_QSL/HFDS_Heisenberg/plot/10x10/layers1_hidd8_feat32_sample4096_lr0.02_iter2000_parityTrue_rotTrue_InitFermi_typecomplex_10"
    ]

    models_vit = [
        "/scratch/f/F.Conoscenti/Thesis_QSL/ViT_Heisenberg/plot/4x4/layers2_d16_heads4_patch2_sample1024_lr0.0075_iter1500_parityTrue_rotTrue_latest_model",
        "/scratch/f/F.Conoscenti/Thesis_QSL/ViT_Heisenberg/plot/6x6/layers3_d40_heads8_patch2_sample1024_lr0.0075_iter3000_parityTrue_rotTrue_latest_model",
        "/scratch/f/F.Conoscenti/Thesis_QSL/ViT_Heisenberg/plot/8x8/layers3_d40_heads8_patch2_sample1024_lr0.0075_iter4000_parityTrue_rotTrue_latest_model"
    ]
    
    plot_entanglement_vs_L(models_hfds, target_J)
