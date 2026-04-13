import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os

def f_lin(x, a, b): return a * x + b
def f_sqrt(x, a, b): return a * np.sqrt(x) + b
def f_log(x, a, b): return a * np.log(x) + b
def f_LlogL(x, a, b): return a * np.sqrt(x) * np.log(x) + b
fit_functions = {'linear': f_lin, 'sqrt': f_sqrt, 'log': f_log, 'LlogL': f_LlogL}

def plot_entropy_scaling_from_pkl(pkl_path, save_path=None, vit_variances=None, hfds_variances=None):
    if not os.path.exists(pkl_path):
        print(f"Error: Could not find the file {pkl_path}")
        return
        
    print(f"Loading data from {pkl_path}...")
    with open(pkl_path, 'rb') as f:
        plot_data = pickle.load(f)
        
    results = plot_data['results']
    xavier_results = plot_data.get('xavier_results')
    variances = plot_data['variances']
    
    # Determine if this is an Entropy vs L plot or an Entropy vs Partition Size plot
    sample_data = list(list(results.values())[0].values())[0]
    is_partition_plot = 'size' in sample_data
    global_L = plot_data.get('L', 6)
    p_type = plot_data.get('partition_type', 'Square')
    
    model_cmaps = {
        'RBM': plt.cm.Blues,
        'ViT': plt.cm.Oranges,
        'HFDS': plt.cm.Greens
    }
    
    plt.figure(figsize=(12, 7))
    
    # Plot the main results
    for name in results:
        cmap = model_cmaps.get(name, plt.cm.Greys)
        present_vars = [v for v in variances if v in results[name]]
        
        if "ViT" in name and vit_variances is not None:
            present_vars = [v for v in present_vars if v in vit_variances]
        elif "HFDS" in name and hfds_variances is not None:
            present_vars = [v for v in present_vars if v in hfds_variances]

        for i, var in enumerate(present_vars):
            if len(present_vars) > 1:
                intensity = 0.4 + 0.6 * (i / (len(present_vars) - 1))
            else:
                intensity = 1.0
            color = cmap(intensity)

            data = results[name][var]
            params_str = ",".join(map(str, data['params']))
            base_label = f"{name} Var={var} (P={params_str})"
            
            if is_partition_plot:
                size_arr = np.array(data['size'])
                N_arr = size_arr * global_L if p_type == "Strip" else size_arr**2
                max_ent = N_arr * np.log(2)
            else:
                L_arr = np.array(data['L'])
                N_arr = L_arr**2
                max_ent = (N_arr / 2.0) * np.log(2)

            y_data = np.array(data['mean']) * max_ent
            y_err = np.array(data['err']) * max_ent

            best_fit_name, best_popt = None, None
            if len(N_arr) > 2:
                min_chisqr = np.inf
                valid_pts = np.isfinite(y_err) & (y_err > 1e-9)
                x_fit, y_fit, y_err_fit = N_arr[valid_pts], y_data[valid_pts], y_err[valid_pts]

                if len(x_fit) > 2:
                    for fit_name, fit_func in fit_functions.items():
                        try:
                            popt, _ = curve_fit(fit_func, x_fit, y_fit, sigma=y_err_fit, absolute_sigma=True)
                            residuals = y_fit - fit_func(x_fit, *popt)
                            chisqr = np.sum((residuals / y_err_fit) ** 2)
                            if chisqr < min_chisqr:
                                min_chisqr, best_fit_name, best_popt = chisqr, fit_name, popt
                        except RuntimeError:
                            continue
            
            fit_label_ext = f" ({best_fit_name} fit)" if best_fit_name else ""
            plt.errorbar(N_arr, y_data, yerr=y_err, 
                         label=base_label + fit_label_ext, color=color, 
                         marker='o', linestyle='none', capsize=5)

            if best_popt is not None:
                x_plot = np.linspace(min(N_arr), max(N_arr), 200)
                plt.plot(x_plot, fit_functions[best_fit_name](x_plot, *best_popt), 
                         color=color, linestyle='-')

    if is_partition_plot:
        plt.xlabel(f'Partition Area ({p_type})')
        plt.title(f'Entanglement Entropy vs Partition Area (L={global_L})')
    else:
        plt.xlabel('Number of Spins N (L^2)')
        plt.title('Entanglement Entropy vs L (Loaded from Pickle)')
        
    plt.ylabel('Renyi-2 Entropy (Unnormalized)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Plot successfully saved to {save_path}")
    else:
        plt.show()

if __name__ == "__main__":
    # Input file to load the data from
    pkl_file_path = "/scratch/f/F.Conoscenti/Thesis_QSL/Entanglement/plots5/Entropy_vs_Partition_Square_L10_data.pkl"
    
    # Where to save the generated plot
    save_image_path = "/scratch/f/F.Conoscenti/Thesis_QSL/Entanglement/plots/Entropy_vs_Partition_Square_L10_Replot1.png"
    
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(save_image_path), exist_ok=True)
    
    # Generate and save the plot
    plot_entropy_scaling_from_pkl(pkl_file_path, save_image_path, vit_variances=None, hfds_variances=None)
