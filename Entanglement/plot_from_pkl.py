import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
import os

def f_lin(x, a, b): return a * x + b
def f_sqrt(x, a, b): return a * np.sqrt(x) + b
def f_log(x, a, b): return a * np.log(x) + b
def f_LlogL(x, a, b): return a * np.sqrt(x) * np.log(x) + b
fit_functions = {'linear': f_lin, 'sqrt': f_sqrt, 'log': f_log, 'LlogL': f_LlogL}

def plot_entropy_scaling_from_pkl(pkl_path, save_path=None, vit_variances=None, hfds_variances=None, max_area=None):
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
        'RBM': plt.cm.Greys,
        'ViT': plt.cm.Reds,
        'HFDS': plt.cm.Greens
    }
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    all_vit_vars = []
    all_hfds_vars = []
    model_params = {}
    
    for name in results:
        cmap = model_cmaps.get(name, plt.cm.Greys)
        present_vars = [v for v in variances if v in results[name]]
        
        if "ViT" in name and vit_variances is not None:
            present_vars = [v for v in present_vars if v in vit_variances]
            all_vit_vars.extend(present_vars)
        elif "HFDS" in name:
            present_vars = [v for v in present_vars if v in hfds_variances]
            all_hfds_vars.extend(present_vars)

        if not present_vars:
            continue

        # Create a normalizer for the variances of the current model
        norm = mcolors.LogNorm(vmin=min(present_vars), vmax=max(present_vars))

        for i, var in enumerate(present_vars):
            color = cmap(norm(var))

            data = results[name][var]
            params_val = data['params'][0] if data['params'] else "N/A"
            if name not in model_params:
                model_params[name] = params_val
            
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

            if max_area is not None:
                mask = N_arr <= max_area
                N_arr = N_arr[mask]
                y_data = y_data[mask]
                y_err = y_err[mask]
                
                if len(N_arr) == 0:
                    continue

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
            
            ax.errorbar(N_arr, y_data, yerr=y_err, 
                         color=color, 
                         marker='o', linestyle='none', capsize=5)

            if best_popt is not None:
                x_plot = np.linspace(min(N_arr), max(N_arr), 200)
                ax.plot(x_plot, fit_functions[best_fit_name](x_plot, *best_popt), 
                         color=color, linestyle='-')

    if is_partition_plot:
        ax.set_xlabel(f'Partition Area ({p_type})')
    else:
        ax.set_xlabel('Number of Spins N')
        
    ax.set_ylabel('Renyi-2 Entropy')
    
    # --- Legend Construction ---
    legend_handles = []
    ordered_names = [n for n in ['ViT', 'HFDS'] if n in results]
    for name in results.keys():
        if name not in ordered_names:
            ordered_names.append(name)

    for name in ordered_names:
        if name in model_params:
            cmap = model_cmaps.get(name)
            if cmap:
                handle = Line2D([0], [0], marker='o', color=cmap(0.7), 
                                label=f'{name} (P={model_params[name]})', linestyle='None')
                legend_handles.append(handle)

    ax.legend(handles=legend_handles, loc='best')
    
    # --- Add Colorbars ---

    # ViT colorbar
    vit_vars_unique = sorted(list(set(all_vit_vars)))
    if vit_vars_unique:
        norm_vit = mcolors.LogNorm(vmin=min(vit_vars_unique), vmax=max(vit_vars_unique))
        sm_vit = plt.cm.ScalarMappable(cmap=model_cmaps['ViT'], norm=norm_vit)
        sm_vit.set_array([])
        cbar_vit = fig.colorbar(sm_vit, ax=ax, orientation='vertical', pad=0.02)
        cbar_vit.set_label('ViT Variance')

    # HFDS colorbar
    hfds_vars_unique = sorted(list(set(all_hfds_vars)))
    if hfds_vars_unique:
        norm_hfds = mcolors.LogNorm(vmin=min(hfds_vars_unique), vmax=max(hfds_vars_unique))
        sm_hfds = plt.cm.ScalarMappable(cmap=model_cmaps['HFDS'], norm=norm_hfds)
        sm_hfds.set_array([])
        cbar_hfds = fig.colorbar(sm_hfds, ax=ax, orientation='vertical', pad=0.08)
        cbar_hfds.set_label('HFDS Variance')

    ax.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Plot successfully saved to {save_path}")
    else:
        plt.show()

if __name__ == "__main__":
    # Input file to load the data from
    pkl_file_path = "/scratch/f/F.Conoscenti/Thesis_QSL/Entanglement/plots/Entropy_vs_L_Scaling_Unnormalized_data_unconstrained_1.pkl"
    pkl_file_path = "/cluster/home/fconoscenti/Thesis_QSL/Entanglement/plots7/Entropy_vs_Partition_Strip_L10_data.pkl"
    #pkl_file_path = "/cluster/home/fconoscenti/Thesis_QSL/Entanglement/plots7/Entropy_vs_Partition_Square_L10_data_1.pkl"
    #pkl_file_path = "/cluster/home/fconoscenti/Thesis_QSL/Entanglement/plots7/Entropy_vs_L_Scaling_Unnormalized_data_unconstrained_1.pkl"
    #pkl_file_path = "/cluster/home/fconoscenti/Thesis_QSL/Entanglement/plots6/Entropy_vs_L_Scaling_Unnormalized_data_unconstrained.pkl"
    #pkl_file_path = "/cluster/home/fconoscenti/Thesis_QSL/Entanglement/plots6/Entropy_vs_Partition_Strip_L10_data.pkl"
    # Where to save the generated plot
    save_image_path = "/cluster/home/fconoscenti/Thesis_QSL/Entanglement/plots7/Entropy_vs_Partition_Square_L10_Replot5.png"
    
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(save_image_path), exist_ok=True)
    
    # Generate and save the plot
    max_area=40
    plot_entropy_scaling_from_pkl(pkl_file_path, save_image_path, vit_variances=[0.001, 0.01, 0.1, 1,10,100], hfds_variances=[0.001, 0.01, 0.1, 1,10,100], max_area=max_area)
