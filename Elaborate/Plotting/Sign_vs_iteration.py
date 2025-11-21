#%%
import sys
import jax
import numpy as np
import jax.numpy as jnp
import netket as nk
from netket.operator import AbstractOperator
from functools import partial  # partial(sum, axis=1)(x) == sum(x, axis=1)
import flax
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import itertools
from matplotlib.lines import Line2D
from pathlib import Path
import re
import matplotlib.patches as mpatches
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

import jax
import jax.numpy as jnp
import numpy as np
import netket as nk
from netket.operator import AbstractOperator
from netket.operator._discrete_operator_jax import DiscreteJaxOperator
from netket.operator._discrete_operator import DiscreteOperator

import netket as nk
import jax
import jax.numpy as jnp

from HFDS_Heisenberg.HFDS_model_spin import HiddenFermion
from Elaborate.Statistics.Error_Stat import Fidelity
from Elaborate.Sign_Obs import *


"""def plot_Sign_full_MCMC(marshall_op, vstate, folder_path, n_samples, hi):

    sign_vstate_MCMC = Marshall_Sign_MCMC(marshall_op, vstate, folder_path, n_samples)
    sign_vstate_full = Marshall_Sign_full_hilbert(vstate, folder_path, hi)
    print("Marshall Sign = ", sign_vstate_full[-1])

    Plot_Sign_full_MCMC(sign_vstate_MCMC, sign_vstate_full, folder_path)

    return sign_vstate_MCMC, sign_vstate_full


def Plot_Sign_full_MCMC(sign_vstate_MCMC, sign_vstate_full, folder_path):

    number_models = len([name for name in os.listdir(f"{folder_path}/models") if os.path.isfile(os.path.join(f"{folder_path}/models", name))])
    sign_vstate_MCMC = np.zeros(number_models)
    sign_vstate_full = np.zeros(number_models)
    x_axis = np.arange(number_models)*20

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(x_axis, sign_vstate_MCMC, marker='o', label='MCMC sampled sign', markersize=8, linewidth=2)
    plt.plot(x_axis, sign_vstate_full, marker='o', label='Full Hilbert sampled sign', markersize=8, linewidth=2)
    plt.title('Sign with Full Hilbert space & MCMC ', fontsize=14)
    plt.ylabel('Sign')
    plt.xlabel('Iterations')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{folder_path}/Sign_plot/Sign_full_MCMC.png")
    plt.show()"""


def plot_Sign_Fidelity(ket_gs, vstate,  hi, folder_path, one_avg):

    sign_vstate_full,_ = Marshall_Sign_full_hilbert(vstate, folder_path, hi)
    sign_exact, _ = Marshall_Sign_exact(ket_gs, hi)
    fidelity = Fidelity_iteration(vstate, ket_gs, folder_path)

    print("⟨Marshall Sign final vstate⟩ = ", sign_vstate_full[-1])
    print("⟨Marshall sign exact gs⟩ =", sign_exact)

    Plot_Sign_Fidelity(sign_vstate_full, sign_exact, fidelity, folder_path, one_avg)

    return sign_vstate_full, sign_exact, fidelity

def _get_iter_from_path(path_str):
    """Extracts iteration number from a path string like '..._iter3000...'."""
    match = re.search(r'_iter(\d+)', path_str)
    if match:
        return int(match.group(1))
    return None

def Plot_Sign_Fidelity(sign_vstate_full, sign_exact, fidelity, folder_path, one_avg, plot_variance=False, sign_vstate_full_var=None, fidelity_var=None):

    num_models = len(sign_vstate_full)
    total_iterations = _get_iter_from_path(str(folder_path))
    if total_iterations and num_models > 1:
        save_every = total_iterations // (num_models-1) if num_models > 1 else total_iterations
    else:
        save_every = 20 # Fallback
    # Determine x-axis length from the data itself to avoid mismatches.
    x_axis = np.arange(num_models) * save_every
    
    plt.figure(figsize=(10, 6))
    #left axis: Sign
    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.plot(x_axis, sign_vstate_full, marker='o', label='Full Hilbert sampled sign',markersize=8, linewidth=2, color='tab:blue')
    ax1.set_xlabel("Iterations", fontsize=12)
    if one_avg == "avg" and plot_variance and sign_vstate_full_var is not None:
        std_dev = np.sqrt(sign_vstate_full_var)
        ax1.errorbar(x_axis, sign_vstate_full, yerr=std_dev, fmt='none', ecolor='tab:blue', capsize=5, alpha=0.5)

    ax1.set_ylabel("Sign", color='tab:blue', fontsize=12)
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.axhline(y=sign_exact, color='tab:blue', linestyle='--', linewidth=1.5, alpha=0.7, label='Exact Sign gs')
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax1.axhline(y=-1*sign_exact, color='tab:blue', linestyle='--', linewidth=1.5, alpha=0.7, label='_nolegend_')

    # right axis: fidelity
    ax2 = ax1.twinx()  # create a second y-axis sharing the same x-axis
    ax2.plot(x_axis, fidelity, marker='s', label='Fidelity',markersize=8, linewidth=2, color='tab:red')
    if one_avg == "avg" and plot_variance and fidelity_var is not None:
        std_dev = np.sqrt(fidelity_var)
        ax2.errorbar(x_axis, fidelity, yerr=std_dev, fmt='none', ecolor='tab:red', capsize=5, alpha=0.5)

    ax2.set_ylabel("Fidelity", color='tab:red', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='tab:red')

    fig.suptitle("Sign & Fidelity", fontsize=14)
    ax1.grid(True, alpha=0.3)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')
    plt.tight_layout()
    if one_avg == "avg":
        folder_path = Path(folder_path)
        save_path = folder_path.parent / "plot_avg" / "Sign_&_Fidelity.png"
        plt.savefig(save_path)
    if one_avg == "one":
        plt.savefig(f"{folder_path}/Sign_plot/Sign_&_Fidelity.png")
    
    plt.show()


def plot_Sign_single_config(ket_gs, vstate, hi, number_states, L, folder_path, one_avg):

    sign_vstate_tot, _ = Marshall_Sign_full_hilbert(vstate, folder_path, hi)
    sign_exact_tot, _ = Marshall_Sign_exact(ket_gs, hi)
    configs, sign_vstate_config, weight_exact, weight_vstate = Marshall_Sign_and_Weights_single_config(ket_gs, vstate, folder_path, L, hi, number_states)

    Plot_Sign_single_config(configs, sign_vstate_config, sign_vstate_tot, sign_exact_tot, weight_exact, weight_vstate, number_states, folder_path, one_avg, plot_variance=False)

    return configs, sign_vstate_config, weight_exact, weight_vstate

def Plot_Sign_single_config(configs, sign_vstate_config,sign_vstate_tot, sign_exact_tot, weight_exact, weight_vstate, number_states, folder_path, one_avg, plot_variance=False, sign_vstate_full_var=None):

    num_models = len(sign_vstate_tot)
    total_iterations = _get_iter_from_path(str(folder_path))
    if total_iterations and num_models > 1:
        save_every = total_iterations // (num_models-1) if num_models > 1 else total_iterations
    else:
        save_every = 20 # Fallback
    # Determine x-axis length from the data itself to avoid mismatches.
    x_axis = np.arange(num_models) * save_every

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))

    # --- Plot the total sign line ---
    ax.plot(x_axis, sign_vstate_tot, marker='o', label='Sign full Hilbert, vstate',markersize=8, alpha=1, linewidth=2, color='tab:blue')
    if one_avg == "avg" and plot_variance and sign_vstate_full_var is not None:
        std_dev = np.sqrt(sign_vstate_full_var)
        ax.errorbar(x_axis, sign_vstate_tot, yerr=std_dev, fmt='none', ecolor='tab:blue', capsize=5, alpha=0.5)

    # Horizontal lines for exact sign
    ax.axhline(y=sign_exact_tot, color='tab:blue', linestyle='--', linewidth=1, alpha=0.4, label='Exact Sign gs full Hilbert')
    ax.axhline(y=-sign_exact_tot, color='tab:blue', linestyle='--', linewidth=1, alpha=0.4, label='_nolegend_')

    # --- Overlay most probable configurations as + / - symbols ---
    colors = ['green', 'orange','brown', 'pink', 'gray', 'cyan', 'magenta', 'olive']
    offsets = np.linspace(0.2, -0.2, number_states)  # stagger vertically

    # For legend: create dummy lines
    legend_elements = [Line2D([0], [0], color='tab:blue', lw=2, marker='o', label='Sign full Hilbert')]
    
    for i in range(number_states):
        # Plot symbols
        for j, x in enumerate(x_axis):
            sym = '+' if sign_vstate_config[i][j] > 0 else '−'
            ax.text(x, offsets[i], sym, ha='center', va='center', fontsize=14, color=colors[i % len(colors)])
        # Add legend entry
        legend_elements.append(Line2D([0], [0], color=colors[i % len(colors)], lw=0, marker='o',
                                      markersize=10, label=f'Sign vstate Config {i+1} with weight = {weight_vstate[i,-1]:.2}'))

    # --- Styling ---
    ax.set_xlabel("Iterations", fontsize=12)
    ax.set_ylabel("Sign", fontsize=12)
    ax.set_title("Marshall Sign: Total and Most Probable Configurations", fontsize=14)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(True, alpha=0.3)

    ax.set_xlim(x_axis[0]-5, x_axis[-1]+5)
    
    # Filter out NaN/Inf values before calculating min/max for y-limits
    valid_signs = sign_vstate_tot[np.isfinite(sign_vstate_tot)]
    if valid_signs.size > 0:
        min_sign_val = valid_signs.min()
        max_sign_val = valid_signs.max()
    else:
        min_sign_val = -1.0 # Default range if all signs are non-finite
        max_sign_val = 1.0
    ymin = min(min_sign_val, min(offsets) - 0.2)
    ymax = max(max_sign_val, 0.5)
    ax.set_ylim(ymin, ymax)

    # Add legend with all lines + dummy symbols
    ax.legend(handles=legend_elements, loc='best')
    plt.tight_layout()
    if one_avg == "avg":
        folder_path = Path(folder_path)
        save_path = folder_path.parent /"plot_avg"/"Sign_single_config.png"
        plt.savefig(save_path)
    if one_avg == "one":
        plt.savefig(f"{folder_path}/Sign_plot/Sign_single_config.png")
    
    
    plt.show()



def plot_Weight_single(ket_gs, vstate, hi, number_states, L, folder_path, one_avg):

    configs, sign_vstate_config, weight_exact, weight_vstate = Marshall_Sign_and_Weights_single_config(ket_gs, vstate, folder_path, L, hi, number_states)
    
    Plot_Weight_single(configs, sign_vstate_config, weight_exact, weight_vstate, number_states, folder_path, one_avg, plot_variance=False)

    return configs, sign_vstate_config, weight_exact, weight_vstate

def Plot_Weight_single(configs, sign_vstate_config, weight_exact, weight_vstate, number_states, folder_path, one_avg, plot_variance=False, weight_vstate_var=None):

    num_models = weight_vstate.shape[1]
    total_iterations = _get_iter_from_path(str(folder_path))
    if total_iterations and num_models > 1:
        save_every = total_iterations // (num_models-1) if num_models > 1 else total_iterations
    else:
        save_every = 20 # Fallback
    # Determine x-axis length from the data itself to avoid mismatches.
    x_axis = np.arange(num_models) * save_every
    spin_config = [[] for _ in range(number_states)] 

    # --- helper to create a colored image from a 1D spin config (+1 = blue, -1 = red) ---
    def make_colored_grid_image(config, cell_pixels=20):
        """
        config: 1D array-like of +1 / -1, length L*L
        cell_pixels: how many pixels per grid cell (bigger -> bigger square)
        returns: (H, W, 3) numpy float image with values in [0,1]
        """
        L = int(np.sqrt(len(config)))
        arr = np.array(config).reshape(L, L)
        img = np.zeros((L, L, 3), dtype=float)
        img[arr == 1] = [0.0, 0.0, 1.0]   # blue
        img[arr == -1] = [1.0, 0.0, 0.0]  # red

        # Upscale each cell to cell_pixels x cell_pixels so the grid is larger and clearer
        img_big = np.kron(img, np.ones((cell_pixels, cell_pixels, 1)))
        return img_big

    # --- plotting (replace or adapt your existing plotting block) ---
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Example style colors for the lines (keeps your previous palette)
    line_colors = ['green', 'orange','brown', 'pink', 'gray', 'cyan', 'magenta', 'olive']

    # Plot the main lines
    for i in range(number_states):
        color = line_colors[i % len(line_colors)]
        ax1.plot(
            x_axis,
            weight_vstate[i] + 0.0003 * i,   # your offset so they don't overlap
            marker='s',
            markersize=8,
            alpha=0.8,
            linewidth=2,
            color=color,
            label=f'Config {i+1}'
        )
        # optional errorbars
        if one_avg == "avg" and plot_variance and (weight_vstate_var is not None):
            std_dev = np.sqrt(weight_vstate_var[i])
            ax1.errorbar(x_axis, weight_vstate[i], yerr=std_dev, fmt='none',
                        ecolor=color, capsize=5, alpha=0.5)
        # exact weight horizontal line
        ax1.axhline(y=weight_exact[i], color=color, linestyle='--',
                    linewidth=1, label=f'Exact weight config {i+1}={weight_exact[i]:.4f}')

    # Axis labels + title
    ax1.set_xlabel("Iterations", fontsize=12)
    ax1.set_ylabel("Weight", fontsize=12)
    ax1.set_title("Weight most probable configuration", fontsize=14)
    ax1.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax1.grid(True, alpha=0.3)

    # Build legend for the lines (we will color the legend text after creating it)
    legend = ax1.legend(loc='upper left', bbox_to_anchor=(0.01, 0.99))
    # Color legend texts to match their lines (if legend entries match number_states)
    for text, i in zip(legend.get_texts(), range(number_states * 2)):  # you have lines + axhlines -> double entries
        # Only color the first number_states text entries that correspond to your "Config i" labels.
        # We try to find the ones that start with "Config " (safe approach).
        if text.get_text().startswith("Config "):
            # extract index -> "Config X" where X is 1-based
            try:
                idx = int(text.get_text().split()[1]) - 1
                col = line_colors[idx % len(line_colors)]
                text.set_color(col)
            except Exception:
                pass

    # === Insert the colored 2D grid images to the right of the plot and label them ===
    # compute vertical spacing and positions so they don't overlap
    right_x = 0.82   # left coordinate of first image in figure coords
    img_width = 0.12 # width fraction for each image block
    img_height = 0.12
    v_space = 0.02   # vertical spacing between images

    # Make sure we have enough vertical space: compute total height and adjust img_height if needed
    total_needed = number_states * img_height + (number_states - 1) * v_space
    if total_needed > 0.9:
        # shrink each to fit
        img_height = (0.9 - (number_states - 1) * v_space) / number_states
        img_width = img_height  # keep roughly square

    for i in range(number_states):
        # position from top downward (figure coordinate system)
        y_top = 0.95 - i * (img_height + v_space)  # top for this image
        # convert y_top (top) to bottom for add_axes
        y_bottom = y_top - img_height

        # Create a small inset axes for the i-th grid
        ax_img = fig.add_axes([right_x, y_bottom, img_width, img_height])
        ax_img.axis('off')  # no ticks, frame is optional
        grid_img = make_colored_grid_image(configs[i], cell_pixels=16)  # increase cell_pixels for bigger squares
        ax_img.imshow(grid_img, interpolation='nearest', origin='upper')
        # Optional framed box around the image to make it stand out
        for spine in ax_img.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(0.5)
            spine.set_visible(True)

        # Put the text label to the right of the inset image, colored the same as the plotted line
        txt_x = right_x + img_width + 0.01
        txt_y = y_bottom + img_height / 2.0
        label_color = line_colors[i % len(line_colors)]
        fig.text(txt_x, txt_y, f"Config {i+1}", va='center', ha='left', color=label_color, fontsize=10)

    plt.tight_layout(rect=[0, 0, 0.80, 1])  # leave room on the right for our insets



    if one_avg == "avg":
        folder_path = Path(folder_path)
        save_path = folder_path.parent / "plot_avg" / "Weight_single_config.png"
        plt.savefig(save_path)
    elif one_avg == "one":
        plt.savefig(f"{folder_path}/Sign_plot/Weight_single_config.png")

    plt.show()



def plot_Amp_overlap_configs(ket_gs, vstate, hi, folder_path, one_avg): 

    amp_overlap = Amplitude_overlap_configs(ket_gs, vstate, folder_path, hi)
    
    Plot_Amp_overlap_configs(amp_overlap, folder_path, one_avg, plot_variance=False)

    return amp_overlap

def Plot_Amp_overlap_configs(error, folder_path, one_avg, plot_variance=False, error_var=None):

    num_models = len(error)
    total_iterations = _get_iter_from_path(str(folder_path))
    if total_iterations and num_models > 1:
        save_every = total_iterations // (num_models-1) if num_models > 1 else total_iterations
    else:
        save_every = 20 # Fallback
    # Determine x-axis length from the data itself to avoid mismatches.
    x_axis = np.arange(num_models) * save_every

    plt.figure(figsize=(10, 6))
    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.plot(x_axis, error, marker='o', label=f'Amplitude Overlap full Hilbert space',
            markersize=8, linewidth=2, color='pink')
    if one_avg == "avg" and plot_variance and error_var is not None:
        std_dev = np.sqrt(error_var)
        ax1.errorbar(x_axis, error, yerr=std_dev, fmt='none', ecolor='pink', capsize=5, alpha=0.5)
    
    ax1.set_xlabel("Iterations", fontsize=12)
    ax1.set_ylabel("Amplitude Overlap configs", fontsize=12)
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

    fig.suptitle("Amplitude Overlap full Hiblert space", fontsize=14)
    ax1.grid(True, alpha=0.3)
    lines1, labels1 = ax1.get_legend_handles_labels()
    ax1.legend(lines1, labels1, loc='best')
    plt.tight_layout()
    
    if one_avg == "avg":
        folder_path = Path(folder_path)
        save_path = folder_path.parent /"plot_avg"/"Amp_Overlap_configs.png"
        plt.savefig(save_path)
    if one_avg == "one":
        plt.savefig(f"{folder_path}/Sign_plot/Amp_Overlap_configs.png")
    
    plt.show()


def plot_Sign_Err_Amplitude_Err_Fidelity(ket_gs, vstate, hi, folder_path, one_avg):

    amplitude_overlap = Amplitude_overlap_configs(ket_gs, vstate, folder_path, hi)
    fidelity = Fidelity_iteration(vstate, ket_gs, folder_path)
    sign_vstate, signs_vstate = Marshall_Sign_full_hilbert(vstate, folder_path, hi)
    sign_exact, signs_exact = Marshall_Sign_exact(ket_gs, hi)
    sign_overlap = Sign_overlap(ket_gs, signs_vstate, signs_exact)

    Plot_Sign_Err_Amplitude_Err_Fidelity(amplitude_overlap, fidelity, sign_overlap, folder_path, one_avg, plot_variance=False, error_var=None, fidelity_var=None, sign_err_var=None)
    
    return amplitude_overlap, fidelity, sign_vstate, sign_exact, sign_overlap

def Plot_Sign_Err_Amplitude_Err_Fidelity(amp_overlap, fidelity, sign_err, folder_path, one_avg, plot_variance=False, error_var=None, fidelity_var=None, sign_err_var=None):

    num_models = len(amp_overlap)
    total_iterations = _get_iter_from_path(str(folder_path))
    if total_iterations and num_models > 1:
        save_every = total_iterations // (num_models-1) if num_models > 1 else total_iterations
    else:
        save_every = 20 # Fallback
    # Determine x-axis length from the data itself to avoid mismatches.
    x_axis = np.arange(num_models) * save_every
 
    plt.figure(figsize=(10, 6))
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Amplitude Overlap
    ax1.plot(x_axis, amp_overlap, marker='o', label='Amplitude Overlap configs',markersize=8, linewidth=2, color='pink')

    if one_avg == "avg" and plot_variance and error_var is not None:
        std_dev = np.sqrt(error_var)
        ax1.errorbar(x_axis, amp_overlap, yerr=std_dev, fmt='none', ecolor='pink', capsize=5, alpha=0.5)

    # Sign Overlap
    ax1.plot(x_axis, sign_err, marker='o', label='Sign overlap',markersize=8, linewidth=2, color='tab:blue')

    if one_avg == "avg" and plot_variance and sign_err_var is not None:
        std_dev = np.sqrt(sign_err_var)
        ax1.errorbar(x_axis, sign_err, yerr=std_dev, fmt='none', ecolor='tab:blue', capsize=5, alpha=0.5)

    ax1.set_xlabel("Iterations", fontsize=12)
    ax1.set_ylabel("Amplitude Overlap / Sign Overlap/ Fidelity", fontsize=12)
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax1.grid(True, alpha=0.3)

    # Fidelity
    ax1.plot(x_axis, fidelity, marker='s', label='Fidelity',markersize=8, linewidth=2, color='tab:red')

    if one_avg == "avg" and plot_variance and fidelity_var is not None:
        std_dev = np.sqrt(fidelity_var)
        ax1.errorbar(x_axis, fidelity, yerr=std_dev, fmt='none', ecolor='tab:red', capsize=5, alpha=0.5)

    # Combine legends from both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    ax1.legend(lines1, labels1, loc='best')
    fig.suptitle("Amplitude Overlap full Hilbert space & Sign Overlap & Fidelity", fontsize=14)
    plt.tight_layout()

    if one_avg == "avg":
        folder_path = Path(folder_path)
        save_path = folder_path.parent /"plot_avg"/"Sign_Err_&_Amplitude_Err_&_Fidelity.png"
        plt.savefig(save_path)
    if one_avg == "one":
        plt.savefig(f"{folder_path}/Sign_plot/Sign_Err_&_Amplitude_Err_&_Fidelity.png")
    
    plt.show()


def plot_Sign_Err_vs_Amplitude_Err_with_iteration(ket_gs, vstate, hi, folder_path, one_avg):

    amplitude_overlap = Amplitude_overlap_configs(ket_gs, vstate, folder_path, hi)
    sign_vstate, signs_vstate = Marshall_Sign_full_hilbert(vstate, folder_path, hi)
    sign_exact, signs_exact = Marshall_Sign_exact(ket_gs, hi)
    #sign_err = Sign_difference(sign_vstate, sign_exact)
    sign_overlap = Sign_overlap(ket_gs, signs_vstate, signs_exact)

    Plot_Sign_Err_vs_Amplitude_Err_with_iteration(amplitude_overlap, sign_overlap, folder_path, one_avg, plot_variance=False)

    return amplitude_overlap, sign_vstate, sign_exact, sign_overlap
                       

def Plot_Sign_Err_vs_Amplitude_Err_with_iteration(amplitude_overlap, sign_overlap, folder_path, one_avg, plot_variance=False, amplitude_overlap_var=None, sign_overlap_var=None):
    
    plt.figure(figsize=(10, 6))
    fig, ax = plt.subplots(figsize=(10, 6))

    if plot_variance and amplitude_overlap_var is not None and sign_overlap_var is not None:
        xerr = np.sqrt(amplitude_overlap_var)
        yerr = np.sqrt(sign_overlap_var)
        ax.errorbar(amplitude_overlap, sign_overlap, xerr=xerr, yerr=yerr,
                    fmt='o', color='purple', capsize=5, alpha=0.7, label='Models with Variance')
    else:
        ax.scatter(amplitude_overlap, sign_overlap, marker='o', color='purple', alpha=0.7, label='Models')

    ax.set_xlabel("Amplitude Overlap (Amplitude Overlap configs)", fontsize=12)
    ax.set_ylabel("Sign Overlap", fontsize=12)
    ax.set_title("Sign Overlap vs Amplitude Overlap", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')

    # Set x and y axis limits from 0 to 1
    ax.set_xlim(0.5, 1)
    ax.set_ylim(0.5, 1)
    plt.tight_layout()

    if one_avg == "avg":
        folder_path = Path(folder_path)
        save_path = folder_path.parent / "plot_avg" / "Sign_Overlap_vs_Amplitude_Overlap.png"
        plt.savefig(save_path)
    if one_avg == "one":
        plt.savefig(f"{folder_path}/Sign_plot/Sign_Overlap_vs_Amplitude_Overlap.png")
    
    plt.show()


def plot_Overlap_vs_iteration(ket_gs, vstate, hi, folder_path, one_avg):
    """
    Calculates and plots Sign Overlap and Amplitude Overlap vs. optimization iterations.
    """
    amplitude_overlap = Amplitude_overlap_configs(ket_gs, vstate, folder_path, hi)
    _, signs_vstate = Marshall_Sign_full_hilbert(vstate, folder_path, hi)
    _, signs_exact = Marshall_Sign_exact(ket_gs, hi)
    sign_overlap = Sign_overlap(ket_gs, signs_vstate, signs_exact)

    Plot_Overlap_vs_iteration(amplitude_overlap, sign_overlap, folder_path, one_avg, plot_variance=False)

    return amplitude_overlap, sign_overlap

def Plot_Overlap_vs_iteration(amplitude_overlap, sign_overlap, folder_path, one_avg, plot_variance=False, amplitude_overlap_var=None, sign_overlap_var=None):
    """
    Plots Sign Overlap and Amplitude Overlap on the y-axis against optimization iterations on the x-axis.
    """
    num_models = len(amplitude_overlap)
    total_iterations = _get_iter_from_path(str(folder_path))
    if total_iterations and num_models > 1:
        save_every = total_iterations // (num_models-1) if num_models > 1 else total_iterations
    else:
        save_every = 20 # Fallback
    x_axis = np.arange(num_models) * save_every
    
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot Amplitude Overlap
    ax.plot(x_axis, amplitude_overlap, marker='o', label='Amplitude Overlap', color='tab:purple', markersize=8, linewidth=2)
    if one_avg == "avg" and plot_variance and amplitude_overlap_var is not None:
        std_dev_amp = np.sqrt(amplitude_overlap_var)
        ax.errorbar(x_axis, amplitude_overlap, yerr=std_dev_amp, fmt='none', ecolor='tab:purple', capsize=5, alpha=0.5)

    # Plot Sign Overlap
    ax.plot(x_axis, sign_overlap, marker='s', label='Sign Overlap', color='tab:green', markersize=8, linewidth=2)
    if one_avg == "avg" and plot_variance and sign_overlap_var is not None:
        std_dev_sign = np.sqrt(sign_overlap_var)
        ax.errorbar(x_axis, sign_overlap, yerr=std_dev_sign, fmt='none', ecolor='tab:green', capsize=5, alpha=0.5)

    ax.set_xlabel("Iterations", fontsize=12)
    ax.set_ylabel("Overlap Value", fontsize=12)
    ax.set_title("Amplitude and Sign Overlap vs. Iterations", fontsize=14)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
    plt.tight_layout()

    if one_avg == "avg":
        save_path = Path(folder_path).parent / "plot_avg" / "Overlaps_vs_Iteration.png"
        plt.savefig(save_path)
    elif one_avg == "one":
        plt.savefig(f"{folder_path}/Sign_plot/Overlaps_vs_Iteration.png")

    plt.show()


def plot_Overlap_vs_Weight(ket_gs, vstate, hi, folder_path, one_avg):
    """
    Calculates and plots per-configuration Sign and Amplitude Overlap vs. exact weight.
    This is done for the final trained model.
    """
    # Get the final model state
    number_models = len([name for name in os.listdir(f"{folder_path}/models") if os.path.isfile(os.path.join(f"{folder_path}/models", name))])
    model_index = number_models - 1
    with open(folder_path + f"/models/model_{model_index}.mpack", "rb") as f:
        vstate.variables = flax.serialization.from_bytes(vstate.variables, f.read())

    # Get all states and amplitudes
    all_states = hi.all_states()
    psi_vstate = vstate.log_value(all_states)
    psi_exact = ket_gs.reshape(psi_vstate.shape) # Ensure shapes match

    # Normalize the variational wavefunction
    psi_vstate_unnorm = np.exp(psi_vstate)
    norm_vstate = np.sqrt(np.sum(np.abs(psi_vstate_unnorm)**2))
    psi_vstate_norm = psi_vstate_unnorm / norm_vstate

    # Calculate weights and overlaps
    weights_exact = np.abs(psi_exact)**2
    
    # Calculate per-configuration amplitude fidelity. This is bounded between 0 and 1.
    # It is 1 if |psi_vstate_norm| = |psi_exact| for a given configuration.
    amp_vstate = np.abs(psi_vstate_norm)
    amp_exact = np.abs(psi_exact)
    numerator = 2 * amp_vstate * amp_exact
    denominator = amp_vstate**2 + amp_exact**2
    amp_overlap_per_config = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator!=0)

    # Calculate sign overlap as sign(<s|psi_var>) * sign(<s|psi_exact>)
    sign_overlap_per_config = (np.sign(psi_vstate_norm.real) * np.sign(psi_exact.real) + 1 ) / 2 # Map to [0,1] for plotting

    # --- Bin weights on a log scale and average overlaps within each bin ---
    # This reduces the number of points for a clearer plot.
    num_bins = 40  # Adjust this number to control the density of points
    
    # Filter out zero weights to avoid issues with log scale
    valid_indices = weights_exact > 1e-12
    weights_to_bin = weights_exact[valid_indices]
    amp_to_bin = amp_overlap_per_config[valid_indices]
    sign_to_bin = sign_overlap_per_config[valid_indices]

    # Create logarithmic bins
    log_bins = np.logspace(np.log10(weights_to_bin.min()), np.log10(weights_to_bin.max()), num_bins + 1)
    
    # Digitize weights to find which bin each weight belongs to
    bin_indices = np.digitize(weights_to_bin, log_bins)

    # Calculate the mean for each bin using np.bincount for efficiency
    binned_weights = np.bincount(bin_indices, weights=weights_to_bin)[1:num_bins+1]
    binned_amp_overlap = np.bincount(bin_indices, weights=amp_to_bin)[1:num_bins+1]
    binned_sign_overlap = np.bincount(bin_indices, weights=sign_to_bin)[1:num_bins+1]
    counts = np.bincount(bin_indices)[1:num_bins+1]
    
    # Avoid division by zero for empty bins
    non_empty = counts > 0
    avg_weights = np.divide(binned_weights, counts, where=non_empty, out=np.zeros_like(binned_weights))
    avg_amp_overlap = np.divide(binned_amp_overlap, counts, where=non_empty, out=np.zeros_like(binned_amp_overlap))
    avg_sign_overlap = np.divide(binned_sign_overlap, counts, where=non_empty, out=np.zeros_like(binned_sign_overlap))

    Plot_Overlap_vs_Weight(avg_weights[non_empty], avg_amp_overlap[non_empty], avg_sign_overlap[non_empty], folder_path, one_avg)

    return avg_weights[non_empty], avg_amp_overlap[non_empty], avg_sign_overlap[non_empty]
def Plot_Overlap_vs_Weight(weights, amp_overlap, sign_overlap, folder_path, one_avg):
    """
    Plots per-configuration Sign and Amplitude Overlap vs. exact weight.
    """
    fig, ax = plt.subplots(figsize=(14, 7))

    x = np.arange(len(weights))  # the label locations
    width = 0.35  # the width of the bars

    # Plot Amplitude Overlap
    ax.bar(x - width/2, amp_overlap, width, label='Amplitude Overlap', color='tab:purple', alpha=0.9)

    # Plot Sign Overlap
    ax.bar(x + width/2, sign_overlap, width, label='Sign Overlap', color='tab:green', alpha=0.9)

    ax.set_xlabel("Binned Exact Weight |<s|ψ_exact>|²", fontsize=12)
    ax.set_ylabel("Overlap Value", fontsize=12)
    ax.set_title("Per-Configuration Overlap vs. Exact Weight", fontsize=14)
    
    # Use bin indices for ticks and label them with the corresponding average weight
    # Show fewer labels to avoid clutter
    tick_indices = np.linspace(0, len(x) - 1, num=10, dtype=int)
    ax.set_xticks(tick_indices)
    ax.set_xticklabels([f'{weights[i]:.1e}' for i in tick_indices], rotation=45, ha="right")

    ax.grid(True, axis='y', linestyle="--", alpha=0.5)
    ax.legend(loc='best')
    plt.tight_layout()

    if one_avg == "avg":
        save_path = Path(folder_path).parent / "plot_avg" / "Overlap_vs_Weight.png"
        plt.savefig(save_path)
    elif one_avg == "one":
        plt.savefig(f"{folder_path}/Sign_plot/Overlap_vs_Weight.png")

    plt.show()
 