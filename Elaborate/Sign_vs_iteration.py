#%%
import jax
import numpy as np
import jax.numpy as jnp
import netket as nk
from netket.operator import AbstractOperator
from functools import partial  # partial(sum, axis=1)(x) == sum(x, axis=1)
import flax
import os
import matplotlib.pyplot as plt
import itertools
from matplotlib.lines import Line2D
from pathlib import Path

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
from Elaborate.Error_Stat import Fidelity
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

    sign_vstate_full = Marshall_Sign_full_hilbert(vstate, folder_path, hi)
    sign_exact = Marshall_Sign_exact(ket_gs, hi)
    fidelity = Fidelity_iteration(vstate, ket_gs, folder_path)

    print("⟨Marshall Sign final vstate⟩ = ", sign_vstate_full[-1])
    print("⟨Marshall sign exact gs⟩ =", sign_exact)

    Plot_Sign_Fidelity(sign_vstate_full, sign_exact, fidelity, folder_path, one_avg)

    return sign_vstate_full, sign_exact, fidelity

def Plot_Sign_Fidelity(sign_vstate_full, sign_exact, fidelity, folder_path, one_avg):

    number_models = len([name for name in os.listdir(f"{folder_path}/models") if os.path.isfile(os.path.join(f"{folder_path}/models", name))])
    x_axis = np.arange(number_models)*20
    
    plt.figure(figsize=(10, 6))
    #left axis: Sign
    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.plot(x_axis, sign_vstate_full, marker='o', label='Full Hilbert sampled sign',markersize=8, linewidth=2, color='tab:blue')
    ax1.set_xlabel("Iterations", fontsize=12)
    ax1.set_ylabel("Sign", color='tab:blue', fontsize=12)
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.axhline(y=sign_exact, color='tab:blue', linestyle='--', linewidth=1.5, alpha=0.7, label='Exact Sign gs')
    ax1.axhline(y=-1*sign_exact, color='tab:blue', linestyle='--', linewidth=1.5, alpha=0.7, label='_nolegend_')

    # right axis: fidelity
    ax2 = ax1.twinx()  # create a second y-axis sharing the same x-axis
    ax2.plot(x_axis, fidelity, marker='s', label='Fidelity',markersize=8, linewidth=2, color='tab:red')
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

    sign_vstate_tot = Marshall_Sign_full_hilbert(vstate, folder_path, hi)
    sign_exact_tot = Marshall_Sign_exact(ket_gs, hi)
    configs, sign_vstate_config, weight_exact, weight_vstate = Marshall_Sign_and_Weights_single_config(ket_gs, vstate, folder_path, L, hi, number_states)

    Plot_Sign_single_config(configs, sign_vstate_config, sign_vstate_tot, sign_exact_tot, weight_exact, weight_vstate, number_states, folder_path, one_avg)

    return configs, sign_vstate_config, weight_exact, weight_vstate

def Plot_Sign_single_config(configs, sign_vstate_config,sign_vstate_tot, sign_exact_tot, weight_exact, weight_vstate, number_states, folder_path, one_avg):

    number_models = len([name for name in os.listdir(f"{folder_path}/models") if os.path.isfile(os.path.join(f"{folder_path}/models", name))])
    x_axis = np.arange(number_models)*20

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))

    # --- Plot the total sign line ---
    ax.plot(x_axis, sign_vstate_tot, marker='o', label='Sign full Hilbert, vstate',markersize=8, alpha=1, linewidth=2, color='tab:blue')

    # Horizontal lines for exact sign
    ax.axhline(y=sign_exact_tot, color='tab:blue', linestyle='--', linewidth=1, alpha=0.4, label='Exact Sign gs full Hilbert')
    ax.axhline(y=-sign_exact_tot, color='tab:blue', linestyle='--', linewidth=1, alpha=0.4, label='_nolegend_')

    # --- Overlay most probable configurations as + / - symbols ---
    colors = ['red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'cyan', 'magenta', 'olive']
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
    ax.grid(True, alpha=0.3)

    ax.set_xlim(x_axis[0]-5, x_axis[-1]+5)
    ymin = min(sign_vstate_tot.min(), min(offsets) - 0.2)
    ymax = max(sign_vstate_tot.max(), 0.5)
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
    
    Plot_Weight_single(configs, sign_vstate_config, weight_exact, weight_vstate, number_states, folder_path, one_avg)

    return configs, sign_vstate_config, weight_exact, weight_vstate

def Plot_Weight_single(configs, sign_vstate_config, weight_exact, weight_vstate, number_states, folder_path, one_avg):

    number_models = len([name for name in os.listdir(f"{folder_path}/models") if os.path.isfile(os.path.join(f"{folder_path}/models", name))])
    x_axis = np.arange(number_models)*20
    spin_config = [[] for _ in range(number_states)] 

    plt.figure(figsize=(10, 6))
    fig, ax1 = plt.subplots(figsize=(10, 6))

    for i in range(number_states):
        for s in configs[i]:
            if s == +1:
                spin_config[i].append("↑")
            elif s == -1:
                spin_config[i].append("↓")

        spin_config[i] = "[" + "".join(spin_config[i]) + "]"

    # Define colors and labels dynamically
    colors = ['yellow', 'orange', 'purple', 'red', 'blue', 'green', 'brown', 'pink', 'gray', 'cyan']
    labels = [f'Sign config {i+1} most prob, vstate {spin_config[i]}' for i in range(number_states)]

    for i in range(number_states):

        ax1.plot(x_axis, weight_vstate[i], marker='o', label=labels[i],
                markersize=8, alpha=0.7, linewidth=2, color=colors[i % len(colors)])
        
        ax1.axhline(y=weight_exact[i], color=colors[i % len(colors)], linestyle='--', 
                        linewidth=1, label=f'Exact weight config {i+1}={weight_exact[i]:.4f}')

    ax1.set_xlabel("Iterations", fontsize=12)
    ax1.set_ylabel("Weight", fontsize=12)

    fig.suptitle("Weight most probable configuration", fontsize=14)
    ax1.grid(True, alpha=0.3)
    lines1, labels1 = ax1.get_legend_handles_labels()
    ax1.legend(lines1, labels1, loc='best')
    plt.tight_layout()
    if one_avg == "avg":
        folder_path = Path(folder_path)
        save_path = folder_path.parent /"plot_avg"/"Weight_single_config.png"
        plt.savefig(save_path)
    if one_avg == "one":
        plt.savefig(f"{folder_path}/Sign_plot/Weight_single_config.png")
    
    plt.show()



def plot_MSE_configs(ket_gs, vstate, hi, folder_path, one_avg):

    error = Mean_Square_Error_configs(ket_gs, vstate, folder_path, hi)
    
    Plot_MSE_configs(error, folder_path, one_avg)

    return error

def Plot_MSE_configs(error, folder_path, one_avg):

    number_models = len([name for name in os.listdir(f"{folder_path}/models") if os.path.isfile(os.path.join(f"{folder_path}/models", name))])
    x_axis = np.arange(number_models)*20

    plt.figure(figsize=(10, 6))
    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.plot(x_axis, error, marker='o', label=f'MSE full Hilbert space',
            markersize=8, linewidth=2, color='pink')
    
    ax1.set_xlabel("Iterations", fontsize=12)
    ax1.set_ylabel("MSE configs", fontsize=12)

    fig.suptitle("MSE full Hiblert space", fontsize=14)
    ax1.grid(True, alpha=0.3)
    lines1, labels1 = ax1.get_legend_handles_labels()
    ax1.legend(lines1, labels1, loc='best')
    plt.tight_layout()
    
    if one_avg == "avg":
        folder_path = Path(folder_path)
        save_path = folder_path.parent /"plot_avg"/"MSE_configs.png"
        plt.savefig(save_path)
    if one_avg == "one":
        plt.savefig(f"{folder_path}/Sign_plot/MSE_configs.png")
    
    plt.show()


def plot_Sign_Err_Amplitude_Err_Fidelity(ket_gs, vstate, hi, folder_path, one_avg):

    error = Mean_Square_Error_configs(ket_gs, vstate, folder_path, hi)
    fidelity = Fidelity_iteration(vstate, ket_gs, folder_path)
    sign_vstate = Marshall_Sign_full_hilbert(vstate, folder_path, hi)
    sign_exact = Marshall_Sign_exact(ket_gs, hi)

    Plot_Sign_Err_Amplitude_Err_Fidelity(error, fidelity, sign_vstate, sign_exact, folder_path, one_avg)
    
    return error, fidelity, sign_vstate, sign_exact


def Plot_Sign_Err_Amplitude_Err_Fidelity(error, fidelity, sign_vstate, sign_exact, folder_path, one_avg):

    number_models = len([name for name in os.listdir(f"{folder_path}/models") if os.path.isfile(os.path.join(f"{folder_path}/models", name))])
    x_axis = np.arange(number_models)*20
    array = np.ones(number_models)

    sign_exact_array= array * sign_exact
    sign_err = np.abs(np.abs(sign_vstate) - np.abs(sign_exact_array))

    plt.figure(figsize=(10, 6))
    fig, ax1 = plt.subplots(figsize=(10, 6))
    # First y-axis (left) - for Error and sign_err
    ax1.plot(x_axis, error, marker='o', label='MSE configs',
            markersize=8, linewidth=2, color='pink')
    ax1.plot(x_axis, sign_err, marker='o', label='Sign error',
            markersize=8, linewidth=2, color='tab:blue')
    ax1.set_xlabel("Iterations", fontsize=12)
    ax1.set_ylabel("MSE / Error Values", fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    # Create second y-axis (right) - for Fidelity
    ax2 = ax1.twinx()
    ax2.plot(x_axis, fidelity, marker='s', label='Fidelity',
            markersize=8, linewidth=2, color='red')
    ax2.set_ylabel("Fidelity", fontsize=12)
    # Combine legends from both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')
    fig.suptitle("MSE full Hilbert space & Sign Error & Fidelity", fontsize=14)
    plt.tight_layout()

    if one_avg == "avg":
        folder_path = Path(folder_path)
        save_path = folder_path.parent /"plot_avg"/"Sign_Err_&_Amplitude_Err_&_Fidelity.png"
        plt.savefig(save_path)
    if one_avg == "one":
        plt.savefig(f"{folder_path}/Sign_plot/Sign_Err_&_Amplitude_Err_&_Fidelity.png")
    
    plt.show()