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


#for debug
"""import sys
sys.path.append('/scratch/f/F.Conoscenti/Thesis_QSL/HFDS_Heisenberg')
from HFDS_model_spin import HiddenFermion"""

L       = 4
n_elecs = L*L # L*L should be half filling
n_dim = 2

dtype_ = jnp.float64
MFinitialization = "Fermi"
bounds  = "PBC"
symmetry = True  #True or False

#Varaitional state param
n_hid_ferm       = 4
features         = 32 #hidden units per layer
hid_layers       = 1

n_dim            = 2





###################################################################################################

"""n_samples        =  32
n_chains         =  n_samples//2
cs               =  n_samples

# --- 5. Usage example ---
lattice = nk.graph.Hypercube(length=L, n_dim=n_dim, pbc=True, max_neighbor_order=2)
hi = nk.hilbert.Spin(s=1 / 2, N=lattice.n_nodes, total_sz=0) 

model = HiddenFermion(n_elecs=n_elecs,network="FFNN",n_hid=n_hid_ferm,Lx=L,Ly=L,layers=hid_layers,features=features,MFinit=MFinitialization,hilbert=hi,stop_grad_mf=False,stop_grad_lower_block=False,bounds=bounds,parity=symmetry,dtype=dtype_)

# ------------- define Hamiltonian ------------------------
    # Heisenberg J1-J2 spin ha
ha = nk.operator.Heisenberg(hilbert=hi, graph=lattice, J=[1.0, 0.0], sign_rule=[False, False]).to_jax_operator()  # No Marshall sign rule
# ---------- define sampler ------------------------
sampler = nk.sampler.MetropolisExchange(hilbert=hi,graph=lattice,d_max=2,n_chains=n_chains,sweep_size=lattice.n_nodes,)
vstate = nk.vqs.MCState(sampler, model, n_samples=n_samples, chunk_size=cs, n_discard_per_chain=128)

# Instantiate operator
marshall_op = MarshallSignOperator(hi)

#calculate sign obs of a model from training
folder_path="/scratch/f/F.Conoscenti/Thesis_QSL/HFDS_Heisenberg/plot/J1J2/spin/layers1_hidd4_feat32_sample1024_lr0.02_iter400_symmTrue_Hannah/J2=0.2_L=4"

# --- Loading ---
number_models = len([name for name in os.listdir(f"{folder_path}/models") if os.path.isfile(os.path.join(f"{folder_path}/models", name))])
sign1 = np.zeros(number_models)
sign2 = np.zeros(number_models)
x = np.arange(number_models)*50

for i in range(number_models):
    with open(folder_path + f"/models/model_{i} .mpack", "rb") as f:
        vstate.variables = flax.serialization.from_bytes(vstate.variables, f.read())

    # Compute expectation value MCMC
    #exp_val = vstate.expect(marshall_op)
    #sign1[i] = exp_val.mean
  

    # Compute expectation value full Hilbert space with reweight
    configs = balanced_combinations_numpy(L*L)
    logpsi = vstate.log_value(configs)
    psi = jnp.exp(logpsi)
    weights = jnp.abs(psi) ** 2
    signs = _marshal_sign_single_full_hilbert( configs, vstate) 

    #print("psi.shape:", psi.shape)            # expect (N,)
    #print("weights.shape:", weights.shape)    # expect (N,)
    #print("signs.shape:", signs.shape)        # expect (N,)
    #print("weights.ndim, signs.ndim:", weights.ndim, signs.ndim)

    # reweighted expectation
    sign2[i] = jnp.sum(weights * signs) / jnp.sum(weights)
    #not weighted expectation
    #sign[i] = np.mean(signs)
    
    #print("⟨Marshall Sign Operator⟩ =", sign1[i])
    print("⟨Marshall Sign Operator⟩ =", sign2[i])


# Create the plot
plt.figure(figsize=(10, 6))

# Plot both lines with dots
plt.plot(x, sign1, marker='o', label='MCMC sampled sign', markersize=8, linewidth=2)
plt.plot(x, sign2, marker='o', label='Full Hilbert sampled sign', markersize=8, linewidth=2)

# Customize the graph
plt.title('Two Lines with Data Points', fontsize=14)
plt.ylabel('Sign')
plt.xlabel('Iterations')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{folder_path}/Sign.png")
plt.show()
"""

#################################################################################

def plot_Sign_full_MCMC(marshall_op, vstate, folder_path, n_samples):

    number_models = len([name for name in os.listdir(f"{folder_path}/models") if os.path.isfile(os.path.join(f"{folder_path}/models", name))])
    sign1 = np.zeros(number_models)
    sign2 = np.zeros(number_models)
    x_axis = np.arange(number_models)*50

    sign1, sign2 = Marshall_Sign(marshall_op, vstate, folder_path, n_samples, L)

    print("Marshall Sign = ", sign2[-1])

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(x_axis, sign1, marker='o', label='MCMC sampled sign', markersize=8, linewidth=2)
    plt.plot(x_axis, sign2, marker='o', label='Full Hilbert sampled sign', markersize=8, linewidth=2)
    plt.title('Sign with Full Hilbert space & MCMC ', fontsize=14)
    plt.ylabel('Sign')
    plt.xlabel('Iterations')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{folder_path}/Sign_full_MCMC.png")
    plt.show()



def plot_Sign_Fidelity(ket_gs, vstate, folder_path, hi):

    number_models = len([name for name in os.listdir(f"{folder_path}/models") if os.path.isfile(os.path.join(f"{folder_path}/models", name))])
    sign_vstate = np.zeros(number_models)
    x_axis = np.arange(number_models)*20

    sign_vstate, sign_exact = Marshall_Sign(ket_gs, vstate, folder_path, L, hi)
    fidelity = Fidelity_iteration(vstate, ket_gs, folder_path)

    print("⟨Marshall Sign final vstate⟩ = ", sign_vstate[-1])
    print("⟨Marshall sign exact gs⟩ =", sign_exact)
    
    plt.figure(figsize=(10, 6))
    #left axis: Sign
    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.plot(x_axis, sign_vstate, marker='o', label='Full Hilbert sampled sign',
            markersize=8, linewidth=2, color='tab:blue')
    ax1.set_xlabel("Iterations", fontsize=12)
    ax1.set_ylabel("Sign", color='tab:blue', fontsize=12)
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.axhline(y=sign_exact, color='tab:blue', linestyle='--', linewidth=1.5, alpha=0.7, label='Exact Sign gs')
    ax1.axhline(y=-1*sign_exact, color='tab:blue', linestyle='--', linewidth=1.5, alpha=0.7, label='_nolegend_')

    # right axis: fidelity
    ax2 = ax1.twinx()  # create a second y-axis sharing the same x-axis
    ax2.plot(x_axis, fidelity, marker='s', label='Fidelity',
            markersize=8, linewidth=2, color='tab:red')
    ax2.set_ylabel("Fidelity", color='tab:red', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='tab:red')

    fig.suptitle("Sign & Fidelity", fontsize=14)
    ax1.grid(True, alpha=0.3)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')
    plt.tight_layout()
    plt.savefig(f"{folder_path}/Sign_&_Fidelity.png")
    plt.show()


from matplotlib.lines import Line2D

def plot_Sign_single(ket_gs, vstate, folder_path, hi, number_states, L):
    # Determine number of models / iterations
    number_models = len([name for name in os.listdir(f"{folder_path}/models") if os.path.isfile(os.path.join(f"{folder_path}/models", name))])
    x_axis = np.arange(number_models)*20

    # Get total sign and exact sign
    sign_vstate_tot, sign_exact_tot = Marshall_Sign(ket_gs, vstate, folder_path, L, hi)

    # Get most probable configurations and their signs
    configs, sign_vstate_config, weight_exact, weight_vstate = Marshall_Sign_and_Weights_single_config(
        ket_gs, vstate, folder_path, L, hi, number_states
    )

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))

    # --- 1️⃣ Plot the total sign line ---
    ax.plot(x_axis, sign_vstate_tot, marker='o', label='Sign full Hilbert, vstate',
            markersize=8, alpha=1, linewidth=2, color='tab:blue')

    # Horizontal lines for exact sign
    ax.axhline(y=sign_exact_tot, color='tab:blue', linestyle='--', linewidth=1, alpha=0.4, label='Exact Sign gs full Hilbert')
    ax.axhline(y=-sign_exact_tot, color='tab:blue', linestyle='--', linewidth=1, alpha=0.4, label='_nolegend_')

    # --- 2️⃣ Overlay most probable configurations as + / - symbols ---
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
                                      markersize=10, label=f'Config {i+1}: +/− of vstate {configs[i]}'))

    # --- 3️⃣ Styling ---
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
    plt.savefig(f"{folder_path}/Sign_single_config_symbols_legend.png")
    plt.show()



def plot_Weight_single(ket_gs, vstate, folder_path, hi, number_states):

    number_models = len([name for name in os.listdir(f"{folder_path}/models") if os.path.isfile(os.path.join(f"{folder_path}/models", name))])
    x_axis = np.arange(number_models)*20

    configs, sign_vstate_config, weight_exact, weight_vstate = Marshall_Sign_and_Weights_single_config(ket_gs, vstate, folder_path, L, hi, number_states)
    
    plt.figure(figsize=(10, 6))
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Define colors and labels dynamically
    colors = ['yellow', 'orange', 'purple', 'red', 'blue', 'green', 'brown', 'pink', 'gray', 'cyan']
    labels = [f'Sign config {i+1} most prob, vstate {configs[i]}' for i in range(number_states)]

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
    plt.savefig(f"{folder_path}/Weight_single_config.png")
    plt.show()

def plot_MSE_configs(ket_gs, vstate, folder_path, hi):

    number_models = len([name for name in os.listdir(f"{folder_path}/models") if os.path.isfile(os.path.join(f"{folder_path}/models", name))])
    x_axis = np.arange(number_models)*20

    Error = Mean_Square_Error_configs(ket_gs, vstate, folder_path, L, hi)
    
    plt.figure(figsize=(10, 6))
    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.plot(x_axis, Error, marker='o', label=f'MSE full Hilbert space',
            markersize=8, linewidth=2, color='pink')
    
    ax1.set_xlabel("Iterations", fontsize=12)
    ax1.set_ylabel("MSE configs", fontsize=12)


    fig.suptitle("MSE full Hiblert space", fontsize=14)
    ax1.grid(True, alpha=0.3)
    lines1, labels1 = ax1.get_legend_handles_labels()
    ax1.legend(lines1, labels1, loc='best')
    plt.tight_layout()
    plt.savefig(f"{folder_path}/MSE_configs.png")
    plt.show()

def plot_Sign_Err_Amplitude_Err_Fidelity(ket_gs, vstate, folder_path, hi):

    number_models = len([name for name in os.listdir(f"{folder_path}/models") if os.path.isfile(os.path.join(f"{folder_path}/models", name))])
    x_axis = np.arange(number_models)*20
    array = np.ones(number_models)

    Error = Mean_Square_Error_configs(ket_gs, vstate, folder_path, L, hi)
    Fidelity = Fidelity_iteration(vstate, ket_gs, folder_path)
    sign_vstate, sign_exact = Marshall_Sign(ket_gs, vstate, folder_path, L, hi)
    
    sign_exact_array= array * sign_exact
    sign_err = np.abs(np.abs(sign_vstate) - np.abs(sign_exact_array))
    #print(sign_err)

    plt.figure(figsize=(10, 6))
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # First y-axis (left) - for Error and sign_err
    ax1.plot(x_axis, Error, marker='o', label='MSE configs',
            markersize=8, linewidth=2, color='pink')
    ax1.plot(x_axis, sign_err, marker='o', label='Sign error',
            markersize=8, linewidth=2, color='tab:blue')
    ax1.set_xlabel("Iterations", fontsize=12)
    ax1.set_ylabel("MSE / Error Values", fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')

    # Create second y-axis (right) - for Fidelity
    ax2 = ax1.twinx()
    ax2.plot(x_axis, Fidelity, marker='s', label='Fidelity',
            markersize=8, linewidth=2, color='red')
    ax2.set_ylabel("Fidelity", fontsize=12)

    # Combine legends from both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')

    fig.suptitle("MSE full Hilbert space & Sign Error & Fidelity", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{folder_path}/Sign_Err_&_Amplitude_Err_&_Fidelity.png")
    plt.show()