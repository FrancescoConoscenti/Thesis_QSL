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
    sign2 = np.zeros(number_models)
    x_axis = np.arange(number_models)*20

    sign2, fidelity, sign_exact = Marshall_Sign_Fidelity(ket_gs, vstate, folder_path, L, hi)

    print("⟨Marshall Sign final vstate⟩ = ", sign2[-1])
    print("⟨Marshall sign exact gs⟩ =", sign_exact)
    
    plt.figure(figsize=(10, 6))
    #left axis: Sign
    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.plot(x_axis, sign2, marker='o', label='Full Hilbert sampled sign',
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
