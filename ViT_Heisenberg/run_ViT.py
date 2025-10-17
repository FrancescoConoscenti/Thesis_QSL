#%%
import matplotlib.pyplot as plt
import netket as nk
import jax
import numpy as np
import jax.numpy as jnp
print(jax.devices())
import flax
from flax import linen as nn
from netket.operator.spin import sigmax, sigmaz, sigmay
import sys
import os
import argparse
import pickle

from ViT_Heisenberg.ViT_model import ViT_sym


from Elaborate.Energy import *
from Elaborate.Corr_Struct import *
from Elaborate.Error_Stat import *
from Elaborate.count_params import *
from Elaborate.Sign_vs_iteration import *
from Elaborate.Sign_vs_iteration import *

 
parser = argparse.ArgumentParser(description="Example script with parameters")
parser.add_argument("--J2", type=float, default=0.5, help="Coupling parameter J2")
parser.add_argument("--seed", type=float, default=0, help="seed")
args = parser.parse_args()

M = 10  # Number of spin configurations to initialize the parameters
L = 4  # Linear size of the lattice
symm = True

n_dim = 2
J2 = args.J2
seed = int(args.seed)

num_layers      = 1     # number of Tranformer layers
d_model         = 2    # dimensionality of the embedding space
n_heads         = 1     # number of heads
patch_size      = 2     # lenght of the input sequence
lr              = 0.01

N_samples       = 1024
N_opt           = 15
save_every       = 5
block_iter = N_opt//save_every


model_name = f"layers{num_layers}_d{d_model}_heads{n_heads}_patch{patch_size}_sample{N_samples}_lr{lr}_iter{N_opt}_symm{symm}"
seed_str = f"seed_{seed}"
lattice_name = f"J={J2}"
folder = f'ViT_Heisenberg/plot/{model_name}/{seed_str}/{lattice_name}'
save_model = f"/scratch/f/F.Conoscenti/Thesis_QSL/ViT_Heisenberg/plot/{model_name}/{seed_str}/{lattice_name}/models"
os.makedirs(save_model, exist_ok=True)
os.makedirs(folder, exist_ok=True)  #create folder for the plots and the output file
os.makedirs(folder+"/physical_obs", exist_ok=True)
os.makedirs(folder+"/Sign_plot", exist_ok=True)
sys.stdout = open(f"{folder}/output.txt", "w") #redirect print output to a file inside the folder

print(f"ViT, J={J2}, L={L}, layers{num_layers}_d{d_model}_heads{n_heads}_patch{patch_size}_sample{N_samples}_lr{lr}_iter{N_opt}")

# Hilbert space of spins on the graph
lattice = nk.graph.Hypercube(length=L, n_dim=n_dim, pbc=True, max_neighbor_order=2)
hilbert = nk.hilbert.Spin(s=1 / 2, N=lattice.n_nodes, total_sz=0)

# Heisenberg J1-J2 spin hamiltonian
hamiltonian = nk.operator.Heisenberg(
    hilbert=hilbert, graph=lattice, J=[1.0, J2], sign_rule=[False, False]
).to_jax_operator()  # No Marshall sign rule

"""
#Ising Hamiltonian
hamiltonian = nk.operator.LocalOperator(hilbert)
for u, v in lattice.edges():
    hamiltonian += sigmaz(hilbert, u) * sigmaz(hilbert, v) 
"""

# Intiialize the ViT variational wave function
vit_module = ViT_sym(
    num_layers=num_layers, 
    d_model=d_model, 
    n_heads=n_heads, 
    patch_size=patch_size, 
    transl_invariant=True, 
    parity=symm
)

key = jax.random.key(seed)
key, subkey = jax.random.split(key)
spin_configs = jax.random.randint(subkey, shape=(M, L * L), minval=0, maxval=1) * 2 - 1
params = vit_module.init(subkey, spin_configs)

# Metropolis Local Sampling
sampler = nk.sampler.MetropolisExchange(
    hilbert=hilbert,
    graph=lattice,
    d_max=2,
    n_chains=N_samples,
    sweep_size=lattice.n_nodes,
)

optimizer = nk.optimizer.Sgd(learning_rate=lr)

key, subkey = jax.random.split(key, 2)
vstate = nk.vqs.MCState(
    sampler=sampler,
    model=vit_module,
    sampler_seed=subkey,
    n_samples=N_samples,
    n_discard_per_chain=16,
    variables=params,
    chunk_size=512,
)

N_params = nk.jax.tree_size(vstate.parameters)
print("Number of parameters = ", N_params, flush=True)


# Variational monte carlo driver
from netket.experimental.driver import VMC_SR

vmc = VMC_SR(
    hamiltonian=hamiltonian,
    optimizer=optimizer,
    diag_shift=1e-4,
    variational_state=vstate,
    mode="complex",
) 

# Optimization
log = nk.logging.RuntimeLog()


for i in range(block_iter):
    vmc.run(n_iter=save_every, out=log)
    #Save
    with open(save_model +"/model_"+ f"{i} "+ ".mpack", "wb") as f:
        bytes_out = flax.serialization.to_bytes(vstate.variables)
        f.write(bytes_out)


#%%
#Energy
E_vs = Energy(log, L, folder)

#Correlation function
vstate.n_samples = 1024
Corr_Struct(lattice, vstate, L, folder, hilbert)

#change to lanczos for Heisenberg
E_exact, ket_gs = Exact_gs(L, J2, hamiltonian, J1J2=True, spin=True)

#comment for ising
fidelity = Fidelity(vstate, ket_gs)
print(f"Fidelity <vstate|exact> = {fidelity}")

Relative_Error(E_vs, E_exact, L)

Magnetization(vstate, lattice, hilbert)

variance = Variance(log)

Vscore(L, variance, E_vs)

count_params = vit_param_count(n_heads, num_layers, patch_size, d_model, L*L)
print(f"params={count_params}")

#Marshall_sign(marshall_op, vstate, folder, n_samples = 64 )
n_sample = 4096
marshall_op = MarshallSignOperator(hilbert)
#plot_Sign_full_MCMC(marshall_op, vstate, folder, n_sample)
plot_Sign_Fidelity(ket_gs, vstate, folder, hilbert)
#sign_vstate_MCMC, sign_vstate_full = plot_Sign_full_MCMC(marshall_op, vstate, str(folder), 64, hi)
sign_vstate_full, sign_exact, fidelity = plot_Sign_Fidelity(ket_gs, vstate, folder, hilbert)
configs, sign_vstate_config, weight_exact, weight_vstate = plot_Sign_single_config(ket_gs, vstate, folder, hilbert, 4, L)
configs, sign_vstate_config, weight_exact, weight_vstate = plot_Weight_single(ket_gs, vstate, folder, hilbert, L)
error = plot_MSE_configs(ket_gs, vstate, folder, hilbert)
error, fidelity, sign_vstate, sign_exact = plot_Sign_Err_Amplitude_Err_Fidelity(ket_gs, vstate, folder, hilbert)


variables = {
        #'sign_vstate_MCMC': sign_vstate_MCMC,
        'sign_vstate_full': sign_vstate_full,
        'sign_exact': sign_exact,
        'fidelity': fidelity,
        'configs': configs,
        'sign_vstate_config': sign_vstate_config,
        'weight_exact': weight_exact,
        'weight_vstate': weight_vstate,
        'error': error
    }

with open(folder+"/variables", 'wb') as f:
    pickle.dump(variables, f)                   


sys.stdout.close()