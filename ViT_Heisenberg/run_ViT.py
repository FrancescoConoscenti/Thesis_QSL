
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

from Elaborate.Statistics.Energy import *
from Elaborate.Statistics.Corr_Struct import *
from Elaborate.Statistics.Error_Stat import *
from Elaborate.Statistics.count_params import *
from Elaborate.Plotting.Sign_vs_iteration import *
from Elaborate.Plotting.Sign_vs_iteration import *
from Elaborate.Plotting.S_matrix_vs_iteration import *

from DMRG.DMRG_NQS_Imp_sampl import Observable_Importance_sampling

from Observables import run_observables

 
parser = argparse.ArgumentParser(description="Example script with parameters")
parser.add_argument("--J2", type=float, default=0.0, help="Coupling parameter J2")
parser.add_argument("--seed", type=float, default=1, help="seed")
args = parser.parse_args()

M = 10  # Number of spin configurations to initialize the parameters
L = 6  # Linear size of the lattice


n_dim = 2
J2 = args.J2
seed = int(args.seed)

# 1k params for L=4 num_layers=2 d_model=8 n_heads=4 patch_size=2
# 3.4k params for L=4 num_layers=1 d_model=16 n_heads=4 patch_size=2
# 6k params for L=4 num_layers=2 d_model=16 n_heads=4 patch_size=2
# 13.6k params for L=4 num_layers=2 d_model=24 n_heads=6 patch_size=2

# 1k params for L=6 num_layers=2 d_model=8 n_heads=4 patch_size=2
# 3k params for L=6 num_layers=1 d_model=16 n_heads=4 patch_size=2
# 6k params for L=6 num_layers=2 d_model=16 n_heads=4 patch_size=2
# 15k params for L=6 num_layers=2 d_model=24 n_heads=6 patch_size=2
# 43k params for L=6 num_layers=3 d_model=36 n_heads=6 patch_size=2
# 36k params for L=6 num_layers=2 d_model=40 n_heads=8 patch_size=2
# 53k params for L=6 num_layers=3 d_model=40 n_heads=8 patch_size=2

num_layers      = 2     # number of Tranformer layers
d_model         = 24    # dimensionality of the embedding space
n_heads         = 4     # number of heads
patch_size      = 2     # lenght of the input sequence
lr              = 0.0075
parity = True
rotation = True

N_samples       = 1024
N_opt           = 500

number_data_points = 20
save_every       = N_opt//number_data_points
block_iter = N_opt//save_every

model_name = f"layers{num_layers}_d{d_model}_heads{n_heads}_patch{patch_size}_sample{N_samples}_lr{lr}_iter{N_opt}_parity{parity}_rot{rotation}_latest_model"
seed_str = f"seed_{seed}"
J_value = f"J={J2}"
model_path = f'ViT_Heisenberg/plot/{L}x{L}/{model_name}/{J_value}'
folder = f'{model_path}/{seed_str}'
folder_energy = f'{folder}/Energy_plot'
save_model = f"{model_path}/{seed_str}/models"

os.makedirs(save_model, exist_ok=True)
os.makedirs(folder, exist_ok=True)  #create folder for the plots and the output file
os.makedirs(folder+"/physical_obs", exist_ok=True)
os.makedirs(folder+"/Sign_plot", exist_ok=True)
os.makedirs(model_path+"/plot_avg", exist_ok=True)
os.makedirs(folder_energy, exist_ok=True)

sys.stdout = open(f"{folder}/output.txt", "w") #redirect print output to a file inside the folder
print(f"ViT, J={J2}, L={L}, layers{num_layers}_d{d_model}_heads{n_heads}_patch{patch_size}_sample{N_samples}_lr{lr}_iter{N_opt}")

# Hilbert space of spins on the graph
lattice = nk.graph.Hypercube(length=L, n_dim=n_dim, pbc=True, max_neighbor_order=2)
hilbert = nk.hilbert.Spin(s=1 / 2, N=lattice.n_nodes, total_sz=0)

# Heisenberg J1-J2 spin hamiltonian
hamiltonian = nk.operator.Heisenberg(
    hilbert=hilbert, graph=lattice, J=[1.0, J2], sign_rule=[False, False]
).to_jax_operator()  # No Marshall sign rule


# Intiialize the ViT variational wave function
vit_module = ViT_sym(
    L=L,
    num_layers=num_layers, 
    d_model=d_model, 
    n_heads=n_heads, 
    patch_size=patch_size, 
    transl_invariant=True, 
    parity=parity, 
    rotation=rotation
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
    #Save model
    with open(save_model +f"/model_{i}.mpack", "wb") as f:
        bytes_out = flax.serialization.to_bytes(vstate.variables)
        f.write(bytes_out)

    vmc.run(n_iter=save_every, out=log)

# Save the final model state after the last optimization step
with open(save_model +f"/model_{block_iter}.mpack", "wb") as f:
    bytes_out = flax.serialization.to_bytes(vstate.variables)
    f.write(bytes_out)

#####################################################################################################

run_observables(log, folder)
    
"""
#Correlation function
vstate.n_samples = 1024
Corr_Struct(lattice, vstate, L, folder, hilbert)
if L == 4:
    #Exact
    E_exact, ket_gs = Exact_gs(L, J2, hamiltonian, J1J2=True, spin=True)
elif L==6:
    E_exact = Exact_gs_en_6x6(J2)

E_vs = Energy(log, L, folder_energy, E_exact=E_exact)
#Rel Err
Relative_Error(E_vs, E_exact, L)
#Magn
Magnetization(vstate, lattice, hilbert)
#Variance
variance = Variance(log, folder_energy)
#Vscore
Vscore(L, variance, E_vs)
#count Params
count_params = vit_param_count(n_heads, num_layers, patch_size, d_model, L*L)
print(f"params={count_params}")

if L == 4:
    #Fidelity
    fidelity = Fidelity(vstate, ket_gs)
    print(f"Fidelity <vstate|exact> = {fidelity}")

    configs, sign_vstate_config, weight_exact, weight_vstate = plot_Sign_single_config(ket_gs, vstate, hilbert, 3, L, folder, one_avg = "one")
    configs, sign_vstate_config, weight_exact, weight_vstate = plot_Weight_single(ket_gs, vstate, hilbert, 8, L, folder, one_avg = "one")
    amp_overlap, fidelity, sign_vstate, sign_exact, sign_overlap = plot_Sign_Err_Amplitude_Err_Fidelity(ket_gs, vstate, hilbert, folder, one_avg = "one")
    amp_overlap, sign_vstate, sign_exact, sign_overlap = plot_Sign_Err_vs_Amplitude_Err_with_iteration(ket_gs, vstate, hilbert, folder, one_avg = "one")
    sorted_weights, sorted_amp_overlap, sorted_sign_overlap = plot_Overlap_vs_Weight(ket_gs, vstate, hilbert, folder, "one")
    S_matrices, eigenvalues = plot_S_matrix_eigenvalues(vstate, folder, hilbert,  part_training = "end", one_avg = "one")

    variables = {
            #'sign_vstate_MCMC': sign_vstate_MCMC,
            'sign_vstate': sign_vstate,
            'sign_exact': sign_exact,
            'fidelity': fidelity,
            'configs': configs,
            'sign_vstate_config': sign_vstate_config,
            'weight_exact': weight_exact,
            'weight_vstate': weight_vstate,
            'amp_overlap': amp_overlap,
            'sign_overlap': sign_overlap,
            #'eigenvalues': eigenvalues
        }

    with open(folder+"/variables", 'wb') as f:
        pickle.dump(variables, f)                   

elif L==6:
    print("6x6")
    Observable_Importance_sampling(J2, NQS_path=None, vstate=vstate)

    variables = {
            
        }

    with open(folder+"/variables", 'wb') as f:
        pickle.dump(variables, f)   

"""
sys.stdout.close()