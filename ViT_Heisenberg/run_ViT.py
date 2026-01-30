
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
import helper

print("Total devices:", jax.device_count())
print("Local devices:", jax.local_device_count())
print("Devices:", jax.devices())

from ViT_Heisenberg.ViT_model import ViT_sym
from Elaborate.Statistics.Energy import Energy, plot_energy

from Observables import run_observables

 
parser = argparse.ArgumentParser(description="Example script with parameters")
parser.add_argument("--J2", type=float, default=0.5, help="Coupling parameter J2")
parser.add_argument("--seed", type=float, default=1, help="seed")
args = parser.parse_args()

M = 10  # Number of spin configurations to initialize the parameters
L = 8  # Linear size of the lattice


n_dim = 2
J2 = args.J2
seed = int(args.seed)
#4x4
# 1k params for L=4 num_layers=2 d_model=8 n_heads=4 patch_size=2
# 3.4k params for L=4 num_layers=1 d_model=16 n_heads=4 patch_size=2
# 6k params for L=4 num_layers=2 d_model=16 n_heads=4 patch_size=2
# 13.6k params for L=4 num_layers=2 d_model=24 n_heads=6 patch_size=2
#6x6
# 1k params for L=6 num_layers=2 d_model=8 n_heads=4 patch_size=2
# 3k params for L=6 num_layers=1 d_model=16 n_heads=4 patch_size=2
# 6k params for L=6 num_layers=2 d_model=16 n_heads=4 patch_size=2
# 15k params for L=6 num_layers=2 d_model=24 n_heads=6 patch_size=2
# 43k params for L=6 num_layers=3 d_model=36 n_heads=6 patch_size=2
# 36k params for L=6 num_layers=2 d_model=40 n_heads=8 patch_size=2
# 53k params for L=6 num_layers=3 d_model=40 n_heads=8 patch_size=2
#8x8
# 53k params for L=6 num_layers=3 d_model=40 n_heads=8 patch_size=2

num_layers      = 3     # number of Tranformer layers
d_model         = 40   # dimensionality of the embedding space
n_heads         = 8     # number of heads
patch_size      = 2     # lenght of the input sequence
lr              = 0.0075
parity = True
rotation = True

N_samples       = 1024  # number of MC samples
n_chains        = N_samples
chunk_size      = 1024
N_opt           = 4000

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
    n_chains=n_chains,
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
    chunk_size=chunk_size,
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

# Load existing log if available to append to it
log_path = os.path.join(folder, "log.pkl")
old_log_data = helper.load_log(folder)

start_block, vstate = helper.load_checkpoint(save_model, block_iter, save_every, vstate)

for i in range(start_block, block_iter):
    #Save model
    with open(save_model +f"/model_{i}.mpack", 'wb') as file:
        file.write(flax.serialization.to_bytes(vstate))

    vmc.run(n_iter=save_every, out=log)
    
    # Save log incrementally
    current_log_data = helper.merge_log_data(old_log_data, log.data)
    with open(log_path, 'wb') as f:
        pickle.dump(current_log_data, f)

# Save the final model state after the last optimization step
with open(save_model +f"/model_{block_iter}.mpack", "wb") as f:
    bytes_out = flax.serialization.to_bytes(vstate)
    f.write(bytes_out)

final_log_data = helper.merge_log_data(old_log_data, log.data)

#####################################################################################################

with open(log_path, 'wb') as f:
    pickle.dump(final_log_data, f)


run_observables(helper.MockLog(log.data), folder)
    
sys.stdout.close()