import sys
import argparse
import jax
from jax import numpy as jnp
import netket as nk
import os
import flax
import helper
import logging
import pickle
os.environ["JAX_PLATFORM_NAME"] = "gpu"

print("Total devices:", jax.device_count())
print("Local devices:", jax.local_device_count())
print("Devices:", jax.devices())

import pickle
sys.path.append(os.path.dirname(os.path.dirname("/scratch/f/F.Conoscenti/Thesis_QSL")))

from HFDS_Heisenberg.HFDS_model_spin import HiddenFermion

from Elaborate.Statistics.Energy import *
from Elaborate.Statistics.Corr_Struct import *
from Elaborate.Statistics.Error_Stat import *
from Elaborate.Statistics.count_params import *
from Elaborate.Plotting.Old.Sign_vs_iteration import *
from Elaborate.Sign_Obs import *
from Elaborate.Plotting.QGT.QGT_vs_iteration import *

from DMRG.DMRG_NQS_Imp_sampl import Observable_Importance_sampling

from Observables import run_observables

parser = argparse.ArgumentParser(description="Example script with parameters")
parser.add_argument("--J2", type=float, default=0.5, help="Coupling parameter J2")
parser.add_argument("--seed", type=float, default=1, help="seed")
args = parser.parse_args()

spin = True

#Physical param
L       = 6
n_elecs = L*L # L*L should be half filling
N_sites = L*L
N_up    = (n_elecs+1)//2
N_dn    = n_elecs//2
n_dim = 2

J1J2 = True
J2 = args.J2
seed = int(args.seed)

dtype   = "complex"
MFinitialization = "Fermi" #G_MF#random #Fermi
determinant_type = "hidden"
bounds  = "PBC"
parity = False
rotation = False


#Varaitional state param
# 1k params for L=4 n_hid=1 features=16 layers=1
# 3.9k params for L=4 n_hid=2 features=64 layers=1
# 6k params for L=4 n_hid=4 features=64 layers=1
# 13k params for L=4 n_hid=6 features=128 layers=1

# 6x6
# 3.8k params for L=6 n_hid=1 features=16 layers=1
# 6k params for L=6 n_hid=2 features=32 layers=1
# 15k params for L=6 n_hid=4 features=64 layers=1
# 40k params for L=6 n_hid=6 features=128 layers=1
# 53k params for L=6 n_hid=8 features=128 layers=1
# 8x8
# 40k params for L=6 n_hid=6 features=64 layers=1
# 50k params for L=6 n_hid=8 features=64 layers=1
# 91k params for L=6 n_hid=8 features=128 layers=1
#10x10
# 68k params for L=6 n_hid=8 features=64 layers=1
# 84k params for L=6 n_hid=8 features=64 layers=1
# 145k params for L=6 n_hid=8 features=128 layers=1
#12x12
# 110k params for L=6 n_hid=8 features=64 layers=1
# 132k params for L=6 n_hid=8 features=64 layers=1

n_hid_ferm       = 1
features         = 1    #hidden units per layer
hid_layers       = 2

#Network param
lr               = 0.02
n_samples        = 128
chunk_size       = 128
N_opt            = 5

number_data_points = 2
save_every       = N_opt//number_data_points
block_iter       = N_opt//save_every

n_chains         = n_samples//4

model_name = f"layers{hid_layers}_hidd{n_hid_ferm}_feat{features}_sample{n_samples}_lr{lr}_iter{N_opt}_parity{parity}_rot{rotation}_Init{MFinitialization}_type{dtype}"
seed_str = f"seed_{seed}"
J_value = f"J={J2}"
if J1J2==True:
   model_path = f'HFDS_Heisenberg/plot/{L}x{L}/{model_name}/{J_value}'
   folder = f'{model_path}/{seed_str}'
   save_model = f"{model_path}/{seed_str}/models"
else:
    folder = f'HFDS_Heisenberg/plot/Ising/spin/{model_name}/{J_value}'
    save_model = f"HFDS_Heisenberg/plot/Ising/spin/{model_name}/{J_value}/models"

os.makedirs(save_model, exist_ok=True)
os.makedirs(folder, exist_ok=True)  #create folder for the plots and the output file
os.makedirs(folder+"/physical_obs", exist_ok=True)
os.makedirs(folder+"/Sign_plot", exist_ok=True)
os.makedirs(model_path+"/plot_avg", exist_ok=True)

sys.stdout = open(f"{folder}/output.txt", "w") #redirect print output to a file inside the folder
print(f"HFDS_spin, J={J2}, L={L}, layers{hid_layers}_hidd{n_hid_ferm}_feat{features}_sample{n_samples}_lr{lr}_iter{N_opt}_try")

# ------------- define lattice and hilbert space ------------------------
boundary_conditions = 'pbc' 
lattice = nk.graph.Hypercube(length=L, n_dim=n_dim, pbc=True, max_neighbor_order=2)
hi = nk.hilbert.Spin(s=1 / 2, N=lattice.n_nodes, total_sz=0) 
print(f"hilbert space size = ",hi.size)

# ------------- define symmetries ------------------------
"""
translation_group_representation = nk.symmetry.canonical_representation(
    hilbert=hi,
    group=lattice.translation_group())
"""
# ------------- define Hamiltonian ------------------------
ha = nk.operator.Heisenberg(hilbert=hi, graph=lattice, J=[1.0, J2], sign_rule=[False, False]).to_jax_operator()  # No Marshall sign rule"""


if dtype=="real": dtype_ = jnp.float64
else: dtype_ = jnp.complex128


model = HiddenFermion(
                     L=L,
                   network="FFNN",
                   n_hid=n_hid_ferm,
                   layers=hid_layers,
                   features=features,
                   MFinit=MFinitialization,
                   hilbert=hi,
                   stop_grad_mf=False,
                   stop_grad_lower_block=False,
                   bounds=bounds,
                   parity=parity,
                   rotation=rotation,
                   dtype=dtype_
                  )

                
# ---------- define sampler ------------------------
sampler = nk.sampler.MetropolisExchange(
    hilbert=hi,
    graph=lattice,
    d_max=2,
    n_chains=n_chains,
    sweep_size=lattice.n_nodes,
)

key = jax.random.key(seed)
key, pkey, skey = jax.random.split(key, 3)
vstate = nk.vqs.MCState(
    sampler, 
    model, 
    n_samples=n_samples, 
    seed=pkey,
    chunk_size=chunk_size,
    n_discard_per_chain=128) #defines the variational state object

"""
vstate_sym = translation_group_representation.project(
    state=vstate, 
    character_index=0)
"""

total_params = sum(p.size for p in jax.tree_util.tree_leaves(vstate.parameters))
print(f'Total number of parameters: {total_params}')

optimizer = nk.optimizer.Sgd(learning_rate=lr)

from netket.experimental.driver.vmc_srt import VMC_SRt

vmc = VMC_SRt(
    hamiltonian=ha,
    optimizer=optimizer,
    diag_shift=1e-4,
    variational_state=vstate,
    jacobian_mode="complex",
)

log = nk.logging.RuntimeLog()

"""sr = nk.optimizer.SR(
    diag_shift=1e-4,     
    holomorphic=True
)

vmc = nk.driver.VMC(
    hamiltonian=ha,
    optimizer=optimizer,
    variational_state=vstate,
    preconditioner=sr
) """


# Load existing log if available to append to it
log_path = os.path.join(folder, "log.pkl")
old_log_data = helper.load_log(folder)

start_block, vstate = helper.load_checkpoint(save_model, block_iter, save_every, vstate)

for i in range(start_block, block_iter):
    #Save model
    with open(save_model +f"/model_{i}.mpack", 'wb') as file:
        file.write(flax.serialization.to_bytes(vstate.variables))

    vmc.run(n_iter=save_every, out=log)
    
    # Save log incrementally
    current_log_data = helper.merge_log_data(old_log_data, log.data)
    with open(log_path, 'wb') as f:
        pickle.dump(current_log_data, f)


# Save the final model state after the last optimization step
with open(save_model +f"/model_{block_iter}.mpack", "wb") as f:
    bytes_out = flax.serialization.to_bytes(vstate.variables)
    f.write(bytes_out)

final_log_data = helper.merge_log_data(old_log_data, log.data)

#################################################################################################################

print("Running observables computation...")
run_observables(helper.MockLog(log.data), folder)


sys.stdout.close()