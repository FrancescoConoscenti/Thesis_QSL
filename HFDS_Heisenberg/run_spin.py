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
import re
os.environ["JAX_PLATFORM_NAME"] = "gpu"

print("Total devices:", jax.device_count())
print("Local devices:", jax.local_device_count())
print("Devices:", jax.devices())

import pickle
sys.path.append(os.path.dirname(os.path.dirname("/scratch/f/F.Conoscenti/Thesis_QSL")))

from netket.driver import VMC_SR

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

from Hamiltonian import build_heisenberg_apbc

parser = argparse.ArgumentParser(description="Example script with parameters")
parser.add_argument("--J2", type=float, default=0.5, help="Coupling parameter J2")
parser.add_argument("--seed", type=float, default=1, help="seed")
parser.add_argument("--L", type=int, default=4, help="Linear size of the lattice")
args = parser.parse_args()

spin = True


#Physical param
L       = args.L
N_sites = L * L

n_elecs = N_sites # L*L should be half filling
N_up    = (n_elecs+1)//2
N_dn    = n_elecs//2
n_dim = 2

J1J2 = True
J2 = args.J2
seed = int(args.seed)

dtype   = "complex"
MFinitialization = "Fermi" #G_MF#random #Fermi
determinant_type = "hidden"

parity = True
rotation = True


n_hid_ferm       = 4
features         = 32   #hidden units per layer
hid_layers       = 1

#Network param
lr               = 0.01
n_samples        = 4096 #total number of samples
#n_samples = 4096  n_chains  = 128  chunk_size = 4096
#n_samples = 8192  n_chains  = 256  chunk_size = 2048  
n_chains         = n_samples//16  #number of parallel Markov chains
chunk_size       = n_samples//16  #samples are divided in chunks to compute observables in parallel
N_iter           = 2000 #N_opt on the top of the one of the loaded model, if any

#---------------------------Load another model -----------------------------------------
#load_path = "/cluster/home/fconoscenti/Thesis_QSL/HFDS_Heisenberg/plot/10x10/layers1_hidd8_feat32_sample4096_bcPBC_PBC_lr0.02_iter4000_parityTrue_rotTrue_InitFermi_typecomplex"
load_path = None #set to None to not load any model and start from scratch
previous_iter = 0

N_opt = N_iter
save_every = N_opt // 10
block_iter = N_opt // save_every

#-------------------------------- Set up model saving -----------------------------------------
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

sys.stdout = open(f"{folder}/output.txt", "w") #redirect print output to a file inside the folder
print(f"HFDS_spin, J={J2}, L={L}, layers{hid_layers}_hidd{n_hid_ferm}_feat{features}_sample{n_samples}_lr{lr}_iter{N_opt}_try")


# ------------- define Hilbert space ------------------------
hi = nk.hilbert.Spin(s=1 / 2, N=L**2, total_sz=0)
lattice = nk.graph.Hypercube(length=L, n_dim=n_dim, pbc=[True, True], max_neighbor_order=2)
print(f"hilbert space size = ",hi.size)


# ------------- define Hamiltonian ------------------------
ha = nk.operator.Heisenberg(hilbert=hi, graph=lattice, J=[1.0, J2], sign_rule=[False, False]).to_jax_operator()  # No Marshall sign rule"""

# ------------- define model ------------------------
if dtype=="real": dtype_ = jnp.float64
else: dtype_ = jnp.complex128


model = HiddenFermion(
                     L=L,
                   N_sites=N_sites,
                   network="FFNN",
                   n_hid=n_hid_ferm,
                   layers=hid_layers,
                   features=features,
                   MFinit=MFinitialization,
                   hilbert=hi,
                   stop_grad_mf=False,
                   stop_grad_lower_block=False,
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

total_params = sum(p.size for p in jax.tree_util.tree_leaves(vstate.parameters))
print(f'Total number of parameters: {total_params}')

log = nk.logging.RuntimeLog()
log_path = os.path.join(folder, "log.pkl")

start_block, vstate = helper.load_checkpoint(save_model, block_iter, save_every, vstate)

if start_block == 0 and load_path:
    print(f"Attempting to load starting model from {load_path}")
    load_J_path = os.path.join(load_path, f"J={J2}")
    if not os.path.exists(load_J_path):
        load_J_path = os.path.join(load_path, f"J2={J2}")

    load_seed_path = os.path.join(load_J_path, f"seed_{seed}")
    load_save_model = os.path.join(load_seed_path, "models")
    
    if os.path.exists(load_save_model):
        files = [f for f in os.listdir(load_save_model) if f.endswith(".mpack")]
        if files:
            files.sort(key=lambda x: int(re.search(r"model_(\d+)", x).group(1)))
            last_model = files[-1]
            last_model_path = os.path.join(load_save_model, last_model)
            print(f"Loading starting model from {last_model_path}")
            with open(last_model_path, 'rb') as f:
                try:
                    vstate = flax.serialization.from_bytes(vstate, f.read())
                except Exception:
                    f.seek(0)
                    vstate.variables = flax.serialization.from_bytes(vstate.variables, f.read())
            
            start_block = previous_iter // save_every
            
            old_log_path_load = os.path.join(load_seed_path, "log.pkl")
        else:
            print("No .mpack files found in the load path.")
    else:
        print(f"Models directory not found in the load path: {load_save_model}")

# Initialize VMC after loading the state so it uses the loaded vstate
optimizer = nk.optimizer.Sgd(learning_rate=lr)


vmc = VMC_SR(
    hamiltonian=ha,
    optimizer=optimizer,
    diag_shift=1e-6,
    variational_state=vstate,
    use_ntk=True,
    momentum=0.9
) 
vmc._step_count = start_block * save_every

for i in range(start_block, block_iter):
    #Save model
    with open(save_model +f"/model_{i}.mpack", 'wb') as file:
        file.write(flax.serialization.to_bytes(vstate))

    vmc.run(n_iter=save_every, out=log)
    
    # Save log incrementally
    current_log_data =  log.data
    with open(log_path, 'wb') as f:
        pickle.dump(current_log_data, f)


# Save the final model state after the last optimization step
with open(save_model +f"/model_{block_iter}.mpack", "wb") as f:
    bytes_out = flax.serialization.to_bytes(vstate)
    f.write(bytes_out)

final_log_data = log.data

#################################################################################################################

print("Running observables computation...")
if final_log_data and "Energy" in final_log_data:
    run_observables(helper.MockLog(final_log_data), folder)
else:
    run_observables(None, folder)


sys.stdout.close()