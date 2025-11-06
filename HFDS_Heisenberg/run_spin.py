try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    import jax
    jax.distributed.initialize()

    print(f"Rank={rank}: Total number of GPUs: {jax.device_count()}, devices: {jax.devices()}")
    print(f"Rank={rank}: Local number of GPUs: {jax.local_device_count()}, devices: {jax.local_devices()}", flush=True)

    # wait for all processes to show their devices
    comm.Barrier()
except:
  pass


import sys
import argparse
import jax
from jax import numpy as jnp
import netket as nk
import os
import flax
os.environ['JAX_TRACEBACK_FILTERING'] = 'off'
import matplotlib.pyplot as plt
import numpy as np 
from scipy.sparse.linalg import eigsh
import pickle
sys.path.append(os.path.dirname(os.path.dirname("/scratch/f/F.Conoscenti/Thesis_QSL")))

from netket.operator.spin import sigmax, sigmaz, sigmay
from netket.experimental.driver import VMC_SR
import netket.operator as op
from HFDS_Heisenberg.HFDS_model_spin import Orbitals
from HFDS_Heisenberg.HFDS_model_spin import HiddenFermion

from Elaborate.Statistics.Energy import *
from Elaborate.Statistics.Corr_Struct import *
from Elaborate.Statistics.Error_Stat import *
from Elaborate.Statistics.count_params import *
from Elaborate.Plotting.Sign_vs_iteration import *
from Elaborate.Sign_Obs import *
from Elaborate.Plotting.S_matrix_vs_iteration import *

parser = argparse.ArgumentParser(description="Example script with parameters")
parser.add_argument("--J2", type=float, default=0.7, help="Coupling parameter J2")
parser.add_argument("--seed", type=float, default=1, help="seed")
args = parser.parse_args()

spin = True

#Physical param
L       = 4
n_elecs = L*L # L*L should be half filling
N_sites = L*L
N_up    = (n_elecs+1)//2
N_dn    = n_elecs//2
n_dim = 2

J1J2 = True
J2 = args.J2
seed = int(args.seed)

dtype   = "real"
MFinitialization = "Fermi" #Hartree #random #Fermi #striped
determinant_type = "hidden"
bounds  = "PBC"
parity = True
rotation = True
translation = False

#Varaitional state param
n_hid_ferm       = 4
features         = 32 #hidden units per layer
hid_layers       = 1

#Network param
lr               = 0.025
n_samples        = 1024
N_opt            = 100

number_data_points = 20
save_every       = N_opt//number_data_points
block_iter       = N_opt//save_every

n_chains         = n_samples//2
chunk_size       =  n_samples#//4   #chunk size for the sampling


model_name = f"layers{hid_layers}_hidd{n_hid_ferm}_feat{features}_sample{n_samples}_lr{lr}_iter{N_opt}_parity{parity}_rot{rotation}_trans{translation}_Init{MFinitialization}_type{dtype}"
seed_str = f"seed_{seed}"
J_value = f"J={J2}"
if J1J2==True:
   model_path = f'HFDS_Heisenberg/plot/spin_new/{model_name}/{J_value}'
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
print(f"HFDS_spin, J={J2}, L={L}, layers{hid_layers}_hidd{n_hid_ferm}_feat{features}_sample{n_samples}_lr{lr}_iter{N_opt}")

# Hilbert space of spins on the graph
boundary_conditions = 'pbc' 
lattice = nk.graph.Hypercube(length=L, n_dim=n_dim, pbc=True, max_neighbor_order=2)
hi = nk.hilbert.Spin(s=1 / 2, N=lattice.n_nodes, total_sz=0) 
print(f"hilbert space size = ",hi.size)


if dtype=="real": dtype_ = jnp.float64
else: dtype_ = jnp.complex128

model = HiddenFermion(n_elecs=n_elecs,
                   network="FFNN",
                   n_hid=n_hid_ferm,
                   Lx=L,
                   Ly=L,
                   layers=hid_layers,
                   features=features,
                   MFinit=MFinitialization,
                   hilbert=hi,
                   stop_grad_mf=False,
                   stop_grad_lower_block=False,
                   bounds=bounds,
                   parity=parity,
                   rotation=rotation,
                   translation=translation,
                   dtype=dtype_)


# ------------- define Hamiltonian ------------------------

if J1J2==True:
    # Heisenberg J1-J2 spin ha
    ha = nk.operator.Heisenberg(hilbert=hi, graph=lattice, J=[1.0, J2], sign_rule=[False, False]).to_jax_operator()  # No Marshall sign rule"""
else:
    #ising hamiltonian
    ha = nk.operator.Ising(hilbert=hi, graph=lattice, h = J2, J=1.0, dtype=jnp.float64).to_jax_operator()
    """ha = op.LocalOperator(hi)
    for i, j in lattice.edges():
        ha += -1 * op.spin.sigmaz(hi, i) * op.spin.sigmaz(hi, j)
    for i in range(L*L):
        ha += -h * op.spin.sigmax(hi, i)"""


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
    chunk_size=chunk_size, n_discard_per_chain=128) #defines the variational state object

total_params = sum(p.size for p in jax.tree_util.tree_leaves(vstate.parameters))
print(f'Total number of parameters: {total_params}')

optimizer = nk.optimizer.Sgd(learning_rate=lr)

vmc = VMC_SR(
    hamiltonian=ha,
    optimizer=optimizer,
    diag_shift=1e-4,
    variational_state=vstate,
    mode = 'real'
) 

log = nk.logging.RuntimeLog()


for i in range(block_iter):
     #Save
    with open(save_model +"/model_"+ f"{i}"+".mpack", "wb") as f:
        bytes_out = flax.serialization.to_bytes(vstate.variables)
        f.write(bytes_out)

    vmc.run(n_iter=save_every, out=log)
    
with open(save_model + f"/model_{block_iter}.mpack", "wb") as f:
    f.write(flax.serialization.to_bytes(vstate.variables))


#Energy
E_vs = Energy(log, L, folder)
#Correlation function
vstate.n_samples = 1024
Corr_Struct(lattice, vstate, L, folder, hi)
#exact diagonalization
E_exact, ket_gs = Exact_gs(L, J2, ha, J1J2, spin)
#Fidelity
if J1J2 == True:
    fidelity = Fidelity(vstate, ket_gs)
    print(f"Fidelity <vstate|exact> = {fidelity}")
#Rel Error
Relative_Error(E_vs, E_exact, L)
#magnetization
Magnetization(vstate, lattice, hi)
#Variance
variance = Variance(log)
#Vscore
Vscore(L, variance, E_vs)
#count number of parameters in the model
hidden_fermion_param_count(n_elecs, n_hid_ferm, L, L, hid_layers, features)

#Marshall_sign(marshall_op, vstate, folder, n_samples = 64 )
#n_sample = 4096
#marshall_op = MarshallSignOperator(hilbert)
#sign_vstate_MCMC, sign_vstate_full = plot_Sign_full_MCMC(marshall_op, vstate, str(folder), 64, hi)
#sign_vstate_full, sign_exact, fidelity = plot_Sign_Fidelity(ket_gs, vstate, hi,  folder, one_avg = "one")
#amp_overlap = plot_Amp_overlap_configs(ket_gs, vstate, hi, folder, one_avg = "one")

configs, sign_vstate_config, weight_exact, weight_vstate = plot_Sign_single_config(ket_gs, vstate, hi, 3, L, folder, one_avg = "one")
configs, sign_vstate_config, weight_exact, weight_vstate = plot_Weight_single(ket_gs, vstate, hi, 8, L, folder, one_avg = "one")
amp_overlap, fidelity, sign_vstate, sign_exact, sign_overlap = plot_Sign_Err_Amplitude_Err_Fidelity(ket_gs, vstate, hi, folder, one_avg = "one")
amp_overlap, sign_vstate, sign_exact, sign_overlap = plot_Sign_Err_vs_Amplitude_Err_with_iteration(ket_gs, vstate, hi, folder, one_avg = "one")
sorted_weights, sorted_amp_overlap, sorted_sign_overlap = plot_Overlap_vs_Weight(ket_gs, vstate, hi, folder, "one")

variables = {
        #'sign_vstate_MCMC': sign_vstate_MCMC,
        #'sign_vstate_full': sign_vstate_full,
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

vstate.n_samples = 256
S_matrices, eigenvalues = plot_S_matrix_eigenvalues(vstate, folder, hi,  one_avg = "one")
   

sys.stdout.close()
