#%%
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
#sys.path.insert(1, '/project/th-scratch/h/Hannah.Lange/PhD/ML/HiddenFermions/src')
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

from netket.operator.spin import sigmax, sigmaz, sigmay
# Variational monte carlo driver
from netket.experimental.driver import VMC_SR
import netket.operator as op

sys.path.append(os.path.dirname(os.path.dirname("/scratch/f/F.Conoscenti/Thesis_QSL")))

from HFDS_Heisenberg.HFDS_model_spin import Orbitals
from HFDS_Heisenberg.HFDS_model_spin import HiddenFermion

from Elaborate.Energy import *
from Elaborate.Corr_Struct import *
from Elaborate.Error_Stat import *
from Elaborate.count_params import *
from Elaborate.Sign_vs_iteration import *
from Elaborate.Sign_Obs import *


parser = argparse.ArgumentParser(description="Example script with parameters")
parser.add_argument("--J2", type=float, default=0.5, help="Coupling parameter J2")
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

dtype   = "real"
MFinitialization = "Fermi"
determinant_type = "hidden"
bounds  = "PBC"
symmetry = True  #True or False

#Varaitional state param
n_hid_ferm       = 4
features         = 64 #hidden units per layer
hid_layers       = 1

#Network param
lr               = 0.02
n_samples        = 1024
N_opt            = 1500
save_every       = 20
block_iter = N_opt//save_every

n_chains         = n_samples//2
chunk_size       =  n_samples#//4   #chunk size for the sampling


model_name = f"layers{hid_layers}_hidd{n_hid_ferm}_feat{features}_sample{n_samples}_lr{lr}_iter{N_opt}_symm{symmetry}_Hannah"
lattice_name = f"J2={J2}_L={L}"
if J1J2==True:
   folder = f'HFDS_Heisenberg/plot/J1J2/spin/{model_name}/{lattice_name}'
   save_model = f"HFDS_Heisenberg/plot/J1J2/spin/{model_name}/{lattice_name}/models"
else:
    folder = f'HFDS_Heisenberg/plot/Ising/spin/{model_name}/{lattice_name}'
    save_model = f"HFDS_Heisenberg/plot/Ising/spin/{model_name}/{lattice_name}/models"
os.makedirs(folder, exist_ok=True)  #create folder for the plots and the output file
os.makedirs(save_model, exist_ok=True)
sys.stdout = open(f"{folder}/output.txt", "w") #redirect print output to a file inside the folder

print(f"HFDS_spin, J={J2}, L={L}, layers{hid_layers}_hidd{n_hid_ferm}_feat{features}_sample{n_samples}_lr{lr}_iter{N_opt}")


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
                   parity=symmetry,
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


vstate = nk.vqs.MCState(sampler, model, n_samples=n_samples, chunk_size=chunk_size, n_discard_per_chain=128) #defines the variational state object
total_params = sum(p.size for p in jax.tree_util.tree_leaves(vstate.parameters))
print(f'Total number of parameters: {total_params}')


optimizer = nk.optimizer.Sgd(learning_rate=lr)

vmc = VMC_SR(
    hamiltonian=ha,
    optimizer=optimizer,
    diag_shift=1e-3,
    variational_state=vstate,
    mode = 'real'
) 

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
Corr_Struct(lattice, vstate, L, folder, hi)

E_exact, ket_gs = Exact_gs(L, J2, ha, J1J2, spin)


if J1J2 == True:
    fidelity = Fidelity(vstate, ket_gs)
    print(f"Fidelity <vstate|exact> = {fidelity}")

Relative_Error(E_vs, E_exact)

Magnetization(vstate, lattice, hi)

variance = Variance(log)

Vscore(L, variance, E_vs)

hidden_fermion_param_count(n_elecs, n_hid_ferm, L, L, hid_layers, features)

#Staggered_and_Striped_Magnetization(vstate, lattice, hi)

n_sample = 2048
marshall_op = MarshallSignOperator(hi)
plot_Sign_full_MCMC(marshall_op, vstate, folder, n_samples)
plot_Sign_Fidelity(marshall_op, ket_gs, vstate, folder)

sys.stdout.close()

