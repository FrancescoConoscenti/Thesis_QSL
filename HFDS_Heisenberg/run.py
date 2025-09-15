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

from netket.operator.fermion import destroy as c
from netket.operator.fermion import create as cdag
from netket.operator.fermion import number as nc

# Variational monte carlo driver
from netket.experimental.driver import VMC_SR


from HFDS_model import Orbitals
from HFDS_model import HiddenFermion


parser = argparse.ArgumentParser()
parser.add_argument("-L" , "--L"   , type=int,  default = 4 , help="length in x dir")
parser.add_argument("-J2"  , "--J2"    , type=float,default = 0 , help="spin-spin interaction")
parser.add_argument("-Ne"  , "--n_elecs"    , type=int,default = 16, help="number of electrons")
parser.add_argument("-b1"  , "--b1"    , type=int,default = 0 , help="boundary for x-dir (0:periodic, 1:open)")
parser.add_argument("-b2"  , "--b2"    , type=int,default = 0 , help="boundary for y-dir (0:periodic, 1:open)")
parser.add_argument("-init"  , "--MFinit"    , type=str, default = "Fermi" , help="initialization for MF")
parser.add_argument("-f"  , "--features"    , type=int, default = 32 , help="number of features for transformer / FFNN")
parser.add_argument("-l"  , "--layers"    , type=int, default = 1 , help="number of layers")
parser.add_argument("-nhid"  , "--nhid"    , type=int, default = 2 , help="number of hidden fermions")
parser.add_argument("-dtype"  , "--dtype"    , type=str, default = "real" , help="complex or real")

load = True

# parse arguments
args = parser.parse_args()
L      = args.L
n_elecs = args.n_elecs
J2      = args.J2
b1      = args.b1
b2      = args.b2
dtype   = args.dtype
MFinitialization = args.MFinit
determinant_type = "hidden"
bounds  = {1:{1:"OBC"}, 0:{0:"PBC"}}[b1][b2]


print("params: J2=", J2, "L=", L, "bounds=", b1, b2)



# more parameters for the physical system
pbc     = [{0: True, 1:False}[b1],{0: True, 1:False}[b2]]
N_sites = L*L
N_up    = (n_elecs+1)//2
N_dn    = n_elecs//2

double_occupancy = False

# network parameters and sampling
lr               = 0.02
n_samples        = 128
n_chains         = n_samples//2
n_steps          = 10
n_hid            = args.nhid
features         = args.features
layers           = args.layers
n_modes          = 2*L*L
cs               = n_samples
n_dim           = 2


# --------------- define the network  -------------------
boundary_conditions = 'pbc' if pbc[0] else 'obc'
folder = f"results/energy_{L}x{L}_{boundary_conditions}x{boundary_conditions}_Nup={N_up}_Ndn={N_dn}J2={J2}_lr={lr}_nlayers={layers}_nfeatures={features}_nhid={n_hid}_nsamples={n_samples}_{determinant_type}_{MFinitialization}_{dtype}"
lattice = nk.graph.Hypercube(length=L, n_dim=n_dim, pbc=True, max_neighbor_order=2)
hi = nk.hilbert.SpinOrbitalFermions(N_sites, s = 1/2, n_fermions_per_spin = (N_up, N_dn))
print(hi.size)


if dtype=="real": dtype_ = jnp.float64
else: dtype_ = jnp.complex128

model = HiddenFermion(n_elecs=n_elecs,
                   network="FFNN",
                   n_hid=n_hid,
                   Lx=L,
                   Ly=L,
                   layers=layers,
                   features=features,
                   double_occupancy_bool=double_occupancy,
                   MFinit=MFinitialization,
                   hilbert=hi,
                   stop_grad_mf=False,
                   stop_grad_lower_block=False,
                   bounds=bounds,
                   dtype=dtype_)


# ------------- define Hamiltonian ------------------------
# Heisenberg J1-J2 spin hamiltonian
hamiltonian = nk.operator.Heisenberg(
    hilbert=hi, graph=lattice, J=[1.0, J2], sign_rule=[False, False]
).to_jax_operator()  # No Marshall sign rule


# ---------- define sampler ------------------------
sampler = nk.sampler.MetropolisExchange(
    hilbert=hi,
    graph=lattice,
    d_max=2,
    n_chains=n_samples,
    sweep_size=lattice.n_nodes,
)

vstate = nk.vqs.MCState(sampler, model, n_samples=n_samples, chunk_size=cs, n_discard_per_chain=32) #defines the variational state object
total_params = sum(p.size for p in jax.tree_util.tree_leaves(vstate.parameters))
print(f'Total number of parameters: {total_params}')

# Draw a sample and compute log amplitude to debug
sample = vstate.samples[0]
val = vstate.log_value(sample)
print("Initial log amplitude:", val)


optimizer = nk.optimizer.Sgd(learning_rate=0.001)

vmc = VMC_SR(
    hamiltonian=hamiltonian,
    optimizer=optimizer,
    diag_shift=1e-4,
    variational_state=vstate,
) 

log = nk.logging.RuntimeLog()

N_opt = 5
vmc.run(n_iter=N_opt, out=log)


#%%
energy_per_site = log.data["Energy"]["Mean"].real / (L * L * 4)

E_vs = energy_per_site[-1]
print("Last value: ", energy_per_site[-1])

plt.plot(energy_per_site)

plt.xlabel("Iterations")
plt.ylabel("Energy per site")


"""
# -------------- start the training ---------------
os.makedirs(folder, exist_ok=True)
with open(folder+".mpack", 'rb') as file:
  print("load vstate parameters")
  vstate.variables = flax.serialization.from_bytes(vstate.variables, file.read())
"""
  
#evaluate any observable...
