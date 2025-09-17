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

# Variational monte carlo driver
from netket.experimental.driver import VMC_SR

from HFDS_model import Orbitals
from HFDS_model import HiddenFermion
#%%
load = True

L      = 4
n_elecs = 16
J2      = 0
b1      = 0
b2      = 0
dtype   = "real"
MFinitialization = "Fermi"
determinant_type = "hidden"
bounds  = {1:{1:"OBC"}, 0:{0:"PBC"}}[b1][b2]

print("params: J2=", J2, "L=", L, "bounds=", b1, b2)

# more parameters for the physical system
pbc     = [{0: True, 1:False}[b1],{0: True, 1:False}[b2]]
N_sites = L*L
N_up    = (n_elecs+1)//2
N_dn    = n_elecs//2

# network parameters and sampling
lr               = 0.02
n_samples        = 4096
n_chains         = n_samples//2
n_steps          = 10
n_hid            = 1
features         = 3
layers           = 1
n_modes          = 2*L*L
cs               = n_samples
n_dim           = 2


# --------------- define the network  -------------------
boundary_conditions = 'pbc' if pbc[0] else 'obc'
folder = f"results/energy_{L}x{L}_{boundary_conditions}x{boundary_conditions}_Nup={N_up}_Ndn={N_dn}J2={J2}_lr={lr}_nlayers={layers}_nfeatures={features}_nhid={n_hid}_nsamples={n_samples}_{determinant_type}_{MFinitialization}_{dtype}"
lattice = nk.graph.Hypercube(length=L, n_dim=n_dim, pbc=True, max_neighbor_order=2)
#hi = nk.hilbert.SpinOrbitalFermions(N_sites, s = 1/2, n_fermions_per_spin = (N_up, N_dn))
hi = nk.hilbert.Spin(s=1 / 2, N=lattice.n_nodes, total_sz=0)
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
                   MFinit=MFinitialization,
                   hilbert=hi,
                   stop_grad_mf=False,
                   stop_grad_lower_block=False,
                   bounds=bounds,
                   dtype=dtype_)

#model = nk.models.RBM(alpha=1)

# ------------- define Hamiltonian ------------------------
# Heisenberg J1-J2 spin hamiltonian
ha = nk.operator.Heisenberg(
    hilbert=hi, graph=lattice, J=[1.0, J2], sign_rule=[False, False]
).to_jax_operator()  # No Marshall sign rule


# ---------- define sampler ------------------------
sampler = nk.sampler.MetropolisExchange(
    hilbert=hi,
    graph=lattice,
    d_max=1,
    n_chains=n_samples,
    sweep_size=lattice.n_nodes,
)


vstate = nk.vqs.MCState(sampler, model, n_samples=n_samples, chunk_size=cs, n_discard_per_chain=128) #defines the variational state object
total_params = sum(p.size for p in jax.tree_util.tree_leaves(vstate.parameters))
print(f'Total number of parameters: {total_params}')

# Draw a sample and compute log amplitude to debug
sample = vstate.samples[0]
print(sample)
val = vstate.log_value(sample)
print("Initial log amplitude:", val)

# calculate observable
obs = vstate.expect(ha)
print("Initial energy:", obs)

#%%

optimizer = nk.optimizer.Sgd(learning_rate=0.01)

vmc = VMC_SR(
    hamiltonian=ha,
    optimizer=optimizer,
    diag_shift=0.1,
    variational_state=vstate,
    mode = 'real'
) 

log = nk.logging.RuntimeLog()

N_opt = 3
vmc.run(n_iter=N_opt, out=log)


#%%
energy_per_site = log.data["Energy"]["Mean"].real / (L * L * 4)

E_vs = energy_per_site[-1]
print("Last value: ", energy_per_site[-1])

plt.plot(energy_per_site)

plt.xlabel("Iterations")
plt.ylabel("Energy per site")
plt.show()

