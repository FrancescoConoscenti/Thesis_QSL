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
import matplotlib.pyplot as plt
sys.path.insert(1, '/project/th-scratch/h/Hannah.Lange/PhD/ML/HiddenFermions/src')
import argparse
from jax import numpy as jnp
import netket as nk
import jax
from netket import experimental as nkx
import os
import flax
os.environ['JAX_TRACEBACK_FILTERING'] = 'off'

from HFDS_model_ferm import Orbitals
from HFDS_model_ferm import HiddenFermion

from netket.experimental.driver import VMC_SR


# parse arguments
L = 4
n_elecs = L*L
J2 = 0
dtype   = "real"
MFinitialization = "Fermi"
determinant_type = "hidden"
bounds  ="PBC"
print("params: J2=", J2, "L=", L)

N_sites = L * L
N_up    = (n_elecs+1)//2
N_dn    = n_elecs//2

double_occupancy = False

# network parameters and sampling
lr               = 0.02
n_samples        = 4096*2
n_chains         = n_samples//2
n_steps          = 1000
n_modes          = 2*L*L
cs               = n_samples

#FFN and hidden
n_hid            = 3
features         = 3
layers           = 3



# --------------- define the network  -------------------
boundary_conditions = 'pbc' 
filename = f"results_sym/energy_{L}x{L}_{boundary_conditions}x{boundary_conditions}_Nup={N_up}_Ndn={N_dn}_lr={lr}_nlayers={layers}_nfeatures={features}_nhid={n_hid}_nsamples={n_samples}_{determinant_type}_"+MFinitialization+"_"+dtype
g = nk.graph.Hypercube(length=L, n_dim=2, pbc=True, max_neighbor_order=2)

hi = nk.hilbert.SpinOrbitalFermions(N_sites, s = 1/2, n_fermions_per_spin = (N_up, N_dn))
print(hi.size)


if dtype=="real": dtype_ = jnp.float64
else: dtype_ = jnp.complex128

ma = HiddenFermion(n_elecs=n_elecs,
                   network="FFNN",
                   n_hid=n_hid,
                   L=L,
                   layers=layers,
                   features=features,
                   MFinit=MFinitialization,
                   hilbert=hi,
                   stop_grad_mf=False,
                   stop_grad_lower_block=False,
                   bounds=bounds,
                   dtype=dtype_)


# ------------- define Hamiltonian ------------------------
# Heisenberg J1-J2 spin hamiltonian
hamiltonian = nk.operator.Heisenberg(
    hilbert=hi, graph=g, J=[1.0, J2], sign_rule=[False, False]).to_jax_operator()  # No Marshall sign rule


# ---------- define sampler ------------------------
exchange_g = nk.graph.disjoint_union(g, g)
print("Exchange graph size:", exchange_g.n_nodes)

sampler = nk.sampler.MetropolisExchange(hi, graph=exchange_g, d_max=1)

vstate = nk.vqs.MCState(sampler, ma, n_samples=n_samples, chunk_size=cs, n_discard_per_chain=32) #defines the variational state object
total_params = sum(p.size for p in jax.tree_util.tree_leaves(vstate.parameters))
print(f'Total number of parameters: {total_params}')


"""# Draw a sample and compute log amplitude to debug
sample = vstate.samples[0]
print(sample)
val = vstate.log_value(sample)
print("Initial log amplitude:", val)

# calculate observable
obs = vstate.expect(hamiltonian)
print("Initial energy:", obs)"""


optimizer = nk.optimizer.Sgd(learning_rate=0.001)

vmc = VMC_SR(
    hamiltonian=hamiltonian,
    optimizer=optimizer,
    diag_shift=1e-3,
    variational_state=vstate,
    mode="real",
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
  
#evaluate any observable...
