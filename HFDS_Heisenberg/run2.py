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

from extra import tJExchangeRule

# Variational monte carlo driver
from netket.experimental.driver import VMC_SR


from HFDS_Heisenberg.HFDS_model_ferm import Orbitals
from HFDS_Heisenberg.HFDS_model_ferm import HiddenFermion

parser = argparse.ArgumentParser()
parser.add_argument("-L" , "--L"   , type=int,  default = 4 , help="length in x dir")
parser.add_argument("-J2"  , "--J2"    , type=float,default = 0 , help="spin-spin interaction")
parser.add_argument("-Ne"  , "--n_elecs"    , type=int,default = 16, help="number of electrons")
parser.add_argument("-b1"  , "--b1"    , type=int,default = 0 , help="boundary for x-dir (0:periodic, 1:open)")
parser.add_argument("-b2"  , "--b2"    , type=int,default = 0 , help="boundary for y-dir (0:periodic, 1:open)")
parser.add_argument("-init"  , "--MFinit"    , type=str, default = "Fermi" , help="initialization for MF")
parser.add_argument("-f"  , "--features"    , type=int, default = 32 , help="number of features for transformer / FFNN")
parser.add_argument("-l"  , "--layers"    , type=int, default = 1 , help="number of layers")
parser.add_argument("-nhid"  , "--nhid"    , type=int, default = 16 , help="number of hidden fermions")
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

# network parameters and sampling
lr               = 0.02
n_samples        = 4096
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
#hi = nk.hilbert.Spin(s=1 / 2, N=lattice.n_nodes, total_sz=0)
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
                   double_occupancy_bool=False,
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
exchange_lattice = nk.graph.disjoint_union(lattice, lattice)
print("Exchange graph size:", exchange_lattice.n_nodes)

sampler = nk.sampler.MetropolisExchange(hi, graph=exchange_lattice, d_max=1)


vstate = nk.vqs.MCState(sampler, model, n_samples=n_samples, chunk_size=cs, n_discard_per_chain=32) #defines the variational state object
total_params = sum(p.size for p in jax.tree_util.tree_leaves(vstate.parameters))
print(f'Total number of parameters: {total_params}')

# Draw a sample and compute log amplitude to debug
sample = vstate.samples[0]
print(sample)
val = vstate.log_value(sample)
print("Initial log amplitude:", val)

# calculate observable
obs = vstate.expect(hamiltonian)
print("Initial energy:", obs)


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