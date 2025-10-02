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
import numpy as np
from scipy.sparse.linalg import eigsh

from netket.operator.spin import sigmax, sigmaz, sigmay
from netket.experimental.driver import VMC_SR
import netket.operator as op

from HFDS_Heisenberg.HFDS_model_ferm import Orbitals
from HFDS_Heisenberg.HFDS_model_ferm import HiddenFermion

from Elaborate.Energy import *
from Elaborate.Corr_Struct import *
from Elaborate.Error_Stat import *
from Elaborate.count_params import *
from Elaborate.order_param import *
from Elaborate.Sign_obs import MarshallSignOperator

from HFDS_Heisenberg.Exchange_sampler import *


parser = argparse.ArgumentParser(description="Example script with parameters")
parser.add_argument("--J2", type=float, default=0.5, help="Coupling parameter J2")
args = parser.parse_args()

J1J2 = False
spin = False

#Physical param
L = 4
n_elecs = L*L
N_sites = L*L
N_up    = n_elecs//2
N_dn    = n_elecs//2
n_dim = 2

J2 = args.J2


dtype   = "real"
MFinitialization = "Fermi"
determinant_type = "hidden"
bounds  ="PBC"
double_occupancy = False
save = False

#Network param
n_hid_ferm       = 4
features         = 64
hid_layers       = 1

n_samples        = 2048
lr               = 0.02
N_opt            = 400

n_chains         = n_samples
n_steps          = 100
n_modes          = 2*L*L
cs               = n_samples
n_dim           = 2

model_name = f"layers{hid_layers}_hidd{n_hid_ferm}_feat{features}_sample{n_samples}_lr{lr}_iter{N_opt}"
lattice_name = f"J={J2}_L={L}"
if J1J2==True:
    folder = f'HFDS_Heisenberg/plot/J1J2/ferm/{model_name}/{lattice_name}'
    save_model = f"HFDS_Heisenberg/plot/J1J2/spin/{model_name}/{lattice_name}/models"
else:
    folder = f'HFDS_Heisenberg/plot/Ising/ferm/{model_name}/{lattice_name}'
    save_model = f"HFDS_Heisenberg/plot/Ising/spin/{model_name}/{lattice_name}/models"
os.makedirs(folder, exist_ok=True)  #create folder for the plots and the output file
os.makedirs(save_model, exist_ok=True)
sys.stdout = open(f"{folder}/output.txt", "w") #redirect print output to a file inside the folder
print(f"HFDS_ferm, J={J2}, L={L}, layers{hid_layers}_hidd{n_hid_ferm}_feat{features}_sample{n_samples}_lr{lr}_iter{N_opt}")

# Lattice and Hilbert space
boundary_conditions = 'pbc' 
g = nk.graph.Hypercube(length=L, n_dim=2, pbc=True, max_neighbor_order=2)
exchange_g = nk.graph.disjoint_union(g, g)
print("Exchange graph size:", exchange_g.n_nodes)

# put some costraint of the hilbert space
hi = nk.hilbert.SpinOrbitalFermions(N_sites, s = 1/2, n_fermions_per_spin = (N_up, N_dn))
print("Hilbert space size",hi.size)


if dtype=="real": dtype_ = jnp.float64
else: dtype_ = jnp.complex128

# Model
ma = HiddenFermion(n_elecs=n_elecs,
                   network="FFNN",
                   n_hid=n_hid_ferm,
                   L=L,
                   layers=hid_layers,
                   features=features,
                   MFinit=MFinitialization,
                   hilbert=hi,
                   stop_grad_mf=False,
                   stop_grad_lower_block=False,
                   bounds=bounds,
                   dtype=dtype_)

#ma = nk.models.RBM(alpha=1, param_dtype=complex)

if J1J2==True:
# Heisenberg J1-J2 spin hamiltonian
    hamiltonian = nk.operator.Heisenberg(hilbert=hi, graph=g, J=[1.0, J2], sign_rule=[False, False]).to_jax_operator()  # No Marshall sign rule
else:
    #ising hamiltonian
    #hamiltonian = nk.operator.Ising(hilbert=hi, graph=g, h = h, J = 1.0, dtype=jnp.float64).to_jax_operator()
    hamiltonian = op.LocalOperator(hi)
    for i, j in g.edges():
        hamiltonian += -1 * op.spin.sigmaz(hi, i) * op.spin.sigmaz(hi, j)
    for i in range(L*L):
        hamiltonian += -J2 * op.spin.sigmax(hi, i)

#sampler
sampler = nk.sampler.MetropolisSampler(hi, n_chains=n_chains, rule=tJExchangeRule(graph=exchange_g))

#vstate
vstate = nk.vqs.MCState(sampler, ma, n_samples=n_samples, chunk_size=cs, n_discard_per_chain=32) #defines the variational state object
total_params = sum(p.size for p in jax.tree_util.tree_leaves(vstate.parameters))
print(f'Total number of parameters: {total_params}')

"""
# Draw a sample and compute log amplitude to debug
for s in range(n_samples):
    sample = vstate.samples[s]
    print(sample)
    val = vstate.log_value(sample)
    print("Initial log amplitude:", val)
    print(" ")

# calculate observable
obs = vstate.expect(hamiltonian)
print("Initial energy:", obs)
"""

#optimization
optimizer = nk.optimizer.Sgd(learning_rate=lr)

vmc = VMC_SR(
    hamiltonian=hamiltonian,
    optimizer=optimizer,
    diag_shift=1e-4,
    variational_state=vstate,
    mode="real",
) 

log = nk.logging.RuntimeLog()

vmc.run(n_iter=N_opt, out=log)

"""
# --- Saving ---
with open(filename + ".mpack", "wb") as f:
    bytes_out = flax.serialization.to_bytes(vstate.variables)
    f.write(bytes_out)

# --- Loading ---
with open(filename + ".mpack", "rb") as f:
    print("load vstate parameters")
    vstate.variables = flax.serialization.from_bytes(vstate.variables, f.read())
"""

#%%

#Energy
E_vs = Energy(log, L, folder)

#Correlation function
vstate.n_samples = 1024
Corr_Struct(g, vstate, L, folder, hi)


E_exact = Exact_gs(L, J2, hamiltonian, J1J2, spin=False)


#Fidelity(vstate, ket_gs)


Relative_Error(E_vs, E_exact)

Magnetization(vstate, g, hi)

variance = Variance(log)

Vscore(L, variance, E_vs)


hidden_fermion_param_count(n_elecs, n_hid_ferm, L, L, hid_layers, features)


Staggered_and_Striped_Magnetization(vstate, g, hi)


"""
#Energy
energy_per_site = log.data["Energy"]["Mean"].real / (L * L * 4)
E_vs = energy_per_site[-1]
print("Last value: ", energy_per_site[-1])
plt.plot(energy_per_site)
plt.xlabel("Iterations")
plt.ylabel("Energy per site")
plt.savefig(f'{folder}/Energy.png')
plt.show()

#Correlation function
vstate.n_samples = 1024
N_tot = g.n_nodes

corr_r = np.zeros((L, L))
counts = np.zeros((L, L))

for i in range(N_tot):
    for j in range(N_tot):
        r = g.positions[i] - g.positions[j]
        corr_ij = 0.25 * (sigmaz(hi, i) * sigmaz(hi, j) + sigmax(hi, i) * sigmax(hi, j) + sigmay(hi, i) * sigmay(hi, j))
        exp = vstate.expect(corr_ij)
        r0, r1 = int(r[0]) % L , int(r[1]) % L #PBC
        corr_r[r0, r1] += exp.mean.real
        counts[r0, r1] += 1
corr_r /= counts 
corr_r[0, 0] = 0  # set C(0) = 0

plt.figure(figsize=(6,5))
plt.imshow(corr_r, origin='lower', cmap='viridis')
plt.colorbar(label='C(r)')
plt.xlabel('dx')
plt.ylabel('dy')
plt.title('Spin-Spin Correlation Function C(r) in 2D')
plt.xticks(np.arange(L))  # integer ticks for x-axis
plt.yticks(np.arange(L)) 
plt.savefig(f'{folder}/Corr.png')
plt.show()


#Structure factor
# Compute the 2D Fourier transform of corr_r
S_q = np.fft.fft2(corr_r)
S_q_periodic = np.zeros((L+1, L+1), dtype=S_q.dtype)
S_q_periodic[:L, :L] = S_q  
S_q_periodic[L, :] = S_q_periodic[0, :]    
S_q_periodic[:, L] = S_q_periodic[:, 0]    

plt.figure(figsize=(6,5))
plt.imshow(np.abs(S_q_periodic), origin='lower', cmap='viridis')
plt.colorbar(label='|S(q)|')
plt.xlabel('q_x')
plt.ylabel('q_y')
plt.title('Structure Factor S(q)')
plt.xticks([0, 1/2*L, L], ['0', 'π', '2π'])
plt.yticks([0, 1/2*L, L], ['0', 'π', '2π'])
plt.savefig(f'{folder}/Struct.png')
plt.show()

#Exact gs
#Define a different hilbert space and hamiltonian for the exact calculation, that is equivalent to the other
#graph_scaled = nk.graph.Hypercube(length=L, n_dim=2, pbc=True, max_neighbor_order=2)
graph_scaled = nk.graph.Hypercube(length=L, n_dim=2, pbc=True)
hi_scaled = nk.hilbert.Spin(s=0.5, N=graph_scaled.n_nodes)
H = nk.operator.Ising(hilbert=hi_scaled, graph=graph_scaled, h=h, J=1.0)
#H = nk.operator.Heisenberg(hilbert=hi_scaled, graph=graph_scaled, J=[1.0, J2], sign_rule=[False, False]).to_jax_operator()
H_sparse = H.to_sparse(jax_=False).tocsc()
E_gs, vecs = eigsh(H_sparse, k=1, which="SA")
E_exact = E_gs[0]/(L*L*4)
print(f"Exact ground state energy per site= {E_exact}")


# Fidelity vstate, exact state
#vstate is too big to be converted in a matrix

vstate_array = vstate.to_array()
overlap_val = vstate_array.conj() @ ket_gs
fidelity_val = np.abs(overlap_val) ** 2 / (np.vdot(vstate_array, vstate_array) * np.vdot(ket_gs, ket_gs))
print(f"Fidelity <vstate|exact> = {fidelity_val}")

#Relative Error
e = np.abs((E_vs - E_exact)/E_exact)
print(f"Relative error = {e}")

import netket.experimental.observable as obs
import netket.operator as op

#Total magnetization on Z
tot_magn = sum([sigmaz(hi, i) for i in g.nodes()])
tot_magn_vstate = vstate.expect(tot_magn).mean.real
print(f"Magnetization = {tot_magn_vstate}" )

#Variance
variance = log.data["Energy"]["Variance"][-1].real
print(f"Variance = {variance}")

#Vscore
v_score = L*L*variance/(E_vs*L*L*4)
print(f"V-score = {v_score}")

"""
sys.stdout.close()

