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

import argparse
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

from netket.operator.spin import sigmax, sigmaz, sigmay
# Variational monte carlo driver
from netket.experimental.driver import VMC_SR

from HFDS_model_spin import Orbitals
from HFDS_model_spin import HiddenFermion
#%%


parser = argparse.ArgumentParser(description="Example script with parameters")
parser.add_argument("--J2", type=float, default=0.5, help="Coupling parameter J2")
args = parser.parse_args()

#Physical param
L       = 4
n_elecs = L*L # L*L should be half filling
N_sites = L*L
N_up    = (n_elecs+1)//2
N_dn    = n_elecs//2
n_dim = 2
J2 = args.J2

dtype   = "real"
MFinitialization = "Fermi"
determinant_type = "hidden"
bounds  = "PBC"
symmetry = True  #True or False

#Varaitional state param
n_hid_ferm       = 16
features         = 32
hid_layers       = 1

#Network param
lr               = 0.02
n_samples        = 4096
N_opt            = 200

n_chains         = n_samples//2
n_steps          = 10
n_modes          = 2*L*L
cs               = n_samples
n_dim            = 2


model_name = f"layers{hid_layers}_hidd{n_hid_ferm}_feat{features}_sample{n_samples}_lr{lr}_iter{N_opt}_symm{symmetry}"
lattice_name = f"J={J2}_L={L}"
folder = f'HFDS_Heisenberg/plot/spin/{model_name}/{lattice_name}'
os.makedirs(folder, exist_ok=True)  #create folder for the plots and the output file
sys.stdout = open(f"{folder}/output.txt", "w") #redirect print output to a file inside the folder

print(f"HFDS_spin, J={J2}, L={L}, layers{hid_layers}_hidd{n_hid_ferm}_feat{features}_sample{n_samples}_lr{lr}_iter{N_opt}")

# --------------- define the network  -------------------
boundary_conditions = 'pbc' 
lattice = nk.graph.Hypercube(length=L, n_dim=n_dim, pbc=True, max_neighbor_order=2)
#hi = nk.hi.SpinOrbitalFermions(N_sites, s = 1/2, n_fermions_per_spin = (N_up, N_dn))
hi = nk.hilbert.Spin(s=1 / 2, N=lattice.n_nodes, total_sz=0)
print(hi.size)


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

#model = nk.models.RBM(alpha=1)

# ------------- define Hamiltonian ------------------------
# Heisenberg J1-J2 spin ha
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

"""
# Draw a sample and compute log amplitude to debug
sample = vstate.samples[0]
print(sample)
val = vstate.log_value(sample)
print("Initial log amplitude:", val)

# calculate observable
obs = vstate.expect(ha)
print("Initial energy:", obs)
"""

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

vmc.run(n_iter=N_opt, out=log)


#%%

#Save log, vstate
import pickle
with open(f"{folder}/log.pkl", "wb") as f:
    pickle.dump(log, f)

params = vstate.parameters  # dictionary of model parameters
np.savez(f"{folder}/vstate_params.npz", **params)

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
N_tot = lattice.n_nodes

corr_r = np.zeros((L, L))
counts = np.zeros((L, L))

for i in range(N_tot):
    for j in range(N_tot):
        r = lattice.positions[i] - lattice.positions[j]
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
E_gs, ket_gs = nk.exact.lanczos_ed(ha, compute_eigenvectors=True)

print(f"Exact ground state energy = {E_gs[0]:.3f}")
E_exact = E_gs[0]/(L*L*4)
print(f"Exact ground state energy per site= {E_exact}")

# Fidelity vstate, exact state
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
tot_magn = sum([sigmaz(hi, i) for i in lattice.nodes()])
tot_magn_vstate = vstate.expect(tot_magn).mean.real
print(f"Magnetization = {tot_magn_vstate}" )

#Variance
variance = log.data["Energy"]["Variance"][-1].real
print(f"Variance = {variance}")

#Vscore
v_score = L*L*variance/(E_vs*L*L*4)
print(f"V-score = {v_score}")


#Count Parameters
def hidden_fermion_param_count(n_elecs, n_hid, Lx, Ly, layers, features):
    # Parameters in Orbitals module
    n_sites = Lx * Ly
    orbitals_mf_params = 2 * n_sites * n_elecs  # orbitals_mf shape (2*Lx*Ly, n_elecs)
    orbitals_hf_params = 2 * n_sites * n_hid    # orbitals_hf shape (2*Lx*Ly, n_hid)
    
    # Parameters in FFNN part of HiddenFermion
    input_dim = n_sites  # Input dimension Lx*Ly
    # Hidden layers (all without bias)
    hidden_params = input_dim * features  # First hidden layer
    hidden_params += (layers - 1) * (features * features)  # Subsequent hidden layers
    
    # Output layer (with bias)
    output_dim = n_hid * (n_elecs + n_hid)
    output_params = features * output_dim + output_dim  # Weights + biases
    
    total_params = orbitals_mf_params + orbitals_hf_params + hidden_params + output_params
    return total_params


count_params = hidden_fermion_param_count(n_elecs, n_hid_ferm, L, L, hid_layers, features)
print(f"params={count_params}")



# Staggered and Striped Magnetization
ops = {}
for i in lattice.nodes():  # just node indices
    x, y = lattice.positions[i]  # get coordinates
    staggered_sign = (-1) ** (x + y)
    ops[f"sz_{i}"] = staggered_sign * nk.operator.spin.sigmaz(hi, i)
M_stag = sum(ops.values()) / lattice.n_nodes

Staggered_Magnetization = vstate.expect(M_stag) 
print(f"Staggered Magnetization = {Staggered_Magnetization.mean.real}")

ops = {}
for i in lattice.nodes():  # just node indices
    x, y = lattice.positions[i]  # get coordinates
    stripe_sign = (-1) ** x  # change to (-1)**y for y-stripes
    ops[f"sz_{i}"] = stripe_sign * nk.operator.spin.sigmaz(hi, i)
M_stripe = sum(ops.values()) / lattice.n_nodes

Striped_Magnetization = vstate.expect(M_stripe) 
print(f"Striped Magnetization = {Striped_Magnetization.mean.real}")

sys.stdout.close()

