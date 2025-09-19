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

from netket.operator.spin import sigmax, sigmaz, sigmay
from netket.experimental.driver import VMC_SRt

from HFDS_model_ferm import Orbitals
from HFDS_model_ferm import HiddenFermion



#Physical param
L = 4
n_elecs = L*L
N_sites = L*L
N_up    = (n_elecs+1)//2
N_dn    = n_elecs//2
n_dim = 2
J2 = 0

dtype   = "real"
MFinitialization = "Fermi"
determinant_type = "hidden"
bounds  ="PBC"
double_occupancy = False
save = False

#Network param
n_hid_ferm       = 16
features         = 8
hid_layers       = 2

n_samples        = 4096
lr               = 0.0075
N_opt            = 100

n_chains         = n_samples//2
n_steps          = 10
n_modes          = 2*L*L
cs               = n_samples
n_dim           = 2

model_name = f"layers{hid_layers}_hidd{n_hid_ferm}_feat{features}_sample{n_samples}_lr{lr}_iter{N_opt}"
lattice_name = f"J={J2}_L={L}"
folder = f'HFDS_Heisenberg/plot/ferm/{lattice_name}/{model_name}'
os.makedirs(folder, exist_ok=True)  #create folder for the plots and the output file
sys.stdout = open(f"{folder}/output.txt", "w") #redirect print output to a file inside the folder

print(f"HFDS_ferm, J={J2}, L={L}, layers{hid_layers}_hidd{n_hid_ferm}_feat{features}_sample{n_samples}_lr{lr}_iter{N_opt}")

# Lattice and Hilbert space
boundary_conditions = 'pbc' 
g = nk.graph.Hypercube(length=L, n_dim=2, pbc=True, max_neighbor_order=2)
exchange_g = nk.graph.disjoint_union(g, g)
print("Exchange graph size:", exchange_g.n_nodes)

hi = nkx.hilbert.SpinOrbitalFermions(N_sites, s = 1/2, n_fermions_per_spin = (N_up, N_dn))
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


# Heisenberg J1-J2 spin hamiltonian
hamiltonian = nk.operator.Heisenberg(
    hilbert=hi, graph=g, J=[1.0, J2], sign_rule=[False, False]).to_jax_operator()  # No Marshall sign rule

#sampler
sampler = nk.sampler.MetropolisExchange(hi, graph=exchange_g, d_max=1)

#vstate
vstate = nk.vqs.MCState(sampler, ma, n_samples=n_samples, chunk_size=cs, n_discard_per_chain=32) #defines the variational state object
total_params = sum(p.size for p in jax.tree_util.tree_leaves(vstate.parameters))
print(f'Total number of parameters: {total_params}')

#optimization
optimizer = nk.optimizer.Sgd(learning_rate=lr)

vmc = VMC_SRt(
    hamiltonian=hamiltonian,
    optimizer=optimizer,
    diag_shift=1e-2,
    variational_state=vstate,
    #mode="real",
) 

log = nk.logging.RuntimeLog()

vmc.run(n_iter=N_opt, out=log)


#%%
#Energy
energy_per_site = log.data["Energy"]["Mean"].real / (L * L * 4)

E_vs = energy_per_site[-1]
print("Last value: ", energy_per_site[-1])

plt.plot(energy_per_site)

plt.xlabel("Iterations")
plt.ylabel("Energy per site")
if(save):
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
if(save):
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
if(save):
    plt.savefig(f'{folder}/Struct.png')
plt.show()

#Exact gs
E_gs, ket_gs = nk.exact.lanczos_ed(hamiltonian, compute_eigenvectors=True)

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
tot_magn = sum([sigmaz(hi, i) for i in g.nodes()])
tot_magn_vstate = vstate.expect(tot_magn).mean.real
print(f"Magnetization = {tot_magn_vstate}" )

#Variance
variance = log.data["Energy"]["Variance"][-1].real
print(f"Variance = {variance}")

#Vscore
v_score = L*L*variance/(E_vs*L*L*4)
print(f"V-score = {v_score}")


sys.stdout.close()

