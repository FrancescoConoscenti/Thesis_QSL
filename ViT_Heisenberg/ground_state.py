#%%
import matplotlib.pyplot as plt
import netket as nk
import jax
import numpy as np
import jax.numpy as jnp
print(jax.devices())
import flax
from flax import linen as nn
from netket.operator.spin import sigmax, sigmaz, sigmay
import sys
import os

from ViT_model import ViT_sym
 

seed = 0
key = jax.random.key(seed)
M = 10  # Number of spin configurations to initialize the parameters
L = 4  # Linear size of the lattice

save = True
seed = 0
key = jax.random.key(seed)

n_dim = 2
J2 = 0

num_layers      = 2     # number of Tranformer layers
d_model         = 16     # dimensionality of the embedding space
n_heads         = 2     # number of heads
patch_size      = 2     # lenght of the input sequence
lr              = 0.0075

N_samples       = 1024
N_opt           = 100


model_name = f"layers{num_layers}_d{d_model}_heads{n_heads}_patch{patch_size}_sample{N_samples}_lr{lr}_iter{N_opt}"
lattice_name = f"J={J2}_L={L}"
folder = f'ViT_Heisenberg/plot/{model_name}/{lattice_name}'
os.makedirs(folder, exist_ok=True)  #create folder for the plots and the output file
sys.stdout = open(f"{folder}/output.txt", "w") #redirect print output to a file inside the folder


lattice = nk.graph.Hypercube(length=L, n_dim=n_dim, pbc=True, max_neighbor_order=2)

# Hilbert space of spins on the graph
hilbert = nk.hilbert.Spin(s=1 / 2, N=lattice.n_nodes, total_sz=0)


# Heisenberg J1-J2 spin hamiltonian
hamiltonian = nk.operator.Heisenberg(
    hilbert=hilbert, graph=lattice, J=[1.0, J2], sign_rule=[False, False]
).to_jax_operator()  # No Marshall sign rule

"""
#Ising Hamiltonian
hamiltonian = nk.operator.LocalOperator(hilbert)
for u, v in lattice.edges():
    hamiltonian += sigmaz(hilbert, u) * sigmaz(hilbert, v) 
"""

# Intiialize the ViT variational wave function
vit_module = ViT_sym(
    num_layers=num_layers, d_model=d_model, n_heads=n_heads, patch_size=patch_size, transl_invariant=True, parity=True
)

key, subkey = jax.random.split(key)
spin_configs = jax.random.randint(subkey, shape=(M, L * L), minval=0, maxval=1) * 2 - 1
params = vit_module.init(subkey, spin_configs)

# Metropolis Local Sampling
sampler = nk.sampler.MetropolisExchange(
    hilbert=hilbert,
    graph=lattice,
    d_max=2,
    n_chains=N_samples,
    sweep_size=lattice.n_nodes,
)

optimizer = nk.optimizer.Sgd(learning_rate=lr)

key, subkey = jax.random.split(key, 2)
vstate = nk.vqs.MCState(
    sampler=sampler,
    model=vit_module,
    sampler_seed=subkey,
    n_samples=N_samples,
    n_discard_per_chain=16,
    variables=params,
    chunk_size=512,
)

N_params = nk.jax.tree_size(vstate.parameters)
print("Number of parameters = ", N_params, flush=True)

# Variational monte carlo driver
from netket.experimental.driver import VMC_SRt

vmc = VMC_SRt(
    hamiltonian=hamiltonian,
    optimizer=optimizer,
    diag_shift=1e-4,
    variational_state=vstate,
    #mode="complex",
) 

# Optimization
log = nk.logging.RuntimeLog()

vmc.run(n_iter=N_opt, out=log)


#%%
energy_per_site = log.data["Energy"]["Mean"].real / (L * L * 4)

E_vs = energy_per_site[-1]
print("Last value: ", energy_per_site[-1])

plt.plot(energy_per_site)

plt.xlabel("Iterations")
plt.ylabel("Energy per site")

if(save):
    plt.savefig(f'{folder}/Energy.png')
plt.show()



vstate.n_samples = 1024
N_tot = lattice.n_nodes


corr_r = np.zeros((L, L))
counts = np.zeros((L, L))

for i in range(N_tot):
    for j in range(N_tot):
        
        r = lattice.positions[i] - lattice.positions[j]

        corr_ij = 0.25 * (sigmaz(hilbert, i) * sigmaz(hilbert, j) + sigmax(hilbert, i) * sigmax(hilbert, j) + sigmay(hilbert, i) * sigmay(hilbert, j))

        exp = vstate.expect(corr_ij)

        #PBC
        r0, r1 = int(r[0]) % L , int(r[1]) % L

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


# Compute the 2D Fourier transform of corr_r
S_q = np.fft.fft2(corr_r)

# Account for periodicity
S_q_periodic = np.zeros((L+1, L+1), dtype=S_q.dtype)
S_q_periodic[:L, :L] = S_q  
S_q_periodic[L, :] = S_q_periodic[0, :]    
S_q_periodic[:, L] = S_q_periodic[:, 0]    

#plot
plt.figure(figsize=(6,5))
plt.imshow(np.abs(S_q_periodic), origin='lower', cmap='viridis')
plt.colorbar(label='|S(q)|')
plt.xlabel('q_x')
plt.ylabel('q_y')
plt.title('Structure Factor S(q)')

# Set integer ticks for axes
plt.xticks([0, 1/2*L, L], ['0', 'π', '2π'])
plt.yticks([0, 1/2*L, L], ['0', 'π', '2π'])

if(save):
    plt.savefig(f'{folder}/Struct.png')
plt.show()

E_gs, ket_gs = nk.exact.lanczos_ed(hamiltonian, compute_eigenvectors=True)

print(f"Exact ground state energy = {E_gs[0]:.3f}")
E_exact = E_gs[0]/(L*L*4)
print(f"Exact ground state energy per site= {E_exact}")


e = np.abs((E_vs - E_exact)/E_exact)
print(f"Relative error = {e}")

import netket.experimental.observable as obs
import netket.operator as op

# I can compute infidelity like this only if I have 2 variational states
""" 

infidelity_op = obs.InfidelityOperator(target_state= vstate)
infidelity = vstate.expect(infidelity_op).mean.real
print(f"Infidelity = {infidelity}")

"""
#Compute fidelity between variational state and exact ground state in case of symmetrical degenaracy
# Create a state superposition of ground state and the all spins flipped ground state, 
# so it become symmetrical in the 2 degenerate gs, and I can apply the fidelity reliably
"""
vstate_array = vstate.to_array()

if J2==0: # degenerate ground state
    
    X_tensor = op.spin.sigmax(hilbert, 0)
    for i in range(1, lattice.n_nodes):
        X_tensor = X_tensor @ op.spin.sigmax(hilbert, i)

    ket_gs_flip = X_tensor @ ket_gs
    vstate_flip = X_tensor @ vstate_array

    vstate_sym = (vstate_array + vstate_flip) / (np.linalg.norm(vstate_array + vstate_flip))
    ket_gs_sym = (ket_gs + ket_gs_flip ) /  (np.linalg.norm(ket_gs + ket_gs_flip))

else:
    vstate_sym = vstate_array / np.linalg.norm(vstate_array)
    ket_gs_sym = ket_gs / np.linalg.norm(ket_gs)


overlap_val = vstate_sym.conj() @ ket_gs_sym
fidelity_val = np.abs(overlap_val) ** 2 / (np.vdot(vstate_sym, vstate_sym) * np.vdot(ket_gs_sym, ket_gs_sym))

print(f"Fidelity = {fidelity_val[0].real}")

"""

# Check the spin parity symmetry of the variational state
# Apply the spin parity operator to the variational state and check if it remain invariant,
# Checking that the fidelity is 1

"""
vstate_array = vstate.to_array()
X_tensor = op.spin.sigmax(hilbert, 0)
for i in range(1, lattice.n_nodes):
    X_tensor = X_tensor @ op.spin.sigmax(hilbert, i)
vstate_flip = X_tensor @ vstate_array
vstate_sym = (vstate_array + vstate_flip) / (np.linalg.norm(vstate_array + vstate_flip))
overlap_val = vstate_sym.conj() @ vstate_array
fidelity_val = np.abs(overlap_val) ** 2 / (np.vdot(vstate_sym, vstate_sym) * np.vdot(vstate_array, vstate_array))

print(f"Fidelity = {fidelity_val}")
"""


#construct the observable total magnetization
tot_magn = sum([sigmaz(hilbert, i) for i in lattice.nodes()])

tot_magn_vstate = vstate.expect(tot_magn).mean.real
print(f"Magnetization = {tot_magn_vstate}" )

#Variance
variance = log.data["Energy"]["Variance"][-1].real
print(f"Variance = {variance}")

#Vscore
v_score = L*L*variance/(E_vs*L*L*4)
print(f"V-score = {v_score}")


sys.stdout.close()