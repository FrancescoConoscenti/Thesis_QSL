#%%
import jax
import numpy as np
import jax.numpy as jnp
import netket as nk
from netket.operator import AbstractOperator
from functools import partial  # partial(sum, axis=1)(x) == sum(x, axis=1)
import flax
import os
import matplotlib.pyplot as plt
import itertools

import jax
import jax.numpy as jnp
import numpy as np
import netket as nk
from netket.operator import AbstractOperator
from netket.operator._discrete_operator_jax import DiscreteJaxOperator
from netket.operator._discrete_operator import DiscreteOperator

import netket as nk
import jax
import jax.numpy as jnp

from HFDS_Heisenberg.HFDS_model_spin import HiddenFermion

L       = 4
n_elecs = L*L # L*L should be half filling
n_dim = 2

dtype_ = jnp.float64
MFinitialization = "Fermi"
bounds  = "PBC"
symmetry = True  #True or False

#Varaitional state param
n_hid_ferm       = 4
features         = 32 #hidden units per layer
hid_layers       = 1

n_dim            = 2


def balanced_combinations_numpy(L):
    """
    Generate all arrays as numpy arrays
    """
    if L % 2 != 0:
        raise ValueError("L must be even for equal numbers of 1 and -1")
    
    positions_to_change = L // 2
    combinations_list = []
    
    for positions in itertools.combinations(range(L), positions_to_change):
        arr = np.full(L, -1, dtype=int)
        arr[list(positions)] = 1
        combinations_list.append(arr)
    
    return np.array(combinations_list)


"""def test_phase_factor_batch():
    # Create batch of test configurations: (4 samples, 4 sites)
    test_configs = jnp.array([
        [1, 1, 1, 1],      # All up - should give +1
        [-1, -1, -1, -1],  # All down - should give +1  
        [1, -1, -1, 1],    # Alternating - should give -1
        [1, -1, 1, -1]     # Mixed - should give +1
    ])
    
    expected_phases = jnp.array([1.0, 1.0, -1.0, 1.0])
    
    # Your function should handle batches
    computed_phases = get_marshal_sign(test_configs)
    
    print("Test configurations:")
    for i, config in enumerate(test_configs):
        print(f"  Config {i}: {config} -> expected {expected_phases[i]}, got {computed_phases[i]}")
    
    assert jnp.allclose(computed_phases, expected_phases), f"Phase mismatch: {computed_phases} vs {expected_phases}"
    print("✓ Batch manual tests passed!")


# --- 5. Usage example ---
hilbert = nk.hilbert.Spin(0.5, 16)
vs = nk.vqs.MCState(nk.sampler.MetropolisLocal(hilbert), nk.models.RBM(alpha=1))
marshall_op = MarshallSignOperator(hilbert)

exp_val = vs.expect(marshall_op)
print("⟨Marshall Sign Operator⟩ =", exp_val.mean)


#test
test_phase_factor_batch()"""


class MarshallSignOperator(nk.operator.AbstractOperator):
    def __init__(self, hilbert):
        super().__init__(hilbert)
    @property
    def dtype(self):
        return float
    @property
    def is_hermitian(self):
        return True

# vectorized function to compute Marshall sign per sample
def _marshal_sign_single(sigma):
    N_sites = sigma.shape[0]
    A_sites = jnp.arange(0, N_sites, 2) # A sublattice
    M_A = jnp.sum(0.5 * sigma[A_sites]) # Magn on A sublattice
    S_A = 0.5 * (N_sites // 2) # sum of the spins in the sublattice

    log_psi = vstate.log_value(sigma) #log coefficient of wf associated with sample sigma
    psi = jnp.exp(log_psi)

    return  (psi * ((-1.0) ** jnp.rint(S_A - M_A))) / jnp.abs(psi)

get_marshal_sign = jax.vmap(_marshal_sign_single, in_axes=0, out_axes=0)

# local estimator
def e_loc(logpsi, pars, sigma, extra_args, *, chunk_size=None):
    return extra_args.astype(float)

# with chunk_size (HFDS/clustered sampler)
@nk.vqs.get_local_kernel.dispatch
def get_local_kernel(vstate: nk.vqs.MCState, op: MarshallSignOperator, chunk_size: int):
    return e_loc

@nk.vqs.get_local_kernel_arguments.dispatch
def get_local_kernel_arguments(vstate: nk.vqs.MCState, op: MarshallSignOperator):
    sigma = vstate.samples
    # wavefunction amplitudes at sampled σ
    #log_psi = vstate.log_value(sigma)
    #psi = jnp.exp(log_psi)
    # compute marshall sign for each sample
    sign = get_marshal_sign(sigma)
    # return product: sign * amplitude
    extra_args = sign #* psi
    return sigma, extra_args


###################################################################################################

MCMC=True

n_samples        = 64
n_chains         =  n_samples//2
cs               =  n_samples

# --- 5. Usage example ---
lattice = nk.graph.Hypercube(length=L, n_dim=n_dim, pbc=True, max_neighbor_order=2)
hi = nk.hilbert.Spin(s=1 / 2, N=lattice.n_nodes, total_sz=0) 

model = HiddenFermion(n_elecs=n_elecs,network="FFNN",n_hid=n_hid_ferm,Lx=L,Ly=L,layers=hid_layers,features=features,MFinit=MFinitialization,hilbert=hi,stop_grad_mf=False,stop_grad_lower_block=False,bounds=bounds,parity=symmetry,dtype=dtype_)

# ------------- define Hamiltonian ------------------------
    # Heisenberg J1-J2 spin ha
ha = nk.operator.Heisenberg(hilbert=hi, graph=lattice, J=[1.0, 0.0], sign_rule=[False, False]).to_jax_operator()  # No Marshall sign rule"""
# ---------- define sampler ------------------------
sampler = nk.sampler.MetropolisExchange(hilbert=hi,graph=lattice,d_max=2,n_chains=n_chains,sweep_size=lattice.n_nodes,)
vstate = nk.vqs.MCState(sampler, model, n_samples=n_samples, chunk_size=cs, n_discard_per_chain=128)

# Instantiate operator
marshall_op = MarshallSignOperator(hi)

#calculate sign obs of a model from training
folder_path="/scratch/f/F.Conoscenti/Thesis_QSL/HFDS_Heisenberg/plot/J1J2/spin/layers1_hidd4_feat32_sample1024_lr0.02_iter400_symmTrue_Hannah/J2=0.5_L=4"

# --- Loading ---
number_models = len([name for name in os.listdir(f"{folder_path}/models") if os.path.isfile(os.path.join(f"{folder_path}/models", name))])
sign1 = np.zeros(number_models)
sign2 = np.zeros(number_models)
x = np.arange(number_models)*50

for i in range(number_models):
    #print(f"model_{i}")
    with open(folder_path + f"/models/model_{i} .mpack", "rb") as f:
        vstate.variables = flax.serialization.from_bytes(vstate.variables, f.read())

    # Compute expectation value MCMC
    exp_val = vstate.expect(marshall_op)
    sign1[i] = exp_val.mean
  

    # Compute expectation value full Hilbert space
    configs = balanced_combinations_numpy(L*L)
    logpsi = vstate.log_value(configs)
    psi = jnp.exp(logpsi)
    weights = jnp.abs(psi) ** 2
    signs = np.real(get_marshal_sign(configs))
    # reweighted expectation
    sign2[i] = jnp.sum(weights * signs) / jnp.sum(weights)
    #not weighted expectation
    #sign[i] = np.mean(signs)
    
    print("⟨Marshall Sign Operator⟩ =", sign1[i])
    print("⟨Marshall Sign Operator⟩ =", sign2[i])

#%%
# Create the plot
plt.figure(figsize=(10, 6))

# Plot both lines with dots
plt.plot(x, sign1, marker='o', label='MCMC sampled sign', markersize=8, linewidth=2)
plt.plot(x, sign2, marker='o', label='Full Hilbert sampled sign', markersize=8, linewidth=2)

# Customize the graph
plt.title('Two Lines with Data Points', fontsize=14)
plt.ylabel('Sign')
plt.xlabel('Iterations')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{folder_path}/Sign.png")
plt.show()