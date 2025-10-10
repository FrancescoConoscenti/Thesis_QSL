import itertools
import math
import jax
import jax.numpy as jnp
import numpy as np
import netket as nk
import flax
import os
from Elaborate.Error_Stat import Fidelity


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
def _marshal_sign_single_MCMC(sigma, vstate):
    #for samples in MCMC the dimension 1 is N, for samples in full hilbert the dimension 0 is N
    N_sites = sigma.shape[1]

    #A_sites = jnp.arange(0, N_sites, 2) # A sublattice
    M_A = jnp.array([sum(sample[::2]) for sample in sigma]) #jnp.sum(0.5 * sigma[A_sites]) # Magn on A sublattice
    S_A = jnp.ones_like(M_A) * 0.5 * (N_sites // 2) # sum of the spins in the sublattice
    
    log_psi = vstate.log_value(sigma) #log coefficient of wf associated with sample sigma
    psi = jnp.exp(log_psi)
    #print("psi = ",psi)
    sign = jnp.real((psi * ((-1.0) ** (S_A - M_A))) / jnp.abs(psi))
    #print("sign MCMC = ",sign.val)
    return  sign #jnp.rint(S_A - M_A)"""


def _marshal_sign_single_full_hilbert(sigma, vstate):

    N_sites = sigma.shape[-1]
    #M_A = 0.5 * jnp.sum(sigma[..., ::2], axis=-1)
    M_A = jnp.array([0.5*sum(sample[::2]) for sample in sigma]) #jnp.sum(0.5 * sigma_test[A_sites]) # Magn on A sublattice
    S_A = 0.5 * (N_sites // 2)
    psi = jnp.exp(vstate.log_value(sigma))
    sign = jnp.real((psi * ((-1.0) ** (S_A - M_A))) / jnp.abs(psi))

    return  sign


#get_marshal_sign_MCMC= jax.vmap(_marshal_sign_single_MCMC, in_axes=0, out_axes=0)
#get_marshal_sign_full_hilbert = jax.vmap(_marshal_sign_single_full_hilbert, in_axes=0, out_axes=0)
get_marshal_sign_MCMC = lambda vstate: jax.vmap(lambda sigma: _marshal_sign_single_MCMC(sigma, vstate), in_axes=0, out_axes=0)
get_marshal_sign_full_hilbert = lambda vstate: _marshal_sign_single_full_hilbert

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
    sign = get_marshal_sign_MCMC(vstate)(sigma)
    # return product: sign * amplitude
    extra_args = sign #* psi
    return sigma, extra_args


#########################################################################################

def Marshall_Sign(marshall_op, vstate, folder_path, n_samples, L):
    
    number_models = len([name for name in os.listdir(f"{folder_path}/models") if os.path.isfile(os.path.join(f"{folder_path}/models", name))])
    sign1 = np.zeros(number_models)
    sign2 = np.zeros(number_models)
    sign3 = np.zeros(number_models)

    vstate.n_samples = n_samples

    for i in range(0, number_models):
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
        signs = _marshal_sign_single_full_hilbert( configs, vstate) 
        #print("sign_full :", signs)
        # reweighted expectation
        sign2[i] = jnp.sum(weights * signs) / jnp.sum(weights)
        #not weighted expectation
        sign3[i] = np.mean(signs)
        
    return sign1, sign2

def Marshall_Sign_Fidelity(marshall_op, ket_gs, vstate, folder_path, L):
    
    number_models = len([name for name in os.listdir(f"{folder_path}/models") if os.path.isfile(os.path.join(f"{folder_path}/models", name))])
    sign2 = np.zeros(number_models)
    fidelity = np.zeros(number_models)

    for i in range(0, number_models):
        
        with open(folder_path + f"/models/model_{i} .mpack", "rb") as f:
            vstate.variables = flax.serialization.from_bytes(vstate.variables, f.read())

        # Compute expectation value full Hilbert space
        configs = balanced_combinations_numpy(L*L)
        logpsi = vstate.log_value(configs)
        psi = jnp.exp(logpsi)
        weights = jnp.abs(psi) ** 2
        signs = _marshal_sign_single_full_hilbert( configs, vstate) 
        # reweighted expectation
        sign2[i] = jnp.sum(weights * signs) / jnp.sum(weights)

        fidelity[i] = Fidelity(vstate, ket_gs)

    return sign2, fidelity