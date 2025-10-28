import itertools
import math
import jax
import jax.numpy as jnp
import numpy as np
import netket as nk
import flax
import os
from Elaborate.Statistics.Error_Stat import Fidelity

def sublattice_sites(N_sites):
    """
    Returns the indices of the sites belonging to sublattice A, assuming a square lattice.
    """
    L = np.sqrt(N_sites)

    A_sites = []
    for idx in range(N_sites):
        x = idx % L
        y = idx // L
        if (x + y) % 2 == 0:
            A_sites.append(idx)
    A_sites = jnp.array(A_sites)
    return A_sites

def balanced_combinations_numpy(L):
    """
    Generates all possible combinations of a 1D array of length L 
    with an equal number of 1s and -1s.
    Returns a numpy array of these combinations.
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

import numpy as np

def extract_config(ket_gs, hi, n):
    """
    Returns the n-th most probable configuration and its coefficient
    from a wavefunction vector `ket_gs`.
    """

    probs = np.abs(ket_gs[:,0]) ** 2
    sorted_indices = np.argsort(probs)[::-1]
    idx_nth = sorted_indices[n - 1]

    nth_most_prob_coeff = ket_gs[idx_nth]
    nth_most_prob_state = hi.all_states()[idx_nth]

    return nth_most_prob_state, nth_most_prob_coeff


def _marshal_sign_exact_single_config(most_prob_config, coeff, hi):

    N_sites = hi.size
    A_sites = sublattice_sites(N_sites)
    coeff=coeff[0]

    # Magnetization on A sublattice
    M_A = jnp.array(0.5 * jnp.sum(most_prob_config[A_sites]))
    S_A = jnp.ones_like(M_A) * 0.5 * (N_sites // 2)
    marshall_phase = (-1.0) ** (S_A - M_A)

    # Compute expectation value full Hilbert space
    sign = jnp.real((coeff * marshall_phase) / jnp.abs(coeff))
    weight = jnp.abs(coeff) ** 2

    return sign, weight

def _marshal_sign_single_config(most_prob_config, vstate, hi):
    
    N_sites = hi.size
    A_sites = sublattice_sites(N_sites)

    M_A = 0.5 * jnp.sum(most_prob_config[A_sites])
    S_A = 0.5 * (N_sites // 2) 
    psi_most = jnp.exp(vstate.log_value(most_prob_config))
    sign_most = jnp.real(psi_most) * ((-1.0) ** (S_A - M_A)) / jnp.abs(jnp.real(psi_most))
    
    configs = hi.all_states()
    psi = jnp.exp(vstate.log_value(configs))
    non_norm_weights = jnp.abs(psi) ** 2
    Z = jnp.sum(non_norm_weights)
    psi_most_weight = jnp.abs(psi_most) ** 2 / Z

    return  sign_most, psi_most_weight

def Amp_overlap_configs(ket_gs, vstate, hi):
    
    configs = hi.all_states()

    # Evaluate both wavefunctions
    psi_var = jnp.exp(vstate.log_value(configs))
    psi_exact = ket_gs[:,0]

    # Normalize both wavefunctions
    psi_var = psi_var / jnp.sqrt(jnp.sum(jnp.abs(psi_var) ** 2))
    psi_exact = psi_exact / jnp.sqrt(jnp.sum(jnp.abs(psi_exact) ** 2))

    """
    # Error difference amplitude
    weights = jnp.abs(psi_exact) ** 2
    diff = jnp.abs(jnp.abs(psi_exact)**2 - jnp.abs(psi_var)**2)
    Error = jnp.sum(weights * diff)
    """

    # Error product amplitude
    Overlap = jnp.sum(np.abs(psi_exact) * np.abs(psi_var))


    return Overlap

#################################################################################################################################################

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
def _marshal_sign_MCMC(sigma, vstate):
    #for samples in MCMC the dimension 1 is N, for samples in full hilbert the dimension 0 is N
    N_sites = sigma.shape[1]
    A_sites = sublattice_sites(N_sites)

    #M_A = jnp.array([sum(sample[::2]) for sample in sigma]) #jnp.sum(0.5 * sigma[A_sites]) # Magn on A sublattice
    M_A = jnp.array(0.5 * jnp.sum(sigma[:, A_sites], axis=1))
    S_A = jnp.ones_like(M_A) * 0.5 * (N_sites // 2) # sum of the spins in the sublattice
    
    log_psi = vstate.log_value(sigma) #log coefficient of wf associated with sample sigma
    psi = jnp.exp(log_psi)
    sign = jnp.real((psi * ((-1.0) ** (S_A - M_A))) / jnp.abs(psi))

    return  sign


def _marshal_sign_full_hilbert(vstate, hi):
    N_sites = hi.size
    configs = hi.all_states()

    A_sites = sublattice_sites(N_sites)

    M_A = jnp.array(0.5 * jnp.sum(configs[:, A_sites], axis=1))
    S_A = jnp.ones_like(M_A) * 0.5 * (N_sites // 2) 
    psi = jnp.exp(vstate.log_value(configs))
    signs = jnp.real(psi) * ((-1.0) ** (S_A - M_A)) / jnp.abs(jnp.real(psi))

    weights = jnp.abs(psi) ** 2
    sign_expect = jnp.sum(weights * signs) / jnp.sum(weights) 

    return  sign_expect, signs


def _marshal_sign_exact(ket_gs, hi):
    N_sites = hi.size
    configs = hi.all_states()

    A_sites = sublattice_sites(N_sites)

    # Magnetization on A sublattice
    M_A = jnp.array(0.5 * jnp.sum(configs[:, A_sites], axis=1))
    S_A = jnp.ones_like(M_A) * 0.5 * (N_sites // 2)
    marshall_phase = (-1.0) ** (S_A - M_A)

    # Compute expectation value full Hilbert space
    psi = ket_gs[:,0]
    signs = jnp.real((psi * marshall_phase) / jnp.abs(psi))

    weights = jnp.abs(psi) ** 2
    marshall_sign_expect = jnp.sum(weights * signs) / jnp.sum(weights)

    return marshall_sign_expect, signs



#get_marshal_sign_MCMC= jax.vmap(_marshal_sign_single_MCMC, in_axes=0, out_axes=0)
#get_marshal_sign_full_hilbert = jax.vmap(_marshal_sign_single_full_hilbert, in_axes=0, out_axes=0)
get_marshal_sign_MCMC = lambda vstate: jax.vmap(lambda sigma: _marshal_sign_MCMC(sigma, vstate), in_axes=0, out_axes=0)
get_marshal_sign_full_hilbert = lambda vstate: _marshal_sign_full_hilbert

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

def Marshall_Sign_MCMC(marshall_op, vstate, folder_path, n_samples):
    
    number_models = len([name for name in os.listdir(f"{folder_path}/models") if os.path.isfile(os.path.join(f"{folder_path}/models", name))])
    sign = np.zeros(number_models)
    vstate.n_samples = n_samples

    for i in range(0, number_models):
        with open(folder_path + f"/models/model_{i} .mpack", "rb") as f:
            vstate.variables = flax.serialization.from_bytes(vstate.variables, f.read())

        # Compute expectation value MCMC
        exp_val = vstate.expect(marshall_op)
        sign[i] = exp_val.mean

    return sign


def Marshall_Sign_full_hilbert(vstate, folder_path, hi):
    
    number_models = len([name for name in os.listdir(f"{folder_path}/models") if os.path.isfile(os.path.join(f"{folder_path}/models", name))])
    sign = np.zeros(number_models)
    signs_vstate = np.zeros((number_models, hi.n_states))

    for i in range(0, number_models):
        with open(folder_path + f"/models/model_{i} .mpack", "rb") as f:
            vstate.variables = flax.serialization.from_bytes(vstate.variables, f.read())

        sign[i], signs_vstate[i,:] = _marshal_sign_full_hilbert(vstate, hi) 

    return sign,  signs_vstate


def Marshall_Sign_exact(ket_gs, hi):

    sign_expected, signs_exact = _marshal_sign_exact(ket_gs, hi)

    return sign_expected, signs_exact


def Fidelity_iteration(vstate, ket_gs, folder_path):

    number_models = len([name for name in os.listdir(f"{folder_path}/models") if os.path.isfile(os.path.join(f"{folder_path}/models", name))])
    fidelity = np.zeros(number_models)

    for i in range(0, number_models):
        with open(folder_path + f"/models/model_{i} .mpack", "rb") as f:
            vstate.variables = flax.serialization.from_bytes(vstate.variables, f.read())
        
        fidelity[i] = Fidelity(vstate, ket_gs)
       
    return fidelity

def Marshall_Sign_and_Weights_single_config(ket_gs, vstate, folder_path, L, hi, number_states):
    
    number_models = len([name for name in os.listdir(f"{folder_path}/models") if os.path.isfile(os.path.join(f"{folder_path}/models", name))])
    
    sign_config_vstate = np.zeros((number_states, number_models))  # Shape: (number_states, number_models)
    weight_vstate = np.zeros((number_states, number_models))       # Shape: (number_states, number_models)

    configs = []
    weight_config_exact = np.zeros(number_states)
    
    # Extract all configurations and exact weights
    for i in range(number_states):
        config, coeff = extract_config(ket_gs, hi, i*2 + 1)
        configs.append(config)
        
        sign_config_exact, weight_config_exact[i] = _marshal_sign_exact_single_config(config, coeff, hi)

    configs = np.array(configs)

    # Calculate variational weights and signs for each model
    for i in range(number_models):
        with open(folder_path + f"/models/model_{i} .mpack", "rb") as f:
            vstate.variables = flax.serialization.from_bytes(vstate.variables, f.read())

        # Calculate for each state
        for j in range(number_states):
            sign_config_vstate[j, i], weight_vstate[j, i] = _marshal_sign_single_config(configs[j], vstate, hi)

    return configs, sign_config_vstate, weight_config_exact, weight_vstate


def Amplitude_overlap_configs(ket_gs, vstate, folder_path, hi):

    number_models = len([name for name in os.listdir(f"{folder_path}/models") if os.path.isfile(os.path.join(f"{folder_path}/models", name))])
    Overlap = np.zeros(number_models)

    for i in range(0, number_models):
        with open(folder_path + f"/models/model_{i} .mpack", "rb") as f:
            vstate.variables = flax.serialization.from_bytes(vstate.variables, f.read())

        Overlap[i] = Amp_overlap_configs(ket_gs, vstate, hi)

    return Overlap

def Sign_difference(sign_vstate, sign_exact):
    """
    Computes the difference between the variational sign and the exact sign.
    sign_vstate: total sign from variational state at each model step (array of shape (number_models,))
    sign_exact: total sign from exact state (scalar)
    """

    array = np.ones_like(sign_vstate) #
    sign_exact_array = array * sign_exact

    sign_error = np.abs(np.abs(sign_vstate) - np.abs(sign_exact_array))

    return sign_error 


def Sign_overlap(ket_gs, signs_vstate, signs_exact):
    """
    Computes the sign overlap between the variational state and the exact state.
    signs_vstate: array of variational signs for each configuration for each model step (array of shape (number_models, N_states))
    signs_exact: array of exact signs for each configuration (array of shape (N_states,))
    Returns the sign overlap (number_models,)
    """

    psi = ket_gs[:, 0]
    weights = jnp.abs(psi) ** 2

    # weights has shape (N_states,)
    # signs_vstate has shape (number_models, N_states)
    # signs_exact has shape (N_states,)
    # The product will be broadcast correctly.
    # We sum along axis=1 to get a result per model.
    sign_overlap = np.abs(np.sum(weights * signs_vstate * signs_exact, axis=1))

    return sign_overlap 