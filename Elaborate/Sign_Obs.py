import itertools
import math
import jax
import jax.numpy as jnp
import numpy as np
import netket as nk
import flax
import os

from tenpy.networks.mps import MPS


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

def extract_config(ket_gs, hi, n):
    """
    Returns the n-th most probable configuration and its coefficient
    from a wavefunction vector `ket_gs`.
    """

    probs = np.abs(ket_gs[:,0]) ** 2
    sorted_indices = np.argsort(probs)[::-1]
    idx_nth = sorted_indices[n]

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

    # Compute sign
    sign = jnp.sign(jnp.real(coeff)) * marshall_phase
    weight = jnp.abs(coeff) ** 2

    return sign, weight

def _marshal_sign_single_config(most_prob_config, vstate, hi):
    
    N_sites = hi.size
    A_sites = sublattice_sites(N_sites)

    M_A = 0.5 * jnp.sum(most_prob_config[A_sites])
    S_A = 0.5 * (N_sites // 2) 
    psi_most = jnp.exp(vstate.log_value(most_prob_config))
    sign_most = jnp.sign(jnp.real(psi_most)) * ((-1.0) ** (S_A - M_A))
    
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

    # Error product amplitude
    Overlap = jnp.sum(np.abs(psi_exact) * np.abs(psi_var))
    overlap_configs = np.abs(psi_exact) * np.abs(psi_var)

    return Overlap, overlap_configs


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


def _marshal_sign_full_hilbert(vstate, hi):
    N_sites = hi.size
    configs = hi.all_states()

    A_sites = sublattice_sites(N_sites)

    M_A = jnp.array(0.5 * jnp.sum(configs[:, A_sites], axis=1))
    S_A = jnp.ones_like(M_A) * 0.5 * (N_sites // 2) 
    psi = jnp.exp(vstate.log_value(configs)) # This can still be NaN/Inf if vstate.log_value is problematic
    signs = jnp.sign(jnp.real(psi)) * ((-1.0) ** (S_A - M_A))

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
    signs = jnp.sign(jnp.real(psi)) * marshall_phase

    weights = jnp.abs(psi) ** 2
    marshall_sign_expect = jnp.sum(weights * signs) / jnp.sum(weights)

    return marshall_sign_expect, signs



#get_marshal_sign_MCMC= jax.vmap(_marshal_sign_single_MCMC, in_axes=0, out_axes=0)
#get_marshal_sign_full_hilbert = jax.vmap(_marshal_sign_single_full_hilbert, in_axes=0, out_axes=0)
get_marshal_sign_full_hilbert = lambda vstate: _marshal_sign_full_hilbert


#########################################################################################


def Marshall_Sign_full_hilbert(vstate, folder_path, hi):
    
    num_files = len([name for name in os.listdir(f"{folder_path}/models") if os.path.isfile(os.path.join(f"{folder_path}/models", name))])
    sign = np.zeros(num_files)
    signs_vstate = np.zeros((num_files, hi.n_states))

    for i in range(num_files):
        with open(folder_path + f"/models/model_{i}.mpack", "rb") as f:
            data = f.read()
            try:
                vstate = flax.serialization.from_bytes(vstate, data)
            except KeyError:
                vstate.variables = flax.serialization.from_bytes(vstate.variables, data)

        sign[i], signs_vstate[i,:] = _marshal_sign_full_hilbert(vstate, hi) 

    return sign,  signs_vstate

def Marshall_Sign_full_hilbert_one(vstate, hi):

    sign, signs_vstate = _marshal_sign_full_hilbert(vstate, hi) 

    return sign,  signs_vstate


def Marshall_Sign_exact(ket_gs, hi):

    sign_expected, signs_exact = _marshal_sign_exact(ket_gs, hi)

    return sign_expected, signs_exact


def Sign_DMRG_full_hilbert(psi: MPS, hi):

    hilbert_states = hi.all_states()
    N_states = hi.n_states

    A_sites = sublattice_sites(hi.size)        # array of A-sublattice indices
    S_A = 0.5 * (len(A_sites))                 # scalar

    signs_out = np.zeros(N_states, dtype=float)
    amps_out = np.zeros(N_states, dtype=complex)

    for i in range(N_states):

        sample = hilbert_states[i]
        M_A = 0.5 * np.sum(sample[A_sites])

        # build product MPS of the sample and compute overlap amplitude <prod|psi>
        prod_labels = [ "up" if s == 1 else "down" for s in sample ]
        prod_mps = MPS.from_product_state(psi.sites, prod_labels, bc=psi.bc)
        amp = psi.overlap(prod_mps)   # complex scalar

        parity = (-1.0) ** (S_A - M_A)
        sample_sign = np.sign(np.real(amp)) * parity

        amps_out[i] = amp
        signs_out[i] = sample_sign
    
    prob_DMRG_full = np.abs(amps_out) **2
    sign_DMRG_full = np.sum(prob_DMRG_full * signs_out.reshape(-1)) / np.sum(prob_DMRG_full)

    return signs_out, amps_out, sign_DMRG_full

# Fidelity vstate, exact state
def Fidelity(vstate, ket_gs):
    vstate_array = vstate.to_array()
    overlap_val = vstate_array.conj() @ ket_gs
    fidelity_val = np.abs(overlap_val) ** 2 / (np.vdot(vstate_array, vstate_array) * np.vdot(ket_gs, ket_gs))
    #print(f"Fidelity <vstate|exact> = {fidelity_val}")
    return np.real(fidelity_val)


def Fidelity_iteration(vstate, ket_gs, folder_path):

    num_files = len([name for name in os.listdir(f"{folder_path}/models") if os.path.isfile(os.path.join(f"{folder_path}/models", name))])
    fidelity = np.zeros(num_files)

    for i in range(num_files):
        with open(folder_path + f"/models/model_{i}.mpack", "rb") as f:
            data = f.read()
            try:
                vstate = flax.serialization.from_bytes(vstate, data)
            except KeyError:
                vstate.variables = flax.serialization.from_bytes(vstate.variables, data)
        
        fidelity[i] = Fidelity(vstate, ket_gs)
       
    return fidelity

def Marshall_Sign_and_Weights_single_config(ket_gs, vstate, folder_path, L, hi, number_states):
    
    num_files = len([name for name in os.listdir(f"{folder_path}/models") if os.path.isfile(os.path.join(f"{folder_path}/models", name))])
    
    sign_config_vstate = np.zeros((number_states, num_files))  # Shape: (number_states, num_files)
    weight_vstate = np.zeros((number_states, num_files))       # Shape: (number_states, num_files)

    configs = []
    weight_config_exact = np.zeros(number_states)
    
    # Extract all configurations and exact weights
    for i in range(number_states):
        config, coeff = extract_config(ket_gs, hi, i)
        configs.append(config)
        
        sign_config_exact, weight_config_exact[i] = _marshal_sign_exact_single_config(config, coeff, hi)

    configs = np.array(configs)

    # Calculate variational weights and signs for each model
    for i in range(num_files):
        with open(folder_path + f"/models/model_{i}.mpack", "rb") as f:
            data = f.read()
            try:
                vstate = flax.serialization.from_bytes(vstate, data)
            except KeyError:
                vstate.variables = flax.serialization.from_bytes(vstate.variables, data)

        # Calculate for each state
        for j in range(number_states):
            sign_config_vstate[j, i], weight_vstate[j, i] = _marshal_sign_single_config(configs[j], vstate, hi)

    return configs, sign_config_vstate, weight_config_exact, weight_vstate


def Amplitude_overlap_configs(ket_gs, vstate, folder_path, hi):

    num_files = len([name for name in os.listdir(f"{folder_path}/models") if os.path.isfile(os.path.join(f"{folder_path}/models", name))])
    Overlap = np.zeros(num_files)

    for i in range(num_files):
        with open(folder_path + f"/models/model_{i}.mpack", "rb") as f:
            data = f.read()
            try:
                vstate = flax.serialization.from_bytes(vstate, data)
            except KeyError:
                vstate.variables = flax.serialization.from_bytes(vstate.variables, data)

        Overlap[i], _ = Amp_overlap_configs(ket_gs, vstate, hi)

    return Overlap


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
    # We sum along axis=1 to get a result per model if signs_vstate is 2D.
    # If it's 1D (for a single model), we sum over the only axis (0).
    sum_axis = 1 if signs_vstate.ndim == 2 else 0
    product = weights * signs_vstate * signs_exact
    
    sign_overlap = np.abs(np.sum(product, axis=sum_axis))
    sign_overlap_configs = signs_vstate * signs_exact

    return sign_overlap, sign_overlap_configs
