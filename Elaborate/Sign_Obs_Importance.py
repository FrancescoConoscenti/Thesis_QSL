from Elaborate.Sign_Obs_MCMC import MarshallSignObs
from Elaborate.Sign_Obs import sublattice_sites
from tenpy.networks.mps import MPS
from jax import numpy as jnp
import jax
import numpy as np
import netket as nk
import matplotlib.pyplot as plt

def importance_Sampling_DMRG(DMRG_vstate, n_samples, N_sites):
    
    ops_z = ['Sigmaz'] * N_sites  # or just 'Sigmaz' if measuring all sites
    samples = np.zeros((n_samples, N_sites), dtype=int)
    psi_DMRG_sampled = np.zeros(n_samples, dtype=np.complex128) 
    for n in range(n_samples):
        sigmas, psi_DMRG = DMRG_vstate.sample_measurements(first_site=0, last_site=N_sites-1, ops = ops_z, complex_amplitude=True)
        samples[n, :] = sigmas
        psi_DMRG_sampled[n] = psi_DMRG

    return samples, psi_DMRG_sampled

def sign_NQS_importance_Sampled(RBM_vstate, samples_DMRG, hi):

    SignObs = MarshallSignObs(hi)
    RBM_vstate.n_samples = samples_DMRG.shape[0]
    kernel = nk.vqs.get_local_kernel(RBM_vstate, SignObs)
    sigma_template, args_template = nk.vqs.get_local_kernel_arguments(RBM_vstate, SignObs)
    logpsi_vals = RBM_vstate.log_value(samples_DMRG)
    sign_RBM_samples = kernel(logpsi_vals, RBM_vstate.parameters, samples_DMRG, args_template)
    
    prob_samples = jnp.exp(2.0 * jnp.abs(logpsi_vals))
    expectation = jnp.sum(prob_samples * sign_RBM_samples.reshape(-1)) / jnp.sum(prob_samples) # weighted mean
    
    return expectation

def Sign_DMRG_samples(psi: MPS, sigma: np.ndarray):
    """
    psi: tenpy MPS
    sigma: ndarray shape (N_samples, N_sites) with entries +1/-1
    returns: signs (N_samples,), amplitudes (N_samples,) complex
    """
    N_samples, N_sites = sigma.shape
    A_sites = sublattice_sites(N_sites)        # array of A-sublattice indices
    S_A = 0.5 * (len(A_sites))     # scalar

    signs_out = np.zeros(N_samples, dtype=float)
    amps_out = np.zeros(N_samples, dtype=complex)

    for i in range(N_samples):

        sample = sigma[i]
        M_A = 0.5 * np.sum(sample[A_sites])

        # build product MPS of the sample and compute overlap amplitude <prod|psi>
        prod_labels = [ "up" if s == 1 else "down" for s in sample ]
        prod_mps = MPS.from_product_state(psi.sites, prod_labels, bc=psi.bc)
        amp = psi.overlap(prod_mps)   # complex scalar

        parity = (-1.0) ** (S_A - M_A)
        sample_sign = np.sign(np.real(amp)) * parity

        amps_out[i] = amp
        signs_out[i] = sample_sign

    return signs_out, amps_out

def sign_DMRG_importance_Sampled(DMRG_vstate, samples_DMRG):

    sign_DMRG_samples, psi_DMRG_sampled_1 = Sign_DMRG_samples(DMRG_vstate, samples_DMRG)
    prob_DMRG_samples = np.abs(psi_DMRG_sampled_1) **2
    sign_DMRG = np.sum(prob_DMRG_samples * sign_DMRG_samples.reshape(-1)) / np.sum(prob_DMRG_samples)
    return sign_DMRG


def sign_overlap_Importance_Sampled(RBM_vstate, DMRG_vstate, samples_DMRG, hi):

    SignObs = MarshallSignObs(hi)
    kernel = nk.vqs.get_local_kernel(RBM_vstate, SignObs)
    sigma_template, args_template = nk.vqs.get_local_kernel_arguments(RBM_vstate, SignObs)
    logpsi_vals = RBM_vstate.log_value(samples_DMRG)
    sign_RBM_samples = kernel(logpsi_vals, RBM_vstate.parameters, samples_DMRG, args_template)
    
    sign_DMRG_samples, psi_DMRG_sampled = Sign_DMRG_samples(DMRG_vstate, samples_DMRG)
    
    # --- DMRG RBM Sign Overlap on sampled configurations ---
    sign_product = sign_DMRG_samples.reshape(-1) * sign_RBM_samples.reshape(-1)
    Overlap_sign_samples = np.mean(sign_product)

    return Overlap_sign_samples

def amp_overlap_Importance_Sampled(RBM_vstate, psi_DMRG_sampled, samples_DMRG):

    logpsi_vals = RBM_vstate.log_value(samples_DMRG)
    psi_RBM_samples = jnp.exp(logpsi_vals)
    
    weights = np.abs(psi_RBM_samples) / np.abs(psi_DMRG_sampled)
    numerator = np.mean(weights)
    norm_RBM_sq = np.mean(weights**2)
    Overlap_amp_samples = numerator / np.sqrt(norm_RBM_sq)

    return Overlap_amp_samples    


############################################################################################à


def importance_Sampling_Exact(Exact_vstate, n_samples):
    
    Exact_vstate.n_samples = n_samples
    Exact_vstate.reset()
    
    samples = Exact_vstate.samples
    samples = samples.reshape(-1, samples.shape[-1])
    
    log_psi = Exact_vstate.log_value(samples)
    psi_exact_sampled = jnp.exp(log_psi)
    
    return np.array(samples), np.array(psi_exact_sampled)

def sign_overlap_Importance_Sampled_Exact(RBM_vstate, Exact_vstate, samples_Exact, hi):

    SignObs = MarshallSignObs(hi)
    
    # RBM Sign
    kernel = nk.vqs.get_local_kernel(RBM_vstate, SignObs)
    sigma_template, args_template = nk.vqs.get_local_kernel_arguments(RBM_vstate, SignObs)
    logpsi_vals = RBM_vstate.log_value(samples_Exact)
    sign_RBM_samples = kernel(logpsi_vals, RBM_vstate.parameters, samples_Exact, args_template)
    
    # Exact Sign
    kernel_exact = nk.vqs.get_local_kernel(Exact_vstate, SignObs)
    sigma_template_exact, args_template_exact = nk.vqs.get_local_kernel_arguments(Exact_vstate, SignObs)
    logpsi_vals_exact = Exact_vstate.log_value(samples_Exact)
    sign_Exact_samples = kernel_exact(logpsi_vals_exact, Exact_vstate.parameters, samples_Exact, args_template_exact)
    
    # Overlap
    sign_product = sign_Exact_samples.reshape(-1) * sign_RBM_samples.reshape(-1)
    Overlap_sign_samples = np.mean(sign_product)

    return Overlap_sign_samples

def amp_overlap_Importance_Sampled_Exact(RBM_vstate, psi_Exact_sampled, samples_Exact):

    logpsi_vals = RBM_vstate.log_value(samples_Exact)
    psi_RBM_samples = jnp.exp(logpsi_vals)
    
    weights = jnp.abs(psi_RBM_samples) / jnp.abs(psi_Exact_sampled)
    numerator = jnp.mean(weights)
    norm_RBM_sq = jnp.mean(weights**2)
    Overlap_amp_samples = numerator / jnp.sqrt(norm_RBM_sq)

    return Overlap_amp_samples

def fidelity_Importance_Sampled_Exact(RBM_vstate, psi_Exact_sampled, samples_Exact):
    
    logpsi_vals = RBM_vstate.log_value(samples_Exact)
    psi_RBM_samples = jnp.exp(logpsi_vals)
    
    ratios = psi_RBM_samples / psi_Exact_sampled
    overlap = jnp.mean(ratios)
    norm_sq = jnp.mean(jnp.abs(ratios)**2)
    
    fidelity = jnp.abs(overlap)**2 / norm_sq
    return fidelity