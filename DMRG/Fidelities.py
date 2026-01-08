from tenpy.models.model import CouplingMPOModel
from tenpy.models.lattice import Square
from tenpy.networks.site import SpinSite
from tenpy.networks.mps import MPS
from tenpy.algorithms import dmrg
from tenpy.tools import hdf5_io
from numpy import linspace
from random import shuffle
import numpy as np
from tenpy.networks.site import SpinHalfSite
from tenpy.algorithms.exact_diag import get_full_wavefunction
from DMRG.plot.Plotting import *
from DMRG.Observable.Corr_Struct import Correlations_Structure_Factor

from netket.experimental.driver import VMC_SR
from DMRG.DMRG import *
from DMRG.plot.Plotting import *


import os
import numpy as np


import matplotlib.pyplot as plt


from jax import numpy as jnp
import netket as nk
import flax

def fidelity_RBM_exact(vstate, ket_gs):
    # Ensure vstate has access to the full Hilbert space for array conversion
    vstate_array = vstate.to_array(normalize=True)

    overlap = np.vdot(vstate_array, ket_gs)
    fidelity = np.abs(overlap)**2
    return fidelity

def fidelity_DMRG_exact(DMRG_vstate, exact_ket):

    dmrg_array = get_full_wavefunction(DMRG_vstate, undo_sort_charge=True)

    overlap = np.vdot(dmrg_array, exact_ket)
    fidelity = np.abs(overlap)**2
    return fidelity

def Fidelity_exact(RBM_vstate, DMRG_vstate):

    # undo_sort_charge=True to ensure that the basis state ordering of the TenPy wavefunction
    #  matches the standard lexicographical ordering used by NetKet's `to_array()` method
    dmrg_array = get_full_wavefunction(DMRG_vstate, undo_sort_charge=True)
    RBM_array = RBM_vstate.to_array(normalize=True)

    overlap = np.vdot(RBM_array, dmrg_array)
    fidelity = np.abs(overlap)**2

    return fidelity


######################################################################################

def Fidelity_sampled(psi_DMRG_sampled, psi_RBM_sampled):

    """
    The term np.mean(np.abs(psi_RBM_sampled)**2) is an estimate of Σ_s |ψ_DMRG(s)|^2 * |ψ_RBM(s)|^2.
    This is not the squared norm ⟨ψ_RBM | ψ_RBM⟩. 
    It's an overlap of probability distributions. 
    Therefore, psi_RBM_sampled_norm is not correctly normalizing ψ_RBM to unit norm.
    """
    psi_DMRG_sampled_norm = psi_DMRG_sampled / np.sqrt(np.mean(np.abs(psi_DMRG_sampled)**2))
    psi_RBM_sampled_norm = psi_RBM_sampled / np.sqrt(np.mean(np.abs(psi_RBM_sampled)**2))

    # 1. Calculate the element-wise ratio X(sigma) = psi_RBM(sigma) / psi_DMRG(sigma)
    ratio = psi_RBM_sampled / psi_DMRG_sampled
    #ratio = psi_RBM_sampled / psi_DMRG_sampled

    
    # 2. Estimate the overlap <psi_DMRG | psi_RBM>
    # This is the mean of the ratio: E[X]
    overlap_est = np.mean(ratio)
    # The numerator of the fidelity is the squared magnitude of the overlap.
    numerator_est = np.abs(overlap_est)**2 
    
    # 3. Estimate the RBM Norm Squared (Denominator): <psi_RBM | psi_RBM>
    # This is the mean of the squared magnitude of the ratio: E[|X|^2]
    denominator_est = np.mean(np.abs(ratio)**2)
    
    # 4. Calculate Fidelity: |E[X]|^2 / E[|X|^2].
    fidelity = numerator_est / denominator_est

    return fidelity

def Sign_Overlap_sampled(psi_DMRG_sampled, psi_RBM_sampled):

    # Extract signs (phases)
    # We add a small epsilon to avoid division by zero, although psi_DMRG_sampled should be non-zero.
    epsilon = 1e-18
    sign_DMRG = psi_DMRG_sampled / (np.abs(psi_DMRG_sampled) + epsilon)
    sign_RBM = psi_RBM_sampled / (np.abs(psi_RBM_sampled) + epsilon)

    # 1. Calculate the element-wise ratio of signs X(sigma) = sgn_RBM(sigma) / sgn_DMRG(sigma)
    ratio = sign_RBM / sign_DMRG
    
    # 2. Estimate the overlap <sgn_DMRG | sgn_RBM>
    # This is the mean of the ratio: E[X]. Since samples are drawn from |psi_DMRG|^2,
    # this corresponds to the overlap weighted by the DMRG probability.
    overlap_est = np.mean(ratio)
    
    return np.abs(overlap_est)

def Amplitude_Overlap_sampled(psi_DMRG_sampled, psi_RBM_sampled):

    abs_DMRG = np.abs(psi_DMRG_sampled)
    abs_RBM = np.abs(psi_RBM_sampled)

    # 1. Calculate the element-wise ratio X(sigma) = |psi_RBM(sigma)| / |psi_DMRG(sigma)|
    ratio = abs_RBM / (abs_DMRG)
    
    # 2. Estimate the overlap <|psi_DMRG| | |psi_RBM|>
    # This is the mean of the ratio: E[X]. Since samples are drawn from |psi_DMRG|^2,
    # this corresponds to the overlap weighted by the DMRG probability.
    overlap_est = np.mean(ratio)
    
    # 3. Estimate the RBM Norm Squared (Denominator): <|psi_RBM| | |psi_RBM|>
    # This is the mean of the squared magnitude of the ratio: E[|X|^2]
    denominator_est = np.mean(ratio**2)
    
    # 4. Calculate Amplitude Overlap: E[X] / sqrt(E[|X|^2])
    amplitude_overlap = overlap_est / np.sqrt(denominator_est)

    return amplitude_overlap

############################################################################################################################
















#############################################################################################################################

def Fidelity_sample_distr_vs_DMRG(P_DMRG_sampled, samples_dmrg_distr):
    overlap = np.mean(np.vdot(P_DMRG_sampled, samples_dmrg_distr))
    fidelity = np.abs(overlap)**2
    return fidelity


def ravel_configs_to_indices(configs_01, local_dims):
        if configs_01.ndim == 1:
            configs_01 = configs_01.reshape(1, -1)
        tuples = tuple(configs_01[:, i] for i in range(configs_01.shape[1]))
        return np.ravel_multi_index(tuples, dims=tuple(local_dims), order='C')


def Fidelity_RBM_sampled_vs_DMRG_exact(psi_RBM_sampled, samples_dmrg, dmrg_array, local_dims):

    # DMRG full-wavefunction
    # --- flat indices in full Hilbert space for each unique config ---
    flat_indices_unique = ravel_configs_to_indices(samples_dmrg, local_dims)  # shape (n_unique,)
    psi_DMRG_exact = dmrg_array[flat_indices_unique]
    psi_DMRG_exact = psi_DMRG_exact / (psi_DMRG_exact.sum() + 1e-16)


    # 1. Calculate the element-wise ratio X(sigma) = psi_RBM(sigma) / psi_DMRG(sigma)
    ratio = psi_RBM_sampled / psi_DMRG_exact
    
    # 2. Estimate the overlap <psi_DMRG | psi_RBM>
    # This is the mean of the ratio: E[X]
    overlap_est = np.mean(ratio)
    # The numerator of the fidelity is the squared magnitude of the overlap.
    numerator_est = np.abs(overlap_est)**2 
    
    # 3. Estimate the RBM Norm Squared (Denominator): <psi_RBM | psi_RBM>
    # This is the mean of the squared magnitude of the ratio: E[|X|^2]
    denominator_est = np.mean(np.abs(ratio)**2)
    
    # 4. Calculate Fidelity: |E[X]|^2 / E[|X|^2].
    fidelity = numerator_est / denominator_est

    return fidelity


def fidelity_DMRG_sampled_vs_full(psi_DMRG_sampled, samples_dmrg, dmrg_array, local_dims):
    """
    Calculates the fidelity between the sampled DMRG amplitudes and the full DMRG wavefunction.
    This serves as a sanity check for the importance sampling fidelity calculation, and should be close to 1.

    Args:
        psi_DMRG_sampled (np.ndarray): Amplitudes from the DMRG sampling process (proposal distribution).
        samples_dmrg (np.ndarray): The configurations {0,1} that were sampled.
        dmrg_array (np.ndarray): The full DMRG wavefunction array (target distribution).
        local_dims (list): List of local dimensions, e.g., [2, 2, ...].

    Returns:
        float: The calculated fidelity.
    """

    # 1. Get the target amplitudes from the full array for each sampled configuration
    flat_indices = ravel_configs_to_indices(samples_dmrg, local_dims)
    psi_target_sampled = dmrg_array[flat_indices]

    # 2. Calculate the ratio X(s) = psi_target(s) / psi_proposal(s)
    # Here, both target and proposal are the DMRG wavefunction.
    ratio = psi_target_sampled / psi_DMRG_sampled

    # 3. Estimate fidelity using the ratio method
    overlap_est = np.mean(ratio)
    denominator_est = np.mean(np.abs(ratio)**2)
    fidelity = np.abs(overlap_est)**2 / denominator_est

    return fidelity

def fidelity_RBM_sampled_vs_full(psi_RBM_sampled, samples_dmrg, RBM_array, local_dims):
    """
    Calculates the fidelity between the sampled RBM amplitudes and the full RBM wavefunction.
    This is a cross-check where samples are drawn from DMRG, but used to evaluate RBM fidelity.

    Args:
        psi_RBM_sampled (np.ndarray): RBM amplitudes evaluated on configurations sampled from DMRG.
        samples_dmrg (np.ndarray): The configurations {0,1} that were sampled from DMRG.
        RBM_array (np.ndarray): The full RBM wavefunction array (target distribution).
        local_dims (list): List of local dimensions, e.g., [2, 2, ...].

    Returns:
        float: The calculated fidelity.
    """

    # 1. Get the target amplitudes from the full array for each sampled configuration
    flat_indices = ravel_configs_to_indices(samples_dmrg, local_dims)
    psi_target_sampled = RBM_array[flat_indices]

    # 2. Calculate the ratio X(s) = psi_target(s) / psi_RBM_sampled(s)
    # NOTE: This is NOT an importance sampling ratio, as samples are not from RBM.
    # It's a direct comparison on a specific subset of configurations.
    overlap = np.vdot(psi_target_sampled, psi_RBM_sampled)
    norm_target_sq = np.vdot(psi_target_sampled, psi_target_sampled)
    norm_sampled_sq = np.vdot(psi_RBM_sampled, psi_RBM_sampled)

    fidelity = np.abs(overlap)**2 / (norm_target_sq * norm_sampled_sq)
    return fidelity.real
