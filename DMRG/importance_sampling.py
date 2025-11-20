#%%
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
from DMRG.Plotting import *
from DMRG.Observable.Corr_Struct import Correlations_Structure_Factor

from netket.experimental.driver import VMC_SR
from scipy.sparse.linalg import eigsh


from jax import numpy as jnp
import matplotlib.pyplot as plt
import netket as nk
import flax
import os

from DMRG.QSL_DMRG import *
from DMRG.Fidelities import *


# ----------------------
# Helper: extract MPS tensors (tries common TenPy field names)
# ----------------------
def extract_mps_tensors(DMRG_vstate):
    """
    Return a list of MPS tensors (B_list) as plain np.ndarray objects with shape (Dl, d, Dr)
    and local_dims list of local dimensions.
    Works for TenPy MPS where tensors may be tenpy.linalg.np_conserved.Array objects.
    """
    B_list = []
    tensors = DMRG_vstate._B

    for t in tensors:
        # Unwrap conserved Array -> ndarray
        arr = t.to_ndarray()
        B_list.append(np.asarray(arr, dtype=np.complex128))

    local_dims = [B.shape[1] for B in B_list]
    return B_list, local_dims

# ----------------------
# Sequential sampler from MPS tensors (left-normalized assumed / canonicalization recommended)
# ----------------------
from numpy.linalg import qr

def left_orthonormalize_mps(B_list):
    """
    Left-orthonormalize an MPS in-place and return the new list.
    Simple QR-based left sweep (no truncation).
    Assumes B_list[i] has shape (D_l, d, D_r).
    Returns new_B_list.
    """
    new_B = []
    carry = None
    for i, A in enumerate(B_list):
        Dl, d, Dr = A.shape
        # reshape (Dl*d, Dr)
        M = A.reshape(Dl * d, Dr)
        # QR: M = Q R
        Q, R = qr(M, mode='reduced')
        # new left-orthonormal tensor
        new_Dr = Q.shape[1]
        A_new = Q.reshape(Dl, d, new_Dr)
        new_B.append(A_new)
        # absorb R into next tensor if exists
        if i + 1 < len(B_list):
            A_next = B_list[i + 1]
            # A_next shape (Dr, d_next, Dr_next)
            # multiply R (new_Dr x Dr) into left index of next tensor
            B_list[i + 1] = np.tensordot(R, A_next, axes=(1,0))  # shape (new_Dr, d_next, Dr_next)
    return new_B

def importance_sample_mps_tensors(B_list, local_dims=None, n_samples=1024, canonicalize=True, eps=1e-16):
    """
    Draw n_samples i.i.d. from distribution p(s)=|psi_MPS(s)|^2 by sequential conditional sampling
    IF canonicalize==True the MPS will be left-orthonormalized first (so weights==1).
    Returns:
      samples: array (n_samples, L) with integers in [0, d-1]
      weights: array (n_samples,) importance weights (should be ~1 if canonicalized)
    """
    rng = np.random.default_rng()

    # copy B_list so we don't mutate user's tensors when canonicalizing
    B_copy = [np.array(A, dtype=np.complex128, copy=True) for A in B_list]
    L = len(B_copy)
    if local_dims is None:
        local_dims = [A.shape[1] for A in B_copy]
    assert len(local_dims) == L

    if canonicalize:
        B_copy = left_orthonormalize_mps(B_copy)

    samples = np.zeros((n_samples, L), dtype=int)
    weights = np.ones(n_samples, dtype=float)

    # true amplitude evaluator (exact contraction) for importance weight numerator if needed
    def amplitude_squared_for_config(config):
        # contract left->right to a scalar amplitude
        vec = np.array([1.0+0.0j])  # left boundary scalar
        for i, s in enumerate(config):
            Ai = B_copy[i][:, s, :]  # shape (Dl, Dr)
            vec = vec.conj().T @ Ai   # new vector shape (Dr,)
        # final vec should be scalar (Dr==1) for open-boundary; take squared norm
        return float(np.vdot(vec, vec).real)

    for n in range(n_samples):
        # Initialize left env vector v
        v = np.array([1.0+0.0j]) if B_copy[0].shape[0] == 1 else np.ones((B_copy[0].shape[0],), dtype=np.complex128)
        q_prod = 1.0  # product of proposal probabilities for this sample
        for i in range(L):
            Ai = B_copy[i]  # shape (D_l, d, D_r)
            d = local_dims[i]
            # compute u_s = v^H @ Ai[:,s,:] for all s
            probs = np.empty(d, dtype=float)
            u_list = [None] * d
            for s in range(d):
                u = np.tensordot(v.conj(), Ai[:, s, :], axes=(0, 0))  # shape (D_r,)
                u_list[s] = u
                val = np.vdot(u, u).real
                # guard against tiny negative rounding
                probs[s] = val if val > 0 else 0.0
            psum = probs.sum()
            if psum <= eps:
                # degenerate: all zero (numerical underflow). Make a tiny uniform fallback
                probs = np.ones(d, dtype=float) / d
                psum = 1.0
            else:
                probs = probs / psum

            s_choice = rng.choice(d, p=probs)
            samples[n, i] = s_choice
            q_prod *= probs[s_choice]
            # update v to unit norm version of u_list[s_choice]
            if probs[s_choice] > 0:
                v = u_list[s_choice] / np.sqrt(probs[s_choice] * psum) if not canonicalize else u_list[s_choice] / np.sqrt(probs[s_choice])
                # Note: If canonicalize==True then psum should equal 1 so dividing by sqrt(probs) suffices.
                # For general tensors we used fallback dividing by sqrt(probs[s_choice]*psum) to keep correct scaling, but
                # this update primarily stabilizes numerics; final weight will correct bias.
            else:
                v = u_list[s_choice]

        # compute importance weight:
        if canonicalize:
            weights[n] = 1.0
        else:
            p_s = amplitude_squared_for_config(samples[n])  # exact |psi|^2
            q_s = q_prod
            weights[n] = (p_s / q_s) if q_s > 0 else 0.0

    return samples, weights


# ----------------------
# Sequential sampler from MPS tensors (left-normalized assumed / canonicalization recommended)
# ----------------------
def importance_sample_mps_tensors_1(B_list, local_dims, n_samples=1024):
    """
    Draw n_samples i.i.d. from distribution p(s)=|psi_MPS(s)|^2 by sequential conditional sampling.
    Return:
      samples: array (n_samples, L) with integers in [0, d-1]

    """
    
    rng = np.random.default_rng()

    L = len(B_list)
    samples = np.zeros((n_samples, L), dtype=int) # array containing the samples

    # For each sample for each site
    for n in range(n_samples):
        # For each sample, start a left boundary vector v. 
        # For a standard open MPS the left boundary is a 1-d array of length D_left (often 1 at the first site); using ones is a generic left-boundary.
        v = np.ones((B_list[0].shape[0],), dtype=np.complex128)  # left boundary
        #log_amp = 0.0 + 0.0j

        for i in range(L):
            Ai = B_list[i]  # shape (D_l, d, D_r)
            d = local_dims[i] # how many eigenvalues in site i

            # for each local value compute u = v^H @ Ai[:,s,:] -> shape (D_r,)
            probs = np.zeros(d, dtype=float)
            u_list = [None] * d
            for s in range(d): # for each eigenvale at site i
                # treat v as column vector; compute u = v.conj().T @ Ai[:,s,:]
                u = np.tensordot(v.conj(), Ai[:, s, :], axes=(0, 0))  # shape (D_r,)
                u_list[s] = u
                probs[s] = np.vdot(u, u).real  # ||u||^2
            #Normalize probability
            psum = probs.sum()
            probs = probs / psum

            #from this probability distribution, draw some samples
            # s_choice → the local spin/configuration value chosen at the current site i for one particular sample. each element is a single spin
            # samples → a 2D array collecting all the s_choice values for all sites and all samples. each row is a onfiguration of spins
            s_choice = rng.choice(d, p=probs)
            samples[n, i] = s_choice
            
           
            # We'll compute full amplitude at end by contracting again for exact logpsi; here accumulate log prob for numerical sanity
            # Update v <- u / sqrt(probs[s_choice]) so that v remains normalized
            if probs[s_choice] > 0:
                v = u_list[s_choice] / np.sqrt(probs[s_choice])
            else:
                v = u_list[s_choice]

    return samples

# ----------------------
# Contract MPS to obtain log amplitude for a single product configuration
# ----------------------
def psi_mps_from_config(B_list, configs):
    """
    Exact complex amplitude psi_MPS(config)
    config: iterable of ints length L
    B_list: list of tensors shape (D_left, d, D_right)
    """
    n_samples = configs.shape[0]
    psi_vals = np.zeros(n_samples, dtype=np.complex128)

    for sample_idx, config in enumerate(configs):
        # start with left boundary vector (1,) no phase
        v = np.ones((B_list[0].shape[0],), dtype=np.complex128)
        for site_idx, s_val in enumerate(config):
            Ai = B_list[site_idx]
            # contract v (D_l,) with Ai[:, s, :] -> shape (D_r,)
            v = np.tensordot(v, Ai[:, int(s_val), :], axes=(0, 0))
        # after last site, v should be scalar (shape (1,) or ())
        # If final v is array, take its only element
        val = np.atleast_1d(v)
        if val.size != 1:
            # if boundary dims not 1, final scalar may still be length>1 -> reduce by normed dot with right boundary ones
            val = np.sum(val)
        psi_vals[sample_idx] = complex(val.flat[0])

    return psi_vals


def logpsi_netket(RBM_vstate, samples):
    """
    Given a NetKet MCState 'RBM_vstate' and samples array shape (n_samples, L),
    return complex logpsi array length n_samples: log(psi_V(s)).
    """

    lv = RBM_vstate.log_value(samples)
                
    return np.array(lv, dtype=np.complex128)



#%%
if __name__ == "__main__":

    model_params = {
        'Lx': 4,
        'Ly': 4,
        'J1': 1.0,
        'J2': 0.0
    }

    n_samples = 2048
    n_iter = 120
    N_sites = model_params['Lx'] **2

    # --- Define file paths for saved models ---
    model_storage_dir = "DMRG/trained_models"
    os.makedirs(model_storage_dir, exist_ok=True)
    dmrg_filename = os.path.join(model_storage_dir, f"dmrg_L{model_params['Lx']}_J2_{model_params['J2']}.pkl.gz")
    rbm_filename = os.path.join(model_storage_dir, f"rbm_L{model_params['Lx']}_J2_{model_params['J2']}.mpack")


    # --- DMRG ---
    hamiltonian = J1J2Heisenberg(model_params=model_params)
    DMRG_vstate = DMRG_vstate_optimization(hamiltonian, model_params, filename=dmrg_filename)
    B_list, local_dims = extract_mps_tensors(DMRG_vstate)
    DMRG_vstate.canonical_form()
    
    # --- RBM ---
    RBM_vstate, ha = RBM_vstate_optimization(model_params, n_iter, filename=rbm_filename) # ha is the NetKet Hamiltonian for exact diagonalization
    
    # --- Importance Sampling ---

    ops_z = ['Sigmaz'] * N_sites  # or just 'Sigmaz' if measuring all sites
    samples = np.zeros((n_samples, N_sites), dtype=int)
    psi_DMRG_sampled = np.zeros(n_samples, dtype=np.complex128) 
    for n in range(n_samples):
        sigmas, psi_DMRG = DMRG_vstate.sample_measurements(first_site=0, last_site=N_sites-1, ops = ops_z, complex_amplitude=True)
        samples[n, :] = sigmas
        psi_DMRG_sampled[n] = psi_DMRG
        

    # Convert samples from {-1, 1} basis (NetKet/Sigmaz) to {0, 1} basis (TenPy/DMRG)
    # This is crucial for functions expecting TenPy-like configurations.
    # +1 -> 0 (up), -1 -> 1 (down)
    samples_netket = samples # Keep original for RBM logpsi if needed
    samples_dmrg_01_basis = ((1 - np.asarray(samples_netket)) / 2).astype(int)
    

    # Evaluate RBM amplitudes on the sampled configurations (NetKet format)
    # Note: logpsi_netket expects samples in {-1, 1} basis, so use samples_netket
    logpsi_RBM_sampled = logpsi_netket(RBM_vstate, samples_netket)
    psi_RBM_sampled = np.exp(logpsi_RBM_sampled)

    # --- Compute full wavefunction arrays for fidelity calculations ---
    dmrg_array = get_full_wavefunction(DMRG_vstate, undo_sort_charge=True)
    RBM_array = RBM_vstate.to_array()

    # --- Fidelity with DMRG and RBM ---
    fidelity_exact_rbm_dmrg = Fidelity_exact(RBM_vstate, DMRG_vstate)
    print("\nFidelity exact (RBM_Full vs DMRG_Full):", fidelity_exact_rbm_dmrg)
    fidelity_sampled_rbm_dmrg = Fidelity_sampled(psi_DMRG_sampled, psi_RBM_sampled)
    print("\nFidelity sampled (RBM_sampled vs DMRG_sampled):", fidelity_sampled_rbm_dmrg)
    fidelity_rbm_sampled__dmrg_exact = Fidelity_RBM_sampled_vs_DMRG_exact(psi_RBM_sampled, samples_dmrg_01_basis, dmrg_array, local_dims)
    print("\nFidelity sampled (RBM_sampled vs DMRG_exact):", fidelity_rbm_sampled__dmrg_exact)


    # --- Fidelity with Exact Ground State ---

    E_gs_vals, ket_gs_matrix = nk.exact.lanczos_ed(ha, k=1, compute_eigenvectors=True)
    """
    if E_gs_vals is None:
        graph = nk.graph.Hypercube(length=L, n_dim=2, pbc=True)
        hi_scaled = nk.hilbert.Spin(s=0.5, N=graph.n_nodes)
        H = nk.operator.Ising(hilbert=hi_scaled, graph=graph, h=jnp.float64(0), J=1.0)
        H_sparse = H.to_sparse(jax_=False).tocsc()
        E_gs_vals, ket_gs_matrix = eigsh(H_sparse, k=1, which="SA")
    """
    ket_gs = ket_gs_matrix[:, 0] # Extract the ground state vector

    fidelity_RBM_exact_val = fidelity_RBM_exact(RBM_vstate, ket_gs)
    print(f"\nFidelity (RBM vs Exact Diagonalization): {fidelity_RBM_exact_val:.6f}")
    fidelity_DMRG_exact_val_0 = fidelity_DMRG_exact(DMRG_vstate, ket_gs)
    print(f"\nFidelity (DMRG vs Exact Diagonallization): {fidelity_DMRG_exact_val_0:.6f}")

    # Corrected calls for sampled vs full fidelities
    fidelity_DMRG_sampled_vs_full_val = fidelity_DMRG_sampled_vs_full(psi_DMRG_sampled, samples_dmrg_01_basis, dmrg_array, local_dims)
    print(f"\nFidelity (DMRG_sampled vs DMRG_Full): {fidelity_DMRG_sampled_vs_full_val:.6f}")
    fidelity_RBM_sampled_vs_full_val = fidelity_RBM_sampled_vs_full(psi_RBM_sampled, samples_dmrg_01_basis, RBM_array, local_dims)
    print(f"\nFidelity (RBM_sampled vs RBM_Full): {fidelity_RBM_sampled_vs_full_val:.6f}")

    # --- Compare DMRG probability distribution with the sample distribution ---
    # This is a sanity check. Since we sampled from DMRG, the distributions should be similar.
    unique_configs, unique_indices, counts = np.unique(samples_dmrg_01_basis, axis=0, return_index=True, return_counts=True)
    P_samples = counts / counts.sum()
    unique_configs, unique_indices, inverse_indices, counts = np.unique(
        samples_dmrg_01_basis, axis=0, return_index=True, return_inverse=True, return_counts=True
    )
    # P_unique_samples contains the probability for each unique configuration
    P_unique_samples = counts / counts.sum()

    psi_DMRG_unique = psi_DMRG_sampled[unique_indices]
    P_DMRG = np.abs(psi_DMRG_unique)**2
    P_DMRG /= P_DMRG.sum()

    classical_fidelity = np.sum(np.sqrt(P_samples * P_DMRG))**2
    classical_fidelity = np.sum(P_unique_samples * P_DMRG)
    print(f"\nClassical Fidelity (Sample Distr. vs DMRG  Distr.): {classical_fidelity:.6f}")

    # --- Overlap Calculation Demonstration ---
    # 1. Exact Overlap
    overlap_exact_val = Overlap_exact(RBM_vstate, DMRG_vstate)
    print(f"\nOverlap exact (RBM vs DMRG): {overlap_exact_val:.6f}")

    # 2. Sampled Overlap (samples from DMRG, target is RBM)
    overlap_sampled_val = Overlap_sampled(psi_proposal=psi_DMRG_sampled, psi_target=psi_RBM_sampled)
    print(f"Overlap sampled (RBM vs DMRG): {overlap_sampled_val:.6f}")


    # --- Fidelity norm ---
    # Create P_sample_sort with the same order as samples_dmrg_01_basis
    # It maps the probability of each unique sample back to all original samples.
    P_sample_sort = P_unique_samples[inverse_indices]
    fidelity_sampled_rbm_dmrg = Fidelity_sampled_norm(psi_DMRG_sampled, psi_RBM_sampled, P_sample_sort)
    print("\nFidelity sampled (RBM_sampled vs DMRG_sampled):", fidelity_sampled_rbm_dmrg)
    fidelity_new = fidelity_new(psi_DMRG_sampled, psi_RBM_sampled)
    print("\nFidelity new (RBM_sampled vs DMRG_sampled):", fidelity_new)
    

    # --- # --- Plotting --- ---
    #plot_probability_distributions(ket_gs, samples_netket, samples_dmrg_01_basis, B_list, RBM_vstate, dmrg_array, RBM_array, model_params, fidelity_exact=fidelity_exact_rbm_dmrg, fidelity_sampled=fidelity_sampled_rbm_dmrg)
    #plot_full_hilbert_distributions(ket_gs, samples_dmrg_01_basis, B_list, RBM_vstate, dmrg_array, RBM_array, model_params, base_output_dir="DMRG/plot", fidelity_exact=fidelity_exact_rbm_dmrg, fidelity_sampled=fidelity_sampled_rbm_dmrg)

##################################################################################################################################################################################################################