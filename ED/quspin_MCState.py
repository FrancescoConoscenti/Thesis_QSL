import matplotlib.pyplot as plt
import netket as nk
import jax
import numpy as np
import jax.numpy as jnp

from quspin.basis import spin_basis_general
from quspin.operators import hamiltonian

import flax
from flax import linen as nn
from netket.operator.spin import sigmax, sigmaz, sigmay
import sys
import os
import argparse
import pickle
from typing import Any

def MCState_quspin(sampler, hamiltonian, L):
    # Load QuSpin ground state if available
    psi_path = "/cluster/home/fconoscenti/Thesis_QSL/ED/psi_output_6x6_all_symm.npy"
    print(f"Loading QuSpin ground state from {psi_path}")


    # Reconstruct basis (Must match quspin_6x6.py)
    N_qs = L**2
    sites_qs = np.arange(N_qs).reshape((L, L))
    tx_qs = np.roll(sites_qs, -1, axis=1).flatten() 
    ty_qs = np.roll(sites_qs, -1, axis=0).flatten()
    rot_qs = np.rot90(sites_qs, k=1).flatten()
    px_qs = sites_qs[:, ::-1].flatten()

    basis_symm = spin_basis_general(N_qs, m=0, S='1/2',
                                kxblock=(tx_qs, 0), kyblock=(ty_qs, 0),
                                rblock=(rot_qs, 0),
                                pblock=(px_qs, 0))

    psi_symm = np.load(psi_path)
    if psi_symm.ndim > 1: psi_symm = psi_symm[:, 0]
    psi_symm = psi_symm / np.linalg.norm(psi_symm)

    # Create basis with only magnetization symmetry
    basis_nosymm = spin_basis_general(N_qs, m=0, S='1/2')
    
    # Project the symmetric state to the non-symmetric basis
    psi_nosymm = basis_nosymm.project_from(psi_symm, basis_symm, dtype=np.complex128)
    psi_nosymm = psi_nosymm / np.linalg.norm(psi_nosymm)



    # Define QuSpin Model for MCMC
    class QuSpinModel(nn.Module):
        basis: Any = flax.struct.field(pytree_node=False)
        psi_vec: np.ndarray = flax.struct.field(pytree_node=False)
        
        @nn.compact
        def __call__(self, x):
            def cb(x_np):
                x_01 = (x_np + 1) // 2
                N = x_np.shape[-1]
                powers = 1 << np.arange(N, dtype=np.int64)
                states = x_01.astype(np.int64).dot(powers)
                amps = self.basis.get_amp(self.psi_vec, states)
                return np.log(amps.astype(np.complex128) + 0j)
            return jax.pure_callback(cb, jnp.zeros(x.shape[0], dtype=jnp.complex128), x)

    # Create Exact VState
    vstate_exact = nk.vqs.MCState(sampler, QuSpinModel(basis_nosymm, psi_nosymm), n_samples=1024)

    E_exact = vstate_exact.expect(hamiltonian).mean.real
    print(f"Exact Energy (MCMC): {E_exact}")

    return E_exact , vstate_exact

def Exact_Correlations_QuSpin(L):
    psi_path = "/cluster/home/fconoscenti/Thesis_QSL/ED/psi_output_6x6_all_symm.npy"
    print(f"Loading QuSpin ground state from {psi_path}")

    N_qs = L**2
    sites_qs = np.arange(N_qs).reshape((L, L))
    tx_qs = np.roll(sites_qs, -1, axis=1).flatten() 
    ty_qs = np.roll(sites_qs, -1, axis=0).flatten()
    rot_qs = np.rot90(sites_qs, k=1).flatten()
    px_qs = sites_qs[:, ::-1].flatten()

    basis_symm = spin_basis_general(N_qs, m=0, S='1/2',
                                kxblock=(tx_qs, 0), kyblock=(ty_qs, 0),
                                rblock=(rot_qs, 0),
                                pblock=(px_qs, 0))

    psi_symm = np.load(psi_path)
    if psi_symm.ndim > 1: psi_symm = psi_symm[:, 0]
    psi_symm = psi_symm / np.linalg.norm(psi_symm)

    correlations = np.zeros(N_qs)
    
    # Helper to generate symmetry orbit of a displacement vector
    def get_sym_displacements(dx, dy):
        orbit = set()
        # D4 group operations on displacement vector (dx, dy)
        # Rotations
        curr_x, curr_y = dx, dy
        for _ in range(4):
            orbit.add((curr_x, curr_y))
            curr_x, curr_y = curr_y, (-curr_x) % L
        # Reflection (x -> -x) and its rotations
        ref_x, ref_y = (-dx) % L, dy
        curr_x, curr_y = ref_x, ref_y
        for _ in range(4):
            orbit.add((curr_x, curr_y))
            curr_x, curr_y = curr_y, (-curr_x) % L
        return list(orbit)

    print("Computing exact correlations...")
    for j in range(N_qs):
        dy_target, dx_target = divmod(j, L)
        sym_disps = get_sym_displacements(dx_target, dy_target)
        
        # Symmetrized operator: sum over symmetry orbit, sum over all sites
        J_list = []
        for (dx, dy) in sym_disps:
            for i in range(N_qs):
                iy, ix = divmod(i, L)
                target = ((iy + dy) % L) * L + ((ix + dx) % L)
                J_list.append([1.0, i, target]) # zz
        
        # Normalize: 1/N (correlation def) * 1/len(orbit) (symmetry avg)
        scale = 1.0 / (N_qs * len(sym_disps))
        static = [["zz", [[v*scale, i, t] for v,i,t in J_list]], 
                  ["+-", [[v*scale*0.5, i, t] for v,i,t in J_list]], 
                  ["-+", [[v*scale*0.5, i, t] for v,i,t in J_list]]]
        
        Op = hamiltonian(static, [], basis=basis_symm, dtype=np.complex128, check_symm=False, check_herm=False)
        correlations[j] = Op.expt_value(psi_symm).real

    # Structure factor S(q)
    q_vals = 2 * np.pi * np.arange(L) / L
    
    structure_factor = np.zeros((L, L))
    for ky in range(L):
        for kx in range(L):
            q = np.array([q_vals[kx], q_vals[ky]])
            # S(q) = sum_j e^{-i q r_j} C(j)
            phase = q[0] * (np.arange(N_qs) % L) + q[1] * (np.arange(N_qs) // L)
            structure_factor[ky, kx] = np.sum(correlations * np.exp(-1j * phase)).real
            
    return correlations, structure_factor

if __name__ == '__main__':

    corr, struct = Exact_Correlations_QuSpin(L=6)