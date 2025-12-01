from jax import numpy as jnp
import netket as nk
import jax
from flax import linen as nn
from jax.nn.initializers import normal

from HFDS_Heisenberg.MF_Init import init_orbitals_mf
from HFDS_Heisenberg.Gutzwiller_MF_Init import update_orbitals_gmf
 

def compute_orbital_selection(x, orbitals_full, N_sites):
  """
  Computes the orbital selection for a single sample `x`.
  x has shape (N_sites,).
  """
  #1.  Convert {-1,+1} → 0/1 occupancy for spin-up, spin-down orbitals
  spin_up = (x == 1)
  spin_dn = (x == -1)
  x_flat = jnp.concatenate([spin_up, spin_dn]) # Concatenate 1D arrays

  #2 & 3. Select occupied orbitals using advanced indexing
  mask = x_flat.astype(bool)
  _, idx = jax.lax.top_k(mask, k = N_sites)
  return orbitals_full[idx, :] # Simple indexing for a single sample


class Orbitals(nn.Module):
  lattice: nk.graph.Graph
  n_elecs: int
  n_hid: int
  MFinit: str
  stop_grad_mf: bool
  bounds: str
  dtype: type = jnp.float64
  U: float=8.0

  def _init_gutzwiller(self, key, shape, dtype):

    #h_opt, phi_opt = 0.06, 0.1

    from HFDS_Heisenberg.Optimized_Gutwiller_MF_Init import optimized_gutzwiller_params
    opt_params = optimized_gutzwiller_params(self.lattice)
    h_opt, phi_opt = opt_params["h"], opt_params["phi"]
    print(f"Optimized Gutzwiller h: {h_opt}, phi: {phi_opt}")
    
    mf = update_orbitals_gmf(self.lattice, dtype=dtype, h=h_opt, phi=phi_opt)
    return mf
      
  def _init_mf(self, key, shape, dtype):
    return init_orbitals_mf(L=int(jnp.sqrt(self.lattice.n_nodes)), bounds=self.bounds, dtype=dtype)
  
  
  @nn.compact
  def __call__(self,x):

    n_samples, N_sites = x.shape

    if self.MFinit=="Fermi":
        orbitals_mfmf = self.param('orbitals_mf',self._init_mf,(N_sites,self.n_elecs), self.dtype)
    elif self.MFinit=="G_MF":
        orbitals_mfmf = self.param('orbitals_mf', self._init_gutzwiller, (N_sites, self.n_elecs), self.dtype)
    elif self.MFinit=="random":
        orbitals_mfmf = self.param('orbitals_mf', normal(0.1),(2*N_sites,self.n_elecs), self.dtype)
    else:
        raise NotImplementedError("This MF initialization is not implemented! Chose one of: Fermi, random")
    
    orbitals_mfhf = self.param('orbitals_hf', normal(0.1),(2*N_sites,self.n_hid), self.dtype)
    orbitals_full = jnp.concatenate((orbitals_mfmf, orbitals_mfhf), axis=1)
    
    # Vectorize the single-sample function to work on a batch of samples `x`.
    # We vectorize over `x` (axis 0), but use the same `orbitals_full` (None) and `N_sites` (None) for all samples.
    vmapped_selection = jax.vmap(compute_orbital_selection, in_axes=(0, None, None))
    orbitals_selected = vmapped_selection(x, orbitals_full, N_sites)
    
    return orbitals_selected
