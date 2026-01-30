from jax import numpy as jnp
import numpy as np
import netket as nk
import jax
from flax import linen as nn
from jax.nn.initializers import normal, zeros
from typing import Callable

from HFDS_Heisenberg.MF_Init import init_orbitals_mf
#from HFDS_Heisenberg.Optimized_Gutwiller_MF_Init import optimized_gutzwiller_params
#from HFDS_Heisenberg.Gutzwiller_MF_Init import update_orbitals_gmf
#from HFDS_Heisenberg.Gutzwiller_MF_Init_old import update_orbitals_gmf_init

import logging
logger = logging.getLogger(__name__)
 

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

 
class Orbitals_ent(nn.Module):
  L: int
  n_elecs: int
  n_hid: int
  MFinit: str
  stop_grad_mf: bool
  bounds: str
  dtype: type = jnp.float64
  U: float = 8.0
  kernel_init: Callable = normal(0.1)

  def _init_gutzwiller(self, key, shape, dtype):
    #logger.info(f"Initializing Gutzwiller orbitals with h={self.h_opt}, phi={self.phi_opt}")
    #mf = update_orbitals_gmf_init(self.lattice, dtype=dtype, h=self.h_opt, phi=self.phi_opt)
    #mf = update_orbitals_gmf(self.L, dtype=dtype, h=self.h_opt, phi=self.phi_opt)
    mf = None

    return mf
      
  def _init_mf(self, key, shape, dtype):
    mf = init_orbitals_mf(L=self.L, bounds=self.bounds, dtype=dtype)
    return mf
  
  def setup(self):
    N_sites = self.L * self.L

    if self.MFinit=="Fermi":
        self.orbitals_mfmf = self.param('orbitals_mf',self._init_mf,(2*N_sites,self.n_elecs), self.dtype)
    elif self.MFinit=="G_MF":
        self.orbitals_mfmf = self.param('orbitals_mf', self._init_gutzwiller, (N_sites, self.n_elecs), self.dtype)
    elif self.MFinit=="random":
        self.orbitals_mfmf = self.param('orbitals_mf', self.kernel_init,(2*N_sites,self.n_elecs), self.dtype)
    else:
        raise NotImplementedError("This MF initialization is not implemented! Chose one of: Fermi, random")
    
    self.orbitals_mfhf = self.param('orbitals_hf', self.kernel_init,(2*N_sites,self.n_hid), self.dtype)
    #self.orbitals_mfhf = self.param('orbitals_hf', zeros,(2*N_sites,self.n_hid), self.dtype)



  def __call__(self,x):
    
    orbitals_full = jnp.concatenate((self.orbitals_mfmf, self.orbitals_mfhf), axis=1)
    # Vectorize the single-sample function to work on a batch of samples `x`.
    # We vectorize over `x` (axis 0), but use the same `orbitals_full` (None) and `N_sites` (None) for all samples.
    vmapped_selection = jax.vmap(compute_orbital_selection, in_axes=(0, None, None))
    orbitals_selected = vmapped_selection(x, orbitals_full, self.L * self.L)
    
    return orbitals_selected