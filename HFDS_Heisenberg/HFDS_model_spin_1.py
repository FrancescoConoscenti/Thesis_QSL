from jax import numpy as jnp
import netket as nk
import jax
from jax.random import PRNGKey, choice, split
from functools import partial
from flax import linen as nn
from jax.nn.initializers import zeros, normal, constant
from netket.utils.dispatch import dispatch
from netket import experimental as nkx
from netket.jax import apply_chunked
import numpy as np
from netket.hilbert.homogeneous import HomogeneousHilbert 
from netket.jax import logsumexp_cplx
from jax import lax
from functools import partial
from typing import Sequence, Tuple

class HiddenFermion(nn.Module):
  n_elecs: int
  network: str
  n_hid: int
  Lx: int
  Ly: int
  layers: int
  features: int
  MFinit: str
  hilbert: HomogeneousHilbert
  stop_grad_mf: bool = False
  stop_grad_lower_block: bool = False
  bounds: str="PBC"
  parity: bool = False
  rotation: bool = False
  translation: bool = False
  dtype: type = jnp.float64
  U: float=8.0

  def setup(self):
    self.n_modes = 2*self.Lx*self.Ly
    self.orbitals = Orbitals(self.n_elecs,self.n_hid,self.Lx, self.Ly, self.MFinit, self.stop_grad_mf, self.bounds, self.dtype, self.U)
    if self.network=="FFNN":
        self.hidden = [nn.Dense(features=self.features,use_bias=False,param_dtype=self.dtype) for i in range(self.layers)]
        self.output = nn.Dense(features=self.n_hid*(self.n_elecs + self.n_hid),use_bias=True,param_dtype=self.dtype)
    else:
        raise NotImplementedError()
    
    if self.rotation:
        idx = jnp.arange(self.Lx * self.Ly).reshape(self.Ly, self.Lx)
        self.idx_rot = jnp.flip(idx.T, axis=1).reshape(-1)
    if self.translation:
        idx = jnp.arange(self.Lx * self.Ly).reshape(self.Ly, self.Lx)
        self.idx_trans = jnp.roll(idx, shift=1, axis=1).reshape(-1)

  def selu(self,x):
    if self.dtype==jnp.float64:
      return jax.nn.selu(x)
    else:
      return jax.nn.selu(x.real) +1j*jax.nn.selu(x.imag)


  def calc_psi(self,x,return_orbs=False):

    #1, 2, 3.
    orbitals = self.orbitals(x)

    # 4. Forward pass through the NN, create x_
    for i in range(self.layers):
        x = self.selu(self.hidden[i](x))
    x_ = self.output(x).reshape(x.shape[0],self.n_hid,self.n_elecs + self.n_hid)

    x_2 = jnp.repeat(jnp.expand_dims(jnp.eye(self.n_hid), axis=0),x.shape[0],axis=0)
    x_ += jnp.concatenate((jnp.zeros((x.shape[0], self.n_hid, self.n_elecs),self.dtype), x_2),axis=2)
    
    # 5. Concatenate the MF orbitals and the NN outputs
    x = jnp.concatenate((orbitals,x_),axis=1)
    sign, logx = jnp.linalg.slogdet(x)
    return logx, jnp.log(sign + 0j)


  def gen_reflected_samples(self,x):
    x_refl = -x
    return x_refl
  

  def gen_rotated_samples(self, x):
    #jax.debug.print("type of x rotation: {x}", x=type(x))
    x_rot1 = x[:, self.idx_rot]
    x_rot2 = x_rot1[:, self.idx_rot]
    x_rot3 = x_rot2[:, self.idx_rot]
    return (x_rot1, x_rot2, x_rot3)
  
  def gen_translated_samples(self, x):
    x_tra = x[:, self.idx_trans]
    return x_tra

  
  
  def remove_duplicate_samples(self, x_sym):

    #print x_sym
    #jax.debug.print("type of x_sym: {x}", x=type(x_sym))
    #jax.debug.print("shape of x_sym: {x}", x=len(x_sym))
    #jax.debug.print("x_sym: {x}", x=x_sym)

    #jax.debug.print("type of x_sym elements: {x}", x=x_sym[0].dtype)
    #jax.debug.print("shape of x_sym elements: {x}", x=x_sym[0].shape)
    #jax.debug.print("x_sym elements: {x}", x=x_sym[0])

    return x_sym
  
  
  def gen_sym_samples(self, x):
    x_sym = [x]
    if self.parity:
        x_refl = self.gen_reflected_samples(x)
        x_sym.append(x_refl)
    
    # --- Step 2: add rotation symmetry ---
    if self.rotation:
        x_rot1, x_rot2, x_rot3 = self.gen_rotated_samples(x)
        x_sym.extend([x_rot1, x_rot2, x_rot3])
        if self.parity:
            x_refl = self.gen_reflected_samples(x)
            x_rot1_refl, x_rot2_refl, x_rot3_refl = self.gen_rotated_samples(x_refl)
            x_sym.extend([x_rot1_refl, x_rot2_refl, x_rot3_refl])
    
    x_sym = self.remove_duplicate_samples(x_sym)
    return x_sym
  

  def __call__(self,x):

    x_sym = self.gen_sym_samples(x)
  
    log_det, log_sign = jax.vmap(self.calc_psi)(jnp.stack(x_sym))
    log_psi_sym = log_det + log_sign
    return logsumexp_cplx(log_psi_sym, axis=0)


class Orbitals(nn.Module):
  n_elecs: int
  n_hid: int
  Lx: int
  Ly: int
  MFinit: str
  stop_grad_mf: bool
  bounds: str
  dtype: type = jnp.float64
  U: float=8.0

  def _init_orbitals_dct(self, key, shape, dtype):

    def ft_local_pbc(x,y,kx,ky):
        if self.dtype == jnp.float64:
            # real-valued Fourier basis
            return jnp.cos(2*jnp.pi*kx*x/self.Lx) * jnp.cos(2*jnp.pi*ky*y/self.Ly)
        else:
            return jnp.exp(1j*2*jnp.pi*(kx*x/self.Lx + ky*y/self.Ly))

    def ft(k_arr, max_val):
        matrix = []
        for (kx, ky) in k_arr[:max_val]:
            kstate = [ft_local_pbc(x,y,kx,ky)
                      for y in range(self.Ly)
                      for x in range(self.Lx)]
            matrix.append(kstate)
        return jnp.array(matrix)

    # number of orbitals = shape[1]
    n_elecs = shape[1]

    # build & sort momentum modes
    k_modes = [(kx, ky) for kx in range(self.Lx) for ky in range(self.Ly)]
    sorted_k_modes = sorted(
        k_modes,
        key=lambda x: (-np.cos(2*np.pi*x[0]/self.Lx) -
                       -np.cos(2*np.pi*x[1]/self.Ly), x)
    )
    k_arr = np.array(sorted_k_modes)

    # SINGLE spin sector now
    matrix = ft(k_arr, n_elecs)
    # return (N_sites, n_elecs)
    mf = matrix.T

    # noise
    noise = normal(0.1)(key, matrix.shape, dtype)
    mf = (matrix + noise).T

    return dtype(mf)


  @nn.compact
  def __call__(self,x):

    n_samples, N_sites = x.shape

    if self.MFinit=="Fermi":
        #orbitals_mfmf = self.param('orbitals_mf',self._init_orbitals_dct,(self.Lx*self.Ly,self.n_elecs), self.dtype)
        orbitals_mfmf = self.param('orbitals_mf',self._init_orbitals_dct,(self.Lx*self.Ly,self.n_elecs), self.dtype)
    elif self.MFinit=="random":
        orbitals_mfmf = self.param('orbitals_mf', normal(0.1),(2*self.Lx*self.Ly,self.n_elecs), self.dtype)
    else:
        raise NotImplementedError("This MF initialization is not implemented! Chose one of: Fermi, random")
    
    orbitals_mfhf = self.param('orbitals_hf', zeros,(self.Lx*self.Ly,self.n_hid), self.dtype)
    #orbitals_mfhf = self.param('orbitals_hf', normal(0.1),(self.Lx*self.Ly,self.n_hid), self.dtype)
    orbitals_full = jnp.concatenate((orbitals_mfmf, orbitals_mfhf), axis=1)
    n_orbs = orbitals_full.shape[1]

    base = jnp.arange(N_sites)          # (N_sites,)
    idx = jnp.broadcast_to(base, (n_samples, N_sites))  # (n_samples, N_sites)

    orbitals_selected = jax.vmap(lambda rows: orbitals_full[rows, :])(idx)

    
    """
    #1.  Convert {-1,+1} → 0/1 occupancy for spin-up, spin-down orbitals
    spin_up = (x == 1).astype(self.dtype)
    spin_dn = (x == -1).astype(self.dtype)
    x_flat = jnp.concatenate([spin_up, spin_dn], axis=1) 

    #2 & 3. Select occupied orbitals using advanced indexing
    # x_flat: (n_samples, 2*N_sites), with 0/1 occupancy
    mask = x_flat.astype(bool)  # (n_samples, 2*N_sites)
    # Get indices of the 1s using top_k
    # Since entries are 0/1, top_k will pick exactly the N_sites "1"s
    _, idx = jax.lax.top_k(mask, k=N_sites)   # shape (n_samples, N_sites)
    orbitals_selected = jax.vmap(lambda i: orbitals_full[i, :])(idx)
    """

    return orbitals_selected # shape: (n_samples, n_elecs, n_orbitals)


    """    # Select first n_elecs occupied orbitals per sample using argsort
    idx_sort = jnp.argsort(-x_flat, axis=1)  # occupied first
    orbitals_full_batched = jnp.broadcast_to(orbitals_full[None, :, :], (n_samples, 2*N_sites, n_orbs))
    orbitals_selected = jax.vmap(lambda mat, idx: mat[idx[:self.n_elecs], :])(orbitals_full_batched, idx_sort)

    return orbitals_selected  # shape: (n_samples, n_elecs, n_orbitals)
    """


    """
    ind1, ind2 = jnp.nonzero(x_flat, size=n_samples*self.n_elecs)
    orbitals_removed = jnp.repeat(jnp.expand_dims(orbitals_full,0),n_samples,axis=0)[ind1,ind2]
    orbitals_removed = jnp.expand_dims(orbitals_removed, axis=0).reshape(n_samples, self.n_elecs, self.n_hid + self.n_elecs)
    
    return orbitals_removed
    """


    """
    # x_flat: (n_samples, 2*N_sites), with 0/1 occupancy
    mask = x_flat.astype(bool)  # (n_samples, 2*N_sites)
    # Get indices of the 1s using top_k
    # Since entries are 0/1, top_k will pick exactly the N_sites "1"s
    _, idx = jax.lax.top_k(mask, k=N_sites)   # shape (n_samples, N_sites)
    x_selected = jax.vmap(lambda i: orbitals_full[i, :])(idx)

    return x_selected # shape: (n_samples, n_elecs, n_orbitals)
    """