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
from jax.scipy.special import logsumexp

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
  
    # Construct rotation permutation once
    idx = jnp.arange(self.Lx * self.Ly).reshape(self.Ly, self.Lx)
    idx_rot = jnp.flip(idx.T, axis=1).reshape(-1)

    # Apply to all batch elements
    x_rot = x[:, idx_rot]
    return x_rot

  def __call__(self,x):
    
    log_psi, sign = self.calc_psi(x)
    #log_psi += sign  # include sign once

    # --- Step 1: add parity symmetry (if enabled) ---
    if self.parity:
        x_refl = self.gen_reflected_samples(x)
        log_psi_refl, sign_refl = self.calc_psi(x_refl)
        #log_psi_refl += sign_refl
        log_psi = logsumexp(jnp.stack([log_psi, log_psi_refl]), axis=0) - jnp.log(2)

    # --- Step 2: add rotation symmetry (on the current symmetrized state) ---
    if self.rotation:
        x_rot1 = self.gen_rotated_samples(x)
        log_psi_rot1, sign_rot = self.calc_psi(x_rot1)

        x_rot2 = self.gen_rotated_samples(x_rot1)
        log_psi_rot2, sign_rot = self.calc_psi(x_rot2)

        x_rot3 = self.gen_rotated_samples(x_rot2)
        log_psi_rot3, sign_rot = self.calc_psi(x_rot3)

        #log_psi_rot += sign_rot
        log_psi = logsumexp(jnp.stack([log_psi, log_psi_rot1, log_psi_rot2, log_psi_rot3 ]), axis=0) - jnp.log(4)

    return log_psi + sign

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
      if self.dtype==jnp.float64:
        if kx<=self.Lx//2 and ky<=self.Ly//2:
            res = jnp.cos(2*jnp.pi*(x)/self.Lx*(kx))*jnp.cos(2*jnp.pi*(y)/self.Ly*(ky))
        elif kx>=self.Lx//2 and ky<=self.Ly//2:
            res = jnp.sin(2*jnp.pi*(x)/self.Lx*(kx))*jnp.cos(2*jnp.pi*(y)/self.Ly*(ky)) 
        elif kx<=self.Lx//2 and ky>=self.Ly//2:
            res = jnp.cos(2*jnp.pi*(x)/self.Lx*(kx))*jnp.sin(2*jnp.pi*(y)/self.Ly*(ky)) 
        elif kx>=self.Lx//2 and ky>=self.Ly//2:
            res = jnp.sin(2*jnp.pi*(x)/self.Lx*(kx))*jnp.sin(2*jnp.pi*(y)/self.Ly*(ky)) 
      else:
        res = jnp.exp(1j*2*jnp.pi*(kx/self.Lx*x + ky/self.Ly*y))
      return res


    def ft(k_arr, max_val, sigmaz):
      if self.bounds=="PBC":
        matrix = []
        for idx,(kx, ky) in enumerate(k_arr[:max_val]):
          kstate = [ft_local_pbc(x,y,kx,ky) for y in range(self.Ly) for x in range(self.Lx)]
          matrix.append(kstate)
          #jax.debug.print("{x}",x=(-np.cos(2*np.pi*kx/self.Lx) - np.cos(2*np.pi*ky/self.Ly),kstate,kx,ky))
      return jnp.array(matrix)

    n_elecs = shape[1]
    k_modes = []
    for kx in range(0, self.Lx):
      for ky in range(0, self.Ly):
        k_modes.append((kx,ky))
    sorted_k_modes = sorted(k_modes, key=lambda x: (-np.cos(2*np.pi*x[0]/self.Lx) - np.cos(2*np.pi*x[1]/self.Ly), x))
    k_arr = np.array(sorted_k_modes)
    upmatrix = ft(k_arr, (n_elecs+1)//2,+1)
    dnmatrix = ft(k_arr, n_elecs//2,-1)
    mf = jnp.block([[upmatrix, jnp.zeros(upmatrix.shape)], [jnp.zeros(dnmatrix.shape),dnmatrix]]).T
    #jax.debug.print("mf={x}",x=mf)
    return dtype(mf)
  
  
  def _init_orbitals_hartree(self, key, shape, dtype):
    mf = np.load("/project/th-scratch/h/Hannah.Lange/PhD/ML/HiddenFermions/src/orbs_8.0_16_14.npy")
    mf = jnp.array(mf)
    jax.debug.print("mf={x}",x=mf)
    return dtype(mf)


  def _init_orbitals_striped(self, key, shape, dtype):
    return 1


  @nn.compact
  def __call__(self,x):

    n_samples, N_sites = x.shape

    if self.MFinit=="Fermi":
        #orbitals_mfmf = self.param('orbitals_mf',self._init_orbitals_dct,(self.Lx*self.Ly,self.n_elecs), self.dtype)
        orbitals_mfmf = self.param('orbitals_mf',self._init_orbitals_dct,(2*self.Lx*self.Ly,self.n_elecs), self.dtype)
    elif self.MFinit=="Hartree":
        orbitals_mfmf = self.param('orbitals_mf',self._init_orbitals_hartree,(2*self.Lx*self.Ly,self.n_elecs), self.dtype)
    elif self.MFinit=="random":
        orbitals_mfmf = self.param('orbitals_mf', normal(0.1),(2*self.Lx*self.Ly,self.n_elecs), self.dtype)
    elif self.MFinit=="striped":
        orbitals_mfmf = self.param('orbitals_mf',self._init_orbitals_striped, (2 * self.Lx * self.Ly, self.n_elecs), self.dtype)
    else:
        raise NotImplementedError("This MF initialization is not implemented! Chose one of: Fermi, random")
    
    orbitals_mfhf = self.param('orbitals_hf', zeros,(2*self.Lx*self.Ly,self.n_hid), self.dtype)

    orbitals_full = jnp.concatenate((orbitals_mfmf, orbitals_mfhf), axis=1)
    n_orbs = orbitals_full.shape[1]

    
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