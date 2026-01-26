from jax import numpy as jnp
import netket as nk
import jax
from flax import linen as nn
import numpy as np
from netket.hilbert.homogeneous import HomogeneousHilbert 
from netket.jax import logsumexp_cplx
import logging
from typing import Callable

from HFDS_Heisenberg.Init_orbitals_ent import Orbitals_ent


logger = logging.getLogger(__name__)

class HiddenFermion_ent(nn.Module):
  L: int
  network: str
  n_hid: int
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
  U: float = 8.0
  kernel_init: Callable = nn.initializers.normal(stddev=0.1)

  def setup(self):
    #logger.info("Setting up HiddenFermion model.")
    # orbital Initialization
    self.n_modes = 2 * self.L*self.L
    self.n_elecs = self.L * self.L
    self.orbitals = Orbitals_ent(self.L, self.n_elecs, self.n_hid, self.MFinit, self.stop_grad_mf, self.bounds, self.dtype, self.U, kernel_init=self.kernel_init)
    # FFNN architecture
    if self.network=="FFNN":
        self.hidden = [nn.Dense(features=self.features,use_bias=False,param_dtype=self.dtype, kernel_init=self.kernel_init) for i in range(self.layers)]
        self.output = nn.Dense(features=self.n_hid*(self.n_elecs + self.n_hid),use_bias=True,param_dtype=self.dtype, kernel_init=self.kernel_init)
    else:
        raise NotImplementedError()
    # Rotation symmetry indices
    if self.rotation:
      #L = int(jnp.sqrt(self.lattice.n_nodes))
      idx = jnp.arange(self.L* self.L).reshape(self.L, self.L)
      self.idx_rot = jnp.flip(idx.T, axis=1).reshape(-1)
    #logger.info("HiddenFermion model setup complete.")


  def selu(self,x):
    if self.dtype==jnp.float64:
      return jax.nn.selu(x)
    else:
      return jax.nn.selu(x.real) +1j*jax.nn.selu(x.imag)


  def calc_psi(self,x):
    #logger.debug("Executing calc_psi.")

    #1, 2, 3.
    orbitals = self.orbitals(x)

    # 4. Forward pass through the NN, create x_
    for i in range(self.layers):
        x = self.selu(self.hidden[i](x))
    x_ = self.output(x).reshape(x.shape[0],self.n_hid,self.n_elecs + self.n_hid)

    x_2 = jnp.repeat(jnp.expand_dims(jnp.eye(self.n_hid), axis=0),x.shape[0],axis=0)
    x_ += jnp.concatenate((jnp.zeros((x.shape[0], self.n_hid, self.n_elecs),self.dtype), x_2),axis=2)
    #x_ = jnp.concatenate((jnp.zeros((x.shape[0], self.n_hid, self.n_elecs),self.dtype), x_2),axis=2)  # just to evaluate energy of G_MF
    
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
    
    return x_sym
  

  def __call__(self,x):
    #logger.debug("Calling HiddenFermion model.")

    x_sym = self.gen_sym_samples(x)
  
    log_det, log_sign = jax.vmap(self.calc_psi)(jnp.stack(x_sym))
    log_psi_sym = log_det + log_sign
    return logsumexp_cplx(log_psi_sym, axis=0)