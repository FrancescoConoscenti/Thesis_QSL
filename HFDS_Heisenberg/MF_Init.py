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


def init_orbitals_mf(L, bounds, dtype):
    def ft_local_pbc(x,y,kx,ky):
      if dtype==jnp.float64:
        if kx<=L//2 and ky<=L//2:
            res = jnp.cos(2*jnp.pi*(x)/L*(kx))*jnp.cos(2*jnp.pi*(y)/L*(ky))
        elif kx>=L//2 and ky<=L//2:
            res = jnp.sin(2*jnp.pi*(x)/L*(kx))*jnp.cos(2*jnp.pi*(y)/L*(ky)) 
        elif kx<=L//2 and ky>=L//2:
            res = jnp.cos(2*jnp.pi*(x)/L*(kx))*jnp.sin(2*jnp.pi*(y)/L*(ky)) 
        elif kx>=L//2 and ky>=L//2:
            res = jnp.sin(2*jnp.pi*(x)/L*(kx))*jnp.sin(2*jnp.pi*(y)/L*(ky)) 
      else:
        res = jnp.exp(1j*2*jnp.pi*(kx/L*x + ky/L*y))
      return res


    def ft(k_arr, max_val, sigmaz):
      if bounds=="PBC":
        matrix = []
        for idx,(kx, ky) in enumerate(k_arr[:max_val]):
          kstate = [ft_local_pbc(x,y,kx,ky) for y in range(L) for x in range(L)]
          matrix.append(kstate)
          #jax.debug.print("{x}",x=(-np.cos(2*np.pi*kx/L) - np.cos(2*np.pi*ky/L),kstate,kx,ky))
      return jnp.array(matrix)

    n_elecs = L*L
    k_modes = []
    for kx in range(0, L):
      for ky in range(0, L):
        k_modes.append((kx,ky))
    sorted_k_modes = sorted(k_modes, key=lambda x: (-np.cos(2*np.pi*x[0]/L) - np.cos(2*np.pi*x[1]/L), x))
    k_arr = np.array(sorted_k_modes)
    upmatrix = ft(k_arr, n_elecs//2, sigmaz = +1)
    dnmatrix = ft(k_arr, n_elecs//2, sigmaz = -1)
    mf = jnp.block([[upmatrix, jnp.zeros(upmatrix.shape)], [jnp.zeros(dnmatrix.shape),dnmatrix]]).T
    #jax.debug.print("MF init orbitals shape: {x}",x=mf.shape)

    return dtype(mf)