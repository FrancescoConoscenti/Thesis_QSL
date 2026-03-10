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

    def ft_local_apc(x, y, kx, ky):
        if dtype == jnp.float64:
            raise NotImplementedError("APC is not implemented for real dtype in MF initialization.")
        else:
            # kx, ky are integers n_x, n_y
            res = jnp.exp(1j * jnp.pi / L * ((2 * kx + 1) * x + (2 * ky + 1) * y))
        return res

    if bounds == "PBC":
        ft_local = ft_local_pbc
        energy_fn = lambda k: -np.cos(2 * np.pi * k[0] / L) - np.cos(2 * np.pi * k[1] / L)
    elif bounds == "APC":
        ft_local = ft_local_apc
        energy_fn = lambda k: -np.cos((2 * k[0] + 1) * np.pi / L) - np.cos((2 * k[1] + 1) * np.pi / L)
    else:
        raise ValueError(f"Unknown bounds: {bounds}")

    def ft(k_arr, max_val):
        matrix = []
        for idx,(kx, ky) in enumerate(k_arr[:max_val]):
            kstate = [ft_local(x, y, kx, ky) for y in range(L) for x in range(L)]
            matrix.append(kstate)
        return jnp.array(matrix)

    n_elecs = L*L
    k_modes = []
    for kx in range(0, L):
      for ky in range(0, L):
        k_modes.append((kx,ky))
    sorted_k_modes = sorted(k_modes, key=lambda k: (energy_fn(k), k))
    k_arr = np.array(sorted_k_modes)
    upmatrix = ft(k_arr, n_elecs//2)
    dnmatrix = ft(k_arr, n_elecs//2)
    mf = jnp.block([[upmatrix, jnp.zeros(upmatrix.shape)], [jnp.zeros(dnmatrix.shape),dnmatrix]]).T

    return dtype(mf)