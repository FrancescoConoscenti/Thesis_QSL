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

    if isinstance(bounds, str):
        bx = bounds
        by = bounds
    else:
        bx, by = bounds

    def get_k_val(k_idx, bc):
        if bc == "PBC":
            return 2 * k_idx
        elif bc == "APC":
            return 2 * k_idx + 1
        else:
            raise ValueError(f"Unknown BC: {bc}")

    def ft_local_mixed(x, y, kx, ky):
        if dtype == jnp.float64:
             if bx == "PBC" and by == "PBC":
                if kx<=L//2 and ky<=L//2:
                    res = jnp.cos(2*jnp.pi*(x)/L*(kx))*jnp.cos(2*jnp.pi*(y)/L*(ky))
                elif kx>=L//2 and ky<=L//2:
                    res = jnp.sin(2*jnp.pi*(x)/L*(kx))*jnp.cos(2*jnp.pi*(y)/L*(ky)) 
                elif kx<=L//2 and ky>=L//2:
                    res = jnp.cos(2*jnp.pi*(x)/L*(kx))*jnp.sin(2*jnp.pi*(y)/L*(ky)) 
                elif kx>=L//2 and ky>=L//2:
                    res = jnp.sin(2*jnp.pi*(x)/L*(kx))*jnp.sin(2*jnp.pi*(y)/L*(ky))
                return res
             else:
                 raise NotImplementedError("Real dtype only implemented for PBC in both directions.")
        else:
             val_x = get_k_val(kx, bx)
             val_y = get_k_val(ky, by)
             return jnp.exp(1j * jnp.pi / L * (val_x * x + val_y * y))

    ft_local = ft_local_mixed
    energy_fn = lambda k: -np.cos(get_k_val(k[0], bx) * np.pi / L) - np.cos(get_k_val(k[1], by) * np.pi / L)

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