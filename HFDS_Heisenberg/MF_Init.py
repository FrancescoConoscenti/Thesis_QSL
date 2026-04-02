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


def init_orbitals_mf(L, bounds, phi, dtype):
    bx, by = bounds

    def get_k_val(k_idx, bc):
        if bc == "PBC":
            return 2 * k_idx
        elif bc == "APC":
            return 2 * k_idx + 1
        else:
            raise ValueError(f"Unknown BC: {bc}")

    def ft_local_mixed(x, y, kx, ky):
        """
        Bloch orbital for the twisted case.

        With flux φ threading the x-direction, the allowed x-momenta shift:
            k̃x = (get_k_val(kx, bx) * π + φ) / L

        This ensures the Bloch condition φ(x+L) = e^{iφ} φ(x) is satisfied,
        consistent with the Peierls phase on the boundary bond of the Hamiltonian.

        For φ=0 and both PBC, real combinations (cos/sin) are used when dtype
        is real; otherwise complex exponentials are used throughout.
        """
        if phi == 0.0 and dtype == jnp.float64:
            # Real orbital basis (original logic)
            if bx == "PBC" and by == "PBC":
                if kx <= L // 2 and ky <= L // 2:
                    res = jnp.cos(2*jnp.pi*(x)/L*(kx)) * jnp.cos(2*jnp.pi*(y)/L*(ky))
                elif kx >= L // 2 and ky <= L // 2:
                    res = jnp.sin(2*jnp.pi*(x)/L*(kx)) * jnp.cos(2*jnp.pi*(y)/L*(ky))
                elif kx <= L // 2 and ky >= L // 2:
                    res = jnp.cos(2*jnp.pi*(x)/L*(kx)) * jnp.sin(2*jnp.pi*(y)/L*(ky))
                else:
                    res = jnp.sin(2*jnp.pi*(x)/L*(kx)) * jnp.sin(2*jnp.pi*(y)/L*(ky))
                return res
            else:
                raise NotImplementedError("Real dtype only implemented for PBC/PBC.")
        else:
            # Complex orbital basis — required whenever phi != 0
            # x-momentum picks up the Peierls shift phi/L
            # y-momentum is unchanged (no twist in y)
            val_x = get_k_val(kx, bx)   # integer: 2*kx or 2*kx+1
            val_y = get_k_val(ky, by)
            kx_phys = (val_x * jnp.pi + phi) / L   # shifted momentum
            ky_phys =  val_y * jnp.pi / L
            return jnp.exp(1j * (kx_phys * x + ky_phys * y))

    def energy_fn(k):
        """
        Single-particle energy ε(kx, ky) = -cos(k̃x) - cos(k̃y).
        The twist shifts the x-dispersion by φ/L in momentum space.
        """
        val_x = get_k_val(k[0], bx)
        val_y = get_k_val(k[1], by)
        kx_phys = (val_x * np.pi + phi) / L    # <-- phi shifts x-momentum
        ky_phys =  val_y * np.pi / L
        return -np.cos(kx_phys) - np.cos(ky_phys)

    def ft(k_arr, max_val):
        matrix = []
        for kx, ky in k_arr[:max_val]:
            kstate = [ft_local_mixed(x, y, kx, ky) for y in range(L) for x in range(L)]
            matrix.append(kstate)
        return jnp.array(matrix)

    # Build and sort k-modes by single-particle energy
    k_modes = [(kx, ky) for kx in range(L) for ky in range(L)]
    sorted_k_modes = sorted(k_modes, key=lambda k: (energy_fn(k), k))
    k_arr = np.array(sorted_k_modes)

    n_elecs = L * L
    upmatrix = ft(k_arr, n_elecs // 2)
    dnmatrix = ft(k_arr, n_elecs // 2)

    mf = jnp.block([
        [upmatrix,                      jnp.zeros(upmatrix.shape, dtype=upmatrix.dtype)],
        [jnp.zeros(dnmatrix.shape, dtype=dnmatrix.dtype), dnmatrix],
    ]).T

    return mf.astype(dtype)
