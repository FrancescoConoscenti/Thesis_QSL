import itertools
import math
import jax
import jax.numpy as jnp
import numpy as np
import netket as nk
import flax
import os

from Elaborate.Sign_Obs import *

from netket.experimental.observable import AbstractObservable

from functools import partial  # partial(sum, axis=1)(x) == sum(x, axis=1)
from netket.operator import AbstractOperator

from tenpy.networks.mps import MPS


class MarshallSignObs(AbstractOperator):
    def __init__(self, hilbert):
        self._hilbert = hilbert

    @property
    def hilbert(self):
        return self._hilbert

    @property
    def dtype(self):
        return float

    @property
    def is_hermitian(self):
        return True


def e_loc(logpsi, pars, sigma, extra_args, *, A_sites=None, chunk_size=None):
    # extra_args is just sigma (passed for chunking purposes), we ignore it
    
    N_sites = sigma.shape[-1]
    
    # Calculate M_A for the batch
    M_A = 0.5 * jnp.sum(sigma[:, A_sites], axis=-1)
    S_A = 0.5 * (N_sites // 2)
    
    marshall_phase = ((-1.0) ** (S_A - M_A))
    
    log_val = logpsi(pars, sigma).reshape(-1)
    psi = jnp.exp(log_val)
    sign_psi = jnp.sign(jnp.real(psi))
    
    return sign_psi * marshall_phase


@nk.vqs.get_local_kernel.dispatch
def get_local_kernel(vstate: nk.vqs.MCState, op: MarshallSignObs, chunk_size: int,):
    return get_local_kernel(vstate, op)

@nk.vqs.get_local_kernel.dispatch
def get_local_kernel(vstate: nk.vqs.MCState, op: MarshallSignObs):
    N_sites = op.hilbert.size
    A_sites = sublattice_sites(N_sites)
    return partial(e_loc, A_sites=A_sites)


@nk.vqs.get_local_kernel_arguments.dispatch
def get_local_kernel_arguments(vstate: nk.vqs.MCState, op: MarshallSignObs):
    sigma = vstate.samples
    # Return sigma as extra_args so it gets chunked properly by NetKet
    return sigma, sigma


# vectorized function to compute Marshall sign per sample
def _marshal_sign_MCMC(sigma, vstate):
    # sigma is already batched: (n_samples, n_sites)
    N_sites = sigma.shape[1]
    A_sites = sublattice_sites(N_sites)
    
    M_A = 0.5 * jnp.sum(sigma[:, A_sites], axis=1)  # (n_samples,)
    S_A = 0.5 * (N_sites // 2)  # scalar
    
    log_psi = vstate.log_value(sigma)  # Batched call - (n_samples,)
    psi = jnp.exp(log_psi)
    sign = jnp.sign(jnp.real(psi)) * ((-1.0) ** (S_A - M_A))
    
    return sign

# Remove the vmap wrapper - not needed!
get_marshal_sign_MCMC = lambda vstate: lambda sigma: _marshal_sign_MCMC(sigma, vstate)
#########################################################################################################################à
