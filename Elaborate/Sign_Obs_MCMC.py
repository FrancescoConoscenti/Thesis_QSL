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


def e_loc(logpsi, pars, sigma, extra_args, *, chunk_size=None):
    sign = extra_args
    #weights = jnp.exp(2.0 * jnp.real(logpsi))  # |psi|^2 weights
    #return jnp.stack([sign, weights], axis=-1)
    return sign


@nk.vqs.get_local_kernel.dispatch
def get_local_kernel(vstate: nk.vqs.MCState, op: MarshallSignObs, chunk_size: int,):
    return e_loc

@nk.vqs.get_local_kernel.dispatch
def get_local_kernel(vstate: nk.vqs.MCState, op: MarshallSignObs):
    return e_loc


@nk.vqs.get_local_kernel_arguments.dispatch
def get_local_kernel_arguments(vstate: nk.vqs.MCState, op: MarshallSignObs, chunk_size: int,):
    sigma = vstate.samples
    # get the connected elements. Reshape the samples because that code only works
    # if the input is a 2D matrix
    extra_args = get_marshal_sign_MCMC(vstate)(sigma)
    return sigma, extra_args

@nk.vqs.get_local_kernel_arguments.dispatch
def get_local_kernel_arguments(vstate: nk.vqs.MCState, op: MarshallSignObs):
    sigma = vstate.samples
    # get the connected elements. Reshape the samples because that code only works
    # if the input is a 2D matrix
    extra_args = get_marshal_sign_MCMC(vstate)(sigma)
    return sigma, extra_args


# vectorized function to compute Marshall sign per sample
def _marshal_sign_MCMC(sigma, vstate):
    #for samples in MCMC the dimension 1 is N, for samples in full hilbert the dimension 0 is N
    N_sites = sigma.shape[1]
    A_sites = sublattice_sites(N_sites)

    #M_A = jnp.array([sum(sample[::2]) for sample in sigma]) #jnp.sum(0.5 * sigma[A_sites]) # Magn on A sublattice
    M_A = jnp.array(0.5 * jnp.sum(sigma[:, A_sites], axis=1))
    S_A = jnp.ones_like(M_A) * 0.5 * (N_sites // 2) # sum of the spins in the sublattice
    
    log_psi = vstate.log_value(sigma) #log coefficient of wf associated with sample sigma
    psi = jnp.exp(log_psi)
    sign = jnp.sign(jnp.real(psi)) * ((-1.0) ** (S_A - M_A))

    return  sign


get_marshal_sign_MCMC = lambda vstate: jax.vmap(lambda sigma: _marshal_sign_MCMC(sigma, vstate), in_axes=0, out_axes=0)

#########################################################################################################################à
