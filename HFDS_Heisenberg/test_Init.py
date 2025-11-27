
from HFDS_Heisenberg.Gutzwiller_MF_Init import update_orbitals
import netket as nk
import jax
import jax.numpy as jnp
import numpy as np


L=4
n_elecs = L*L
dtype = "complex"

orbitals_mfmf = update_orbitals(lattice=nk.graph.Hypercube(length=L, n_dim=2, pbc=True), dtype=dtype)

