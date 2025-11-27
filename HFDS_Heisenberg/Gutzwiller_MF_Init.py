import sys
sys.path.insert(1, '/project/th-scratch/h/Hannah.Lange/PhD/ML/GutzwillerWaveFunctions/tJ_projection/src')
import argparse
import numpy as np
from jax import numpy as jnp
import netket as nk
import jax
from netket import experimental as nkx
import json
import optax
import os
from jax import numpy as jnp
import netket as nk
import jax
from jax.random import PRNGKey, choice, split
from functools import partial
from flax import linen as nn
from jax.nn.initializers import ones,zeros, normal, constant
from netket.utils.dispatch import dispatch
from netket import experimental as nkx
from netket.jax import apply_chunked
import numpy as np
from netket.hilbert.homogeneous import HomogeneousHilbert



def Hk(sigmaz, phi, h, N_sites, positions, lattice):

  def determine_nns(graph):
    positions =  graph.positions
    nearest_neighbors = []
    for i in positions:
        nearest_neighbors.append({"x": [], "y":[]})
    for x,y in graph.edges(): #enumerate(positions): 
        coord_x = graph.positions[x]
        coord_y = graph.positions[y]
        if coord_x[1]-coord_y[1] == 0: nearest_neighbors[x]["x"].append(y)
        if coord_x[0]-coord_y[0] == 0: nearest_neighbors[x]["y"].append(y)
    return nearest_neighbors

  def hopping_x_conditions(inp):
      H, x, y, ix = inp
      #ix = int(((x + 1) % Lx) * Ly + y)
      flux = (-1)**(jnp.floor(x) + jnp.floor(y)) * 1j * phi * jnp.pi 
      H = H.at[i, ix].add(-0.5 *jnp.exp(flux))
      H = H.at[ix, i].add(-0.5 *jnp.exp(-flux))
      return H

  def hopping_y_conditions(inp):
      H, x, y, iy = inp
      #iy = int(x * Ly + (y + 1) % Ly)
      flux = -1 * (-1)**(jnp.floor(x) + jnp.floor(y)) * 1j * phi * jnp.pi  # here I removed +1 in the y axis and I add -1* in front of everything, it should be the same
      H = H.at[i, iy].add(-0.5 * jnp.exp(-flux))
      H = H.at[iy, i].add(-0.5 *jnp.exp(flux))
      return H
  


  H = jnp.zeros([N_sites,N_sites],dtype=jnp.complex128)
  nearest_neighbors = determine_nns(lattice)

  for i, (x,y) in enumerate(positions):
      
      # hopping terms
      ix = nearest_neighbors[i]["x"]
      iy = nearest_neighbors[i]["y"]
      if len(ix)>0: 
        for r in range(len(ix)):
          H = hopping_x_conditions((H,x,y,ix[r]))
      if len(iy)>0: 
        for r in range(len(iy)):
          H = hopping_y_conditions((H,x,y,iy[r]))

      # staggered magnetic field terms
      H = H.at[i, i].add(jnp.where((x + y) % 2 == 0, -sigmaz * h, sigmaz * h))

      #pert = 1e-5*i/N  # add small perturbation to lift degeneracy
      #H = H.at[i, i].add(jnp.where((x + y) % 2 == 0, -sigmaz * pert, sigmaz * pert)) 

  H = (H + H.transpose().conjugate())/2 # ensure Hermiticity

  # Compute eigenvalues and eigenvectors
  energies, eigenvectors = jnp.linalg.eig(H)
  # Sort eigenvalues and eigenvectors
  indices = jnp.argsort(energies) 
  energies = energies[indices]
  eigenvectors = eigenvectors[:, indices]
  
  return energies, eigenvectors

#################################################################################################################################################################################

def update_orbitals(lattice, dtype):
  
  h = 0.055 #magnetic field
  phi = 0.1 #flux per plaquette
  positions = lattice.positions
  N_sites = len(lattice.positions)
  n_elecs = N_sites

  def initialize_real(n_elecs, sigmaz):

    # Initialize the matrix and the Energy with zeros
    mat = jnp.zeros((n_elecs, len(positions)), dtype=jnp.complex128)
    E = 0
    # Get single particle eigenenergies and states solving the MF Hamiltonian
    energies, eigenstates = Hk(sigmaz, phi, h, N_sites, positions, lattice)

    def body_fn(i, x):
        mat_block, E, eigenstates, energies = x
        def process_r(row_count, x):
            mat_block, eigenstates, energies = x
            # grabs the amplitude of the i-th eigenvector at site rcnt. Get the wave function coefficient for this state and position
            psi = eigenstates[row_count, i]
            #places this amplitude into the output  mat_block
            return  mat_block.at[i, row_count].set(psi), eigenstates, energies
        #it iterates row_count (row count) over every site in the lattice
        x = jax.lax.fori_loop(0, len(positions), process_r, ( mat_block, eigenstates, energies))
        mat_block, eigenstates, energies = x
        return ( mat_block, E, eigenstates, energies)

    # This iterates i from 0 to n_elecs​. It selects the i-th lowest energy orbital.
    x = jax.lax.fori_loop(0, n_elecs, body_fn, (mat, E, eigenstates, energies))

    mat_block, E, eigenstates, energies = x
    E += np.sum(energies[:n_elecs])
    jax.debug.print("MF energy: {E}", E=E)

    return mat_block


  upmatrix = initialize_real(n_elecs, sigmaz = +1)
  dnmatrix = initialize_real(n_elecs,sigmaz = -1)
  mf = jnp.block([[upmatrix, jnp.zeros(upmatrix.shape)], [jnp.zeros(dnmatrix.shape),dnmatrix]]).T

  return dtype(mf)
