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


def update_orbitals_gmf(lattice, dtype, h, phi):
    
    positions = lattice.positions
    N = len(lattice.positions)
    n_elecs = N
    apbc_phase = jnp.exp(1j * jnp.pi)
    Lx, Ly = int(np.sqrt(N)), int(np.sqrt(N))


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

    nearest_neighbors = determine_nns(lattice)


    def Hk(sigmaz):

      def hopping_x_conditions(inp):
          H, x, y, ix = inp
          #ix = int(((x + 1) % Lx) * Ly + y)
          flux = (-1)**(jnp.floor(x) + jnp.floor(y)) * 1j * phi * jnp.pi 
          H = H.at[i, ix].add(-jnp.exp(flux))
          H = H.at[ix, i].add(-jnp.exp(-flux))
          return H

      def hopping_y_conditions(inp):
          H, x, y, iy = inp
          #iy = int(x * Ly + (y + 1) % Ly)
          flux = (-1)**(jnp.floor(x) + jnp.floor(y) + 1) * 1j * phi * jnp.pi
          H = H.at[i, iy].add(-jnp.exp(flux))
          H = H.at[iy, i].add(-jnp.exp(-flux))
          return H

      H = jnp.zeros([N,N],dtype=jnp.complex128)
      for i, (x,y) in enumerate(positions):
          # hopping 
          ix = nearest_neighbors[i]["x"]
          iy = nearest_neighbors[i]["y"]
          if len(ix)>0: 
            for r in range(len(ix)):
              H = hopping_x_conditions((H,x,y,ix[r]))
          if len(iy)>0: 
            for r in range(len(iy)):
              H = hopping_y_conditions((H,x,y,iy[r]))
          # staggered magnetic field
          H = H.at[i, i].add(jnp.where((x + y) % 2 == 0, -sigmaz * h, sigmaz * h))
          #pert = 1e-5*i/N #add small perturbation to lift degeneracy
          #H = H.at[i, i].add(jnp.where((x + y) % 2 == 0, -sigmaz * pert, sigmaz * pert)) 
      #H = (H+H.transpose().conjugate())/2
      # Compute eigenvalues and eigenvectors

      en, us = jnp.linalg.eig(H)
      indices = jnp.argsort(en)  # Use jnp.argsort for JAX compatibility
      energies = en[indices]
      us = us[:, indices]
      return en, us


    def initialize_real(num_particles,sigmaz):
      def body_fn(i, x):
        mat, E, us, energies = x
        def process_r(rcnt, x):
            mat, us, energies = x
            psi = us[rcnt, i]  # Get the wave function coefficient for this state and position
            x = mat.at[i, rcnt].set(psi), us, energies
            return x
        # Use jax.lax.fori_loop to iterate over rs
        x = jax.lax.fori_loop(0, len(positions), process_r, (mat, us, energies))
        mat, us, energies = x
        E += energies[i]
        return (mat, E, us, energies)


      # 'orbitals' are now just eigenstates of single particle Hamiltonian (run from 0 to mX*mY-1) 
      ks = jnp.arange(N)
      mat = np.zeros([num_particles,len(positions)])
      # get single particle eigenenergies and states
      energies, us = Hk(sigmaz)
      # Initialize the matrix with zeros
      mat = jnp.zeros((num_particles, len(positions)), dtype=jnp.complex128)
      E = 0
      x = jax.lax.fori_loop(0, num_particles, body_fn, (mat, E, us, energies))
      mat, E, us, energies = x
      jax.debug.print("MF energy: {E}", E=E)
      return mat

    upmatrix = initialize_real((n_elecs+1)//2,+1)
    dnmatrix = initialize_real(n_elecs//2,-1)
    mf = jnp.block([[upmatrix, jnp.zeros(upmatrix.shape)], [jnp.zeros(dnmatrix.shape),dnmatrix]]).T
    #jax.debug.print("mf: {x}",x=mf)
    return mf


class GutzwillerWaveFunction(nn.Module):
  n_elecs: int
  Lx: int
  Ly: int
  double_occupancy_bool: bool
  hilbert: HomogeneousHilbert
  graph: nk.graph.AbstractGraph
  lattice: str

  def setup(self):
    self.n_modes = 2*len(self.graph.positions)
    self.nearest_neighbors = self.determine_nns(self.graph, self.Lx, self.Ly)
    self.phi = jax.lax.stop_gradient(jnp.abs(self.param("phi", normal(0.2), (), jnp.float64)))+0j
    self.h = jax.lax.stop_gradient(jnp.abs(self.param("h", normal(0.2), (), jnp.float64)))+0j
    self.mf = jax.lax.stop_gradient(self.param("mf", self._update_orbitals, (self.n_modes,self.n_elecs)))
    #add two dummy variables since no of parameters has to be divisable by number of nodes
    self.a = self.param("a", zeros, (2,), jnp.float64)

  def double_occupancy(self,x):
    x = x[:,:x.shape[-1]//2] + x[:,x.shape[-1]//2:]
    return jnp.where(jnp.any(x > 1.5,axis=-1),True,False)

  def _update_orbitals(self, key, shape):
    return update_orbitals_gmf(shape, self.h, self.phi, self.n_elecs, self.graph.positions, self.nearest_neighbors, self.lattice)

  @classmethod
  def determine_nns(self, graph, Lx, Ly):
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


  def __call__(self,x):
    ind1, ind2 = jnp.nonzero(x,size=x.shape[0]*self.n_elecs)
    orbitals = jnp.repeat(jnp.expand_dims(self.mf,0),x.shape[0],axis=0)[ind1,ind2]
    orbitals = orbitals.reshape(-1,self.n_elecs,orbitals.shape[1])
    do = self.double_occupancy(x)
    sign, x = jnp.linalg.slogdet(orbitals)
    if self.double_occupancy_bool: 
      return x + jnp.log(sign + 0j)
    else:
      return x + jnp.log(sign + 0j) - 1e14*do

