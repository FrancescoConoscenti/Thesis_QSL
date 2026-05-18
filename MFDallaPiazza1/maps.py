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
import flax
from tqdm import tqdm
from datetime import datetime

from netket.experimental.operator.fermion import destroy as c
from netket.experimental.operator.fermion import create as cdag
from netket.experimental.operator.fermion import number as nc
from netket.operator.spin import sigmax,sigmay,sigmaz
from netket.jax import apply_chunked
from functools import partial
from collections.abc import Callable
import matplotlib.pyplot as plt



from MFDallaPiazza1.gutzwiller import *
from MFDallaPiazza1.tJ_sampler import *

def grid_search(vstate, ha, positions, nearest_neighbors, lattice):

    Es = []
    hs = []
    phis = []

    # Initialise mf before the first sample() call so parameters are consistent
    # from the start.  The _update_orbitals initialiser in gutzwiller.py returns
    # zeros to avoid a JAX sharding error; the real mf is set here instead.
    _h0 = jnp.array(0.0); _phi0 = jnp.array(0.0)
    _init = vstate.parameters.copy()
    _init["h"]   = _h0
    _init["phi"] = _phi0
    _init["mf"]  = update_orbitals_apbc(
        (2*N_sites, n_elecs), _h0, _phi0, n_elecs, positions, nearest_neighbors, lattice)
    vstate.parameters = _init
    vstate.sample()

    # Extended h range to 0.15 to match the full axis of Fig. 2.14
    h_vals = jnp.arange(0.001, 0.15, 0.001)
    phi_vals = jnp.arange(0.005, 0.49, 0.002)

    for h in h_vals:
        E_ = []
        for phi in phi_vals:
            new_pars = vstate.parameters.copy()
            new_pars["h"]   = h
            new_pars["phi"] = phi
            new_pars["mf"]  = update_orbitals_apbc(
                (2*N_sites, n_elecs), h, phi, n_elecs, positions, nearest_neighbors, lattice)
            vstate.parameters = new_pars
            E = vstate.expect(ha)
            E_.append(np.array([E.mean, E.error_of_mean]))
            print(h, phi, E.mean, E.error_of_mean, E.R_hat, E.tau_corr)
            hs.append(h)
            phis.append(phi)
        Es.append(jnp.stack(E_))

    energy = jnp.stack(Es)
    hs     = jnp.stack(hs)
    phis   = jnp.stack(phis)
    jnp.save(filename + "_energy_map", energy)
    jnp.save(filename + "_h_map",      hs)
    jnp.save(filename + "_phi_map",    phis)

    # Plot the energy map
    X, Y = np.meshgrid(h_vals, phi_vals)
    plt.figure(figsize=(8, 6))
    c = plt.pcolormesh(X, Y, energy[..., 0].real, shading='auto', cmap='viridis')
    plt.colorbar(c, label='Mean Energy')
    plt.xlabel(r'$\phi$')
    plt.ylabel('h')
    plt.title('Energy Map')
    plt.savefig(filename + "_energy_map.png", dpi=300)
    plt.close()

def Fidelity(vstate_1, vstate_2, n_samples=4096):
    """
    Computes the fidelity |<psi_1|psi_2>|^2 between two variational states
    defined on the SAME Hilbert space.
    """
    # Estimate the overlap <psi_1|psi_2> by sampling from psi_1's distribution
    # <psi_1|psi_2> = E_{x ~ |psi_1|^2} [ psi_2(x) / psi_1(x) ]
    samples = vstate_1.sample(n_samples=n_samples).reshape(-1, vstate_1.hilbert.size)

    log_psi_1 = vstate_1.log_value(samples)
    log_psi_2 = vstate_2.log_value(samples)

    log_ratios = log_psi_2 - log_psi_1
    
    ratios = jnp.exp(log_ratios)
    overlap = jnp.mean(ratios)
    
    fidelity = jnp.abs(overlap)**2
    
    # Error of the mean of the ratios
    err = jnp.std(ratios) / jnp.sqrt(n_samples)
    return fidelity, err




parser = argparse.ArgumentParser()
parser.add_argument("-Nx" , "--Nx"   , type=int,  default = 10 , help="length in x dir")
parser.add_argument("-Ny" , "--Ny"   , type=int,  default = 10 , help="length in y dir")
parser.add_argument("-Jz"  , "--Jz"    , type=float,default = 1. , help="spin-spin interaction")
parser.add_argument("-Jp"  , "--Jp"    , type=float,default = 1. , help="spin-spin interaction")
parser.add_argument("-t"  , "--t"    , type=float,default = 3. , help="hopping amplitude")
parser.add_argument("-lattice"  , "--lattice"    , type=str,default = "square" , help="Lattice type: square, Lieb")

args = parser.parse_args()
L1      = args.Nx
L2      = args.Ny
n_elecs = L1*L2
Jz      = args.Jz
Jp      = args.Jp
t       = args.t
lattice = args.lattice


# more parameters for the physical system
N_up    = (n_elecs+1)//2
N_dn    = n_elecs//2

double_occupancy = False

# network parameters and sampling
n_samples        = 4096
n_chains         = n_samples   #4096//4
cs               = n_samples

# --------------- define the network -------------------
filename = f"5test_apbc_alt_{L1}x{L2}_Nup={N_up}_Ndn={N_dn}_t={t}_Jz={Jz}_Jp={Jp}_"+lattice

if lattice=="square":
  g = nk.graph.Grid([L1,L2],pbc=True)
elif lattice=="Lieb":
  basis_vectors = [[1, 0], [0, 1]] # Basis vectors defining the unit cell
  atom_positions = [[0, 0], [0.5, 0], [0.5, 0.5]] # Atom positions within the unit cell
  dimensions = [L1, L2]
  # Define the graph
  g = nk.graph.Lattice(basis_vectors=basis_vectors, site_offsets=atom_positions, extent=dimensions, pbc=True)
positions =  g.positions
N_sites   = len(positions)
# Hilbert space of N_up + N_dn fermions on N_sites sites. The number of
# fermions in each spin sector is conserved. Double occupations can occur.
# Therefore, the dimension of this Hilbert space is
# N! / (N_up! * (N-N_up)!) * N! / (N_dn! * (N-N_dn)!)
hi = nkx.hilbert.SpinOrbitalFermions(N_sites, s = 1/2, n_fermions_per_spin = (N_up, N_dn))
# hi = nkx.hilbert.SpinOrbitalFermions(N_sites, s = 1/2, n_fermions = N_up+N_dn)

def Sz(site):
    return 1/2*(nc(hi, site, up) - nc(hi, site, down))
def Splus(site):
    return cdag(hi, site,up)*c(hi, site,down)
def Sminus(site):
    return cdag(hi, site,down)*c(hi, site,up)

up, down = +1, -1
ha = 0.0
for sz in (up, down):
    for u, v in g.edges():
        ha += -t*(cdag(hi, u, sz) * c(hi, v, sz)  + cdag(hi, v, sz) * c(hi, u, sz))

for u,v in g.edges():
    ha += Jz*Sz(u)*Sz(v)
    ha += 1/2*Jp*Splus(u)*Sminus(v)
    ha += 1/2*Jp*Sminus(u)*Splus(v)
    #ha -= 1/4*Jz*(nc(hi,u,up) + nc(hi,u,down))*(nc(hi,v,up) + nc(hi,v,down))


# staggered magnetization
Sz_staggered = 0.0
for u in range(N_sites):
   (x, y) = positions[u]
   Sz_staggered += (1-2*((x+y)%2))*Sz(u)


# ---------- define sampler ------------------------
if double_occupancy:
    sa = nk.sampler.MetropolisSampler(hi, n_chains=n_chains, rule=ExchangeRule(graph=g))
else:
    sa = nk.sampler.MetropolisSampler(hi, n_chains=n_chains, rule=tJExchangeRule(graph=g))
# machine_power = 2, since the GutzwillerWaveFunction class returns log Psi = 1/2 log pdf
print("sampler: ", sa)


print("------------ Generate model ------------")
ma = GutzwillerWaveFunction(n_elecs=n_elecs,
                   Lx=L1,
                   Ly=L2,
                   double_occupancy_bool=double_occupancy,
                   hilbert=hi,
                   graph = g,
                   lattice = lattice)
nearest_neighbors = ma.determine_nns(g, L1, L2)
print("List of NN: ", nearest_neighbors)

print("------------ Initialize MCstate ------------")
vstate = nk.vqs.MCState(sa, ma, n_samples=n_samples, n_discard_per_chain=8, chunk_size=32) #defines the variational state object
total_params = sum(p.size for p in jax.tree_util.tree_leaves(vstate.parameters))
print(f'Total number of parameters: {total_params}')

print("------------ Grid Search ------------")
#grid_search(vstate, ha, positions, nearest_neighbors, lattice)


print("------------ Fidelity ------------")
h, phi = 0.08, 0.1
new_pars = vstate.parameters.copy()
new_pars["h"]   = h
new_pars["phi"] = phi
new_pars["mf"]  = update_orbitals_apbc((2*N_sites, n_elecs), h, phi, n_elecs, positions, nearest_neighbors, lattice)
vstate.parameters = new_pars

print("------------ Cross-Basis Fidelity ------------")

# Ensure project root is in the path to import HFDS components
sys.path.append("/scratch/f/F.Conoscenti/Thesis_QSL")
from HFDS_Heisenberg.HFDS_model_spin import HiddenFermion

def transform_spin_to_fermion(samples_spin):
    """ Maps nk.hilbert.Spin (-1, 1) to SpinOrbitalFermions single-occupancy format ([up_spins...], [dn_spins...]) """
    up_occ = (samples_spin == 1).astype(jnp.float64)
    dn_occ = (samples_spin == -1).astype(jnp.float64)
    return jnp.concatenate([up_occ, dn_occ], axis=-1)

def Fidelity_cross_basis(vstate_fermion, vstate_spin, n_samples=4096):
    """ Samples from the Spin state, transforms, and computes overlap with the Fermion state. """
    samples_spin = vstate_spin.sample(n_samples=n_samples).reshape(-1, vstate_spin.hilbert.size)
    log_psi_spin = vstate_spin.log_value(samples_spin)
    
    samples_fermion = transform_spin_to_fermion(samples_spin)
    log_psi_fermion = vstate_fermion.log_value(samples_fermion)
    
    log_ratios = log_psi_fermion - log_psi_spin
    ratios = jnp.exp(log_ratios)
    overlap = jnp.mean(ratios)
    
    fidelity = jnp.abs(overlap)**2
    err = jnp.std(ratios) / jnp.sqrt(n_samples)
    return fidelity, err

# Create appropriate Hilbert space and sampler for the HiddenFermion model
hi_spin = nk.hilbert.Spin(s=1/2, N=N_sites, total_sz=0)
sa_spin = nk.sampler.MetropolisExchange(hi_spin, graph=g, d_max=2, n_chains=128)

# Instantiate the HiddenFermion model matching the requested parameters in the path
model_HF = HiddenFermion(
    L=L1, N_sites=N_sites, network="FFNN", n_hid=8, layers=1, features=32,
    MFinit="Fermi", hilbert=hi_spin, stop_grad_mf=False, stop_grad_lower_block=False,
    bounds="PBC", parity=True, rotation=True, dtype=jnp.complex128
)

vstate_HF = nk.vqs.MCState(sa_spin, model_HF, n_samples=4096)

vstate_HF_folder = "/scratch/f/F.Conoscenti/Thesis_QSL/HFDS_Heisenberg/plot/10x10/layers1_hidd8_feat32_sample4096_lr0.02_iter2000_parityTrue_rotTrue_InitFermi_typecomplex_10"
models_dir = os.path.join(vstate_HF_folder, "J=0.5", "seed_1", "models")
model_path = None

import re
if os.path.exists(models_dir):
    model_files = [f for f in os.listdir(models_dir) if f.startswith("model_") and f.endswith(".mpack")]
    if model_files:
        model_files.sort(key=lambda x: int(re.search(r"model_(\d+)\.mpack", x).group(1)))
        model_path = os.path.join(models_dir, model_files[-1])

try:
    if model_path is None:
        raise FileNotFoundError(f"No .mpack models found in {models_dir}")
    with open(model_path, 'rb') as f:
        data = f.read()
        try:
            vstate_HF = flax.serialization.from_bytes(vstate_HF, data)
        except KeyError:
            vstate_HF.variables = flax.serialization.from_bytes(vstate_HF.variables, data)
    print(f"Successfully loaded HF parameters from {model_path}")
    
    fid, err = Fidelity_cross_basis(vstate, vstate_HF, n_samples=4096)
    print(f"Fidelity between Gutzwiller and HF states: {fid} ± {err}")
except FileNotFoundError:
    print(f"Error: Model file not found at {model_path}. Please check the folder structure.")
except Exception as e:
    print(f"An error occurred while computing fidelity: {e}")
