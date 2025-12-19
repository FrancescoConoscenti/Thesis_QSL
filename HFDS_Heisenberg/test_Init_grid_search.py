try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    import jax
    jax.distributed.initialize()

    print(f"Rank={rank}: Total number of GPUs: {jax.device_count()}, devices: {jax.devices()}")
    print(f"Rank={rank}: Local number of GPUs: {jax.local_device_count()}, devices: {jax.local_devices()}", flush=True)

    # wait for all processes to show their devices
    comm.Barrier()
except:
  pass


import sys
import argparse
import jax
from jax import numpy as jnp
import netket as nk
import os
import logging
import numpy as np
import matplotlib.pyplot as plt
from netket.operator.spin import sigmax, sigmay, sigmaz, sigmap, sigmam

os.environ['JAX_TRACEBACK_FILTERING'] = 'off'
sys.path.append(os.path.dirname(os.path.dirname("/scratch/f/F.Conoscenti/Thesis_QSL")))

from HFDS_Heisenberg.HFDS_model_spin import HiddenFermion
from HFDS_Heisenberg.Gutzwiller_MF_Init import update_orbitals_gmf
import flax

# Setup logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    stream=sys.stdout)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description="Grid search for Gutzwiller h and phi parameters.")
parser.add_argument("--J2", type=float, default=0.0, help="Coupling parameter J2")
parser.add_argument("--seed", type=int, default=1, help="seed")
parser.add_argument("--grid_points", type=int, default=20, help="Number of points for h and phi in the grid search.")
logger.info("Parsing command-line arguments.")
args = parser.parse_args()

#Physical param
L       = 4
n_dim = 2
J1=1.0
J2 = args.J2
t = 3.0
seed = args.seed
grid_points = args.grid_points

dtype   = "real"
MFinitialization = "G_MF"
bounds  = "PBC"

#Varaitional state param
n_hid_ferm       = 1
features         = 1    #hidden units per layer
hid_layers       = 1

#Network param
n_samples        = 1024
n_chains         = n_samples//2

logger.info("Script starting execution.")

folder = f'HFDS_Heisenberg/plot/Gutzwiller_grid_search/J={J2}'
os.makedirs(folder, exist_ok=True)


lattice = nk.graph.Hypercube(length=L, n_dim=n_dim, pbc=True, max_neighbor_order=2) 
#lattice = nk.graph.Grid([L,L],pbc=True)

hi = nk.hilbert.Spin(s=1 / 2, N=lattice.n_nodes, total_sz=0)
                     
ha = nk.operator.Heisenberg(hilbert=hi, graph=lattice, J=[1.0, J2], sign_rule=[False, False]).to_jax_operator() # No Marshall sign rule
"""def Sz(site):
    return 0.5 * sigmaz(hi, site)
def Splus(site):
    return sigmap(hi, site)
def Sminus(site):
    return sigmam(hi, site)

up, down = +1, -1
ha = 0.0

for u,v in lattice.edges():
    ha += J1*Sz(u)*Sz(v)
    ha += 1/2*J1*Splus(u)*Sminus(v)
    ha += 1/2*J1*Sminus(u)*Splus(v)

"""
if dtype=="real": dtype_ = jnp.float64
else: dtype_ = jnp.complex128

h_min = 0.001  # Minimum value for the staggered magnetic field h
h_max = 0.5  # Maximum value for the staggered magnetic field h
phi_min = 0.0 # Minimum value for the flux phi
phi_max = 0.5   # Maximum value for the flux phi

# --- Grid Search Setup ---
h_values = np.linspace(h_min, h_max, grid_points)
phi_values = np.linspace(phi_min, phi_max, grid_points)
energy_map = np.zeros((grid_points, grid_points))

logger.info(f"Starting grid search for h and phi over a {grid_points}x{grid_points} grid.")

# --- Create sampler once, outside the loop ---
sampler = nk.sampler.MetropolisExchange(
    hilbert=hi,
    graph=lattice,
    d_max=2,
    n_chains=n_chains,
    sweep_size=lattice.n_nodes,
)
key = jax.random.key(seed)

# --- Construct HiddenFermion model and vstate once ---
model = HiddenFermion(lattice=lattice,
                    network="FFNN",
                    n_hid=n_hid_ferm,
                    layers=hid_layers,
                    features=features,
                    MFinit=MFinitialization,
                    hilbert=hi,
                    bounds=bounds,
                    dtype=dtype_,
                    h_opt=0.0, 
                    phi_opt=0.0)

vstate = nk.vqs.MCState(sampler, model, n_samples=n_samples, seed=key, n_discard_per_chain=128)

for i, h_opt in enumerate(h_values):
    for j, phi_opt in enumerate(phi_values):
        logger.info(f"Calculating for h={h_opt:.3f}, phi={phi_opt:.3f}")

        # Update parameters
        new_mf = update_orbitals_gmf(lattice, dtype_, h_opt, phi_opt)
        params = flax.core.unfreeze(vstate.parameters)
        params['orbitals']['orbitals_mf'] = new_mf
        vstate.parameters = params

        """for key in params:
            if key != 'orbitals':
                params[key] = jax.tree_map(jnp.zeros_like, params[key])
        vstate.parameters = params"""
        
        vstate.reset()
        energy_stats = vstate.expect(ha)
        energy_map[j, i] = energy_stats.mean.real

logger.info("Grid search finished.")

# --- Plotting the results ---
fig, ax = plt.subplots()
im = ax.imshow(energy_map, extent=[0, h_max, 0, phi_max], cmap='viridis', aspect='auto', origin='lower')
ax.set_xlabel("h")
ax.set_ylabel("phi")
ax.set_title(f"Initial Energy per site for J2={J2}")
fig.colorbar(im, ax=ax, label="Energy per site")
plt.savefig(os.path.join(folder, f"initial_energy_map_J2_{J2}_netket_Ha_real.png"))
logger.info(f"Energy map plot saved to {os.path.join(folder, f'initial_energy_map_J2_{J2}.png')}")
plt.close(fig)

logger.info("Script finished successfully.")

