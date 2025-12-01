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
import flax
os.environ['JAX_TRACEBACK_FILTERING'] = 'off'
import pickle
sys.path.append(os.path.dirname(os.path.dirname("/scratch/f/F.Conoscenti/Thesis_QSL")))

from netket.experimental.driver import VMC_SR

from Elaborate.Statistics.Energy import *
from Elaborate.Statistics.Corr_Struct import *
from Elaborate.Statistics.Error_Stat import *
from Elaborate.Statistics.count_params import *
from Elaborate.Plotting.Sign_vs_iteration import *
from Elaborate.Sign_Obs import *
from Elaborate.Plotting.S_matrix_vs_iteration import *

from jax import numpy as jnp
import netket as nk
import jax
from jax.random import PRNGKey, choice, split
from functools import partial
from flax import linen as nn, struct
from jax.nn.initializers import zeros, normal, constant
from netket.utils.dispatch import dispatch
from netket import experimental as nkx
from netket.jax import apply_chunked
import numpy as np
from netket.hilbert.homogeneous import HomogeneousHilbert 
from netket.jax import logsumexp_cplx
from jax import Array

from HFDS_Heisenberg.MF_Init import init_orbitals_mf
from HFDS_Heisenberg.Gutzwiller_MF_Init import update_orbitals_gmf

from HFDS_Heisenberg.HFDS_model_spin import Orbitals



parser = argparse.ArgumentParser(description="Example script with parameters")
parser.add_argument("--J2", type=float, default=0.0, help="Coupling parameter J2")
parser.add_argument("--seed", type=float, default=1, help="seed")
args = parser.parse_args()

spin = True

#Physical param
L       = 4
n_elecs = L*L # L*L should be half filling
N_sites = L*L
N_up    = (n_elecs+1)//2
N_dn    = n_elecs//2
n_dim = 2

J1J2 = True
J2 = args.J2
seed = int(args.seed)

dtype   = "complex"
MFinitialization = "G_MF" #G_MF#random #Fermi
determinant_type = "hidden"
bounds  = "PBC"
parity = True
rotation = True

#Varaitional state param
n_hid_ferm       = 2
features         = 4    #hidden units per layer
hid_layers       = 1

#Network param
lr               = 0.025
n_samples        = 1024
N_opt            = 100

number_data_points = 20
save_every       = N_opt//number_data_points
block_iter       = N_opt//save_every

n_chains         = n_samples//2


model_name = f"layers{hid_layers}_hidd{n_hid_ferm}_feat{features}_sample{n_samples}_lr{lr}_iter{N_opt}_parity{parity}_rot{rotation}_Init{MFinitialization}_type{dtype}_amp"
seed_str = f"seed_{seed}"
J_value = f"J={J2}"
if J1J2==True:
   model_path = f'HFDS_Heisenberg/plot/spin_new/{model_name}/{J_value}'
   folder = f'{model_path}/{seed_str}'
   save_model = f"{model_path}/{seed_str}/models"
else:
    folder = f'HFDS_Heisenberg/plot/Ising/spin/{model_name}/{J_value}'
    save_model = f"HFDS_Heisenberg/plot/Ising/spin/{model_name}/{J_value}/models"

os.makedirs(save_model, exist_ok=True)
os.makedirs(folder, exist_ok=True)  #create folder for the plots and the output file
os.makedirs(folder+"/physical_obs", exist_ok=True)
os.makedirs(folder+"/Sign_plot", exist_ok=True)
os.makedirs(model_path+"/plot_avg", exist_ok=True)

sys.stdout = open(f"{folder}/output.txt", "w") #redirect print output to a file inside the folder
print(f"HFDS_spin, J={J2}, L={L}, layers{hid_layers}_hidd{n_hid_ferm}_feat{features}_sample{n_samples}_lr{lr}_iter{N_opt}")

# Hilbert space of spins on the graph
boundary_conditions = 'pbc' 
lattice = nk.graph.Hypercube(length=L, n_dim=n_dim, pbc=True, max_neighbor_order=2)
hi = nk.hilbert.Spin(s=1 / 2, N=lattice.n_nodes, total_sz=0) 
print(f"hilbert space size = ",hi.size)


# ------------- define Hamiltonian ------------------------
ha = nk.operator.Heisenberg(hilbert=hi, graph=lattice, J=[1.0, J2], sign_rule=[False, False]).to_jax_operator()  # No Marshall sign rule"""

# --- Calculate exact ground state before model initialization ---
_, ket_gs = nk.exact.lanczos_ed(ha, compute_eigenvectors=True)



class HiddenFermion_amp(nn.Module):
  n_elecs: int
  network: str
  n_hid: int
  Lx: int
  Ly: int
  layers: int
  features: int
  MFinit: str
  hilbert: HomogeneousHilbert
  stop_grad_mf: bool = False
  stop_grad_lower_block: bool = False
  bounds: str="PBC"
  parity: bool = False
  rotation: bool = False
  dtype: type = jnp.float64
  U: float=8.0

  def setup(self):
    # orbital Initialization
    self.n_modes = 2*self.Lx*self.Ly
    self.orbitals = Orbitals(self.n_elecs,self.n_hid,self.Lx, self.Ly, self.MFinit, self.stop_grad_mf, self.bounds, self.dtype, self.U)
    # FFNN architecture
    if self.network=="FFNN":
        self.hidden = [nn.Dense(features=self.features,use_bias=False,param_dtype=self.dtype) for i in range(self.layers)]
        self.output = nn.Dense(features=self.n_hid*(self.n_elecs + self.n_hid),use_bias=True,param_dtype=self.dtype)
    else:
        raise NotImplementedError()
    # Rotation symmetry indices
    if self.rotation:
      idx = jnp.arange(self.Lx * self.Ly).reshape(self.Ly, self.Lx)
      self.idx_rot = jnp.flip(idx.T, axis=1).reshape(-1)


  def selu(self,x):
    if self.dtype==jnp.float64:
      return jax.nn.selu(x)
    else:
      return jax.nn.selu(x.real) +1j*jax.nn.selu(x.imag)


  def calc_psi(self,x,return_orbs=False):

    #1, 2, 3.
    orbitals = self.orbitals(x)

    # 4. Forward pass through the NN, create x_
    for i in range(self.layers):
        x = self.selu(self.hidden[i](x))
    x_ = self.output(x).reshape(x.shape[0],self.n_hid,self.n_elecs + self.n_hid)

    x_2 = jnp.repeat(jnp.expand_dims(jnp.eye(self.n_hid), axis=0),x.shape[0],axis=0)
    x_ += jnp.concatenate((jnp.zeros((x.shape[0], self.n_hid, self.n_elecs),self.dtype), x_2),axis=2)
    
    # 5. Concatenate the MF orbitals and the NN outputs
    x = jnp.concatenate((orbitals,x_),axis=1)
    sign, logx = jnp.linalg.slogdet(x)
    return logx, jnp.log(sign + 0j)


  def gen_reflected_samples(self,x):
    x_refl = -x
    return x_refl
  

  def gen_rotated_samples(self, x):
    #jax.debug.print("type of x rotation: {x}", x=type(x))
    x_rot1 = x[:, self.idx_rot]
    x_rot2 = x_rot1[:, self.idx_rot]
    x_rot3 = x_rot2[:, self.idx_rot]
    idx = jnp.arange(self.Lx * self.Ly).reshape(self.Ly, self.Lx)
    idx_rot = jnp.flip(idx.T, axis=1).reshape(-1)

    x_rot1 = x[:, idx_rot]
    x_rot2 = x_rot1[:, idx_rot]
    x_rot3 = x_rot2[:, idx_rot]
    return (x_rot1, x_rot2, x_rot3)


  
  def gen_sym_samples(self, x):
    x_sym = [x]
    if self.parity:
        x_refl = self.gen_reflected_samples(x)
        x_sym.append(x_refl)
    
    # --- Step 2: add rotation symmetry ---
    if self.rotation:
        x_rot1, x_rot2, x_rot3 = self.gen_rotated_samples(x)
        x_sym.extend([x_rot1, x_rot2, x_rot3])
        if self.parity:
            x_refl = self.gen_reflected_samples(x)
            x_rot1_refl, x_rot2_refl, x_rot3_refl = self.gen_rotated_samples(x_refl)
            x_sym.extend([x_rot1_refl, x_rot2_refl, x_rot3_refl])
    
    return x_sym
  
  @nn.compact
  def __call__(self,x):

    #x_sym = self.gen_sym_samples(x)
    log_amp, log_sign = self.calc_psi(x)

    # --- EXACT LOOKUP ---
    exact_log_amps = jnp.log(jnp.array(np.abs(ket_gs), dtype=self.dtype) + 0j)
    exact_log_signs = jnp.angle(ket_gs)

    x_bits = (x + 1) / 2
    powers = 2**jnp.arange(x.shape[-1] - 1, -1, -1)
    indices = jnp.sum(x_bits * powers, axis=-1).astype(int)
    
    exact_log_amp = exact_log_amps[indices].reshape(-1) 
    exact_log_sign = exact_log_signs[indices].reshape(-1)
    #exact_log_amp = jnp.repeat(exact_log_amp[None, :], repeats=8, axis=0)

    log_psi_exact_sign = log_amp  + 1j * exact_log_sign
    log_psi_exact_amp = exact_log_amp + 1j * log_sign

    return log_psi_exact_amp


if dtype=="real": dtype_ = jnp.float64
else: dtype_ = jnp.complex128


model = HiddenFermion_amp(n_elecs=n_elecs,
                        network="FFNN",
                        n_hid=n_hid_ferm,
                        Lx=L,
                        Ly=L,
                        layers=hid_layers,
                        features=features,
                        MFinit=MFinitialization,
                        hilbert=hi,
                        stop_grad_mf=False,
                        stop_grad_lower_block=False,
                        bounds=bounds,
                        parity=parity,
                        rotation=rotation,
                        dtype=dtype_,)



# ---------- define sampler ------------------------
sampler = nk.sampler.MetropolisExchange(
    hilbert=hi,
    graph=lattice,
    d_max=2,
    n_chains=n_chains,
    sweep_size=lattice.n_nodes,
)

key = jax.random.key(seed)
key, pkey, skey = jax.random.split(key, 3)
vstate = nk.vqs.MCState(
    sampler, 
    model, 
    n_samples=n_samples, 
    seed=pkey,
    n_discard_per_chain=128) #defines the variational state object

total_params = sum(p.size for p in jax.tree_util.tree_leaves(vstate.parameters))
print(f'Total number of parameters: {total_params}')

optimizer = nk.optimizer.Sgd(learning_rate=lr)

vmc = VMC_SR(
    hamiltonian=ha,
    optimizer=optimizer,
    diag_shift=1e-4,
    variational_state=vstate,
    mode = 'complex'
) 

log = nk.logging.RuntimeLog()


for i in range(block_iter):
     #Save
    with open(save_model +"/model_"+ f"{i}"+".mpack", "wb") as f:
        bytes_out = flax.serialization.to_bytes(vstate.variables)
        f.write(bytes_out)

    vmc.run(n_iter=save_every, out=log)
    
with open(save_model + f"/model_{block_iter}.mpack", "wb") as f:
    f.write(flax.serialization.to_bytes(vstate.variables))



E_init = get_initial_energy(log, L)
print(f"E_init = {E_init}")
E_vs = Energy(log, L, folder)
#Correlation function
vstate.n_samples = 1024
Corr_Struct(lattice, vstate, L, folder, hi)
#exact diagonalization
E_exact, ket_gs = Exact_gs(L, J2, ha, J1J2, spin)
#Fidelity
fidelity = Fidelity(vstate, ket_gs)
print(f"Fidelity <vstate|exact> = {fidelity}")
#Rel Error
Relative_Error(E_vs, E_exact, L)
#magnetization
Magnetization(vstate, lattice, hi)
#Variance
variance = Variance(log)
#Vscore
Vscore(L, variance, E_vs)
#count number of parameters in the model
hidden_fermion_param_count(n_elecs, n_hid_ferm, L, L, hid_layers, features)

#Marshall_sign(marshall_op, vstate, folder, n_samples = 64 )
#n_sample = 4096
#marshall_op = MarshallSignOperator(hilbert)
#sign_vstate_MCMC, sign_vstate_full = plot_Sign_full_MCMC(marshall_op, vstate, str(folder), 64, hi)
sign_vstate_full, sign_exact, fidelity = plot_Sign_Fidelity(ket_gs, vstate, hi,  folder, one_avg = "one")
#amp_overlap = plot_Amp_overlap_configs(ket_gs, vstate, hi, folder, one_avg = "one")

configs, sign_vstate_config, weight_exact, weight_vstate = plot_Sign_single_config(ket_gs, vstate, hi, 3, L, folder, one_avg = "one")
configs, sign_vstate_config, weight_exact, weight_vstate = plot_Weight_single(ket_gs, vstate, hi, 8, L, folder, one_avg = "one")
amp_overlap, fidelity, sign_vstate, sign_exact, sign_overlap = plot_Sign_Err_Amplitude_Err_Fidelity(ket_gs, vstate, hi, folder, one_avg = "one")
amp_overlap, sign_vstate, sign_exact, sign_overlap = plot_Sign_Err_vs_Amplitude_Err_with_iteration(ket_gs, vstate, hi, folder, one_avg = "one")
sorted_weights, sorted_amp_overlap, sorted_sign_overlap = plot_Overlap_vs_Weight(ket_gs, vstate, hi, folder, "one")

variables = {
        'E_init': E_init,
        'E_exact': E_exact,
        'Energy_iter': E_vs, # Renamed from E_vs for clarity
        #'sign_vstate_MCMC': sign_vstate_MCMC,
        'sign_vstate': sign_vstate_full,
        'sign_exact': sign_exact,
        'fidelity': fidelity,
        'configs': configs,
        'sign_vstate_config': sign_vstate_config,
        'weight_exact': weight_exact,
        'weight_vstate': weight_vstate,
        'amp_overlap': amp_overlap,
        'sign_overlap': sign_overlap,
        #'eigenvalues': eigenvalues
    }

with open(folder+"/variables", 'wb') as f:
    pickle.dump(variables, f)

vstate.n_samples = 256
#S_matrices, eigenvalues = plot_S_matrix_eigenvalues(vstate, folder, hi,  one_avg = "one")
   

sys.stdout.close()




