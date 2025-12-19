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

import matplotlib.pyplot as plt
import netket as nk
import jax
import numpy as np
import jax.numpy as jnp
print(jax.devices())
import flax
from flax import linen as nn
from netket.operator.spin import sigmax, sigmaz, sigmay
import logging
import sys
import os
import argparse
import pickle
from netket.hilbert.homogeneous import HomogeneousHilbert 

# Setup logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    stream=sys.stdout)
logger = logging.getLogger(__name__)

from ViT_Heisenberg.ViT_model import ViT_sym

from Elaborate.Statistics.Energy import *
from Elaborate.Statistics.Corr_Struct import *
from Elaborate.Statistics.Error_Stat import *
from Elaborate.Statistics.count_params import *
from Elaborate.Plotting.Sign_vs_iteration import *
from Elaborate.Plotting.Sign_vs_iteration import *
from Elaborate.Plotting.S_matrix_vs_iteration import *
 
parser = argparse.ArgumentParser(description="Example script with parameters")
parser.add_argument("--J2", type=float, default=0.5, help="Coupling parameter J2", required=False)
parser.add_argument("--seed", type=float, default=0, help="seed", required=False)
parser.add_argument("--output", type=str, default='psi', help="output mode", required=False)


args = parser.parse_args()

M = 10  # Number of spin configurations to initialize the parameters
L = 4  # Linear size of the lattice


n_dim = 2
J2 = args.J2
seed = int(args.seed)
output = args.output #psi #exact_amp #exact_sign

logger.info(f"Script parameters: J2={J2}, seed={seed}")

num_layers      = 2     # number of Tranformer layers
d_model         = 8   # dimensionality of the embedding space
n_heads         = 4     # number of heads
patch_size      = 2     # lenght of the input sequence
lr              = 0.0075
parity = True
rotation = True

N_samples       = 1024
N_opt           = 1

number_data_points = 1
save_every       = N_opt//number_data_points
block_iter = N_opt//save_every

model_name = f"layers{num_layers}_d{d_model}_heads{n_heads}_patch{patch_size}_sample{N_samples}_lr{lr}_iter{N_opt}_parity{parity}_rot{rotation}_mode{output}_sym4"
seed_str = f"seed_{seed}"
J_value = f"J={J2}"
model_path = f'ViT_Heisenberg/plot/Vision_new/{model_name}/{J_value}'
folder = f'{model_path}/{seed_str}'
save_model = f"{model_path}/{seed_str}/models"

os.makedirs(save_model, exist_ok=True)
os.makedirs(folder, exist_ok=True)  #create folder for the plots and the output file
os.makedirs(folder+"/physical_obs", exist_ok=True)
os.makedirs(folder+"/Sign_plot", exist_ok=True)
os.makedirs(model_path+"/plot_avg", exist_ok=True)

logger.info(f"Output will be saved to: {folder}")
sys.stdout = open(f"{folder}/output.txt", "w") #redirect print output to a file inside the folder
print(f"ViT, J={J2}, L={L}, layers{num_layers}_d{d_model}_heads{n_heads}_patch{patch_size}_sample{N_samples}_lr{lr}_iter{N_opt}")

# Hilbert space of spins on the graph
logger.info("Creating Hilbert space and Hamiltonian...")
lattice = nk.graph.Hypercube(length=L, n_dim=n_dim, pbc=True, max_neighbor_order=2)
hilbert = nk.hilbert.Spin(s=1 / 2, N=lattice.n_nodes, total_sz=0)
logger.info(f"Hilbert space size = {hilbert.size}")

# Heisenberg J1-J2 spin hamiltonian
hamiltonian = nk.operator.Heisenberg(
    hilbert=hilbert, graph=lattice, J=[1.0, J2], sign_rule=[False, False]
).to_jax_operator()  # No Marshall sign rule
logger.info("Hamiltonian created.")

# --- Calculate exact ground state before model initialization ---
logger.info("Calculating exact ground state...")
_, ket_gs = nk.exact.lanczos_ed(hamiltonian, compute_eigenvectors=True)
logger.info(f"Exact ground state calculated. Shape: {ket_gs.shape}")
logger.info(f"ket_gs values out ViT (first 10): {ket_gs.flatten()[:10]}")

####################################################################################################################

import matplotlib.pyplot as plt
import netket as nk
import jax
import jax.numpy as jnp
print(jax.devices())
import flax
from flax import linen as nn
from einops import rearrange

seed = 0
key = jax.random.key(seed)



def extract_patches2d(x, patch_size):
    batch = x.shape[0]
    n_patches = int((x.shape[1] // patch_size**2) ** 0.5)
    x = x.reshape(batch, n_patches, patch_size, n_patches, patch_size)
    x = x.transpose(0, 1, 3, 2, 4)
    x = x.reshape(batch, n_patches, n_patches, -1)
    x = x.reshape(batch, n_patches * n_patches, -1)
    return x


class Embed(nn.Module):
    d_model: int  # dimensionality of the embedding space
    patch_size: int  # linear patch size
    param_dtype = jnp.float64

    def setup(self):
        self.embed = nn.Dense(
            self.d_model,
            kernel_init=nn.initializers.xavier_uniform(),
            param_dtype=self.param_dtype,
        )

    def __call__(self, x):
        x = extract_patches2d(x, self.patch_size)
        x = self.embed(x)

        return x


class FactoredAttention(nn.Module):
    n_patches: int  # lenght of the input sequence
    d_model: int  # dimensionality of the embedding space (d in the equations)

    def setup(self):
        self.alpha = self.param(
            "alpha", nn.initializers.xavier_uniform(), (self.n_patches, self.n_patches)
        )
        self.V = self.param(
            "V", nn.initializers.xavier_uniform(), (self.d_model, self.d_model)
        )

    def __call__(self, x):
        y = jnp.einsum("i j, a b, M j b-> M i a", self.alpha, self.V, x)
        return y
    


from functools import partial


@partial(jax.vmap, in_axes=(None, 0, None), out_axes=1)
@partial(jax.vmap, in_axes=(None, None, 0), out_axes=1)
def roll2d(spins, i, j):
    side = int(spins.shape[-1] ** 0.5)
    spins = spins.reshape(spins.shape[0], side, side)
    spins = jnp.roll(jnp.roll(spins, i, axis=-2), j, axis=-1)
    return spins.reshape(spins.shape[0], -1)


class FMHA(nn.Module):
    d_model: int  # dimensionality of the embedding space
    n_heads: int  # number of heads
    n_patches: int  # lenght of the input sequence
    transl_invariant: bool = False
    param_dtype = jnp.float64

    def setup(self):
        self.v = nn.Dense(
            self.d_model,
            kernel_init=nn.initializers.xavier_uniform(),
            param_dtype=self.param_dtype,
        )
        self.W = nn.Dense(
            self.d_model,
            kernel_init=nn.initializers.xavier_uniform(),
            param_dtype=self.param_dtype,
        )
        if self.transl_invariant:
            self.alpha = self.param(
                "alpha",
                nn.initializers.xavier_uniform(),
                (self.n_heads, self.n_patches),
                self.param_dtype,
            )
            sq_n_patches = int(self.n_patches**0.5)
            assert sq_n_patches * sq_n_patches == self.n_patches
            self.alpha = roll2d(
                self.alpha, jnp.arange(sq_n_patches), jnp.arange(sq_n_patches)
            )
            self.alpha = self.alpha.reshape(self.n_heads, -1, self.n_patches)
        else:
            self.alpha = self.param(
                "alpha",
                nn.initializers.xavier_uniform(),
                (self.n_heads, self.n_patches, self.n_patches),
                self.param_dtype,
            )

    def __call__(self, x):
        # apply the value matrix in paralell for each head
        v = self.v(x)

        # split the representations of the different heads
        v = rearrange(
            v,
            "batch n_patches (n_heads d_eff) -> batch n_patches n_heads d_eff",
            n_heads=self.n_heads,
        )

        # factored attention mechanism
        v = rearrange(
            v, "batch n_patches n_heads d_eff -> batch n_heads n_patches d_eff"
        )
        x = jnp.matmul(self.alpha, v)
        x = rearrange(
            x, "batch n_heads n_patches d_eff  -> batch n_patches n_heads d_eff"
        )

        # concatenate the different heads
        x = rearrange(
            x, "batch n_patches n_heads d_eff ->  batch n_patches (n_heads d_eff)"
        )

        # the representations of the different heads are combined together
        x = self.W(x)

        return x
    


class EncoderBlock(nn.Module):
    d_model: int  # dimensionality of the embedding space
    n_heads: int  # number of heads
    n_patches: int  # lenght of the input sequence
    transl_invariant: bool = False
    param_dtype = jnp.float64

    def setup(self):
        self.attn = FMHA(
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_patches=self.n_patches,
            transl_invariant=self.transl_invariant,
        )

        self.layer_norm_1 = nn.LayerNorm(param_dtype=self.param_dtype)
        self.layer_norm_2 = nn.LayerNorm(param_dtype=self.param_dtype)

        self.ff = nn.Sequential(
            [
                nn.Dense(
                    4 * self.d_model,
                    kernel_init=nn.initializers.xavier_uniform(),
                    param_dtype=self.param_dtype,
                ),
                nn.gelu,
                nn.Dense(
                    self.d_model,
                    kernel_init=nn.initializers.xavier_uniform(),
                    param_dtype=self.param_dtype,
                ),
            ]
        )

    def __call__(self, x):
        x = x + self.attn(self.layer_norm_1(x))

        x = x + self.ff(self.layer_norm_2(x))
        return x
    


class Encoder(nn.Module):
    num_layers: int  # number of layers
    d_model: int  # dimensionality of the embedding space
    n_heads: int  # number of heads
    n_patches: int  # lenght of the input sequence
    transl_invariant: bool = False

    def setup(self):
        self.layers = [
            EncoderBlock(
                d_model=self.d_model,
                n_heads=self.n_heads,
                n_patches=self.n_patches,
                transl_invariant=self.transl_invariant,
            )
            for _ in range(self.num_layers)
        ]

    def __call__(self, x):

        for l in self.layers:
            x = l(x)

        return x
    


log_cosh = (
    nk.nn.activation.log_cosh
)  # Logarithm of the hyperbolic cosine, implemented in a more stable way


class OuputHead(nn.Module):
    d_model: int  # dimensionality of the embedding space
    param_dtype = jnp.float64

    def setup(self):
        self.out_layer_norm = nn.LayerNorm(param_dtype=self.param_dtype)

        self.norm2 = nn.LayerNorm(
            use_scale=True, use_bias=True, param_dtype=self.param_dtype
        )
        self.norm3 = nn.LayerNorm(
            use_scale=True, use_bias=True, param_dtype=self.param_dtype
        )

        self.output_layer0 = nn.Dense(
            self.d_model,
            param_dtype=self.param_dtype,
            kernel_init=nn.initializers.xavier_uniform(),
            bias_init=jax.nn.initializers.zeros,
        )
        self.output_layer1 = nn.Dense(
            self.d_model,
            param_dtype=self.param_dtype,
            kernel_init=nn.initializers.xavier_uniform(),
            bias_init=jax.nn.initializers.zeros,
        )

    def __call__(self, x):

        z = self.out_layer_norm(x.sum(axis=1))

        out_real = self.norm2(self.output_layer0(z))
        out_imag = self.norm3(self.output_layer1(z))

        out = out_real + 1j * out_imag

        return jnp.sum(log_cosh(out), axis=-1)
    

class ViT(nn.Module):
    num_layers: int  # number of layers
    d_model: int  # dimensionality of the embedding space
    n_heads: int  # number of heads
    patch_size: int  # linear patch size
    hilbert: HomogeneousHilbert 
    transl_invariant: bool = False
    output_mode: str = 'model_psi' # 'exact_amp', 'exact_sign', 'model_psi'
    


    @nn.compact
    def __call__(self, spins):
        x = jnp.atleast_2d(spins)

        Ns = x.shape[-1]  # number of sites
        n_patches = Ns // self.patch_size**2  # lenght of the input sequence

        x = Embed(d_model=self.d_model, patch_size=self.patch_size)(x)

        y = Encoder(
            num_layers=self.num_layers,
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_patches=n_patches,
            transl_invariant=self.transl_invariant,
        )(x)

        log_psi = OuputHead(d_model=self.d_model)(y)


        return log_psi





from netket.jax import logsumexp_cplx

class ViT_sym(nn.Module):
    L: int  
    num_layers: int  # number of layers
    d_model: int  # dimensionality of the embedding space
    n_heads: int  # number of heads
    patch_size: int  # linear patch size
    hilbert: HomogeneousHilbert
    transl_invariant: bool = False
    parity: bool = True  # parity symmetry operation
    rotation: bool = False  # rotational symmetry operation
    
    output_mode: str = 'model_psi'


    def setup(self):
        if self.rotation:
            idx = jnp.arange(self.L * self.L).reshape(self.L, self.L)
            self.idx_rot = jnp.flip(idx.T, axis=1).reshape(-1)

    @nn.compact
    def __call__(self, spins):

        vit = ViT(  self.num_layers,
                    self.d_model,
                    self.n_heads,
                    self.patch_size,
                    hilbert=self.hilbert,
                    transl_invariant=self.transl_invariant,
                    output_mode=self.output_mode
                    )

        def gen_reflected_samples(spins):
            return -spins

        def gen_rotated_samples(spins):
            spins_rot1 = spins[:, self.idx_rot]
            spins_rot2 = spins_rot1[:, self.idx_rot]
            spins_rot3 = spins_rot2[:, self.idx_rot]
            return spins_rot1, spins_rot2, spins_rot3
        
        def gen_sym_samples(spins):

            spins_sym = []
            spins_sym.append(spins)

            if self.parity:
                spins_refl = gen_reflected_samples(spins)
                spins_sym.append(spins_refl)  

            if self.rotation:
                spins_rot1, spins_rot2, spins_rot3 = gen_rotated_samples(spins)
                spins_sym.extend([spins_rot1, spins_rot2, spins_rot3])          
                if self.parity:
                    spins_refl = gen_reflected_samples(spins)
                    spins_rot1_refl, spins_rot2_refl, spins_rot3_refl = gen_rotated_samples(spins_refl)
                    spins_sym.extend([spins_rot1_refl, spins_rot2_refl, spins_rot3_refl])

            return spins_sym


        spins_sym = gen_sym_samples(spins)

        log_psi = jax.vmap(vit)(jnp.stack(spins_sym))
        
        log_psi_sym = logsumexp_cplx(log_psi, axis=0)

        if self.output_mode == 'exact_amp':
            log_psi_exact_amps = jnp.log(jnp.abs(ket_gs))
            indices = self.hilbert.states_to_numbers(spins)
            log_psi_exact_amp = log_psi_exact_amps[indices].reshape(-1)
            return log_psi_exact_amp + 1j * jnp.imag(log_psi_sym)
        
        elif self.output_mode == 'exact_sign':
            log_psi_exact_signs = jnp.angle(ket_gs)
            indices = self.hilbert.states_to_numbers(spins) 
            log_psi_exact_sign = log_psi_exact_signs[indices].reshape(-1)
            return jnp.real(log_psi_sym) + 1j * log_psi_exact_sign
        
        elif self.output_mode == 'psi':
            return log_psi_sym




####################################################################################################################

logger.info("Initializing ViT model...")
# Intiialize the ViT variational wave function

vit_module = ViT_sym(
    L=L,
    num_layers=num_layers, 
    d_model=d_model, 
    n_heads=n_heads, 
    patch_size=patch_size, 
    hilbert=hilbert,
    transl_invariant=True, 
    parity=parity, 
    rotation=rotation,
    output_mode=output
)
"""

vit_module = ViT(
    num_layers=num_layers, 
    d_model=d_model, 
    n_heads=n_heads, 
    patch_size=patch_size,
    hilbert=hilbert,
    transl_invariant=True,
    output_mode= output # You can change this to 'exact_sign' or 'model_psi'

)
"""
key = jax.random.key(seed)
key, subkey = jax.random.split(key)
# Generate initial spin configurations that satisfy the total_sz=0 constraint
n_sites = L * L
n_up = n_sites // 2

# Create a single valid configuration
base_config = jnp.array([1] * n_up + [-1] * (n_sites - n_up))
# Create a batch of M identical valid configurations and then shuffle each one
spin_configs = jax.vmap(jax.random.permutation)(jax.random.split(subkey, M), jnp.tile(base_config, (M, 1)))
params = vit_module.init(subkey, spin_configs)

# Metropolis Local Sampling
logger.info("Creating sampler...")
sampler = nk.sampler.MetropolisExchange(
    hilbert=hilbert,
    graph=lattice,
    d_max=2,
    n_chains=N_samples,
    sweep_size=lattice.n_nodes,
)

optimizer = nk.optimizer.Sgd(learning_rate=lr)

key, subkey = jax.random.split(key, 2)
logger.info("Creating MCState (vstate)...")
vstate = nk.vqs.MCState(
    sampler=sampler,
    model=vit_module,
    sampler_seed=subkey,
    n_samples=N_samples,
    n_discard_per_chain=16,
    variables=params,
    chunk_size=512,
)

N_params = nk.jax.tree_size(vstate.parameters)
print("Number of parameters = ", N_params, flush=True)


# Variational monte carlo driver
from netket.experimental.driver import VMC_SR

vmc = VMC_SR(
    hamiltonian=hamiltonian,
    optimizer=optimizer,
    diag_shift=1e-4,
    variational_state=vstate,
    mode="complex",
) 

# Optimization
log = nk.logging.RuntimeLog()


for i in range(block_iter):
    #Save model
    with open(save_model +f"/model_{i}.mpack", "wb") as f:
        bytes_out = flax.serialization.to_bytes(vstate.variables)
        f.write(bytes_out)

    vmc.run(n_iter=save_every, out=log)

# Save the final model state after the last optimization step
with open(save_model +f"/model_{block_iter}.mpack", "wb") as f:
    bytes_out = flax.serialization.to_bytes(vstate.variables)
    f.write(bytes_out)


    
#Energy
E_vs = Energy(log, L, folder)
#Correlation function
vstate.n_samples = 1024
Corr_Struct(lattice, vstate, L, folder, hilbert)
#change to lanczos for Heisenberg
E_exact, ket_gs = Exact_gs(L, J2, hamiltonian, J1J2=True, spin=True)
#Fidelity
fidelity = Fidelity(vstate, ket_gs)
print(f"Fidelity <vstate|exact> = {fidelity}")
#Rel Err
Relative_Error(E_vs, E_exact, L)
#Magn
Magnetization(vstate, lattice, hilbert)
#Variance
variance = Variance(log)
#Vscore
Vscore(L, variance, E_vs)
#count Params
count_params = vit_param_count(n_heads, num_layers, patch_size, d_model, L*L)
print(f"params={count_params}")

#Marshall_sign(marshall_op, vstate, folder, n_samples = 64 )
#n_sample = 4096
#marshall_op = MarshallSignOperator(hilbert)
#sign_vstate_MCMC, sign_vstate_full = plot_Sign_full_MCMC(marshall_op, vstate, str(folder), 64, hi)
#sign_vstate_full, sign_exact, fidelity = plot_Sign_Fidelity(ket_gs, vstate, hilbert,  folder, one_avg = "one")
#amp_overlap = plot_Amp_overlap_configs(ket_gs, vstate, hilbert, folder, one_avg = "one")

configs, sign_vstate_config, weight_exact, weight_vstate = plot_Sign_single_config(ket_gs, vstate, hilbert, 3, L, folder, one_avg = "one")
configs, sign_vstate_config, weight_exact, weight_vstate = plot_Weight_single(ket_gs, vstate, hilbert, 8, L, folder, one_avg = "one")
amp_overlap, fidelity, sign_vstate, sign_exact, sign_overlap = plot_Sign_Err_Amplitude_Err_Fidelity(ket_gs, vstate, hilbert, folder, one_avg = "one")
amp_overlap, sign_vstate, sign_exact, sign_overlap = plot_Sign_Err_vs_Amplitude_Err_with_iteration(ket_gs, vstate, hilbert, folder, one_avg = "one")
sorted_weights, sorted_amp_overlap, sorted_sign_overlap = plot_Overlap_vs_Weight(ket_gs, vstate, hilbert, folder, "one")

eigenvalues_start, number_relevant_S_eigenvalues_start = plot_S_matrix_eigenvalues(vstate, folder, hilbert, part_training='start', one_avg="one")
eigenvalues_end, number_relevant_S_eigenvalues_end = plot_S_matrix_eigenvalues(vstate, folder, hilbert, part_training='end', one_avg="one")


variables = {
        #'sign_vstate_MCMC': sign_vstate_MCMC,
        'sign_vstate': sign_vstate,
        'sign_exact': sign_exact,
        'fidelity': fidelity,
        'configs': configs,
        'sign_vstate_config': sign_vstate_config,
        'weight_exact': weight_exact,
        'weight_vstate': weight_vstate,
        'amp_overlap': amp_overlap,
        'sign_overlap': sign_overlap,
        'eigenvalues_start': eigenvalues_start,
        'eigenvalues_end': eigenvalues_end,
        'number_relevant_S_eigenvalues_start': number_relevant_S_eigenvalues_start,
        'number_relevant_S_eigenvalues_end': number_relevant_S_eigenvalues_end

    }

print(variables)

with open(folder+"/variables", 'wb') as f:
    pickle.dump(variables, f)                   

logger.info("Script finished successfully.")
sys.stdout.close()