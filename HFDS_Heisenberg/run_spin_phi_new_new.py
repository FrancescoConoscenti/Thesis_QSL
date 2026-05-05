import sys
import argparse
import jax
from jax import numpy as jnp
import netket as nk
import os
import flax
import helper
import logging
import pickle
import re
os.environ["JAX_PLATFORM_NAME"] = "gpu"

print("Total devices:", jax.device_count())
print("Local devices:", jax.local_device_count())
print("Devices:", jax.devices())

import pickle
sys.path.append(os.path.dirname(os.path.dirname("/scratch/f/F.Conoscenti/Thesis_QSL")))

from netket.driver import VMC_SR

from HFDS_Heisenberg.HFDS_model_spin import HiddenFermion

from Elaborate.Statistics.Energy import *
from Elaborate.Statistics.Corr_Struct import *
from Elaborate.Statistics.Error_Stat import *
from Elaborate.Statistics.count_params import *
from Elaborate.Plotting.Old.Sign_vs_iteration import *
from Elaborate.Sign_Obs import *
from Elaborate.Plotting.QGT.QGT_vs_iteration import *

from DMRG.DMRG_NQS_Imp_sampl import Observable_Importance_sampling

from Observables import run_observables

from Hamiltonian import build_heisenberg_apbc, build_heisenberg_twisted

parser = argparse.ArgumentParser(description="Example script with parameters")
parser.add_argument("--J2", type=float, default=0.5, help="Coupling parameter J2")
parser.add_argument("--seed", type=float, default=1, help="seed")
parser.add_argument("--L", type=int, default=4, help="Linear size of the lattice")
parser.add_argument("--N_iter_first", type=int, default=1000, help="Number of iterations first")
parser.add_argument("--N_iter_adiabatic", type=int, default=200, help="Number of iterations adiabatic")
parser.add_argument("--phi_list", nargs='+', type=float, default=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0], help="Twist angle in radians for the twisted BC")
args = parser.parse_args()

spin = True 


#Physical param
L       = args.L
N_sites = L * L

n_elecs = N_sites # L*L should be half filling
N_up    = (n_elecs+1)//2
N_dn    = n_elecs//2
n_dim = 2

J1J2 = True
J2 = args.J2
seed = int(args.seed)

dtype   = "complex"
MFinitialization = "Fermi" #random #Fermi
determinant_type = "hidden"

bounds = "PBC"

N_iter_first = args.N_iter_first
phi_list = args.phi_list
N_iter_adiabatic = args.N_iter_adiabatic


parity = True
rotation = True

#Varaitional state param
# 1k params for L=4 n_hid=1 features=16 layers=1
# 3.9k params for L=4 n_hid=2 features=64 layers=1
# 6k params for L=4 n_hid=4 features=64 layers=1
# 13k params for L=4 n_hid=6 features=128 layers=1

# 6x6
# 3.8k params for L=6 n_hid=1 features=16 layers=1
# 6k params for L=6 n_hid=2 features=32 layers=1
# 15k params for L=6 n_hid=4 features=64 layers=1
# 40k params for L=6 n_hid=6 features=128 layers=1
# 53k params for L=6 n_hid=8 features=128 layers=1
# 8x8
# 40k params for L=8 n_hid=6 features=64 layers=1
# 50k params for L=8 n_hid=8 features=64 layers=1
# 91k params for L=8 n_hid=8 features=128 layers=1
# ??k params for L=8 n_hid=10 features=64 layers=1
#10x10
# 68k params for L=6 n_hid=8 features=64 layers=1
# 84k params for L=6 n_hid=8 features=64 layers=1
# 145k params for L=6 n_hid=8 features=128 layers=1

n_hid_ferm       = 2
features         = 32    #hidden units per layer
hid_layers       = 1

#Network param
lr               = 0.02
n_samples        = 4096 #total number of samples
#n_samples = 4096  n_chains  = 128  chunk_size = 4096
#n_samples = 8192  n_chains  = 256  chunk_size = 2048  
n_chains         = n_samples  #number of parallel Markov chains
chunk_size       = n_samples #samples are divided in chunks to compute observables in parallel


# ── State carried across phis ─────────────────────────────────────────────────
original_stdout    = sys.stdout
previous_variables = None

for idx, phi in enumerate(args.phi_list):

    print(f"\n=== Starting phi = {phi} ===")

    N_iter = args.N_iter_first if idx == 0 else args.N_iter_adiabatic

    # ── Build folder / file paths ──────────────────────────────────────────────
    model_name = (
        f"layers{hid_layers}_hidd{n_hid_ferm}_feat{features}"
        f"_sample{n_samples}"
        f"_phi{phi}"
        f"_lr{lr}_iter{N_iter}"
        f"_Init{MFinitialization}_type{dtype}_phi"
    )
    seed_str   = f"seed_{seed}"
    J_value    = f"J={J2}"
    model_path = f"HFDS_Heisenberg/plot/{L}x{L}/phi/{model_name}/{J_value}"
    folder     = f"{model_path}/{seed_str}"
    save_model = f"{folder}/models"

    os.makedirs(save_model,                      exist_ok=True)
    os.makedirs(folder,                          exist_ok=True)
    os.makedirs(folder + "/physical_obs",        exist_ok=True)
    os.makedirs(folder + "/Sign_plot",           exist_ok=True)

    save_every = max(1, N_iter // 10)

    # Discover existing checkpoints
    existing_models = []
    if os.path.exists(save_model):
        for fname in os.listdir(save_model):
            m = re.search(r"model_(\d+)\.mpack", fname)
            if m:
                existing_models.append(int(m.group(1)))

    next_block  = max(existing_models) + 1 if existing_models else 0
    block_iter  = next_block + (N_iter // save_every)   # exclusive end block index

    sys.stdout = open(f"{folder}/output.txt", "a")
    try:
        print(
            f"HFDS_spin, J={J2}, L={L}, "
            f"layers{hid_layers}_hidd{n_hid_ferm}_feat{features}"
            f"_sample{n_samples}_lr{lr}_iter{N_iter}_phi{phi} "
            f"(parity={parity}, rotation={rotation})"
        )

        # ── Hilbert space & graph ──────────────────────────────────────────────────
        hi      = nk.hilbert.Spin(s=1/2, N=L**2, total_sz=0)
        lattice = nk.graph.Hypercube(length=L, n_dim=n_dim, pbc=[True, True], max_neighbor_order=2)
        print("Hilbert space size =", hi.size)

        # ── Hamiltonian ────────────────────────────────────────────────────────────
        if phi == 0.0:
            ha = nk.operator.Heisenberg(hilbert=hi, graph=lattice, J=[1.0, J2], sign_rule=[False, False]).to_jax_operator()  # No Marshall sign rule""" 
        else:
            ha = build_heisenberg_twisted(
                L, L, J1=1.0, J2=J2, phi=phi, apbc_y=False
            ).to_jax_operator()

        # ── Model ──────────────────────────────────────────────────────────────────
        dtype_ = jnp.float64 if dtype == "real" else jnp.complex128

        model = HiddenFermion(
                            L=L,
                        network="FFNN",
                        n_hid=n_hid_ferm,
                        layers=hid_layers,
                        features=features,
                        MFinit=MFinitialization,
                        hilbert=hi,
                        stop_grad_mf=False,
                        stop_grad_lower_block=False,
                        bounds=bounds,
                        parity=parity,
                        rotation=rotation,
                        dtype=dtype_
                        )

        # ── Sampler & variational state ────────────────────────────────────────────
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
            chunk_size=chunk_size,
            n_discard_per_chain=128,
        )

        total_params = sum(p.size for p in jax.tree_util.tree_leaves(vstate.parameters))
        print(f"Total number of parameters: {total_params}")

        # ── Load checkpoint or warm-start from previous phi ────────────────────────
        log_path     = os.path.join(folder, "log.pkl")

        start_block, vstate = helper.load_checkpoint(save_model, block_iter, save_every, vstate)

        if start_block == 0:
            if previous_variables is not None:
                print("Warm-starting from previous phi parameters...")
                vstate.variables = previous_variables
            

        # Guard: never step backward
        start_block = max(start_block, next_block)

        # ── Optimizer & VMC driver ────────────────────────────────────────────────
        optimizer = nk.optimizer.Sgd(learning_rate=lr)
        vmc = VMC_SR(
            hamiltonian=ha,
            optimizer=optimizer,
            diag_shift=1e-6,
            variational_state=vstate,
            use_ntk=True,
            momentum=0.8,
        )

        log = nk.logging.RuntimeLog()

        # ── Training loop ─────────────────────────────────────────────────────────
        for i in range(start_block, block_iter):
            with open(save_model + f"/model_{i}.mpack", "wb") as file:
                file.write(flax.serialization.to_bytes(vstate))

            vmc.run(n_iter=save_every, out=log)

        # Save the final model
        with open(save_model + f"/model_{block_iter}.mpack", "wb") as f:
            f.write(flax.serialization.to_bytes(vstate))

        final_log_data = log.data

        # ── Observables ───────────────────────────────────────────────────────────
        print("Running observables computation...")
        if final_log_data and "Energy" in final_log_data:
            run_observables(helper.MockLog(final_log_data), folder)
        else:
            run_observables(None, folder)

        # ── Carry parameters forward ──────────────────────────────────────────────
        previous_variables = vstate.variables
        
    except Exception as e:
        sys.stdout.close()
        sys.stdout = original_stdout
        raise
    finally:
        if sys.stdout != original_stdout:
            sys.stdout.close()
            sys.stdout = original_stdout

    print("Finished phi =", phi)