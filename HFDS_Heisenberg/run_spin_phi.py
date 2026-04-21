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
import shutil
os.environ["JAX_PLATFORM_NAME"] = "gpu"

print("Total devices:", jax.device_count())
print("Local devices:", jax.local_device_count())
print("Devices:", jax.devices())

sys.path.append(os.path.dirname(os.path.dirname("/scratch/f/F.Conoscenti/Thesis_QSL")))

from netket.driver import VMC_SR

from HFDS_Heisenberg.HFDS_model_spin import HiddenFermion_phi

from Elaborate.Statistics.Energy import *
from Elaborate.Statistics.Corr_Struct import *
from Elaborate.Statistics.Error_Stat import *
from Elaborate.Statistics.count_params import *
from Elaborate.Plotting.Old.Sign_vs_iteration import *
from Elaborate.Sign_Obs import *
from Elaborate.Plotting.QGT.QGT_vs_iteration import *

from DMRG.DMRG_NQS_Imp_sampl import Observable_Importance_sampling

from Observables import run_observables
from Hamiltonian import build_heisenberg_twisted

parser = argparse.ArgumentParser(description="HFDS phi sweep")
parser.add_argument("--J2",               type=float, default=0.5,  help="Coupling parameter J2")
parser.add_argument("--seed",             type=float, default=1,     help="Random seed")
parser.add_argument("--L",                type=int,   default=4,     help="Linear size of the lattice")
parser.add_argument("--bc_x",            type=str,   default="PBC", choices=["PBC", "APC"], help="Boundary condition x")
parser.add_argument("--bc_y",            type=str,   default="PBC", choices=["PBC", "APC"], help="Boundary condition y")
parser.add_argument("--parity",          type=lambda x: x.lower() == "true", default=True,  help="Use parity symmetry")
parser.add_argument("--rotation",        type=lambda x: x.lower() == "true", default=True,  help="Use rotation symmetry")
parser.add_argument("--phi_list",        type=float, nargs='+',
                    default=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0],
                    help="List of twist angles in radians")
parser.add_argument("--model_path",      type=str,   default="/scratch/f/F.Conoscenti/Thesis_QSL/HFDS_Heisenberg/plot/8x8/phi/layers1_hidd2_feat32_sample4096_bcPBC_PBC_phi0.0_lr0.02_iter1100_InitFermi_typecomplex_phi",  help="Path to initial model (e.g., phi=0.0) to warm-start from")
parser.add_argument("--N_iter_first",    type=int,   default=100,    help="Number of iterations for the first phi")
parser.add_argument("--N_iter_adiabatic",type=int,   default=100,     help="Number of iterations for subsequent phis")
args = parser.parse_args()

# ── Physical parameters ────────────────────────────────────────────────────────
spin    = True
L       = args.L
N_sites = L * L
n_elecs = N_sites
N_up    = (n_elecs + 1) // 2
N_dn    = n_elecs // 2
n_dim   = 2

J2              = args.J2
seed            = int(args.seed)
dtype           = "complex"
MFinitialization = "Fermi"
bc_x            = args.bc_x
bc_y            = args.bc_y
bounds          = (bc_x, bc_y)
parity          = args.parity
rotation        = args.rotation

# ── Network / sampler parameters ──────────────────────────────────────────────
n_hid_ferm  = 2
features    = 32
hid_layers  = 1
lr          = 0.02
n_samples   = 4096
n_chains    = n_samples
chunk_size  = n_samples // 16

# ── State carried across phis ─────────────────────────────────────────────────
original_stdout    = sys.stdout
previous_variables = None

for idx, phi in enumerate(args.phi_list):

    print(f"\n=== Starting phi = {phi} ===")

    N_iter = args.N_iter_first if idx == 0 else args.N_iter_adiabatic

    # ── Build folder / file paths ──────────────────────────────────────────────
    model_name = (
        f"layers{hid_layers}_hidd{n_hid_ferm}_feat{features}"
        f"_sample{n_samples}_bc{bc_x}_{bc_y}"
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
        ha = build_heisenberg_twisted(
            L, L, J1=1.0, J2=J2, phi=phi, apbc_y=False
        ).to_jax_operator()

        # ── Model ──────────────────────────────────────────────────────────────────
        dtype_ = jnp.float64 if dtype == "real" else jnp.complex128

        model = HiddenFermion_phi(
            L=L,
            N_sites=N_sites,
            network="FFNN",
            n_hid=n_hid_ferm,
            layers=hid_layers,
            features=features,
            MFinit=MFinitialization,
            hilbert=hi,
            stop_grad_mf=False,
            stop_grad_lower_block=False,
            bounds=bounds,
            phi=phi,
            parity=parity,
            rotation=rotation,
            dtype=dtype_,
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
        old_log_data = helper.load_log(folder, "log.pkl")

        start_block, vstate = helper.load_checkpoint(save_model, block_iter, save_every, vstate)

        if start_block == 0:
            if previous_variables is not None:
                print("Warm-starting from previous phi parameters...")
                vstate.variables = previous_variables
            elif args.model_path:
                print(f"Attempting to warm-start from {args.model_path}")
                load_J_path = os.path.join(args.model_path, f"J={J2}")
                if not os.path.exists(load_J_path):
                    load_J_path = os.path.join(args.model_path, f"J2={J2}")
            
                load_seed_path = os.path.join(load_J_path, f"seed_{seed}")
                load_save_model = os.path.join(load_seed_path, "models")
                
                if os.path.exists(load_save_model):
                    files = [f for f in os.listdir(load_save_model) if f.endswith(".mpack")]
                    if files:
                        files.sort(key=lambda x: int(re.search(r"model_(\d+)", x).group(1)))
                        last_model = files[-1]
                        last_model_path = os.path.join(load_save_model, last_model)
                        print(f"Loading starting model from {last_model_path}")
                        with open(last_model_path, 'rb') as f:
                            try:
                                vstate = flax.serialization.from_bytes(vstate, f.read())
                            except Exception:
                                f.seek(0)
                                vstate.variables = flax.serialization.from_bytes(vstate.variables, f.read())
                    else:
                        print(f"No .mpack files found in {load_save_model}.")
                else:
                    print(f"Models directory not found at {load_save_model}.")

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

            with open(log_path, "wb") as f:
                current_log_data = helper.merge_log_data(old_log_data, log.data)
                pickle.dump(current_log_data, f)

        # Save the final model
        with open(save_model + f"/model_{block_iter}.mpack", "wb") as f:
            f.write(flax.serialization.to_bytes(vstate))

        final_log_data = helper.merge_log_data(old_log_data, log.data)

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