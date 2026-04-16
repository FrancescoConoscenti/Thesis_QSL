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

import pickle
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

from Hamiltonian import build_heisenberg_apbc, build_heisenberg_twisted

parser = argparse.ArgumentParser(description="Example script with parameters")
parser.add_argument("--J2", type=float, default=0.5, help="Coupling parameter J2")
parser.add_argument("--seed", type=float, default=1, help="seed")
parser.add_argument("--L", type=int, default=4, help="Linear size of the lattice")
parser.add_argument("--bc_x", type=str, default="PBC", choices=["PBC", "APC"], help="Boundary condition x")
parser.add_argument("--bc_y", type=str, default="PBC", choices=["PBC", "APC"], help="Boundary condition y")
parser.add_argument("--phi_list", type=float, nargs='+', default=[0.0], help="List of twist angles in radians for the twisted BC")
parser.add_argument("--N_iter_conv", type=int, default=0, help="Number of iterations for the first phi (no symmetries)")
parser.add_argument("--N_iter_symm", type=int, default=1000, help="Number of iterations with symmetries (first phi only)")
parser.add_argument("--N_iter_adiabatic", type=int, default=200, help="Number of iterations for subsequent phis")
parser.add_argument("--load_path", type=str, default=None, help="Path to load model and resume training")
parser.add_argument("--load_path_phi", type=str, default=None, help="Path to load model to start optimizing for new phi values")
args = parser.parse_args()

spin = True

#Physical param
L       = args.L
N_sites = L * L

n_elecs = N_sites
N_up    = (n_elecs+1)//2
N_dn    = n_elecs//2
n_dim = 2

J1J2 = True
J2 = args.J2
seed = int(args.seed)

dtype   = "complex"
MFinitialization = "Fermi"
determinant_type = "hidden"

bc_x = args.bc_x
bc_y = args.bc_y
bounds = (bc_x, bc_y)

#Variational state param
n_hid_ferm       = 2
features         = 32
hid_layers       = 1

#Network param
lr               = 0.02
n_samples        = 1024
n_chains         = n_samples
chunk_size       = n_samples

#---------------------------Load another model -----------------------------------------
load_path = args.load_path
load_path_phi = args.load_path_phi
previous_iter = 0 # This will track iterations across a full phi sweep.

original_stdout = sys.stdout
previous_variables = None

for idx, phi in enumerate(args.phi_list):

    # -----------------------------------------------------------------------
    # Compute the canonical total iteration count and phase list for this phi.
    # For phi[0]: two phases (no-symm conv + symm), total = N_iter_conv + N_iter_symm.
    # For phi[1+]: one phase (symm, adiabatic), total = N_iter_adiabatic.
    # The folder name is fixed for the whole phi run and does NOT change between phases.
    # -----------------------------------------------------------------------
    if idx == 0:
        N_total_phi = args.N_iter_conv + args.N_iter_symm
        phases = [
            {"N_iter": args.N_iter_conv, "parity": False, "rotation": False},
            {"N_iter": args.N_iter_symm, "parity": True,  "rotation": True},
        ]
    else:
        N_total_phi = args.N_iter_adiabatic
        phases = [
            {"N_iter": args.N_iter_adiabatic, "parity": True, "rotation": True},
        ]

    folder_iter = N_total_phi # The folder name always reflects the budget for this phi.
    # Build a single canonical model name for this phi value.
        # folder_iter encodes the full training budget for this phi.
    model_name = (
        f"layers{hid_layers}_hidd{n_hid_ferm}_feat{features}"
        f"_sample{n_samples}_bc{bc_x}_{bc_y}"
        f"_phi{phi}"
        f"_lr{lr}_iter{folder_iter}"
        f"_Init{MFinitialization}_type{dtype}_phi" 
    )
    seed_str  = f"seed_{seed}"
    J_value   = f"J={J2}"

    model_path = f"HFDS_Heisenberg/plot/{L}x{L}/phi/{model_name}/{J_value}"
    folder     = f"{model_path}/{seed_str}"
    save_model = f"{folder}/models"

    os.makedirs(save_model, exist_ok=True)
    os.makedirs(folder, exist_ok=True)
    os.makedirs(folder + "/physical_obs", exist_ok=True)
    os.makedirs(folder + "/Sign_plot", exist_ok=True)
    # -----------------------------------------------------------------------
    # Phase loop: both phases share the same folder / save_model path.
    # We track a per-phi iteration offset so checkpoint numbering is contiguous.
    # -----------------------------------------------------------------------
    phase_iter_offset = previous_iter   # absolute iteration count before this phi starts

    for phase in phases:
        N_iter   = phase["N_iter"]
        parity   = phase["parity"]
        rotation = phase["rotation"]

        if N_iter <= 0:
            continue

        # Save exactly 10 models per phase independently
        save_every = max(1, N_iter // 10)

        N_opt = phase_iter_offset + N_iter

        # Discover existing checkpoints to ensure continuous block numbers
        existing_models = []
        if os.path.exists(save_model):
            for f in os.listdir(save_model):
                m = re.search(r"model_(\d+)\.mpack", f)
                if m:
                    existing_models.append(int(m.group(1)))
        
        next_block = max(existing_models) + 1 if existing_models else 0
        
        if existing_models:
            target_block_iter = next_block + (N_iter // save_every)
        else:
            target_block_iter = (phase_iter_offset // save_every) + (N_iter // save_every)

        phase_name = "symm" if (parity or rotation) else "run"

        sys.stdout = open(f"{folder}/output.txt", "a")
        print(
            f"HFDS_spin, J={J2}, L={L}, "
            f"layers{hid_layers}_hidd{n_hid_ferm}_feat{features}"
            f"_sample{n_samples}_lr{lr}_iter{N_opt}_phi{phi} "
            f"(parity={parity}, rotation={rotation})"
        )

        # ------------- define Hilbert space ------------------------
        hi      = nk.hilbert.Spin(s=1/2, N=L**2, total_sz=0)
        lattice = nk.graph.Hypercube(length=L, n_dim=n_dim, pbc=[True, True], max_neighbor_order=2)
        print(f"hilbert space size = ", hi.size)

        # ------------- define Hamiltonian ------------------------
        ha = build_heisenberg_twisted(
            L, L, J1=1.0, J2=J2, phi=phi, apbc_y=False
        ).to_jax_operator()

        # ------------- define model ------------------------
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
            chunk_size=chunk_size,
            n_discard_per_chain=128,
        )

        total_params = sum(p.size for p in jax.tree_util.tree_leaves(vstate.parameters))
        print(f"Total number of parameters: {total_params}")

        log = nk.logging.RuntimeLog()
        log_path = os.path.join(folder, f"log_{phase_name}.pkl")
        old_log_data = helper.load_log(folder, f"log_{phase_name}.pkl")

        start_block, vstate = helper.load_checkpoint(save_model, target_block_iter, save_every, vstate)

        if start_block == 0:
            if previous_variables is not None:
                print("Initializing with parameters from previous phase/phi run...")
                vstate.variables = previous_variables
                start_block = next_block
            elif load_path or load_path_phi:
                active_load_path = load_path if load_path else load_path_phi
                print(f"Attempting to load starting model from {active_load_path}")
                load_J_path = os.path.join(active_load_path, f"J={J2}")
                if not os.path.exists(load_J_path):
                    load_J_path = os.path.join(active_load_path, f"J2={J2}")

                load_seed_path  = os.path.join(load_J_path, f"seed_{seed}")
                load_save_model = os.path.join(load_seed_path, "models")

                if os.path.exists(load_save_model):
                    files = [f for f in os.listdir(load_save_model) if f.endswith(".mpack")]
                    if files:
                        files.sort(key=lambda x: int(re.search(r"model_(\d+)", x).group(1)))
                        last_model_path = os.path.join(load_save_model, files[-1])
                        print(f"Loading starting model from {last_model_path}")
                        with open(last_model_path, "rb") as f:
                            try:
                                vstate = flax.serialization.from_bytes(vstate, f.read())
                            except Exception:
                                f.seek(0)
                                vstate.variables = flax.serialization.from_bytes(vstate.variables, f.read())

                        if load_path or load_path_phi:
                            # After loading, we will start the new run from block 0 in the new folder.
                            start_block = next_block 
                    else:
                        print("No .mpack files found in the load path.")
                else:
                    print(f"Models directory not found in the load path: {load_save_model}")

        # Ensure start_block doesn't step backward (e.g. if load_checkpoint fails)
        start_block = max(start_block, next_block if previous_variables is not None else 0)
        block_iter = start_block + (N_iter // save_every)

        # Initialize VMC
        optimizer = nk.optimizer.Sgd(learning_rate=lr)
        vmc = VMC_SR(
            hamiltonian=ha,
            optimizer=optimizer,
            diag_shift=1e-5,
            variational_state=vstate,
            use_ntk=True,
            momentum=0.8,
        )

        # Determine actual steps already done in this phase for accurate log/step count
        if existing_models and start_block >= next_block:
            blocks_done_in_phase = start_block - next_block
        else:
            blocks_done_in_phase = max(0, start_block - (phase_iter_offset // save_every))
            
        vmc._step_count = phase_iter_offset + (blocks_done_in_phase * save_every)

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

        print("Running observables computation...")
        if final_log_data and "Energy" in final_log_data:
            run_observables(helper.MockLog(final_log_data), folder)
        else:
            run_observables(None, folder)

        # Move generated plots/data to phase-specific subfolders
        for sub_name in ["Energy_plot", "physical_obs", "Sign_plot", "QGT_plot"]:
            sub_dir = os.path.join(folder, sub_name)
            if os.path.exists(sub_dir):
                phase_sub_dir = os.path.join(sub_dir, phase_name)
                os.makedirs(phase_sub_dir, exist_ok=True)
                for filename in os.listdir(sub_dir):
                    file_path = os.path.join(sub_dir, filename)
                    if os.path.isfile(file_path):
                        shutil.move(file_path, os.path.join(phase_sub_dir, filename))

        # Copy variables.pkl so it's available uniquely per phase
        var_path = os.path.join(folder, "variables.pkl")
        if os.path.exists(var_path):
            if phase_name == "run":
                shutil.copy2(var_path, os.path.join(folder, "variables_run.pkl"))

        # Hand off parameters to the next phase (or next phi)
        previous_variables = vstate.variables
        phase_iter_offset  = N_opt   # advance offset for the next phase within this phi

        sys.stdout.close()
        sys.stdout = original_stdout

    # After all phases for this phi are done, advance the global iteration counter
    previous_iter += N_total_phi