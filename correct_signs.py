import os
from pathlib import Path
import netket as nk
from HFDS_Heisenberg.HFDS_model_spin import HiddenFermion
from ViT_Heisenberg.ViT_model import ViT_sym
import re
import jax.numpy as jnp
from Elaborate.Error_Stat import *
from Elaborate.Sign_vs_iteration import *
from Elaborate.Sign_Obs import *
import re

import re
from pathlib import Path
import re
from pathlib import Path
import jax.numpy as jnp
import netket as nk
from pathlib import Path

# ---------------- Helper Functions ----------------

def extract_variables_from_title(path: Path, model):
    """
    Extract key-value pairs from a folder name like:
    'layers1_hidd4_feat16_sample1024_lr0.02_iter700_symmTrue_Hannah'
    
    Returns a dictionary of parsed variables.
    """
    path = Path(path)
    title = path.name  # Get folder name

    # Pattern to capture name-value pairs (letters + number/word/float)
    if model == "HFDS":
        pattern = re.compile(r'([a-zA-Z]+)([0-9.]+|True|False|[A-Za-z]+)')
    if model == "ViT":
        pattern = re.compile(r'([a-zA-Z]+)(-?\d+(?:\.\d+)?|True|False)')

    variables = {}
    for match in pattern.findall(title):
        key, value = match
        # Convert to int, float, or bool
        if value.isdigit():
            value = int(value)
        elif re.match(r'^\d+\.\d+$', value):
            value = float(value)
        elif value in ['True', 'False']:
            value = value == 'True'
        variables[key] = value

    return variables


def initialize_vstate(first_level_folder: Path, J2: float, model):
    """
    Initialize NetKet MCState and Hamiltonian using variables extracted
    from the FIRST LEVEL folder (first_level_folder).
    """
    L = 4
    n_dim = 2
    dtype_ = jnp.float64

    # Extract variables from first_level folder
    try:
        vars_dict = extract_variables_from_title(first_level_folder, model)
        if model == "HFDS":
            hidden_ferm = vars_dict["hidd"]
            hid_layers = vars_dict["layers"]
            features = vars_dict["feat"]
        if model == "ViT":
            num_layers = vars_dict["layers"]
            d_model = vars_dict["d"]
            n_heads = vars_dict["heads"]
            patch_size = vars_dict["patch"]
            symm = vars_dict["symm"]

    except KeyError as e:
        raise ValueError(f"Missing key in folder name '{first_level_folder}': {e}")

    lattice = nk.graph.Hypercube(length=L, n_dim=n_dim, pbc=True, max_neighbor_order=2)
    hi = nk.hilbert.Spin(s=1/2, N=lattice.n_nodes, total_sz=0)

    if model == "HFDS":
        model = HiddenFermion(n_elecs=L*L,network="FFNN",n_hid=hidden_ferm,Lx=L,Ly=L,layers=hid_layers,features=features,MFinit="Fermi",hilbert=hi,stop_grad_mf=False,stop_grad_lower_block=False,bounds="PBC",parity=True,dtype=dtype_)
    if model == "ViT":
        model = ViT_sym(num_layers=num_layers, d_model=d_model, n_heads=n_heads, patch_size=patch_size, transl_invariant=True, parity=True)
    ha = nk.operator.Heisenberg(hilbert=hi,graph=lattice,J=[1.0, J2],sign_rule=[False, False]).to_jax_operator()
    sampler = nk.sampler.MetropolisExchange(hilbert=hi,graph=lattice,d_max=2,n_chains=1024,sweep_size=lattice.n_nodes)

    vstate = nk.vqs.MCState(sampler, model, n_samples=1024, chunk_size=1024, n_discard_per_chain=128)
    return vstate, ha, hi

# ---------------- Main Loop ----------------

def apply_to_files(root_folder: str, model):
    """
    Traverse two levels of subfolders and apply calculation to each.
    """
    root = Path(root_folder)
    if not root.exists() or not root.is_dir():
        print(f"❌ The path '{root_folder}' is invalid.")
        return

    count = 0
    missing_j2_folders = []

    for first_level in root.iterdir():
        if first_level.is_dir():
            for second_level in first_level.iterdir():
                if second_level.is_dir():
                    folder_name = second_level.name

                    #if any(second_level.glob("Sign*Fidelity.png")):
                    #    print(f"⏩ Skipping {second_level}, Sign*Fidelity.png already present.")
                    #    continue

                    # Extract J2
                    if model == "ViT":
                        match = re.search(r"J=([-\d.]+)", folder_name)
                    if model == "HFDS":
                        match = re.search(r"J2=([-\d.]+)", folder_name)
                    if not match:
                        print(f"⚠️ Could not find J2 in folder: {folder_name}")
                        missing_j2_folders.append(second_level)
                        continue

                    J2 = float(match.group(1))
                    print(f"Processing {second_level} with J2={J2}")

                    try:
                        J1J2 = True
                        spin = True

                        vstate, ha, hi = initialize_vstate(first_level, J2, model)

                        E_exact, ket_gs = Exact_gs(4, J2, ha, J1J2, spin)
                        #plot_Sign_Fidelity(ket_gs, vstate, str(second_level), hi)
                        plot_Sign_single(ket_gs, vstate, str(second_level), hi)
                        
                        count += 1

                        print(f"✅ Processed folders.")

                    except Exception as e:
                        print(f"⚠️ Error processing {second_level}: {e}")

    print(f"✅ Processed {count} folders in '{root}'.")
    if missing_j2_folders:
        print("⚠️ Folders missing J2:", [str(f) for f in missing_j2_folders])


# ---------------- Script Entry ----------------

if __name__ == "__main__":

    model = "ViT"
    L=4

    if model == "ViT":
        folder = "/scratch/f/F.Conoscenti/Thesis_QSL/ViT_Heisenberg/plot"
    if model == "HFDS":
        folder = "/scratch/f/F.Conoscenti/Thesis_QSL/HFDS_Heisenberg/plot/J1J2/spin"

    #apply_to_files(folder, model)

    path = "/scratch/f/F.Conoscenti/Thesis_QSL/ViT_Heisenberg/plot/layers2_d32_heads2_patch2_sample1024_lr0.01_iter1000_symmTrue"
    #path = "/scratch/f/F.Conoscenti/Thesis_QSL/HFDS_Heisenberg/plot/J1J2/spin/layers1_hidd1_feat4_sample512_lr0.02_iter100_symmTrue_Hannah"
    J2=0.5
    first_level = path
    second_level = first_level + f"/J={J2}_L=4"

    vstate, ha, hi = initialize_vstate(first_level, J2, model)
    E_exact, ket_gs = Exact_gs(4, J2, ha, J1J2=True, spin=True)
    #plot_Sign_Fidelity(ket_gs, vstate, str(second_level), hi)
    plot_Sign_single(ket_gs, vstate, second_level, hi, 4, L)
    #plot_Weight_single(ket_gs, vstate, second_level, hi, L)
    #plot_MSE_configs(ket_gs, vstate, second_level, hi)
    #plot_Sign_Err_Amplitude_Err_Fidelity(ket_gs, vstate, second_level, hi)

    #sign_expect = get_marshal_sign_full_hilbert(vstate, hi) 
                        
