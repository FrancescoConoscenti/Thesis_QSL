import os
from pathlib import Path
import netket as nk
from HFDS_Heisenberg.HFDS_model_spin import HiddenFermion
from ViT_Heisenberg.ViT_model import ViT_sym
import re
import jax.numpy as jnp
from Elaborate.Statistics.Energy import *
from Elaborate.Statistics.Corr_Struct import *
from Elaborate.Statistics.Error_Stat import *
from Elaborate.Plotting.Sign_vs_iteration import *
from Elaborate.Sign_Obs import *
from Elaborate.Plotting.S_matrix_vs_iteration import plot_S_matrix_eigenvalues
import pickle
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

    # Extract variables from first_level folder
    try:
        L = 4 # Assuming L=4, can be parsed if needed
        n_dim = 2
        n_elecs = L*L
        n_samples = 1024 # Default, can be parsed

        vars_dict = extract_variables_from_title(first_level_folder, model)
        if model == "HFDS":
            hidden_ferm = vars_dict["hidd"]
            hid_layers = vars_dict["layers"]
            features = vars_dict["feat"]
            parity = vars_dict.get("parity", False)
            rotation = vars_dict.get("rotation", False)
            MFinit = vars_dict.get("Init", "Fermi")
            dtype_str = vars_dict.get("type", "real")
            dtype_ = jnp.float64 if dtype_str == "real" else jnp.complex128
            n_samples = vars_dict.get("sample", 1024)

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
        model_obj = HiddenFermion(
            n_elecs=n_elecs,
            network="FFNN",
            n_hid=hidden_ferm,
            Lx=L,
            Ly=L,
            layers=hid_layers,
            features=features,
            MFinit=MFinit,
            hilbert=hi,
            stop_grad_mf=False,
            stop_grad_lower_block=False,
            bounds="PBC",
            parity=parity,
            rotation=rotation,
            dtype=dtype_
        )
    if model == "ViT":
        model_obj = ViT_sym(num_layers=num_layers, d_model=d_model, n_heads=n_heads, patch_size=patch_size, transl_invariant=True, parity=True)
    
    ha = nk.operator.Heisenberg(hilbert=hi,graph=lattice,J=[1.0, J2],sign_rule=[False, False]).to_jax_operator()
    
    n_chains = n_samples // 2 if n_samples > 1 else 1
    sampler = nk.sampler.MetropolisExchange(hilbert=hi,graph=lattice,d_max=2,n_chains=n_chains,sweep_size=lattice.n_nodes)

    vstate = nk.vqs.MCState(sampler, model_obj, n_samples=n_samples, chunk_size=n_samples, n_discard_per_chain=128)
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

                        eigenvalues, rel_1, rel_2, rel_3 = plot_S_matrix_eigenvalues(vstate, folder, hi,  one_avg = "one")
  
                        count += 1

                        print(f"✅ Processed folders.")

                    except Exception as e:
                        print(f"⚠️ Error processing {second_level}: {e}")

    print(f"✅ Processed {count} folders in '{root}'.")
    if missing_j2_folders:
        print("⚠️ Folders missing J2:", [str(f) for f in missing_j2_folders])


# ---------------- Script Entry ----------------

if __name__ == "__main__":
    
    #apply_to_files(folder, model)

    model = "HFDS"
    base_path = "/cluster/home/fconoscenti/Thesis_QSL/HFDS_Heisenberg/plot/spin_new/layers1_hidd4_feat32_sample1024_lr0.025_iter500_parityTrue_rotationFalse_InitFermi_typereal"
    if not os.path.exists(base_path):
        base_path = base_path.replace("/cluster/home/fconoscenti/Thesis_QSL", "/scratch/f/F.Conoscenti/Thesis_QSL")
    
    for j_folder in os.listdir(base_path):
        if j_folder.startswith("J="):
            j_path = os.path.join(base_path, j_folder)
            if not os.path.isdir(j_path):
                continue
            
            try:
                j_value_str = j_folder.split('=')[1]
                j_value = float(j_value_str)
            except (ValueError, IndexError):
                print(f"Could not parse J value from folder name: {j_folder}")
                continue

            for seed_folder in os.listdir(j_path):
                if seed_folder.startswith("seed_"):
                    seed_path = os.path.join(j_path, seed_folder)
                    if not os.path.isdir(seed_path):
                        continue
                    
                    print(f"Processing J={j_value} in {seed_path}")

                    try:
                        # Correctly call initialize_vstate with the model
                        vstate, ha, hi = initialize_vstate(Path(base_path), j_value, model)
                        #E_exact, ket_gs = Exact_gs(4, j_value, ha, J1J2=True, spin=True)
                        #sign_vstate_full, sign_exact, fidelity = plot_Sign_Fidelity(ket_gs, vstate, hi, seed_path, one_avg = "one")
                        #configs, sign_vstate_config, weight_exact, weight_vstate = plot_Sign_single_config(ket_gs, vstate, hi, 3, 4, seed_path, one_avg = "one")
                        #amp_overlap, fidelity, sign_vstate, sign_exact, sign_overlap = plot_Sign_Err_Amplitude_Err_Fidelity(ket_gs, vstate, hi, seed_path, one_avg = "one")
                        #amp_overlap, sign_vstate, sign_exact, sign_overlap = plot_Sign_Err_vs_Amplitude_Err_with_iteration(ket_gs, vstate, hi, seed_path, one_avg = "one")
                        #sorted_weights, sorted_amp_overlap, sorted_sign_overlap = plot_Overlap_vs_Weight(ket_gs, vstate, hi, seed_path, "one")
                        eigenvalues, rel_1, rel_2, rel_3 = plot_S_matrix_eigenvalues(vstate, seed_path, hi,  one_avg = "one")
                        
                        """variables = {
                                #'sign_vstate_MCMC': sign_vstate_MCMC,
                                'sign_vstate_full': sign_vstate_full,
                                'sign_exact': sign_exact,
                                'fidelity': fidelity,
                                'configs': configs,
                                'sign_vstate_config': sign_vstate_config,
                                'weight_exact': weight_exact,
                                'weight_vstate': weight_vstate,
                                'amp_overlap': amp_overlap,
                                'sign_overlap': sign_overlap,
                                'eigenvalues': eigenvalues
                            }

                        with open(seed_path+"/variables.pkl", 'wb') as f:
                            pickle.dump(variables, f)"""
                        
                        print(f"✅ Successfully generated plot for {seed_path}")
                            
                    except Exception as e:
                        print(f"Error processing {seed_path}: {e}")