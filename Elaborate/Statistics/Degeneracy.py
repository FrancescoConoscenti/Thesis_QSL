import matplotlib.pyplot as plt
import os
import sys
import pickle
import re
import numpy as np
import netket as nk
import jax
import jax.numpy as jnp
import flax

# Add path to project root
sys.path.append("/scratch/f/F.Conoscenti/Thesis_QSL")

try:
    from ViT_Heisenberg.ViT_model import ViT_sym
    from HFDS_Heisenberg.HFDS_model_spin import HiddenFermion, HiddenFermion_phi
except ImportError:
    print("Warning: Could not import project models. Ensure path is correct.")

def parse_model_path_local(model_path):
    params = {}
    if "4x4" in model_path: params['L'] = 4
    if "6x6" in model_path: params['L'] = 6
    if "8x8" in model_path: params['L'] = 8
    if "10x10" in model_path: params['L'] = 10
    
    match_J = re.search(r"J=([\d\.]+)", model_path)
    params['J2'] = float(match_J.group(1)) if match_J else 0.5
    
    match_phi = re.search(r"_phi([\d\.]+)_", model_path)
    if match_phi:
        params['phi'] = float(match_phi.group(1))
        params['is_phi_model'] = True
    else:
        params['phi'] = 0.0
        params['is_phi_model'] = "_phi" in model_path
    
    if "hidd" in model_path:
        params['model_type'] = 'HFDS'
        params['n_hid'] = int(re.search(r"hidd(\d+)", model_path).group(1))
        params['features'] = int(re.search(r"feat(\d+)", model_path).group(1))
        params['layers'] = int(re.search(r"layers(\d+)", model_path).group(1))
        match_init = re.search(r"Init((?:(?!_type)[a-zA-Z_])+)", model_path)
        params['MFinit'] = match_init.group(1) if match_init else "random"
        match_type = re.search(r"type([a-zA-Z]+)", model_path)
        params['dtype'] = match_type.group(1) if match_type else "complex"
        match_bc = re.search(r"bc([A-Z_]+)", model_path)
        if match_bc:
            bc_parts = [p for p in match_bc.group(1).split('_') if p]
            if len(bc_parts) == 2:
                params['bc_x'], params['bc_y'] = bc_parts
            elif len(bc_parts) == 1:
                params['bc_x'] = params['bc_y'] = bc_parts[0]
    else:
        params['model_type'] = 'ViT'
        params['num_layers'] = int(re.search(r"layers(\d+)", model_path).group(1)) if re.search(r"layers(\d+)", model_path) else 2
        params['d_model'] = int(re.search(r"_d(\d+)", model_path).group(1)) if re.search(r"_d(\d+)", model_path) else 8
        params['n_heads'] = int(re.search(r"heads(\d+)", model_path).group(1)) if re.search(r"heads(\d+)", model_path) else 4
        params['patch_size'] = int(re.search(r"patch(\d+)", model_path).group(1)) if re.search(r"patch(\d+)", model_path) else 2
    return params

def load_vstate_local(folder, n_samples=16):
    params = parse_model_path_local(folder)
    L = params.get('L', 4)
    
    # Setup System
    n_dim = 2
    # Keep the graph periodic so the edges exist at the boundaries
    lattice = nk.graph.Hypercube(length=L, n_dim=n_dim, pbc=[True, True], max_neighbor_order=2)
    hilbert = nk.hilbert.Spin(s=1 / 2, N=lattice.n_nodes, total_sz=0)
    
    # Setup Model
    if params['model_type'] == 'ViT':
        model = ViT_sym(L=L, num_layers=params['num_layers'], d_model=params['d_model'], 
                        n_heads=params['n_heads'], patch_size=params['patch_size'], 
                        transl_invariant=True, parity=True, rotation=True)
    elif params['model_type'] == 'HFDS':
        dtype_ = jnp.float64 if params.get('dtype') == "real" else jnp.complex128
        if params.get('is_phi_model', False):
            model = HiddenFermion_phi(L=L, network="FFNN", n_hid=params['n_hid'], layers=params['layers'], 
                                      features=params['features'], MFinit=params['MFinit'], hilbert=hilbert, 
                                      bounds=(params.get('bc_x', 'PBC'), params.get('bc_y', 'PBC')),
                                      phi=params.get('phi', 0.0),
                                      parity=True, rotation=True, dtype=dtype_)
        else:
            model = HiddenFermion(L=L, network="FFNN", n_hid=params['n_hid'], layers=params['layers'], 
                                  features=params['features'], MFinit=params['MFinit'], hilbert=hilbert, 
                                  bounds=params.get('bc_x', 'PBC'),
                                  parity=True, rotation=True, dtype=dtype_)
    else:
        return None

    sampler = nk.sampler.MetropolisExchange(hilbert=hilbert, graph=lattice, d_max=2, n_chains=16)
    vstate = nk.vqs.MCState(sampler=sampler, model=model, n_samples=n_samples)
    
    models_dir = os.path.join(folder, "models")
    if os.path.exists(models_dir):
        files = [f for f in os.listdir(models_dir) if f.endswith(".mpack")]
        if files:
            files.sort(key=lambda x: int(re.search(r"model_(\d+)", x).group(1)))
            last_model = files[-1]
            with open(os.path.join(models_dir, last_model), 'rb') as f:
                data = f.read()
                try:
                    vstate = flax.serialization.from_bytes(vstate, data)
                except KeyError:
                    vstate.variables = flax.serialization.from_bytes(vstate.variables, data)
            
            # Ensure consistent sample size and chunk size to prevent shape mismatches and OOM
            vstate.n_samples = n_samples
            vstate.chunk_size = vstate.n_samples//4
            return vstate
    return None


@jax.jit
def fast_overlap_kernel(log_psi_x, log_phi_x, log_psi_y, log_phi_y):
    """
    Computes the fidelity using the log-amplitudes.
    x ~ samples from psi
    y ~ samples from phi
    """
    # Ratio psi: E_{x~psi} [phi(x)/psi(x)]
    ratio_psi = jnp.mean(jnp.exp(log_phi_x - log_psi_x))
    
    # Ratio phi: E_{y~phi} [psi(y)/phi(y)]
    ratio_phi = jnp.mean(jnp.exp(log_psi_y - log_phi_y))
    
    # Fidelity F = |<psi|phi>|^2 / (<psi|psi><phi|phi>)
    return jnp.abs(ratio_psi * ratio_phi)

def calculate_overlap_vmap(vstate_psi, vstate_phi):
    # 1. Generate samples
    # Reshape to (Total_Samples, Hilbert_Size)
    s_psi = vstate_psi.sample().reshape(-1, vstate_psi.hilbert.size)
    s_phi = vstate_phi.sample().reshape(-1, vstate_phi.hilbert.size)

    # 2. Evaluate log values
    # NetKet's log_value is already vectorized over the first dimension
    log_psi_x = vstate_psi.log_value(s_psi)
    log_phi_x = vstate_phi.log_value(s_psi)
    
    log_psi_y = vstate_psi.log_value(s_phi)
    log_phi_y = vstate_phi.log_value(s_phi)

    # 3. Compute overlap
    return fast_overlap_kernel(log_psi_x, log_phi_x, log_psi_y, log_phi_y)


def plot_En_degeneracy(models, target_J=0.5):
    plot_data = {} # Key: bc_label, Value: dict with lists for inv_N, means, stds

    for model_path in models:
        # Fix path if running locally vs cluster
        if not os.path.exists(model_path):
            model_path = model_path.replace("/cluster/home/fconoscenti", "/scratch/f/F.Conoscenti")
        
        if not os.path.exists(model_path):
            print(f"Warning: Path not found, skipping: {model_path}")
            continue

        # Extract parameters for grouping
        params = parse_model_path_local(model_path)
        L = params.get('L')
        if L is None:
            print(f"Warning: Could not determine L for {model_path}, skipping.")
            continue

        bc_x = params.get('bc_x', 'PBC')
        bc_y = params.get('bc_y', 'PBC')
        bc_label = f"{bc_x}, {bc_y}"

        seed_energies = []
        
        # Find specific J folder
        j_folder = None

        if os.path.exists(model_path):
            for d in os.listdir(model_path):
                if d.startswith("J=") or d.startswith("J2="):
                    try:
                        part = d.split('=')[1]
                        val_str = part.split('_')[0] if '_' in part else part
                        if abs(float(val_str) - target_J) < 1e-5:
                            j_folder = os.path.join(model_path, d)
                            break
                    except ValueError:
                        continue
        
        if j_folder and os.path.exists(j_folder):
            seeds = sorted([s for s in os.listdir(j_folder) if os.path.isdir(os.path.join(j_folder, s)) and "seed" in s])
            for seed_sub in seeds:
                full_seed_path = os.path.join(j_folder, seed_sub)

                if os.path.isdir(full_seed_path):
                    E_final = None
                    var_path = os.path.join(full_seed_path, "variables.pkl")
                    if os.path.exists(var_path):
                        try:
                            with open(var_path, "rb") as f:
                                vars_data = pickle.load(f)
                            E_pkl = vars_data.get('E_vs_final')
                            if E_pkl is not None:
                                E_final = float(np.real(E_pkl))
                        except Exception as e:
                            print(f"Error reading {var_path}: {e}")

                    # Compare with output.txt
                    out_path = os.path.join(full_seed_path, "output.txt")
                    if os.path.exists(out_path):
                        try:
                            with open(out_path, "r") as f:
                                content = f.read()
                            matches = re.findall(r"Final Energy from VMC:\s*([-\d\.e]+)", content)
                            if matches:
                                E_txt = float(matches[-1])
                                print(f"Found E={E_txt:.6f} in output.txt of {seed_sub}")
                                if E_final is None:
                                    E_final = E_txt
                                elif abs(E_final - E_txt) > 1e-5:
                                    print(f"  Mismatch in {seed_sub}: pkl={E_final:.6f}, txt={E_txt:.6f}. Using output.txt.")
                                    E_final = E_txt
                        except Exception as e:
                            print(f"Error reading output.txt in {seed_sub}: {e}")

                    if E_final is not None:
                        if L in [6, 8, 10] and bc_label in ["APC, PBC", "APC, APC", "PBC, APC"]:
                            E_final *= 4.0
                        print(f"Found E={E_final:.6f} in {full_seed_path}")
                        seed_energies.append(E_final)
        
        if seed_energies:
            mean_E = np.mean(seed_energies)
            std_E = np.std(seed_energies)
            inv_N = 1.0 / (L*L)
            
            if bc_label not in plot_data:
                plot_data[bc_label] = {'inv_N': [], 'means': [], 'stds': []}
            
            plot_data[bc_label]['inv_N'].append(inv_N)
            plot_data[bc_label]['means'].append(mean_E)
            plot_data[bc_label]['stds'].append(std_E)
            print(f"Model: {os.path.basename(model_path)}, L={L}, BC={bc_label}")
            print(f"  Found {len(seed_energies)} seeds. Mean E: {mean_E:.6f}, Std E: {std_E:.6f}\n")
        else:
            print(f"Warning: No energy data found for model: {model_path}\n")

    # --- PLOT 1: Energies ---
    plt.figure(figsize=(10, 6))
    markers = ['o', 's', '^', 'D', 'v', '<', '>']
    colors = plt.cm.viridis(np.linspace(0, 0.9, max(len(plot_data), 1)))
    
    for i, (bc_label, data) in enumerate(sorted(plot_data.items())):
        inv_N_vals = data['inv_N']
        means = data['means']
        stds = data['stds']
        
        # Sort by inv_N for clean plotting of lines
        sort_indices = np.argsort(inv_N_vals)
        inv_N_sorted = np.array(inv_N_vals)[sort_indices]
        means_sorted = np.array(means)[sort_indices]
        stds_sorted = np.array(stds)[sort_indices]

        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]
        
        # Plot error bars for each point
        plt.errorbar(inv_N_sorted, means_sorted, yerr=stds_sorted, 
                     fmt=marker, color=color, capsize=5, label=bc_label, markersize=8, linestyle='none')
        
        # Plot a connecting line
        plt.plot(inv_N_sorted, means_sorted, color=color, linestyle='--', alpha=0.5)

    plt.legend(loc='best')

    plt.xlabel("1/L²")
    plt.ylabel("Energy per site")
    plt.title(f"Finite Size Scaling / Degeneracy Check (J={target_J})")
    plt.grid(True)
    
    save_dir = "/scratch/f/F.Conoscenti/Thesis_QSL/Elaborate/plot/Degeneracy"
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, "Degeneracy_check_Energy.png"))
    print(f"Plot saved to {os.path.join(save_dir, 'Degeneracy_check_Energy.png')}")
    plt.show()


def plot_Fidelity_Degeneracy(models, target_J=0.5, n_samples=16):
    vstates_by_L = {}

    for model_path in models:
        # Fix path if running locally vs cluster
        if not os.path.exists(model_path):
            model_path = model_path.replace("/cluster/home/fconoscenti", "/scratch/f/F.Conoscenti")
        
        if not os.path.exists(model_path):
            print(f"Warning: Path not found, skipping: {model_path}")
            continue

        # Extract parameters for grouping
        params = parse_model_path_local(model_path)
        L = params.get('L')
        if L is None:
            print(f"Warning: Could not determine L for {model_path}, skipping.")
            continue

        bc_x = params.get('bc_x', 'PBC')
        bc_y = params.get('bc_y', 'PBC')
        bc_label = f"{bc_x}, {bc_y}"

        # Find specific J folder
        j_folder = None
        current_vstate = None

        if os.path.exists(model_path):
            for d in os.listdir(model_path):
                if d.startswith("J=") or d.startswith("J2="):
                    try:
                        part = d.split('=')[1]
                        val_str = part.split('_')[0] if '_' in part else part
                        if abs(float(val_str) - target_J) < 1e-5:
                            j_folder = os.path.join(model_path, d)
                            break
                    except ValueError:
                        continue
        
        if j_folder and os.path.exists(j_folder):
            seeds = sorted([s for s in os.listdir(j_folder) if os.path.isdir(os.path.join(j_folder, s)) and "seed" in s])
            for seed_sub in seeds:
                full_seed_path = os.path.join(j_folder, seed_sub)
                
                # Load vstate from the first valid seed if not already loaded
                if current_vstate is None:
                    try:
                        current_vstate = load_vstate_local(full_seed_path, n_samples=n_samples)
                    except Exception as e:
                        print(f"Error loading vstate from {full_seed_path}: {e}")
            
            if current_vstate is not None:
                if L not in vstates_by_L:
                    vstates_by_L[L] = {}
                vstates_by_L[L][bc_label] = current_vstate
                print(f"Loaded vstate for Model: {os.path.basename(model_path)}, L={L}, BC={bc_label}")
            else:
                print(f"Warning: No valid vstate found for model: {model_path}\n")

    # Compute fidelities between all pairs of BCs for each L
    fidelity_plot_data = {}
    for L in sorted(vstates_by_L.keys()):
        bc_map = vstates_by_L[L]
        bcs = sorted(list(bc_map.keys()))
        for i in range(len(bcs)):
            for j in range(i + 1, len(bcs)):
                bc1 = bcs[i]
                bc2 = bcs[j]
                pair_label = f"{bc1} vs {bc2}"
                try:
                    fid = float(calculate_overlap_vmap(bc_map[bc1], bc_map[bc2]))
                    if pair_label not in fidelity_plot_data:
                        fidelity_plot_data[pair_label] = {'inv_N': [], 'fidelity': []}
                    fidelity_plot_data[pair_label]['inv_N'].append(1.0 / (L * L))
                    fidelity_plot_data[pair_label]['fidelity'].append(fid)
                    print(f"Fidelity L={L} {bc1}-{bc2} = {fid}")
                except Exception as e:
                    print(f"Failed fidelity calc L={L} {bc1}-{bc2}: {e}")

    # --- PLOT 2: Fidelities ---
    if fidelity_plot_data:
        plt.figure(figsize=(10, 6))
        markers = ['o', 's', '^', 'D', 'v', '<', '>']
        colors = plt.cm.viridis(np.linspace(0, 0.9, max(len(fidelity_plot_data), 1)))
        
        for i, (pair_label, data) in enumerate(sorted(fidelity_plot_data.items())):
            inv_N_vals = data['inv_N']
            fidelities = data['fidelity']
            
            # Sort by inv_N for clean plotting of lines
            sort_indices = np.argsort(inv_N_vals)
            inv_N_sorted = np.array(inv_N_vals)[sort_indices]
            fid_sorted = np.array(fidelities)[sort_indices]

            color = colors[i % len(colors)]
            marker = markers[i % len(markers)]
            
            plt.plot(inv_N_sorted, fid_sorted, marker=marker, color=color, label=pair_label, markersize=8, linestyle='--')
            
        plt.legend(loc='best')
        plt.xlabel("1/L²")
        plt.ylabel("Fidelity |<ψ|φ>|²")
        plt.title(f"Ground State Fidelity between BCs (J={target_J})")
        plt.grid(True)
        
        plt.savefig(os.path.join(save_dir, "Degeneracy_check_Fidelity.png"))
        print(f"Plot saved to {os.path.join(save_dir, 'Degeneracy_check_Fidelity.png')}")
        plt.show()


if __name__ == "__main__": 
    
    models = ["/scratch/f/F.Conoscenti/Thesis_QSL/HFDS_Heisenberg/plot/4x4/layers1_hidd4_feat64_sample1024_bcAPC_APC_lr0.02_iter2000_parityTrue_rotFalse_Initrandom_typecomplex_newBC_no_k_shift",
              "/scratch/f/F.Conoscenti/Thesis_QSL/HFDS_Heisenberg/plot/4x4/layers1_hidd4_feat64_sample1024_bcAPC_PBC_lr0.02_iter2000_parityTrue_rotFalse_Initrandom_typecomplex_newBC_no_k_shift",
              "/scratch/f/F.Conoscenti/Thesis_QSL/HFDS_Heisenberg/plot/4x4/layers1_hidd4_feat64_sample1024_bcPBC_PBC_lr0.02_iter2000_parityTrue_rotFalse_Initrandom_typecomplex_newBC_no_k_shift",
            "/scratch/f/F.Conoscenti/Thesis_QSL/HFDS_Heisenberg/plot/4x4/layers1_hidd4_feat64_sample1024_bcPBC_APC_lr0.02_iter2000_parityTrue_rotFalse_Initrandom_typecomplex_newBC_no_k_shift",


              "/scratch/f/F.Conoscenti/Thesis_QSL/HFDS_Heisenberg/plot/6x6/BC/layers1_hidd8_feat64_sample2048_bcAPC_APC_lr0.02_iter2000_parityTrue_rotFalse_InitFermi_typecomplex_newBC",
              "/scratch/f/F.Conoscenti/Thesis_QSL/HFDS_Heisenberg/plot/6x6/BC/layers1_hidd8_feat64_sample2048_bcAPC_PBC_lr0.02_iter2000_parityTrue_rotFalse_InitFermi_typecomplex_newBC",
              "/scratch/f/F.Conoscenti/Thesis_QSL/HFDS_Heisenberg/plot/6x6/BC/layers1_hidd8_feat64_sample2048_bcPBC_PBC_lr0.02_iter2000_parityTrue_rotFalse_InitFermi_typecomplex_newBC",
            "/scratch/f/F.Conoscenti/Thesis_QSL/HFDS_Heisenberg/plot/6x6/BC/layers1_hidd8_feat64_sample2048_bcPBC_APC_lr0.02_iter2000_parityTrue_rotFalse_InitFermi_typecomplex_newBC",

            
            "/scratch/f/F.Conoscenti/Thesis_QSL/HFDS_Heisenberg/plot/8x8/layers1_hidd8_feat64_sample2048_bcAPC_APC_lr0.02_iter2000_parityTrue_rotFalse_InitFermi_typecomplex_newBC",
            "/scratch/f/F.Conoscenti/Thesis_QSL/HFDS_Heisenberg/plot/8x8/layers1_hidd8_feat64_sample2048_bcAPC_PBC_lr0.02_iter2000_parityTrue_rotFalse_InitFermi_typecomplex_newBC",
            "/scratch/f/F.Conoscenti/Thesis_QSL/HFDS_Heisenberg/plot/8x8/layers1_hidd8_feat64_sample2048_bcPBC_PBC_lr0.02_iter2000_parityTrue_rotFalse_InitFermi_typecomplex_newBC",
            "/scratch/f/F.Conoscenti/Thesis_QSL/HFDS_Heisenberg/plot/8x8/layers1_hidd8_feat64_sample2048_bcPBC_APC_lr0.02_iter2000_parityTrue_rotFalse_InitFermi_typecomplex_newBC",

            
            "/scratch/f/F.Conoscenti/Thesis_QSL/HFDS_Heisenberg/plot/10x10/layers1_hidd8_feat64_sample2048_bcAPC_APC_lr0.02_iter2000_parityTrue_rotFalse_InitFermi_typecomplex_newBC",
            "/scratch/f/F.Conoscenti/Thesis_QSL/HFDS_Heisenberg/plot/10x10/layers1_hidd8_feat64_sample2048_bcAPC_PBC_lr0.02_iter2000_parityTrue_rotFalse_InitFermi_typecomplex_newBC",
            "/scratch/f/F.Conoscenti/Thesis_QSL/HFDS_Heisenberg/plot/10x10/layers1_hidd8_feat64_sample2048_bcPBC_APC_lr0.02_iter2000_parityTrue_rotFalse_InitFermi_typecomplex_newBC",
            "/scratch/f/F.Conoscenti/Thesis_QSL/HFDS_Heisenberg/plot/10x10/layers1_hidd8_feat64_sample2048_bcPBC_PBC_lr0.02_iter2000_parityTrue_rotFalse_InitFermi_typecomplex_newBC",
            
            ]
    plot_En_degeneracy(models, target_J=0.5)
    plot_Fidelity_Degeneracy(models, target_J=0.5, n_samples=1024)