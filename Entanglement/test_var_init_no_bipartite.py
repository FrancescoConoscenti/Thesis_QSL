import netket as nk
import numpy as np
import jax
import jax.numpy as jnp
from jax.nn.initializers import normal
import sys
import matplotlib.pyplot as plt
import os
sys.path.append("/scratch/f/F.Conoscenti/Thesis_QSL")
from ViT_Heisenberg.ViT_model_ent import ViT_ent
from HFDS_Heisenberg.entanglement_model.HFDS_model_spin_ent import HiddenFermion_ent
from Entanglement.Entanglement import compute_renyi2_entropy, clean_up

def get_unique_path(directory, filename):
    base, ext = os.path.splitext(filename)
    counter = 1
    new_filename = filename
    while os.path.exists(os.path.join(directory, new_filename)):
        new_filename = f"{base}_{counter}{ext}"
        counter += 1
    return os.path.join(directory, new_filename)

def get_strip_partition(L, width):
    indices = []
    for row in range(width):
        for col in range(L):
            indices.append(row * L + col)
    return jnp.array(indices)

def get_square_partition(L, side):
    indices = []
    for row in range(side):
        for col in range(side):
            indices.append(row * L + col)
    return jnp.array(indices)

def plot_entropy_vs_partition_variance(L=6, n_seeds=10, n_samples=4096, models_to_plot=None):
    print(f"\n--- Plotting Entropy vs Partition Size (L={L}) ---")
    save_dir = "/cluster/home/fconoscenti/Thesis_QSL/Entanglement/plots"
    if not os.path.exists(save_dir):
        save_dir = "/scratch/f/F.Conoscenti/Thesis_QSL/Entanglement/plots"
    os.makedirs(save_dir, exist_ok=True)
    if models_to_plot is None:
        models_to_plot = ["RBM", "ViT", "HFDS"]
    variances = [5e-3, 1e-2, 1, 10, 50]
    
    partition_types = ["Strip", "Square"]
    
    results = {} # results[name][var] = {'L': [], 'mean': [], 'err': [], 'params': []}
    xavier_results = {'L': [], 'mean': [], 'err': [], 'params': []} if (models_to_plot is None or "ViT" in models_to_plot) else None

    N = L*L
    g = nk.graph.Hypercube(length=L, n_dim=2, pbc=True)
    
    hi_free = nk.hilbert.Spin(s=1/2, N=N)
    hi_constrained = nk.hilbert.Spin(s=1/2, N=N, total_sz=0)

    for p_type in partition_types:
        print(f"Processing Partition Type: {p_type}")
        results = {} 
        xavier_results = {'size': [], 'mean': [], 'err': [], 'params': []} if (models_to_plot is None or "ViT" in models_to_plot) else None

        if p_type == "Strip":
            sizes = range(1, L // 2 + 1)
        else:
            sizes = range(1, L)

        # Precompute partitions
        partitions_map = {}
        for size in sizes:
            if p_type == "Strip":
                partition = get_strip_partition(L, size)
                partitions_map[size] = get_strip_partition(L, size)
            else:
                partition = get_square_partition(L, size)
            
            print(f"  Size: {size}, Partition len: {len(partition)}")
            partitions_map[size] = get_square_partition(L, size)

        for var in variances:
            print(f"  Processing Variance: {var}")
            std = np.sqrt(var)
            init_fun = normal(stddev=std)
            
            temp_results = {}

            for seed in range(n_seeds):
                # Models
                rbm = nk.models.RBM(alpha=1, param_dtype=complex, kernel_init=init_fun, hidden_bias_init=init_fun, visible_bias_init=init_fun)
                vit = ViT_ent(num_layers=2, d_model=8, n_heads=4, patch_size=2, kernel_init=init_fun)
                hfds = HiddenFermion_ent(L=L, network="FFNN", n_hid=2, layers=1, features=8, MFinit="Fermi", hilbert=hi_constrained, kernel_init=init_fun, dtype=jax.numpy.complex128)
                
                all_models_list = [
                    ("RBM", rbm, hi_free, nk.sampler.MetropolisLocal(hi_free)),
                    ("ViT", vit, hi_free, nk.sampler.MetropolisLocal(hi_free)),
                    ("HFDS", hfds, hi_constrained, nk.sampler.MetropolisExchange(hi_constrained, graph=g))
                ]
                
                models_list = [m for m in all_models_list if models_to_plot is None or m[0] in models_to_plot]

                for name, model, hi, sampler in models_list:
                    vstate = nk.vqs.MCState(sampler, model, n_samples=n_samples, seed=seed)
                    if name not in temp_results:
                        temp_results[name] = {'s2_matrix': np.zeros((n_seeds, len(sizes))), 'params': 0}
                    
                    if seed == 0:
                        temp_results[name]['params'] = nk.jax.tree_size(vstate.parameters)

                    for idx, size in enumerate(sizes):
                        partition = partitions_map[size]
                        s2, _ = compute_renyi2_entropy(vstate, partition_indices=partition, n_samples=n_samples)
                        temp_results[name]['s2_matrix'][seed, idx] = s2

            for name, data in temp_results.items():
                mean = np.mean(data['s2_matrix'], axis=0)
                err = np.std(data['s2_matrix'], axis=0) / np.sqrt(n_seeds)
                param_count = data['params']

                if name not in results: results[name] = {}
                if var not in results[name]: results[name][var] = {'size': [], 'mean': [], 'err': [], 'params': []}

                results[name][var]['size'] = list(sizes)
                results[name][var]['mean'] = list(mean)
                results[name][var]['err'] = list(err)
                results[name][var]['params'] = [param_count] * len(sizes)
                print(f"    {name} Var={var}: S2 mean over sizes={np.mean(mean):.4f}")
        
        # ViT Xavier
        if models_to_plot is None or "ViT" in models_to_plot:
            print(f"    ViT Xavier...")
            init_xavier = jax.nn.initializers.xavier_uniform()
            vit_xavier = ViT_ent(num_layers=2, d_model=8, n_heads=4, patch_size=2, kernel_init=init_xavier)
            sampler_vit = nk.sampler.MetropolisLocal(hi_free)
            
            s2_vals_x = []
            s2_matrix_x = np.zeros((n_seeds, len(sizes)))
            param_count_x = 0
            
            for seed in range(n_seeds):
                 vit_xavier = ViT_ent(num_layers=2, d_model=8, n_heads=4, patch_size=2, kernel_init=init_xavier)
                 sampler_vit = nk.sampler.MetropolisLocal(hi_free)
                 vstate = nk.vqs.MCState(sampler_vit, vit_xavier, n_samples=n_samples, seed=seed)
                 if param_count_x == 0:
                     param_count_x = nk.jax.tree_size(vstate.parameters)
                 s2, _ = compute_renyi2_entropy(vstate, partition_indices=partition, n_samples=n_samples)
                 s2_vals_x.append(s2)
                 
                 for idx, size in enumerate(sizes):
                     partition = partitions_map[size]
                     s2, _ = compute_renyi2_entropy(vstate, partition_indices=partition, n_samples=n_samples)
                     s2_matrix_x[seed, idx] = s2
            
            xavier_results['size'].append(size)
            xavier_results['mean'].append(np.mean(s2_vals_x))
            xavier_results['err'].append(np.std(s2_vals_x)/np.sqrt(n_seeds))
            xavier_results['params'].append(param_count_x)
            print(f"    ViT Xavier: S2={np.mean(s2_vals_x):.4f}")
            xavier_results['size'] = list(sizes)
            xavier_results['mean'] = list(np.mean(s2_matrix_x, axis=0))
            xavier_results['err'] = list(np.std(s2_matrix_x, axis=0)/np.sqrt(n_seeds))
            xavier_results['params'] = [param_count_x] * len(sizes)
            print(f"    ViT Xavier: S2 mean over sizes={np.mean(xavier_results['mean']):.4f}")

        colors = {'RBM': 'blue', 'ViT': 'orange', 'HFDS': 'green'}
        markers = {1e-3: 'o', 1e-2: 's', 1e-1: 'D', 1: '^'}
        linestyles = {1e-3: '-', 1e-2: '--', 1e-1: '-.', 1: ':'}
        
        # Plotting Unnormalized
        plt.figure(figsize=(10, 6))
        for name in results:
            for var in variances:
                data = results[name][var]
                params_str = ",".join(map(str, data['params']))
                label = f"{name} Var={var} (P={params_str})"
                
                size_arr = np.array(data['size'])
                if p_type == "Strip":
                    partition_size = size_arr * L
                else:
                    partition_size = size_arr ** 2
                
                max_ent = partition_size * np.log(2)
                
                plt.errorbar(size_arr, np.array(data['mean']) * max_ent, yerr=np.array(data['err']) * max_ent, 
                            label=label, color=colors[name], 
                            marker=markers.get(var, 'o'), linestyle=linestyles.get(var, '-'), capsize=5)
        
        if xavier_results is not None and len(xavier_results['size']) > 0:
            size_x = np.array(xavier_results['size'])
            if p_type == "Strip":
                partition_size_x = size_x * L
            else:
                partition_size_x = size_x ** 2
            max_ent_x = partition_size_x * np.log(2)
            
            params_str_x = ",".join(map(str, xavier_results['params']))
            plt.errorbar(size_x, np.array(xavier_results['mean']) * max_ent_x, yerr=np.array(xavier_results['err']) * max_ent_x,
                        label=f'ViT Xavier (P={params_str_x})', color='red', marker='*', linestyle='-', capsize=5)
        
        plt.xlabel(f'Partition Size ({p_type})')
        plt.ylabel('Renyi-2 Entropy')
        plt.title(f'Entanglement Entropy vs Partition Size ({p_type}, L={L}, Unnormalized)')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        save_path = get_unique_path(save_dir, f"Entropy_vs_Partition_{p_type}_L{L}_Unnormalized.png")
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")

def plot_entropy_vs_partition_hidden_size(L=6, n_seeds=10, n_samples=4096, models_to_plot=None):
    print(f"\n--- Plotting Entropy vs Partition Size (L={L}, Varying Hidden Size) ---")
    save_dir = "/cluster/home/fconoscenti/Thesis_QSL/Entanglement/plots"
    if not os.path.exists(save_dir):
        save_dir = "/scratch/f/F.Conoscenti/Thesis_QSL/Entanglement/plots"
    os.makedirs(save_dir, exist_ok=True)
    if models_to_plot is None:
        models_to_plot = ["RBM", "ViT", "HFDS", "HFDS Random"]
    var = 1
    std = np.sqrt(var)
    init_fun = normal(stddev=std)
    
    configs = {
        'Small':  {'RBM': 2, 'HFDS': 1, 'ViT': 8},
        'Medium': {'RBM': 4, 'HFDS': 2, 'ViT': 16},
        'Large':  {'RBM': 8, 'HFDS': 3, 'ViT': 32}
    }
    
    partition_types = ["Strip", "Square"]
    
    N = L*L
    g = nk.graph.Hypercube(length=L, n_dim=2, pbc=True)
    hi_free = nk.hilbert.Spin(s=1/2, N=N)
    hi_constrained = nk.hilbert.Spin(s=1/2, N=N, total_sz=0)

    for p_type in partition_types:
        print(f"Processing Partition Type: {p_type}")
        results = {}

        if p_type == "Strip":
            sizes = range(1, L // 2 + 1)
        else:
            sizes = range(1, L)

        partitions_map = {}
        for size in sizes:
            if p_type == "Strip":
                partitions_map[size] = get_strip_partition(L, size)
            else:
                partitions_map[size] = get_square_partition(L, size)

        for size_label in ['Small', 'Medium', 'Large']:
            print(f"  Processing Model Size: {size_label}")
            params = configs[size_label]

            # RBM
            alpha = params['RBM']
            rbm = nk.models.RBM(alpha=alpha, param_dtype=complex, kernel_init=init_fun, hidden_bias_init=init_fun, visible_bias_init=init_fun)

            # ViT
            d_model = params['ViT']
            vit = ViT_ent(num_layers=2, d_model=d_model, n_heads=4, patch_size=2, kernel_init=init_fun)

            # HFDS
            n_hid = params['HFDS']
            hfds = HiddenFermion_ent(L=L, network="FFNN", n_hid=n_hid, layers=1, features=16, MFinit="Fermi", hilbert=hi_constrained, kernel_init=init_fun, dtype=jax.numpy.complex128)
            hfds_rand = HiddenFermion_ent(L=L, network="FFNN", n_hid=n_hid, layers=1, features=16, MFinit="random", hilbert=hi_constrained, kernel_init=init_fun, dtype=jax.numpy.complex128)

            all_models_list = [
                ("RBM", rbm, hi_free, nk.sampler.MetropolisLocal(hi_free)),
                ("ViT", vit, hi_free, nk.sampler.MetropolisLocal(hi_free)),
                ("HFDS", hfds, hi_constrained, nk.sampler.MetropolisExchange(hi_constrained, graph=g)),
                ("HFDS Random", hfds_rand, hi_constrained, nk.sampler.MetropolisExchange(hi_constrained, graph=g))
            ]

            models_list = [m for m in all_models_list if models_to_plot is None or m[0] in models_to_plot]

            for name, model, hi, sampler in models_list:
                s2_matrix = np.zeros((n_seeds, len(sizes)))
                param_count = 0

                for seed in range(n_seeds):
                    vstate = nk.vqs.MCState(sampler, model, n_samples=n_samples, seed=seed)
                    if seed == 0:
                        param_count = nk.jax.tree_size(vstate.parameters)

                    for idx, size in enumerate(sizes):
                        partition = partitions_map[size]
                        s2, _ = compute_renyi2_entropy(vstate, partition_indices=partition, n_samples=n_samples)
                        s2_matrix[seed, idx] = s2

                mean = np.mean(s2_matrix, axis=0)
                err = np.std(s2_matrix, axis=0) / np.sqrt(n_seeds)

                if name not in results: results[name] = {}
                if size_label not in results[name]: results[name][size_label] = {'size': [], 'mean': [], 'err': [], 'params': []}

                results[name][size_label]['size'] = list(sizes)
                results[name][size_label]['mean'] = list(mean)
                results[name][size_label]['err'] = list(err)
                results[name][size_label]['params'] = [param_count] * len(sizes)
                print(f"    {name} ({size_label}): S2 mean over sizes={np.mean(mean):.4f}")

            # ViT Xavier
            if models_to_plot is None or "ViT" in models_to_plot:
                d_model = params['ViT']
                init_xavier = jax.nn.initializers.xavier_uniform()

                s2_matrix_x = np.zeros((n_seeds, len(sizes)))
                param_count_x = 0

                for seed in range(n_seeds):
                    vit_xavier = ViT_ent(num_layers=2, d_model=d_model, n_heads=4, patch_size=2, kernel_init=init_xavier)
                    sampler = nk.sampler.MetropolisLocal(hi_free)
                    vstate = nk.vqs.MCState(sampler, vit_xavier, n_samples=n_samples, seed=seed)
                    if param_count_x == 0:
                        param_count_x = nk.jax.tree_size(vstate.parameters)

                    for idx, size in enumerate(sizes):
                        partition = partitions_map[size]
                        s2, _ = compute_renyi2_entropy(vstate, partition_indices=partition, n_samples=n_samples)
                        s2_matrix_x[seed, idx] = s2

                mean = np.mean(s2_matrix_x, axis=0)
                err = np.std(s2_matrix_x, axis=0) / np.sqrt(n_seeds)

                name_x = "ViT Xavier"
                if name_x not in results: results[name_x] = {}
                if size_label not in results[name_x]: results[name_x][size_label] = {'size': [], 'mean': [], 'err': [], 'params': []}

                results[name_x][size_label]['size'] = list(sizes)
                results[name_x][size_label]['mean'] = list(mean)
                results[name_x][size_label]['err'] = list(err)
                results[name_x][size_label]['params'] = [param_count_x] * len(sizes)
                print(f"    {name_x} ({size_label}): S2 mean over sizes={np.mean(mean):.4f}")

        colors = {'RBM': 'blue', 'ViT': 'orange', 'HFDS': 'green', 'ViT Xavier': 'red', 'HFDS Random': 'purple'}
        markers = {'Small': 'o', 'Medium': 's', 'Large': '^'}
        linestyles = {'Small': ':', 'Medium': '--', 'Large': '-'}

        plt.figure(figsize=(10, 6))
        for name in results:
            for size_label in ['Small', 'Medium', 'Large']:
                if size_label in results[name]:
                    data = results[name][size_label]
                    params_str = ",".join(map(str, data['params']))
                    label = f"{name} {size_label} (P={params_str})"

                    size_arr = np.array(data['size'])
                    if p_type == "Strip":
                        partition_size = size_arr * L
                    else:
                        partition_size = size_arr ** 2

                    max_ent = partition_size * np.log(2)

                    plt.errorbar(size_arr, np.array(data['mean']) * max_ent, yerr=np.array(data['err']) * max_ent,
                                 label=label, color=colors.get(name, 'black'),
                                 marker=markers[size_label], linestyle=linestyles[size_label], capsize=5)

        plt.xlabel(f'Partition Size ({p_type})')
        plt.ylabel('Renyi-2 Entropy')
        plt.title(f'Entanglement Entropy vs Partition Size ({p_type}, L={L}, Unnormalized)')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        filename = f"Entropy_vs_Partition_HiddenSize_{p_type}_L{L}.png"
        save_path = get_unique_path(save_dir, filename)
        plt.savefig(save_path)
        print(f"Plot ({p_type}) saved to {save_path}")
        plt.close()


def plot_entropy_vs_L_hidden_size_map(n_seeds=10, n_samples=65536, models_to_plot=None, var=10):
    print("\n--- Plotting Entropy vs L & Hidden Size Map ---")
    save_dir = "/cluster/home/fconoscenti/Thesis_QSL/Entanglement/plots"
    if not os.path.exists(save_dir):
        save_dir = "/scratch/f/F.Conoscenti/Thesis_QSL/Entanglement/plots"
    os.makedirs(save_dir, exist_ok=True)
    
    L_values = [4, 6, 8, 10]
    
    models_defs = {
        "RBM": {
            "param_name": "alpha",
            "h_values": [1, 2, 4, 6, 8, 10],
            "builder": lambda L, h, init: nk.models.RBM(alpha=h, param_dtype=complex, kernel_init=init, hidden_bias_init=init, visible_bias_init=init),
            "hilbert_fn": lambda L: nk.hilbert.Spin(s=1/2, N=L*L),
            "sampler_fn": lambda hi, L: nk.sampler.MetropolisLocal(hi)
        },
        "ViTrandom": {
            "param_name": "d_model",
            "h_values": [4, 8, 16, 32, 64],
            "builder": lambda L, h, init: ViT_ent(num_layers=2, d_model=h, n_heads=4, patch_size=2, kernel_init=init),
            "hilbert_fn": lambda L: nk.hilbert.Spin(s=1/2, N=L*L),
            "sampler_fn": lambda hi, L: nk.sampler.MetropolisLocal(hi)
        },
        "ViTXavier": {
            "param_name": "d_model",
            "h_values":  [4, 8, 16, 32, 64],
            "builder": lambda L, h, _: ViT_ent(num_layers=2, d_model=h, n_heads=4, patch_size=2, kernel_init=jax.nn.initializers.xavier_uniform()),
            "hilbert_fn": lambda L: nk.hilbert.Spin(s=1/2, N=L*L),
            "sampler_fn": lambda hi, L: nk.sampler.MetropolisLocal(hi)
        },
        "HFDSrandom": {
            "param_name": "n_hid",
            "h_values": [2,4,8,12],
            "builder": lambda L, h, init: HiddenFermion_ent(L=L, network="FFNN", n_hid=h, layers=1, features=64, MFinit="random", hilbert=nk.hilbert.Spin(s=1/2, N=L*L, total_sz=0), kernel_init=init, dtype=jax.numpy.complex128),
            "hilbert_fn": lambda L: nk.hilbert.Spin(s=1/2, N=L*L, total_sz=0),
            "sampler_fn": lambda hi, L: nk.sampler.MetropolisExchange(hi, graph=nk.graph.Hypercube(length=L, n_dim=2, pbc=True))
        },
        "HFDSFermi": {
            "param_name": "n_hid",
            "h_values": [2,4,8,12],
            "builder": lambda L, h, init: HiddenFermion_ent(L=L, network="FFNN", n_hid=h, layers=1, features=64, MFinit="Fermi", hilbert=nk.hilbert.Spin(s=1/2, N=L*L, total_sz=0), kernel_init=init, dtype=jax.numpy.complex128),
            "hilbert_fn": lambda L: nk.hilbert.Spin(s=1/2, N=L*L, total_sz=0),
            "sampler_fn": lambda hi, L: nk.sampler.MetropolisExchange(hi, graph=nk.graph.Hypercube(length=L, n_dim=2, pbc=True))
        }
    }

    results = []
    all_grids = []

    for model_name, config in models_defs.items():
        if models_to_plot is not None and model_name not in models_to_plot:
            continue
        print(f"Running sweep for {model_name}...")
        h_vals = config["h_values"]
        entropy_grid = np.zeros((len(h_vals), len(L_values)))
        
        for i, h in enumerate(h_vals):
            for j, L in enumerate(L_values):
                s2_acc = 0.0
                hi = config["hilbert_fn"](L)
                sampler = config["sampler_fn"](hi, L)
                
                for seed in range(n_seeds):
                    init = normal(stddev=np.sqrt(var))
                    model = config["builder"](L, h, init)
                    vstate = nk.vqs.MCState(sampler, model, n_samples=n_samples, seed=seed)
                    s2, _ = compute_renyi2_entropy(vstate, n_samples=n_samples)
                    s2_acc += s2
                entropy_grid[i, j] = s2_acc / n_seeds
                print(f"  {model_name} h={h} L={L} -> S2={entropy_grid[i, j]:.4f}")
                clean_up()
        
        # Un-normalize
        for j, L in enumerate(L_values):
            max_ent = (L*L // 2) * np.log(2)
            entropy_grid[:, j] *= max_ent
        
        extent = [0, len(L_values), 0, len(h_vals)]
        
        results.append({
            'name': model_name,
            'grid': entropy_grid,
            'extent': extent,
            'ylabel': f'Hidden Size ({config["param_name"]})',
            'h_vals': h_vals,
            'L_values': L_values
        })
        all_grids.append(entropy_grid)
        
    if results:
        vmin = min(np.min(g) for g in all_grids)
        vmax = max(np.max(g) for g in all_grids)
        
        n_models = len(results)
        fig, axes = plt.subplots(1, n_models, figsize=(4 * n_models, 5), constrained_layout=True)
        if n_models == 1: axes = [axes]
        
        for ax, res in zip(axes, results):
            im = ax.imshow(res['grid'], cmap='viridis', origin='lower', extent=res['extent'], 
                           aspect='auto', interpolation='none', vmin=vmin, vmax=vmax)
            ax.set_xlabel('System Size N')
            ax.set_ylabel(res['ylabel'])
            ax.set_title(res['name'])
            
            ax.set_xticks(np.arange(len(res['L_values'])) + 0.5)
            ax.set_xticklabels([str(L*L) for L in res['L_values']])
            
            ax.set_yticks(np.arange(len(res['h_vals'])) + 0.5)
            ax.set_yticklabels([str(h) for h in res['h_vals']])
            
        fig.colorbar(im, ax=axes, label='Renyi-2 Entropy', location='right')
        fig.suptitle(f'Entanglement Entropy vs L & Hidden Size (var={var}, samples={n_samples}, seeds={n_seeds})')
        
        filename = 'Entanglement_Sweep_L_All.png'
        save_path = get_unique_path(save_dir, filename)
        plt.savefig(save_path)
        print(f"Combined plot saved to {save_path}")

        # Save individual plots
        for res in results:
            fig_s, ax_s = plt.subplots(figsize=(6, 5), constrained_layout=True)
            im_s = ax_s.imshow(res['grid'], cmap='viridis', origin='lower', extent=res['extent'], 
                           aspect='auto', interpolation='none')
            ax_s.set_xlabel('System Size N')
            ax_s.set_ylabel(res['ylabel'])
            ax_s.set_title(res['name'])
            
            ax_s.set_xticks(np.arange(len(res['L_values'])) + 0.5)
            ax_s.set_xticklabels([str(L*L) for L in res['L_values']])
            
            ax_s.set_yticks(np.arange(len(res['h_vals'])) + 0.5)
            ax_s.set_yticklabels([str(h) for h in res['h_vals']])
            
            fig_s.colorbar(im_s, ax=ax_s, label='Renyi-2 Entropy')
            
            fname_s = f'Entanglement_Sweep_L_{res["name"]}.png'
            save_path_s = get_unique_path(save_dir, fname_s)
            plt.savefig(save_path_s)
            print(f"Individual plot saved to {save_path_s}")
            plt.close(fig_s)

def main():

    plot_entropy_vs_partition_variance(L=6, n_seeds=1, n_samples=16, models_to_plot=["ViT", "HFDS"])
    plot_entropy_vs_partition_hidden_size(L=6, n_seeds=1, n_samples=16, models_to_plot=[ "ViT", "HFDS", "HFDS Random"])
    
    plot_entropy_vs_L_hidden_size_map(L=6, n_seeds=1, n_samples=16, models_to_plot=[ "ViTrandom", "ViTXavier", "HFDSrandom", "HFDSFermi"], var=0.01)

if __name__ == "__main__":
    main()