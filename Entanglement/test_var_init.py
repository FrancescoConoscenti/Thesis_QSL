import netket as nk
import numpy as np
import jax
from jax.nn.initializers import normal
import sys
import matplotlib.pyplot as plt
sys.path.append("/scratch/f/F.Conoscenti/Thesis_QSL")
from ViT_Heisenberg.ViT_model_ent import ViT_ent
from HFDS_Heisenberg.entanglement_model.HFDS_model_spin_ent import HiddenFermion_ent
from Entanglement.Entanglement import compute_renyi2_entropy

# --- Common Setup ---
target_variance = 0.01
std_dev = np.sqrt(target_variance)

# Setup Hilbert spaces
# For RBM tests (1D)
g_rbm = nk.graph.Hypercube(length=10, n_dim=1, pbc=True)
hi_rbm = nk.hilbert.Spin(s=1/2, N=g_rbm.n_nodes)

# For ViT/HFDS tests (2D 4x4)
g_vit = nk.graph.Hypercube(length=4, n_dim=2, pbc=True)
hi_vit = nk.hilbert.Spin(s=1/2, N=g_vit.n_nodes)
hi_hfds = nk.hilbert.Spin(s=1/2, N=g_vit.n_nodes, total_sz=0)
sampler_hfds = nk.sampler.MetropolisExchange(hi_hfds, graph=g_vit)

# --- Helper Functions ---

def check_params_zero(params, path=""):
    all_zero = True
    if hasattr(params, 'items'):
        for k, v in params.items():
            if not check_params_zero(v, f"{path}/{k}"):
                all_zero = False
    else:
        # Leaf node
        # LayerNorm scale is initialized to 1.0, so we expect 1.0 there.
        if "scale" in path:
            if not np.allclose(params, 1.0):
                print(f"FAILURE: {path} is {np.max(np.abs(params))} (expected 1.0)")
                all_zero = False
        # Bias and Kernel should be 0.0
        elif "bias" in path or "kernel" in path or "alpha" in path or "V" in path or "W" in path:
            if not np.allclose(params, 0.0):
                print(f"FAILURE: {path} is {np.max(np.abs(params))} (expected 0.0)")
                all_zero = False
    return all_zero

def check_params_hfds(params, path=""):
    all_zero = True
    if hasattr(params, 'items'):
        for k, v in params.items():
            if not check_params_hfds(v, f"{path}/{k}"):
                all_zero = False
    else:
        # Leaf node
        # 'orbitals_mf' is the Fermi sea, not zero.
        if "orbitals_mf" in path:
            if np.allclose(params, 0.0):
                print(f"FAILURE: {path} is all zero! Fermi sea should be non-zero.")
                all_zero = False
        # 'orbitals_hf' and Neural Network weights should be zero
        elif "orbitals_hf" in path or "kernel" in path or "bias" in path:
            if not np.allclose(params, 0.0):
                print(f"FAILURE: {path} is {np.max(np.abs(params))} (expected 0.0)")
                all_zero = False
    return all_zero

def collect_vit_weights(params, container):
    if hasattr(params, 'items'):
        for k, v in params.items():
            if k in ['kernel', 'alpha', 'V']:
                container.append(np.array(v).flatten())
            else:
                collect_vit_weights(v, container)

def collect_hfds_weights(params, container):
    if hasattr(params, 'items'):
        for k, v in params.items():
            if k in ['kernel', 'orbitals_hf']:
                container.append(np.array(v).flatten())
            else:
                collect_hfds_weights(v, container)

def run_entropy_test(model_name, model_builder, hilbert, sampler_ctor, variances, n_samples):
    print(f"\nTesting Entropy for {model_name}")
    for var in variances:
        if var == 0:
            init = jax.nn.initializers.zeros
        else:
            init = normal(stddev=np.sqrt(var))
        
        model = model_builder(init)
        sampler = sampler_ctor(hilbert)
        
        # Use a consistent seed
        vstate = nk.vqs.MCState(sampler, model, n_samples=2048, seed=42)
        
        # Compute entropy
        s2, err = compute_renyi2_entropy(vstate, n_samples=n_samples)
        print(f"  Var={var:.1e}: S2 (norm) = {s2:.5f} ± {err:.5f}")

# --- Test Functions ---

def test_rbm_finite_variance():
    print("--- Test 1: Finite Variance (0.01) ---")
    # Define RBM with finite variance
    rbm = nk.models.RBM(
        alpha=1,
        param_dtype=complex,
        kernel_init=normal(stddev=std_dev),
        hidden_bias_init=normal(stddev=std_dev),
        visible_bias_init=normal(stddev=std_dev)
    )

    # Initialize
    vstate = nk.vqs.MCState(nk.sampler.MetropolisLocal(hi_rbm), model=rbm, n_samples=10)
    params = vstate.parameters

    # Extract weights
    weights = params['Dense']['kernel']
    empirical_variance = np.var(weights)

    print(f"Target Variance:    {target_variance}")
    print(f"Empirical Variance: {empirical_variance:.5f}")

def test_rbm_zero_variance():
    print("\n--- Test 2: Zero Variance ---")
    # Define RBM with zero variance (stddev=0)
    rbm_zero = nk.models.RBM(
        alpha=1,
        param_dtype=complex,
        kernel_init=normal(stddev=0.0),       # Variance = 0
        hidden_bias_init=normal(stddev=0.0),  # Variance = 0
        visible_bias_init=normal(stddev=0.0)  # Variance = 0
    )

    # Initialize
    vstate_zero = nk.vqs.MCState(nk.sampler.MetropolisLocal(hi_rbm), model=rbm_zero, n_samples=10)
    params_zero = vstate_zero.parameters

    # Extract parameters
    weights_zero = params_zero['Dense']['kernel']
    hidden_bias_zero = params_zero['Dense']['bias']
    visible_bias_zero = params_zero['visible_bias']

    # Verification Logic
    is_w_zero = np.all(weights_zero == 0)
    is_hb_zero = np.all(hidden_bias_zero == 0)
    is_vb_zero = np.all(visible_bias_zero == 0)

    print(f"Weights are all zero?       {is_w_zero}")
    print(f"Hidden biases are all zero? {is_hb_zero}")
    print(f"Visible biases are all zero?{is_vb_zero}")

    if is_w_zero and is_hb_zero and is_vb_zero:
        print(">> SUCCESS: Zero variance initialization produced exact zeros.")
    else:
        print(">> FAILURE: Some parameters are non-zero.")

def test_vit_zero_variance():
    print("\n--- Test 3: ViT Zero Variance ---")
    # Define ViT with zero variance
    vit_zero = ViT_ent(
        num_layers=2,
        d_model=8,
        n_heads=4,
        patch_size=2,
        kernel_init=jax.nn.initializers.zeros
    )

    # Initialize
    vstate_vit = nk.vqs.MCState(nk.sampler.MetropolisLocal(hi_vit), model=vit_zero, n_samples=10)
    params_vit = vstate_vit.parameters

    print("Checking ViT parameters...")
    is_vit_zero = check_params_zero(params_vit)

    # Explicit checks for main components
    print("Explicit checks:")
    print(f"  Embed weights zero? {np.allclose(params_vit['Embed_0']['embed']['kernel'], 0.0)}")
    print(f"  Encoder layer 0 weights zero? {np.allclose(params_vit['Encoder_0']['layers_0']['attn']['v']['kernel'], 0.0)}")
    print(f"  OutputHead weights zero? {np.allclose(params_vit['OuputHead_0']['output_layer0']['kernel'], 0.0)}")
    print(f"  OutputHead bias zero? {np.allclose(params_vit['OuputHead_0']['output_layer0']['bias'], 0.0)}")

    # Check output
    samples = vstate_vit.sample(n_samples=10)
    samples = samples.reshape(-1, hi_vit.size)
    log_psi = vstate_vit.log_value(samples)
    print(f"Max |Log Psi|: {np.max(np.abs(log_psi))}")
    print(f"Is output zero? {np.allclose(log_psi, 0.0, atol=1e-7)}")

def test_hfds_zero_variance():
    print("\n--- Test 4: HFDS Zero Variance ---")
    
    hfds_zero = HiddenFermion_ent(
        L=4,
        network="FFNN",
        n_hid=2,
        layers=1,
        features=8,
        MFinit="Fermi",
        hilbert=hi_hfds,
        kernel_init=jax.nn.initializers.zeros,
        dtype=jax.numpy.complex128
    )

    # Initialize
    vstate_hfds = nk.vqs.MCState(sampler_hfds, model=hfds_zero, n_samples=10)
    params_hfds = vstate_hfds.parameters

    print("Checking HFDS parameters...")
    is_hfds_zero = check_params_hfds(params_hfds)

    if is_hfds_zero:
        print(">> SUCCESS: HFDS zero variance initialization produced zeros in expected parameters.")
    else:
        print(">> FAILURE: Some HFDS parameters are non-zero.")

    # Explicit checks for main components
    print("Explicit checks:")
    print(f"  Orbitals HF zero? {np.allclose(params_hfds['orbitals']['orbitals_hf'], 0.0)}")
    print(f"  Hidden layer 0 weights zero? {np.allclose(params_hfds['hidden_0']['kernel'], 0.0)}")
    print(f"  Output layer weights zero? {np.allclose(params_hfds['output']['kernel'], 0.0)}")
    print(f"  Output layer bias zero? {np.allclose(params_hfds['output']['bias'], 0.0)}")

    # Check output
    samples = vstate_hfds.sample(n_samples=10)
    samples = samples.reshape(-1, hi_hfds.size)
    log_psi = vstate_hfds.log_value(samples)
    print(f"Max |Log Psi|: {np.max(np.abs(log_psi))}")
    # HFDS initialized with Fermi sea is not zero (it's the free fermion state).
    # We check for finiteness to ensure the determinant is well-defined (non-singular).
    print(f"Is output finite? {np.all(np.isfinite(log_psi))}")

def test_vit_finite_variance():
    print("\n--- Test 5: ViT Finite Variance (0.01) ---")
    vit_finite = ViT_ent(
        num_layers=2,
        d_model=16,
        n_heads=4,
        patch_size=2,
        kernel_init=normal(stddev=std_dev)
    )

    vstate_vit_finite = nk.vqs.MCState(nk.sampler.MetropolisLocal(hi_vit), model=vit_finite, n_samples=10)
    params_vit_finite = vstate_vit_finite.parameters

    vit_weights = []
    collect_vit_weights(params_vit_finite, vit_weights)
    
    if len(vit_weights) > 0:
        all_vit_weights = np.concatenate(vit_weights)
        vit_var = np.var(all_vit_weights)
        print(f"Target Variance:    {target_variance}")
        print(f"Empirical Variance: {vit_var:.5f}")
    else:
        print("FAILURE: No weights found for ViT.")

def test_hfds_finite_variance():
    print("\n--- Test 6: HFDS Finite Variance (0.01) ---")
    hfds_finite = HiddenFermion_ent(
        L=4,
        network="FFNN",
        n_hid=4,
        layers=2,
        features=16,
        MFinit="Fermi",
        hilbert=hi_hfds,
        kernel_init=normal(stddev=std_dev),
        dtype=jax.numpy.complex128
    )

    vstate_hfds_finite = nk.vqs.MCState(sampler_hfds, model=hfds_finite, n_samples=10)
    params_hfds_finite = vstate_hfds_finite.parameters

    hfds_weights = []
    collect_hfds_weights(params_hfds_finite, hfds_weights)
    
    if len(hfds_weights) > 0:
        all_hfds_weights = np.concatenate(hfds_weights)
        hfds_var = np.var(all_hfds_weights)
        print(f"Target Variance:    {target_variance}")
        print(f"Empirical Variance: {hfds_var:.5f}")
    else:
        print("FAILURE: No weights found for HFDS.")

def test_entanglement_entropy_rbm(n_samples):
    print("\n--- Test 7a: RBM Entanglement Entropy Checks ---")
    variances_to_test = [0.0, 1e-4, 0.01]
    def build_rbm(init):
        return nk.models.RBM(alpha=1, param_dtype=complex, kernel_init=init, hidden_bias_init=init, visible_bias_init=init)
    run_entropy_test("RBM", build_rbm, hi_vit, nk.sampler.MetropolisLocal, variances_to_test, n_samples)

def test_entanglement_entropy_vit(n_samples):
    print("\n--- Test 7b: ViT Entanglement Entropy Checks ---")
    variances_to_test = [0.0, 1e-4, 0.01]
    def build_vit(init):
        return ViT_ent(num_layers=2, d_model=8, n_heads=4, patch_size=2, kernel_init=init)
    run_entropy_test("ViT", build_vit, hi_vit, nk.sampler.MetropolisLocal, variances_to_test, n_samples)

def test_entanglement_entropy_hfds(n_samples):
    print("\n--- Test 7c: HFDS Entanglement Entropy Checks ---")
    variances_to_test = [0.0, 1e-4, 0.01]
    def build_hfds(init):
        return HiddenFermion_ent(L=4, network="FFNN", n_hid=2, layers=1, features=8, MFinit="Fermi", hilbert=hi_hfds, kernel_init=init, dtype=jax.numpy.complex128)
    def hfds_sampler_ctor(hi):
        return nk.sampler.MetropolisExchange(hi, graph=g_vit)
    run_entropy_test("HFDS", build_hfds, hi_hfds, hfds_sampler_ctor, variances_to_test, n_samples)

def test_entanglement_entropy_vit_xavier(n_samples):
    print("\n--- Test 7d: ViT Entanglement Entropy Checks (Xavier Init) ---")
    
    init = jax.nn.initializers.xavier_uniform()
    
    model = ViT_ent(
        num_layers=2,
        d_model=8,
        n_heads=4,
        patch_size=2,
        kernel_init=init
    )
    
    sampler = nk.sampler.MetropolisLocal(hi_vit)
    
    # Use a consistent seed
    vstate = nk.vqs.MCState(sampler, model, n_samples=2048, seed=42)
    
    # Compute entropy
    s2, err = compute_renyi2_entropy(vstate, n_samples=n_samples)
    print(f"  Xavier Uniform: S2 (norm) = {s2:.5f} ± {err:.5f}")

def test_hfds_random_init():
    print("\n--- Test 8: HFDS Random Initialization ---")

    # 1. Random Init with Zero Variance (Expect Singular/Zero params)
    print("Checking Random Init with Zero Variance...")
    hfds_random_zero = HiddenFermion_ent(
        L=4,
        network="FFNN",
        n_hid=2,
        layers=1,
        features=8,
        MFinit="random",
        hilbert=hi_hfds,
        kernel_init=jax.nn.initializers.zeros,
        dtype=jax.numpy.complex128
    )
    vstate_hfds_rand_zero = nk.vqs.MCState(sampler_hfds, model=hfds_random_zero, n_samples=10)
    params_rand_zero = vstate_hfds_rand_zero.parameters

    # Check if orbitals_mf are zero
    orb_mf_zero = params_rand_zero['orbitals']['orbitals_mf']
    print(f"  Orbitals MF zero? {np.allclose(orb_mf_zero, 0.0)}")

    # Check output (Expect finite)
    samples = vstate_hfds_rand_zero.sample(n_samples=10)
    samples = samples.reshape(-1, hi_hfds.size)
    log_psi = vstate_hfds_rand_zero.log_value(samples)
    print(f"  Max |Log Psi|: {np.max(np.abs(log_psi))}")
    print(f"  Is output zero? {np.allclose(log_psi, 0.0)}")

    # 2. Random Init with Finite Variance
    print("Checking Random Init with Finite Variance...")
    hfds_random_finite = HiddenFermion_ent(
        L=4,
        network="FFNN",
        n_hid=2,
        layers=1,
        features=8,
        MFinit="random",
        hilbert=hi_hfds,
        kernel_init=normal(stddev=0.1),
        dtype=jax.numpy.complex128
    )
    vstate_hfds_rand_finite = nk.vqs.MCState(sampler_hfds, model=hfds_random_finite, n_samples=10)
    params_rand_finite = vstate_hfds_rand_finite.parameters

    orb_mf_finite = params_rand_finite['orbitals']['orbitals_mf']
    print(f"  Orbitals MF variance: {np.var(orb_mf_finite):.5f}")

    # Check output (Expect finite)
    samples = vstate_hfds_rand_finite.sample(n_samples=10)
    samples = samples.reshape(-1, hi_hfds.size)
    log_psi = vstate_hfds_rand_finite.log_value(samples)
    print(f"  Max |Log Psi|: {np.max(np.abs(log_psi))}")
    print(f"  Is output finite? {np.all(np.isfinite(log_psi))}")

def plot_entropy_vs_variance(n_seeds=10, n_samples=4096):
    print("\n--- Plotting Entropy vs Variance ---")
    variances = [0.0, 1e-7, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 2.0, 5.0]
    
    results = {}
    
    # Define configurations: (Name, ModelBuilder, Hilbert, SamplerBuilder)
    # Using N=16 (hi_vit/hi_hfds) for all models for consistency.
    models_config = [
        ("RBM", 
         lambda init: nk.models.RBM(alpha=2, param_dtype=complex, kernel_init=init, hidden_bias_init=init, visible_bias_init=init), 
         hi_vit, 
         nk.sampler.MetropolisLocal),
        ("ViT", 
         lambda init: ViT_ent(num_layers=2, d_model=8, n_heads=4, patch_size=2, kernel_init=init), 
         hi_vit, 
         nk.sampler.MetropolisLocal),
        ("HFDS", 
         lambda init: HiddenFermion_ent(L=4, network="FFNN", n_hid=2, layers=1, features=8, MFinit="Fermi", hilbert=hi_hfds, kernel_init=init, dtype=jax.numpy.complex128), 
         hi_hfds, 
         lambda h: nk.sampler.MetropolisExchange(h, graph=g_vit))
    ]

    for name, model_builder, hilbert, sampler_builder in models_config:
        print(f"Processing {name}...")
        means = []
        errors = []
        
        for var in variances:
            s2_vals = []
            for seed in range(n_seeds):
                if var == 0:
                    init = jax.nn.initializers.zeros
                else:
                    init = normal(stddev=np.sqrt(var))
                
                model = model_builder(init)
                sampler = sampler_builder(hilbert)
                
                vstate = nk.vqs.MCState(sampler, model, n_samples=n_samples, seed=seed)
                
                s2, _ = compute_renyi2_entropy(vstate, n_samples=n_samples)
                s2_vals.append(s2)
            
            mean_s2 = np.mean(s2_vals)
            err_s2 = np.std(s2_vals) / np.sqrt(n_seeds)
            means.append(mean_s2)
            errors.append(err_s2)
            print(f"  Var={var:.1e}: S2={mean_s2:.4f} +/- {err_s2:.4f}")
        
        results[name] = (means, errors)

    # Calculate Xavier for ViT
    print("Processing ViT Xavier...")
    xavier_s2_vals = []
    init_xavier = jax.nn.initializers.xavier_uniform()
    for seed in range(n_seeds):
        model = ViT_ent(num_layers=2, d_model=8, n_heads=4, patch_size=2, kernel_init=init_xavier)
        sampler = nk.sampler.MetropolisLocal(hi_vit)
        vstate = nk.vqs.MCState(sampler, model, n_samples=n_samples, seed=seed)
        s2, _ = compute_renyi2_entropy(vstate, n_samples=n_samples)
        xavier_s2_vals.append(s2)
    mean_xavier = np.mean(xavier_s2_vals)
    print(f"  ViT Xavier: S2={mean_xavier:.4f}")

    # Plotting Normalized
    plt.figure(figsize=(10, 6))
    for name, (means, errors) in results.items():
        plt.errorbar(variances, means, yerr=errors, label=name, marker='o', capsize=5)
    
    plt.axhline(y=mean_xavier, color='r', linestyle='--', label=f'ViT Xavier ({mean_xavier:.3f})')
    
    plt.xscale('symlog', linthresh=1e-5)
    plt.xlabel('Variance of Initialization')
    plt.ylabel('Renyi-2 Entropy (Normalized)')
    plt.title(f'Entanglement Entropy vs Initialization Variance (N=16, Normalized, Avg {n_seeds} seeds)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    save_path = "Entropy_vs_Variance_Init_Normalized.png"
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")

    # Plotting Unnormalized
    max_ent = 8 * np.log(2) # N=16, half system partition
    
    plt.figure(figsize=(10, 6))
    for name, (means, errors) in results.items():
        means_un = np.array(means) * max_ent
        errors_un = np.array(errors) * max_ent
        plt.errorbar(variances, means_un, yerr=errors_un, label=name, marker='o', capsize=5)
    
    mean_xavier_un = mean_xavier * max_ent
    plt.axhline(y=mean_xavier_un, color='r', linestyle='--', label=f'ViT Xavier ({mean_xavier_un:.3f})')
    
    plt.xscale('symlog', linthresh=1e-5)
    plt.xlabel('Variance of Initialization')
    plt.ylabel('Renyi-2 Entropy')
    plt.title(f'Entanglement Entropy vs Initialization Variance (N=16, Unnormalized, Avg {n_seeds} seeds)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    save_path = "Entropy_vs_Variance_Init_Unnormalized.png"
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")

def plot_entropy_vs_L(n_seeds=10, n_samples=4096):
    print("\n--- Plotting Entropy vs L ---")
    variances = [1e-5, 1e-3, 1e-1]
    L_values = [4, 6, 8]
    
    results = {} # results[name][var] = {'L': [], 'mean': [], 'err': []}
    xavier_results = {'L': [], 'mean': [], 'err': []}

    for L in L_values:
        print(f"Processing L={L} (N={L*L})...")
        N = L*L
        g = nk.graph.Hypercube(length=L, n_dim=2, pbc=True)
        
        hi_free = nk.hilbert.Spin(s=1/2, N=N)
        hi_constrained = nk.hilbert.Spin(s=1/2, N=N, total_sz=0)
        
        for var in variances:
            std = np.sqrt(var)
            init_fun = normal(stddev=std)
            
            # Models
            rbm = nk.models.RBM(alpha=1, param_dtype=complex, kernel_init=init_fun, hidden_bias_init=init_fun, visible_bias_init=init_fun)
            vit = ViT_ent(num_layers=2, d_model=8, n_heads=4, patch_size=2, kernel_init=init_fun)
            hfds = HiddenFermion_ent(L=L, network="FFNN", n_hid=2, layers=1, features=8, MFinit="Fermi", hilbert=hi_constrained, kernel_init=init_fun, dtype=jax.numpy.complex128)
            
            models_list = [
                ("RBM", rbm, hi_free, nk.sampler.MetropolisLocal(hi_free)),
                ("ViT", vit, hi_free, nk.sampler.MetropolisLocal(hi_free)),
                ("HFDS", hfds, hi_constrained, nk.sampler.MetropolisExchange(hi_constrained, graph=g))
            ]
            
            for name, model, hi, sampler in models_list:
                s2_vals = []
                for seed in range(n_seeds):
                    vstate = nk.vqs.MCState(sampler, model, n_samples=n_samples, seed=seed)
                    s2, _ = compute_renyi2_entropy(vstate, n_samples=n_samples)
                    s2_vals.append(s2)
                
                mean = np.mean(s2_vals)
                err = np.std(s2_vals) / np.sqrt(n_seeds)
                
                if name not in results: results[name] = {}
                if var not in results[name]: results[name][var] = {'L': [], 'mean': [], 'err': []}
                
                results[name][var]['L'].append(L)
                results[name][var]['mean'].append(mean)
                results[name][var]['err'].append(err)
                print(f"  {name} Var={var} L={L}: S2={mean:.4f}")
        
        # ViT Xavier
        print(f"  ViT Xavier L={L}...")
        init_xavier = jax.nn.initializers.xavier_uniform()
        vit_xavier = ViT_ent(num_layers=2, d_model=8, n_heads=4, patch_size=2, kernel_init=init_xavier)
        sampler_vit = nk.sampler.MetropolisLocal(hi_free)
        
        s2_vals_x = []
        for seed in range(n_seeds):
             vstate = nk.vqs.MCState(sampler_vit, vit_xavier, n_samples=n_samples, seed=seed)
             s2, _ = compute_renyi2_entropy(vstate, n_samples=n_samples)
             s2_vals_x.append(s2)
        
        xavier_results['L'].append(L)
        xavier_results['mean'].append(np.mean(s2_vals_x))
        xavier_results['err'].append(np.std(s2_vals_x)/np.sqrt(n_seeds))
        print(f"  ViT Xavier L={L}: S2={np.mean(s2_vals_x):.4f}")

    # Plotting Normalized
    plt.figure(figsize=(10, 6))
    colors = {'RBM': 'blue', 'ViT': 'orange', 'HFDS': 'green'}
    markers = {1e-5: 'o', 1e-3: 's', 1e-1: '^'}
    linestyles = {1e-5: '-', 1e-3: '--', 1e-1: ':'}
    
    for name in results:
        for var in variances:
            data = results[name][var]
            label = f"{name} Var={var}"
            plt.errorbar(np.array(data['L'])**2, data['mean'], yerr=data['err'], 
                         label=label, color=colors[name], 
                         marker=markers[var], linestyle=linestyles[var], capsize=5)
    
    plt.errorbar(np.array(xavier_results['L'])**2, xavier_results['mean'], yerr=xavier_results['err'],
                 label='ViT Xavier', color='red', marker='*', linestyle='-', capsize=5)
    
    plt.xlabel('Number of Spins N (L^2)')
    plt.ylabel('Renyi-2 Entropy (Normalized)')
    plt.title(f'Entanglement Entropy vs L (Normalized, Avg {n_seeds} seeds)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    save_path = "Entropy_vs_L_Scaling_Normalized.png"
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")

    # Plotting Unnormalized
    plt.figure(figsize=(10, 6))
    for name in results:
        for var in variances:
            data = results[name][var]
            label = f"{name} Var={var}"
            L_arr = np.array(data['L'])
            max_ent = (L_arr**2 / 2.0) * np.log(2)
            plt.errorbar(L_arr**2, np.array(data['mean']) * max_ent, yerr=np.array(data['err']) * max_ent, 
                         label=label, color=colors[name], 
                         marker=markers[var], linestyle=linestyles[var], capsize=5)
    
    L_x = np.array(xavier_results['L'])
    max_ent_x = (L_x**2 / 2.0) * np.log(2)
    plt.errorbar(L_x**2, np.array(xavier_results['mean']) * max_ent_x, yerr=np.array(xavier_results['err']) * max_ent_x,
                 label='ViT Xavier', color='red', marker='*', linestyle='-', capsize=5)
    
    plt.xlabel('Number of Spins N (L^2)')
    plt.ylabel('Renyi-2 Entropy')
    plt.title(f'Entanglement Entropy vs L (Unnormalized, Avg {n_seeds} seeds)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    save_path = "Entropy_vs_L_Scaling_Unnormalized.png"
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")

def plot_entropy_vs_L_hidden_size(n_seeds=10, n_samples=4096):
    print("\n--- Plotting Entropy vs L (Varying Hidden Size) ---")
    # Fixed variance for initialization
    var = 1e-3
    std = np.sqrt(var)
    init_fun = normal(stddev=std)
    
    L_values = [4, 6, 8]
    
    # Define parameter sets: (Label, {Model: Value})
    param_sets = [
        ("Small",  {'RBM': 1, 'HFDS': 1, 'ViT': 4}),
        ("Medium", {'RBM': 2, 'HFDS': 2, 'ViT': 8}),
        ("Large",  {'RBM': 4, 'HFDS': 4, 'ViT': 16})
    ]
    
    results = {} # results[model_name][size_label] = {'L': [], 'mean': [], 'err': []}
    
    for L in L_values:
        N = L*L
        print(f"Processing L={L} (N={N})...")
        
        # Hilberts
        g = nk.graph.Hypercube(length=L, n_dim=2, pbc=True)
        hi_free = nk.hilbert.Spin(s=1/2, N=N)
        hi_constrained = nk.hilbert.Spin(s=1/2, N=N, total_sz=0)
        
        for size_label, params in param_sets:
            # RBM
            alpha = params['RBM']
            rbm = nk.models.RBM(alpha=alpha, param_dtype=complex, kernel_init=init_fun, hidden_bias_init=init_fun, visible_bias_init=init_fun)
            
            # ViT
            d_model = params['ViT']
            vit = ViT_ent(num_layers=2, d_model=d_model, n_heads=4, patch_size=2, kernel_init=init_fun)
            
            # HFDS
            n_hid = params['HFDS']
            hfds = HiddenFermion_ent(L=L, network="FFNN", n_hid=n_hid, layers=1, features=8, MFinit="Fermi", hilbert=hi_constrained, kernel_init=init_fun, dtype=jax.numpy.complex128)
            
            models_list = [
                ("RBM", rbm, hi_free, nk.sampler.MetropolisLocal(hi_free)),
                ("ViT", vit, hi_free, nk.sampler.MetropolisLocal(hi_free)),
                ("HFDS", hfds, hi_constrained, nk.sampler.MetropolisExchange(hi_constrained, graph=g))
            ]
            
            for name, model, hi, sampler in models_list:
                s2_vals = []
                for seed in range(n_seeds):
                    vstate = nk.vqs.MCState(sampler, model, n_samples=n_samples, seed=seed)
                    s2, _ = compute_renyi2_entropy(vstate, n_samples=n_samples)
                    s2_vals.append(s2)
                
                mean = np.mean(s2_vals)
                err = np.std(s2_vals) / np.sqrt(n_seeds)
                
                if name not in results: results[name] = {}
                if size_label not in results[name]: results[name][size_label] = {'L': [], 'mean': [], 'err': []}
                
                results[name][size_label]['L'].append(L)
                results[name][size_label]['mean'].append(mean)
                results[name][size_label]['err'].append(err)
                print(f"  {name} ({size_label}) L={L}: S2={mean:.4f}")
            
            # ViT Xavier
            d_model = params['ViT']
            init_xavier = jax.nn.initializers.xavier_uniform()
            vit_xavier = ViT_ent(num_layers=2, d_model=d_model, n_heads=4, patch_size=2, kernel_init=init_xavier)
            sampler = nk.sampler.MetropolisLocal(hi_free)
            
            s2_vals = []
            for seed in range(n_seeds):
                vstate = nk.vqs.MCState(sampler, vit_xavier, n_samples=n_samples, seed=seed)
                s2, _ = compute_renyi2_entropy(vstate, n_samples=n_samples)
                s2_vals.append(s2)
            
            mean = np.mean(s2_vals)
            err = np.std(s2_vals) / np.sqrt(n_seeds)
            
            name_x = "ViT Xavier"
            if name_x not in results: results[name_x] = {}
            if size_label not in results[name_x]: results[name_x][size_label] = {'L': [], 'mean': [], 'err': []}
            
            results[name_x][size_label]['L'].append(L)
            results[name_x][size_label]['mean'].append(mean)
            results[name_x][size_label]['err'].append(err)
            print(f"  {name_x} ({size_label}) L={L}: S2={mean:.4f}")

    # Plotting Normalized
    plt.figure(figsize=(10, 6))
    colors = {'RBM': 'blue', 'ViT': 'orange', 'HFDS': 'green', 'ViT Xavier': 'red'}
    markers = {'Small': 'o', 'Medium': 's', 'Large': '^'}
    linestyles = {'Small': ':', 'Medium': '--', 'Large': '-'}
    
    for name in results:
        for size_label in ['Small', 'Medium', 'Large']:
            if size_label in results[name]:
                data = results[name][size_label]
                # Get param value for label
                if name == "ViT Xavier":
                    p_val = next(p['ViT'] for l, p in param_sets if l == size_label)
                else:
                    p_val = next(p[name] for l, p in param_sets if l == size_label)
                label = f"{name} {size_label} (p={p_val})"
                
                plt.errorbar(np.array(data['L'])**2, data['mean'], yerr=data['err'], 
                             label=label, color=colors[name], 
                             marker=markers[size_label], linestyle=linestyles[size_label], capsize=5)
    
    plt.xlabel('Number of Spins N (L^2)')
    plt.ylabel('Renyi-2 Entropy (Normalized)')
    plt.title(f'Entanglement Entropy vs L (Varying Hidden Size, Normalized, Avg {n_seeds} seeds)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    save_path = "Entropy_vs_L_HiddenSize_Normalized.png"
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")

    # Plotting Unnormalized
    plt.figure(figsize=(10, 6))
    for name in results:
        for size_label in ['Small', 'Medium', 'Large']:
            if size_label in results[name]:
                data = results[name][size_label]
                # Get param value for label
                if name == "ViT Xavier":
                    p_val = next(p['ViT'] for l, p in param_sets if l == size_label)
                else:
                    p_val = next(p[name] for l, p in param_sets if l == size_label)
                label = f"{name} {size_label} (p={p_val})"
                
                L_arr = np.array(data['L'])
                max_ent = (L_arr**2 / 2.0) * np.log(2)
                
                plt.errorbar(L_arr**2, np.array(data['mean']) * max_ent, yerr=np.array(data['err']) * max_ent, 
                             label=label, color=colors[name], 
                             marker=markers[size_label], linestyle=linestyles[size_label], capsize=5)
    
    plt.xlabel('Number of Spins N (L^2)')
    plt.ylabel('Renyi-2 Entropy')
    plt.title(f'Entanglement Entropy vs L (Varying Hidden Size, Unnormalized, Avg {n_seeds} seeds)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    save_path = "Entropy_vs_L_HiddenSize_Unnormalized.png"
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")

def main():

    #test_rbm_finite_variance()
    #test_rbm_zero_variance()
    #test_vit_zero_variance()
    #test_hfds_zero_variance()
    #test_vit_finite_variance()
    #test_hfds_finite_variance()
    #test_entanglement_entropy_rbm(n_samples=65536)
    #test_entanglement_entropy_vit(n_samples=65536)
    #test_entanglement_entropy_hfds(n_samples=65536)
    #test_entanglement_entropy_vit_xavier(n_samples=65536)
    #test_hfds_random_init()
    #plot_entropy_vs_variance(n_seeds=20, n_samples=8096)
    #plot_entropy_vs_L(n_seeds=20, n_samples=8096)
    plot_entropy_vs_L_hidden_size(n_seeds=20, n_samples=8096)

if __name__ == "__main__":
    main()