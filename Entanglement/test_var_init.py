import netket as nk
import numpy as np
import jax
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

# --- Common Setup ---
target_variance = 0.01
std_dev = np.sqrt(target_variance)

# Setup Hilbert spaces
# For RBM tests (1D)
g_rbm = nk.graph.Hypercube(length=16, n_dim=1, pbc=True)
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
                print(f"{path} is all zero!")
            else:
                print(f"{path} is non-zero (Expected for Fermi init).")
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
    vstate = nk.vqs.MCState(nk.sampler.MetropolisLocal(hi_rbm), model=rbm, n_samples=16)
    params = vstate.parameters

    # Extract weights
    weights = params['Dense']['kernel']
    empirical_variance = np.var(weights)

    print(f"Target Variance:    {target_variance}")
    print(f"Empirical Variance: {empirical_variance:.5f}")

def test_rbm_zero_variance():
    print("\n--- Test 2: Zero Variance RBM ---")
    # Define RBM with zero variance (stddev=0)
    rbm_zero = nk.models.RBM(
        alpha=1,
        param_dtype=complex,
        kernel_init=normal(stddev=0.0),       # Variance = 0
        hidden_bias_init=normal(stddev=0.0),  # Variance = 0
        visible_bias_init=normal(stddev=0.0)  # Variance = 0
    )

    # Initialize
    vstate_zero = nk.vqs.MCState(nk.sampler.MetropolisLocal(hi_rbm), model=rbm_zero, n_samples=16)
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

    # Check output
    samples = vstate_zero.sample(n_samples=16)
    samples = samples.reshape(-1, hi_rbm.size)
    log_psi = vstate_zero.log_value(samples)
    print(f"Max |Log Psi|: {np.max(np.abs(log_psi))}")
    print(f"Is output zero? {np.allclose(log_psi, 0.0, atol=1e-7)}")
    print("  -> Note: RBM log_psi=0 corresponds to psi=1 (Uniform Superposition).")

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
    vstate_vit = nk.vqs.MCState(nk.sampler.MetropolisLocal(hi_vit), model=vit_zero, n_samples=16)
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
    samples = vstate_vit.sample(n_samples=16)
    samples = samples.reshape(-1, hi_vit.size)
    log_psi = vstate_vit.log_value(samples)
    print(f"Max |Log Psi|: {np.max(np.abs(log_psi))}")
    print(f"Is output zero? {np.allclose(log_psi, 0.0, atol=1e-7)}")
    print("  -> Note: ViT log_psi=0 corresponds to psi=1 (Uniform Superposition).")

def test_hfds_zero_variance_random_init():
    print("\n--- Test 4: HFDS Zero Variance random init---")
    
    hfds_zero = HiddenFermion_ent(
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

    # Initialize
    vstate_hfds = nk.vqs.MCState(sampler_hfds, model=hfds_zero, n_samples=16)
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
    samples = vstate_hfds.sample(n_samples=16)
    samples = samples.reshape(-1, hi_hfds.size)
    log_psi = vstate_hfds.log_value(samples)
    print(f"Max |Log Psi|: {np.max(np.abs(log_psi))}")
    # HFDS initialized with Fermi sea is not zero (it's the free fermion state).
    # We check for finiteness to ensure the determinant is well-defined (non-singular).
    print(f"Is output finite? {np.all(np.isfinite(log_psi))} (Expected True for Fermi init)")
    if not np.all(np.isfinite(log_psi)):
        print("  -> Note: HFDS log_psi=-inf corresponds to psi=0 (Singular Determinant).")

def test_hfds_zero_variance_Fermi_init():
    print("\n--- Test 4: HFDS Zero Variance Fermi init ---")
    
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
    vstate_hfds = nk.vqs.MCState(sampler_hfds, model=hfds_zero, n_samples=16)
    params_hfds = vstate_hfds.parameters

    print("Checking HFDS parameters...")
    is_hfds_zero = check_params_hfds(params_hfds)

    if is_hfds_zero: # This check expects ALL params to be zero, which fails for Fermi
        print(">> SUCCESS: HFDS zero variance initialization produced zeros in expected parameters.")
    else:
        print(">> NOTE: Some HFDS parameters are non-zero (Expected for Fermi init).")

    # Explicit checks for main components
    print("Explicit checks:")
    print(f"  Orbitals HF zero? {np.allclose(params_hfds['orbitals']['orbitals_hf'], 0.0)}")
    print(f"  Hidden layer 0 weights zero? {np.allclose(params_hfds['hidden_0']['kernel'], 0.0)}")
    print(f"  Output layer weights zero? {np.allclose(params_hfds['output']['kernel'], 0.0)}")
    print(f"  Output layer bias zero? {np.allclose(params_hfds['output']['bias'], 0.0)}")

    # Check output
    samples = vstate_hfds.sample(n_samples=16)
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

    vstate_vit_finite = nk.vqs.MCState(nk.sampler.MetropolisLocal(hi_vit), model=vit_finite, n_samples=16)
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

    vstate_hfds_finite = nk.vqs.MCState(sampler_hfds, model=hfds_finite, n_samples=16)
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
    vstate_hfds_rand_zero = nk.vqs.MCState(sampler_hfds, model=hfds_random_zero, n_samples=16)
    params_rand_zero = vstate_hfds_rand_zero.parameters

    # Check if orbitals_mf are zero
    orb_mf_zero = params_rand_zero['orbitals']['orbitals_mf']
    print(f"  Orbitals MF zero? {np.allclose(orb_mf_zero, 0.0)}")

    # Check output (Expect finite)
    samples = vstate_hfds_rand_zero.sample(n_samples=16)
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
    vstate_hfds_rand_finite = nk.vqs.MCState(sampler_hfds, model=hfds_random_finite, n_samples=16)
    params_rand_finite = vstate_hfds_rand_finite.parameters

    orb_mf_finite = params_rand_finite['orbitals']['orbitals_mf']
    print(f"  Orbitals MF variance: {np.var(orb_mf_finite):.5f}")

    # Check output (Expect finite)
    samples = vstate_hfds_rand_finite.sample(n_samples=16)
    samples = samples.reshape(-1, hi_hfds.size)
    log_psi = vstate_hfds_rand_finite.log_value(samples)
    print(f"  Max |Log Psi|: {np.max(np.abs(log_psi))}")
    print(f"  Is output finite? {np.all(np.isfinite(log_psi))}")


def test_hfds_Fermi_init():
    print("\n--- Test 8: HFDS Random Initialization ---")

    # 1. Random Init with Zero Variance (Expect Singular/Zero params)
    print("Checking Random Init with Zero Variance...")
    hfds_random_zero = HiddenFermion_ent(
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
    vstate_hfds_rand_zero = nk.vqs.MCState(sampler_hfds, model=hfds_random_zero, n_samples=16)
    params_rand_zero = vstate_hfds_rand_zero.parameters

    # Check if orbitals_mf are zero
    orb_mf_zero = params_rand_zero['orbitals']['orbitals_mf']
    print(f"  Orbitals MF zero? {np.allclose(orb_mf_zero, 0.0)}")

    # Check output (Expect finite)
    samples = vstate_hfds_rand_zero.sample(n_samples=16)
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
        MFinit="Fermi",
        hilbert=hi_hfds,
        kernel_init=normal(stddev=0.1),
        dtype=jax.numpy.complex128
    )
    vstate_hfds_rand_finite = nk.vqs.MCState(sampler_hfds, model=hfds_random_finite, n_samples=16)
    params_rand_finite = vstate_hfds_rand_finite.parameters

    orb_mf_finite = params_rand_finite['orbitals']['orbitals_mf']
    print(f"  Orbitals MF variance: {np.var(orb_mf_finite):.5f}")

    # Check output (Expect finite)
    samples = vstate_hfds_rand_finite.sample(n_samples=16)
    samples = samples.reshape(-1, hi_hfds.size)
    log_psi = vstate_hfds_rand_finite.log_value(samples)
    print(f"  Max |Log Psi|: {np.max(np.abs(log_psi))}")
    print(f"  Is output finite? {np.all(np.isfinite(log_psi))}")



def plot_entropy_vs_variance(n_seeds=10, n_samples=4096, models_to_plot=None):
    print("\n--- Plotting Entropy vs Variance ---")
    save_dir = "/scratch/f/F.Conoscenti/Thesis_QSL/Entanglement/plots"
    os.makedirs(save_dir, exist_ok=True)
    if models_to_plot is None:
        models_to_plot = ["RBM", "ViT", "HFDS", "HFDS Random"]
    variances = [0.0, 1e-7, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 2.0, 3.0, 5.0, 10, 20]
    
    results = {}
    
    # Define configurations: (Name, ModelBuilder, Hilbert, SamplerBuilder)
    # Using N=16 (hi_vit/hi_hfds) for all models for consistency.
    all_models_config = [
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
         lambda h: nk.sampler.MetropolisExchange(h, graph=g_vit)),
        ("HFDS Random", 
         lambda init: HiddenFermion_ent(L=4, network="FFNN", n_hid=2, layers=1, features=8, MFinit="random", hilbert=hi_hfds, kernel_init=init, dtype=jax.numpy.complex128), 
         hi_hfds, 
         lambda h: nk.sampler.MetropolisExchange(h, graph=g_vit))
    ]

    models_config = [m for m in all_models_config if m[0] in models_to_plot]

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
    mean_xavier = None
    if "ViT" in models_to_plot:
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

    # Plotting Unnormalized
    max_ent = 8 * np.log(2) # N=16, half system partition
    
    plt.figure(figsize=(10, 6))
    for name, (means, errors) in results.items():
        means_un = np.array(means) * max_ent
        errors_un = np.array(errors) * max_ent
        plt.errorbar(variances, means_un, yerr=errors_un, label=name, marker='o', capsize=5)
    
    if mean_xavier is not None:
        mean_xavier_un = mean_xavier * max_ent
        plt.axhline(y=mean_xavier_un, color='r', linestyle='--', label=f'ViT Xavier ({mean_xavier_un:.3f})')
    
    plt.xscale('symlog', linthresh=1e-5)
    plt.xlim(left=0)
    plt.xlabel('Variance of Initialization')
    plt.ylabel('Renyi-2 Entropy')
    plt.title(f'Entanglement Entropy vs Initialization Variance (N=16, Unnormalized, Avg {n_seeds} seeds, samples={n_samples})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    save_path = get_unique_path(save_dir, "Entropy_vs_Variance_Init_Unnormalized1.png")
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")

def plot_entropy_vs_L(n_seeds=10, n_samples=4096, models_to_plot=None):
    print("\n--- Plotting Entropy vs L ---")
    save_dir = "/scratch/f/F.Conoscenti/Thesis_QSL/Entanglement/plots"
    os.makedirs(save_dir, exist_ok=True)
    if models_to_plot is None:
        models_to_plot = ["RBM", "ViT", "HFDS"]
    variances = [1e-2, 1e-1, 1]
    L_values = [4, 6, 8, 10]
    
    results = {} # results[name][var] = {'L': [], 'mean': [], 'err': []}
    xavier_results = {'L': [], 'mean': [], 'err': []} if "ViT" in models_to_plot else None

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
            
            all_models_list = [
                ("RBM", rbm, hi_free, nk.sampler.MetropolisLocal(hi_free)),
                ("ViT", vit, hi_free, nk.sampler.MetropolisLocal(hi_free)),
                ("HFDS", hfds, hi_constrained, nk.sampler.MetropolisExchange(hi_constrained, graph=g))
            ]
            
            models_list = [m for m in all_models_list if m[0] in models_to_plot]
            
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
        if "ViT" in models_to_plot:
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

    colors = {'RBM': 'blue', 'ViT': 'orange', 'HFDS': 'green'}
    markers = {1e-3: 'o', 1e-2: 's', 1e-1: 'D', 1: '^'}
    linestyles = {1e-3: '-', 1e-2: '--', 1e-1: '-.', 1: ':'}
    
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
    
    if xavier_results is not None:
        L_x = np.array(xavier_results['L'])
        max_ent_x = (L_x**2 / 2.0) * np.log(2)
        plt.errorbar(L_x**2, np.array(xavier_results['mean']) * max_ent_x, yerr=np.array(xavier_results['err']) * max_ent_x,
                     label='ViT Xavier', color='red', marker='*', linestyle='-', capsize=5)
    
    plt.xlabel('Number of Spins N (L^2)')
    plt.ylabel('Renyi-2 Entropy')
    plt.title(f'Entanglement Entropy vs L (Unnormalized, Avg {n_seeds} seeds, samples={n_samples})')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    save_path = get_unique_path(save_dir, "Entropy_vs_L_Scaling_Unnormalized.png")
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")

def plot_entropy_vs_L_hidden_size(n_seeds=10, n_samples=4096, models_to_plot=None):
    print("\n--- Plotting Entropy vs L (Varying Hidden Size) ---")
    save_dir = "/scratch/f/F.Conoscenti/Thesis_QSL/Entanglement/plots"
    os.makedirs(save_dir, exist_ok=True)
    if models_to_plot is None:
        models_to_plot = ["RBM", "ViT", "HFDS", "HFDS Random"]
    # Fixed variance for initialization
    var = 1e-1
    std = np.sqrt(var)
    init_fun = normal(stddev=std)
    
    L_values = [4, 6, 8, 10]
    
    configs = {
        '4x4': {
            'Small':  {'RBM': 2, 'HFDS': 2, 'ViT': 8},
            'Medium': {'RBM': 4, 'HFDS': 4, 'ViT': 16},
            'Large':  {'RBM': 8, 'HFDS': 8, 'ViT': 32}
        },
        '6x6': {
            'Small':  {'RBM': 2, 'HFDS': 1, 'ViT': 8},
            'Medium': {'RBM': 4, 'HFDS': 3, 'ViT': 16},
            'Large':  {'RBM': 8, 'HFDS': 5, 'ViT': 32}
        },
        '8x8': {
            'Small':  {'RBM': 2, 'HFDS': 1, 'ViT': 8},
            'Medium': {'RBM': 4, 'HFDS': 2, 'ViT': 16},
            'Large':  {'RBM': 8, 'HFDS': 3, 'ViT': 32}
        }
    }
    
    results = {} # results[model_name][size_label] = {'L': [], 'mean': [], 'err': []}
    
    for L in L_values:
        N = L*L
        print(f"Processing L={L} (N={N})...")
        
        # Hilberts
        g = nk.graph.Hypercube(length=L, n_dim=2, pbc=True)
        hi_free = nk.hilbert.Spin(s=1/2, N=N)
        hi_constrained = nk.hilbert.Spin(s=1/2, N=N, total_sz=0)
        
        config_key = f"{L}x{L}"
        current_config = configs.get(config_key)

        for size_label in ['Small', 'Medium', 'Large']:
            params = current_config[size_label]
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
            
            models_list = [m for m in all_models_list if m[0] in models_to_plot]
            
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
            if "ViT" in models_to_plot:
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

    colors = {'RBM': 'blue', 'ViT': 'orange', 'HFDS': 'green', 'ViT Xavier': 'red', 'HFDS Random': 'purple'}
    markers = {'Small': 'o', 'Medium': 's', 'Large': '^'}
    linestyles = {'Small': ':', 'Medium': '--', 'Large': '-'}
    
    # Plotting Unnormalized
    plt.figure(figsize=(10, 6))
    for name in results:
        for size_label in ['Small', 'Medium', 'Large']:
            if size_label in results[name]:
                data = results[name][size_label]
                label = f"{name} {size_label}"
                
                L_arr = np.array(data['L'])
                max_ent = (L_arr**2 / 2.0) * np.log(2)
                
                plt.errorbar(L_arr**2, np.array(data['mean']) * max_ent, yerr=np.array(data['err']) * max_ent, 
                             label=label, color=colors[name], 
                             marker=markers[size_label], linestyle=linestyles[size_label], capsize=5)
    
    plt.xlabel('Number of Spins N (L^2)')
    plt.ylabel('Renyi-2 Entropy')
    plt.title(f'Entanglement Entropy vs L (Varying Hidden Size, Unnormalized, Avg {n_seeds} seeds, samples={n_samples})')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    save_path = get_unique_path(save_dir, "Entropy_vs_L_HiddenSize_Unnormalized.png")
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")

def plot_entropy_vs_variance_hidden_size_map(n_seeds=10, n_samples=4096, models_to_plot=None):
    print("\n--- Plotting Entropy vs Variance & Hidden Size Map ---")
    save_dir = "/scratch/f/F.Conoscenti/Thesis_QSL/Entanglement/plots"
    os.makedirs(save_dir, exist_ok=True)
    
    # Ranges
    variances = np.linspace(0.0, 2, 20)
    
    models_defs = {
        "RBM": {
            "param_name": "alpha",
            "h_values": [1, 2, 4, 6, 8, 10],
            "builder": lambda h, init: nk.models.RBM(alpha=h, param_dtype=complex, kernel_init=init, hidden_bias_init=init, visible_bias_init=init),
            "hilbert": hi_vit,
            "sampler": nk.sampler.MetropolisLocal(hi_vit)
        },
        "ViTrandom": {
            "param_name": "d_model",
            "h_values": [4, 8, 16, 32, 64],
            "builder": lambda h, init: ViT_ent(num_layers=2, d_model=h, n_heads=4, patch_size=2, kernel_init=init),
            "hilbert": hi_vit,
            "sampler": nk.sampler.MetropolisLocal(hi_vit)
        },
        "HFDSrandom": {
            "param_name": "n_hid",
            "h_values":[2,  4,  8, 16],
            "builder": lambda h, init: HiddenFermion_ent(L=4, network="FFNN", n_hid=h, layers=1, features=64, MFinit="random", hilbert=hi_hfds, kernel_init=init, dtype=jax.numpy.complex128),
            "hilbert": hi_hfds,
            "sampler": nk.sampler.MetropolisExchange(hi_hfds, graph=g_vit)
        },
        "HFDSFermi": {
            "param_name": "n_hid",
            "h_values": [2, 4, 8, 16],
            "builder": lambda h, init: HiddenFermion_ent(L=4, network="FFNN", n_hid=h, layers=1, features=16, MFinit="Fermi", hilbert=hi_hfds, kernel_init=init, dtype=jax.numpy.complex128),
            "hilbert": hi_hfds,
            "sampler": nk.sampler.MetropolisExchange(hi_hfds, graph=g_vit)
        }
    }

    results = []
    all_grids = []

    for model_name, config in models_defs.items():
        if models_to_plot is not None and model_name not in models_to_plot:
            continue
        print(f"Running sweep for {model_name}...")
        h_vals = config["h_values"]
        entropy_grid = np.zeros((len(h_vals), len(variances)))
        
        for i, h in enumerate(h_vals):
            for j, var in enumerate(variances):
                s2_acc = 0.0
                for seed in range(n_seeds):
                    if var == 0:
                        init = jax.nn.initializers.zeros
                    else:
                        init = normal(stddev=np.sqrt(var))
                    
                    model = config["builder"](h, init)
                    sampler = config["sampler"]
                    vstate = nk.vqs.MCState(sampler, model, n_samples=n_samples, seed=seed)
                    s2, _ = compute_renyi2_entropy(vstate, n_samples=n_samples)
                    s2_acc += s2
                entropy_grid[i, j] = s2_acc / n_seeds
                print(f"  {model_name} h={h} var={var:.2f} -> S2={entropy_grid[i, j]:.4f}")
                clean_up()
        
        dv = (variances[1] - variances[0]) / 2. if len(variances) > 1 else 0.5
        dnh = (np.array(h_vals)[1] - np.array(h_vals)[0]) / 2. if len(h_vals) > 1 else 0.5
        extent = [np.min(variances) - dv, np.max(variances) + dv, np.min(h_vals) - dnh, np.max(h_vals) + dnh]
        
        results.append({
            'name': model_name,
            'grid': entropy_grid,
            'extent': extent,
            'ylabel': f'Hidden Size ({config["param_name"]})'
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
            ax.set_xlabel('Variance')
            ax.set_ylabel(res['ylabel'])
            ax.set_title(res['name'])
            
        fig.colorbar(im, ax=axes, label='Renyi-2 Entropy', location='right')
        fig.suptitle(f'Entanglement Entropy vs Variance & Hidden Size (N=16, samples={n_samples}, seeds={n_seeds})')
        
        filename = 'Entanglement_Sweep_Variance_All.png'
        save_path = get_unique_path(save_dir, filename)
        plt.savefig(save_path)
        print(f"Combined plot saved to {save_path}")

        # Save individual plots
        for res in results:
            fig_s, ax_s = plt.subplots(figsize=(6, 5), constrained_layout=True)
            im_s = ax_s.imshow(res['grid'], cmap='viridis', origin='lower', extent=res['extent'], 
                           aspect='auto', interpolation='none')
            ax_s.set_xlabel('Variance')
            ax_s.set_ylabel(res['ylabel'])
            ax_s.set_title(res['name'])
            fig_s.colorbar(im_s, ax=ax_s, label='Renyi-2 Entropy')
            
            fname_s = f'Entanglement_Sweep_Variance_{res["name"]}.png'
            save_path_s = get_unique_path(save_dir, fname_s)
            plt.savefig(save_path_s)
            print(f"Individual plot saved to {save_path_s}")
            plt.close(fig_s)

def plot_entropy_vs_L_hidden_size_map(n_seeds=10, n_samples=4096, models_to_plot=None):
    print("\n--- Plotting Entropy vs L & Hidden Size Map ---")
    save_dir = "/scratch/f/F.Conoscenti/Thesis_QSL/Entanglement/plots"
    os.makedirs(save_dir, exist_ok=True)
    
    L_values = [4, 6, 8, 10]
    var = 1e-1
    
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
            "h_values": [1, 2, 3, 4, 6, 8],
            "builder": lambda L, h, init: HiddenFermion_ent(L=L, network="FFNN", n_hid=h, layers=1, features=64, MFinit="random", hilbert=nk.hilbert.Spin(s=1/2, N=L*L, total_sz=0), kernel_init=init, dtype=jax.numpy.complex128),
            "hilbert_fn": lambda L: nk.hilbert.Spin(s=1/2, N=L*L, total_sz=0),
            "sampler_fn": lambda hi, L: nk.sampler.MetropolisExchange(hi, graph=nk.graph.Hypercube(length=L, n_dim=2, pbc=True))
        },
        "HFDSFermi": {
            "param_name": "n_hid",
            "h_values": [1, 2, 3, 4, 6, 8],
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
        
        dL = (L_values[1] - L_values[0]) / 2. if len(L_values) > 1 else 0.5
        dnh = (np.array(h_vals)[1] - np.array(h_vals)[0]) / 2. if len(h_vals) > 1 else 0.5
        extent = [np.min(L_values) - dL, np.max(L_values) + dL, np.min(h_vals) - dnh, np.max(h_vals) + dnh]
        
        results.append({
            'name': model_name,
            'grid': entropy_grid,
            'extent': extent,
            'ylabel': f'Hidden Size ({config["param_name"]})'
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
            ax.set_xticks([4, 6, 8])
            ax.set_xticklabels([16, 36, 64])
            
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
            ax_s.set_xticks([4, 6, 8])
            ax_s.set_xticklabels([16, 36, 64])
            fig_s.colorbar(im_s, ax=ax_s, label='Renyi-2 Entropy')
            
            fname_s = f'Entanglement_Sweep_L_{res["name"]}.png'
            save_path_s = get_unique_path(save_dir, fname_s)
            plt.savefig(save_path_s)
            print(f"Individual plot saved to {save_path_s}")
            plt.close(fig_s)

def main():

    
    #test_rbm_zero_variance()
    #test_vit_zero_variance()
    #test_hfds_zero_variance_random_init()
    #test_hfds_zero_variance_Fermi_init()

    #test_rbm_finite_variance()
    #test_vit_finite_variance()
    #test_hfds_finite_variance()
    #test_hfds_random_init()
    #test_hfds_Fermi_init()

    #test_entanglement_entropy_rbm(n_samples=65536)
    #test_entanglement_entropy_vit(n_samples=65536)
    #test_entanglement_entropy_hfds(n_samples=65536)
    #test_entanglement_entropy_vit_xavier(n_samples=65536)

    #plot_entropy_vs_variance(n_seeds=1, n_samples=16, models_to_plot = ["ViT"])
    plot_entropy_vs_L(n_seeds=1, n_samples=16, models_to_plot = ["ViT"])
    #plot_entropy_vs_L_hidden_size(n_seeds=1, n_samples=16, models_to_plot = ["ViT"])
    
    #plot_entropy_vs_variance_hidden_size_map(n_seeds=1, n_samples=16, models_to_plot = ["ViTrandom", "ViTXavier"])
    #plot_entropy_vs_L_hidden_size_map(n_seeds=1, n_samples=16, models_to_plot = ["ViTrandom",  "ViTXavier"])

if __name__ == "__main__":
    main()