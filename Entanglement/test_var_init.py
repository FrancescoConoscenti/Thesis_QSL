import netket as nk
import numpy as np
import jax
from jax.nn.initializers import normal
import sys
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
    test_entanglement_entropy_vit_xavier(n_samples=65536)
    #test_hfds_random_init()

if __name__ == "__main__":
    main()