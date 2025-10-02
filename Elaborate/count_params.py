#Count Parameters
def hidden_fermion_param_count(n_elecs, n_hid, Lx, Ly, layers, features):
    # Parameters in Orbitals module
    n_sites = Lx * Ly
    orbitals_mf_params = 2 * n_sites * n_elecs  # orbitals_mf shape (2*Lx*Ly, n_elecs)
    orbitals_hf_params = 2 * n_sites * n_hid    # orbitals_hf shape (2*Lx*Ly, n_hid)
    
    # Parameters in FFNN part of HiddenFermion
    input_dim = n_sites  # Input dimension Lx*Ly
    # Hidden layers (all without bias)
    hidden_params = input_dim * features  # First hidden layer
    hidden_params += (layers - 1) * (features * features)  # Subsequent hidden layers
    
    # Output layer (with bias)
    output_dim = n_hid * (n_elecs + n_hid)
    output_params = features * output_dim + output_dim  # Weights + biases
    
    total_params = orbitals_mf_params + orbitals_hf_params + hidden_params + output_params

    print(f"params={total_params}")

#Count parameters
def vit_param_count(num_heads, num_layers, patch_size, d_model, Ns):
    """
    Returns (total_params, breakdown_dict).
    Provide either Ns (total input sites) OR n_patches directly.
    """

    n_patches = Ns // (patch_size**2)

    # Embed layer (Dense with bias)
    embed_params = (patch_size**2) * d_model + d_model  # kernel + bias
    # FMHA
    alpha_params = num_heads * n_patches
    v_params = d_model * d_model + d_model  # kernel + bias
    W_params = d_model * d_model + d_model  # kernel + bias
    fmha_params = v_params + W_params + alpha_params  # = 2*(d^2 + d) + alpha
    # FFN (two Dense layers with biases) -- corrected
    ff_dense1 = d_model * (4 * d_model) + (4 * d_model)   # weights + bias
    ff_dense2 = (4 * d_model) * d_model + d_model         # weights + bias
    ffn_params = ff_dense1 + ff_dense2                   # = 8*d^2 + 5*d
    # LayerNorms in EncoderBlock (2 LNs per block, each has scale+bias)
    norm_block = 4 * d_model
    block_params = fmha_params + ffn_params + norm_block
    encoder_params = num_layers * block_params
    # OutputHead: 3 LayerNorms + 2 Dense(d_model -> d_model)
    output_params = 3 * (2 * d_model) + 2 * (d_model * d_model + d_model)
    total = embed_params + encoder_params + output_params

    return total