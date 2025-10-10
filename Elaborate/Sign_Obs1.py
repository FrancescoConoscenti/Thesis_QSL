def _marshal_sign_single_full_hilbert(sigma, vstate):
    N_sites = sigma.shape[-1]
    #M_A = 0.5 * jnp.sum(sigma[..., ::2], axis=-1)
    M_A = jnp.array([0.5*sum(sample[::2]) for sample in sigma]) #jnp.sum(0.5 * sigma_test[A_sites]) # Magn on A sublattice
    S_A = 0.5 * (N_sites // 2)
    psi = jnp.exp(vstate.log_value(sigma))
    sign = jnp.real((psi * ((-1.0) ** (S_A - M_A))) / jnp.abs(psi))

    return  sign


get_marshal_sign_full_hilbert = lambda vstate: _marshal_sign_single_full_hilbert

#########################################################################################

def Marshall_Sign(vstate, folder_path, n_samples, L):
    
    number_models = len([name for name in os.listdir(f"{folder_path}/models") if os.path.isfile(os.path.join(f"{folder_path}/models", name))])
    sign2 = np.zeros(number_models)
    vstate.n_samples = n_samples

    for i in range(0, number_models):
        with open(folder_path + f"/models/model_{i} .mpack", "rb") as f:
            vstate.variables = flax.serialization.from_bytes(vstate.variables, f.read())

        # Compute expectation value full Hilbert space
        configs = balanced_combinations_numpy(L*L)
        logpsi = vstate.log_value(configs)
        psi = jnp.exp(logpsi)
        weights = jnp.abs(psi) ** 2
        signs = _marshal_sign_single_full_hilbert( configs, vstate) 
        sign2[i] = jnp.sum(weights * signs) / jnp.sum(weights)
        
    return sign1