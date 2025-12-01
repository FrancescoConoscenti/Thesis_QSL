from HFDS_Heisenberg.Gutzwiller_MF_Init import update_orbitals_gmf
from HFDS_Heisenberg.Init_orbitals import compute_orbital_selection

import numpy as np
import netket as nk
import jax
import jax.numpy as jnp
from functools import partial

def optimized_gutzwiller_params(lattice):

    def get_log_psi(params, x, L, dtype):

        lattice = nk.graph.Hypercube(length=4, n_dim=2, pbc=True, max_neighbor_order=2)

        mf = update_orbitals_gmf(lattice, dtype, params["h"], params["phi"])
        
        mf_x = compute_orbital_selection(x, mf, lattice.n_nodes)

        sign, log_abs_det = jnp.linalg.slogdet(mf_x)
        log_val = log_abs_det + jnp.log(sign + 0j)
        
        return log_val


    def energy_gradient(params, x, Eloc, lattice, dtype):

        L = jnp.sqrt(lattice.n_nodes).astype(int)

        # Since params are real and log_psi is complex, we use jax.vjp as suggested
        # by the JAX error message for non-holomorphic functions.

        # Create a vectorized version of the gradient function to handle a batch of samples 'x'
        # holomorphic=True is needed if your parameters/wavefunction are complex
        grad_log_psi_fn = jax.vmap(jax.grad(get_log_psi, argnums=0, holomorphic=True), in_axes=(None, 0, None, None))

        # Inside your training loop:
        # params = {"h": ..., "phi": ...}
        # x = samples

        # This computes \nabla_p log(Psi) for every sample in the batch
        grads_per_sample = grad_log_psi_fn(params, x, L, dtype)

        # gradients for h will be shape (n_samples, )
        O_k_h = grads_per_sample["h"] 
        O_k_phi = grads_per_sample["phi"]

        # gradient is \nabla log \Psi
        # Eloc is the Local Energy vector
        
        # Implements the formula above:
        Egrad_h =  2*jnp.real(jnp.mean(jnp.conj(O_k_h)*(Eloc-jnp.mean(Eloc))))
        Egrad_phi =  2*jnp.real(jnp.mean(jnp.conj(O_k_phi)*(Eloc-jnp.mean(Eloc))))
        
        Egrad = {"h": Egrad_h, "phi": Egrad_phi}
        
 
        return Egrad


    def update_params(params, Egrads, learning_rate):

        new_params = {}
        
        new_params["h"] = params["h"] - learning_rate * Egrads["h"]
        new_params["phi"] = params["phi"] - learning_rate * Egrads["phi"]
        
        return new_params


    # Initial guess for parameters
    params = {"h": jnp.array(0.06, dtype=jnp.complex128),
              "phi": jnp.array(0.1, dtype=jnp.complex128)}
    
    learning_rate = 0.01
    n_iterations = 2

    dtype = jnp.complex128

    for i in range(n_iterations):
        key = jax.random.PRNGKey(i)
        x = jax.random.choice(key, jnp.array([-1, 1]), shape=(100, lattice.n_nodes))
        
        # Dummy local energy for testing
        Eloc = jax.random.normal(key, shape=(x.shape[0],))

        Egrads = energy_gradient(params, x, Eloc, lattice, dtype)
        
        params = update_params(params, Egrads, learning_rate)
        
        jax.debug.print("Iteration: {i}", i=i)
        print(f"Iteration: {i}")
    


    params["h"] = jnp.real(params["h"])
    params["phi"] = jnp.real(params["phi"])
            
    return params
