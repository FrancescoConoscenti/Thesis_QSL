from HFDS_Heisenberg.Gutzwiller_MF_Init import update_orbitals_gmf
#from HFDS_Heisenberg.Init_orbitals import compute_orbital_selection

import numpy as np
import netket as nk
import jax
import jax.numpy as jnp
import logging
from functools import partial
import flax.linen as nn
import matplotlib.pyplot as plt
import os


logger = logging.getLogger(__name__)


def compute_orbital_selection(x, orbitals_full, N_sites):
  """
  Computes the orbital selection for a single sample `x`.
  x has shape (N_sites,).
  """
  #1.  Convert {-1,+1} → 0/1 occupancy for spin-up, spin-down orbitals
  spin_up = (x == 1)
  spin_dn = (x == -1)
  x_flat = jnp.concatenate([spin_up, spin_dn]) # Concatenate 1D arrays

  #2 & 3. Select occupied orbitals using advanced indexing
  mask = x_flat.astype(bool)
  _, idx = jax.lax.top_k(mask, k = N_sites)
  return orbitals_full[idx, :] # Simple indexing for a single sample

#####################################################################################################################################################################

class GutzwillerWaveFunction(nn.Module):
    lattice: nk.graph.Graph
    dtype: type

    def setup(self):
        # Define h and phi as parameters here
        self.h = self.param("h", nn.initializers.constant(0.06), (), self.dtype)
        self.phi = self.param("phi", nn.initializers.constant(0.1), (), self.dtype)

    @nn.compact
    def __call__(self, x):
        mf = update_orbitals_gmf(self.lattice, self.dtype, self.h, self.phi)
        # Vectorize the single-sample function to work on a batch of samples `x`.
        vmapped_selection = jax.vmap(compute_orbital_selection, in_axes=(0, None, None))
        mf_x = vmapped_selection(x, mf, self.lattice.n_nodes)
        mf_x = mf_x + 1e-8 * jnp.eye(mf_x.shape[-1], dtype=mf_x.dtype)
        sign, log_abs_det = jax.vmap(jnp.linalg.slogdet)(mf_x)
        return log_abs_det + jnp.log(sign + 0j)

########################################################################################################################################################################

def plot_gutzwiller_optimization_history(iterations, energy_history, h_history, phi_history, output_folder):
    """
    Plots the optimization history of Gutzwiller parameters (energy, h, and phi).

    Args:
        iterations (range): The range of iterations.
        energy_history (list): List of real part of mean energies at each iteration.
        h_history (list): List of 'h' parameter values at each iteration.
        phi_history (list): List of 'phi' parameter values at each iteration.
        output_folder (str): Path to the folder where plots will be saved.
    """
    os.makedirs(output_folder, exist_ok=True)
    
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # Plot Energy on the primary y-axis (left)
    color = 'tab:red'
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Real part of Mean Energy', color=color)
    line1, = ax1.plot(iterations, energy_history, linestyle='-', color=color, label='Energy')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, axis='y', linestyle='--', alpha=0.7)

    # Create a secondary y-axis (right) for h and phi
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Parameter value', color=color)
    line2, = ax2.plot(iterations, h_history, linestyle='--', color='tab:green', label='h')
    line3, = ax2.plot(iterations, phi_history, linestyle='-.', color='tab:purple', label='phi')
    ax2.tick_params(axis='y', labelcolor=color)

    # Add a title and a combined legend
    plt.title('Gutzwiller Optimization History')
    lines = [line1, line2, line3]
    ax1.legend(lines, [l.get_label() for l in lines], loc='upper right')
    fig.tight_layout()
    plt.savefig(os.path.join(output_folder, 'gutzwiller_optimization_history.png'))
    plt.close(fig)
    logger.info(f"Gutzwiller optimization plots saved in {output_folder}")

###############################################################################################

def optimized_gutzwiller_params(lattice, ha, output_folder=None):
    logger.info("Starting optimization of Gutzwiller parameters.")

    # Initial guess for parameters
    params = {"h": jnp.array(0.06, dtype=jnp.float64),
              "phi": jnp.array(0.1, dtype=jnp.float64)}
    
    n_iterations = 2000 # Increased for better convergence
    n_samples = 1024
    dtype = jnp.complex128
    
    # Define a simple model and vstate for Eloc calculation
    gutz_model = GutzwillerWaveFunction(lattice=lattice, dtype=dtype)
    hi = nk.hilbert.Spin(s=1/2, N=lattice.n_nodes, total_sz=0)
    sampler = nk.sampler.MetropolisLocal(hilbert=hi)
    vstate = nk.vqs.MCState(sampler, gutz_model, n_samples=n_samples)
    vstate.parameters = params # Set initial parameters

    # Use NetKet's built-in functionality for optimization
    # This replaces the manual gradient calculation and parameter updates.
    # We can use different learning rates for different parameters with optax.
    import optax
    # Use a single Adam optimizer for all parameters
    optimizer = optax.adam(learning_rate=0.001)

    # Lists to store history for plotting
    energy_history = []
    h_history = []
    phi_history = []

    logger.info(f"Running Gutzwiller optimization for {n_iterations} iterations.")
    opt_state = optimizer.init(vstate.parameters)

    for i in range(n_iterations):
        # This one line computes the mean energy and its gradient
        e_mean, Egrads = vstate.expect_and_grad(ha)
        e_var = vstate.expect(ha @ ha).mean.real - e_mean.mean.real**2

        logger.info(f"Iteration {i}: Mean Energy = {e_mean}, Variance = {e_var}")
        logger.info(f"Iteration {i}: Gradients = {Egrads}")
        
        # Update the optimizer state and parameters
        updates, opt_state = optimizer.update(Egrads, opt_state, vstate.parameters)
        vstate.parameters = optax.apply_updates(vstate.parameters, updates)

        logger.info(f"Iteration {i}: Updated params = {vstate.parameters}")

        # Store values for plotting
        energy_history.append(e_mean.mean.real)
        h_history.append(vstate.parameters["h"])
        phi_history.append(vstate.parameters["phi"])


    logger.info(f"Gutzwiller optimization finished. Final params: {params}")

    if output_folder:
        os.makedirs(output_folder, exist_ok=True)
        
        plot_gutzwiller_optimization_history(range(n_iterations), energy_history, h_history, phi_history, output_folder)
            
    return vstate.parameters


"""
def optimized_gutzwiller_params(lattice, ha, output_folder=None):
    logger.info("Starting optimization of Gutzwiller parameters.")

    def get_log_psi(params, x, L, dtype):

        # x is a batch of samples, shape (n_samples, n_sites)
        mf = update_orbitals_gmf(lattice, dtype, params["h"].astype(dtype), params["phi"].astype(dtype))
        
        vmapped_selection = jax.vmap(compute_orbital_selection, in_axes=(0, None, None))
        mf_x = vmapped_selection(x, mf, lattice.n_nodes)
        mf_x = mf_x + 1e-8 * jnp.eye(mf_x.shape[-1], dtype=mf_x.dtype)
        sign, log_abs_det = jax.vmap(jnp.linalg.slogdet)(mf_x)
        log_val = log_abs_det + jnp.log(sign + 0j)
        
        return log_val


    def energy_gradient(params, x, Eloc, lattice, dtype):

        L = jnp.sqrt(lattice.n_nodes).astype(int)

        # Define a function that computes log_psi for a SINGLE sample.
        # Note: x will have shape (n_sites,)
        log_psi_single_sample_fn = lambda p, sample: get_log_psi(p, sample[jnp.newaxis, :], L, dtype)[0]

        # Differentiate the single-sample function, then vmap it over the batch of samples.
        # in_axes=(None, 0) means we don't map over params, but we map over the samples `x`.
        grad_log_psi_fn = jax.vmap(jax.grad(lambda p, sample: log_psi_single_sample_fn(p, sample).real, argnums=0), in_axes=(None, 0))
        grads_per_sample = grad_log_psi_fn(params, x)

        logger.info(f"grads_per_sample keys: {grads_per_sample.keys()}")

        # gradients for h will be shape (n_samples, )
        O_k_h = grads_per_sample["h"] 
        O_k_phi = grads_per_sample["phi"]

        # gradient is \nabla log \Psi
        # Eloc is the Local Energy vector
        logger.info(f"Ok_h sample values: {O_k_h}, O_k_phi sample values: {O_k_phi}")
        logger.info(f"Mean of Eloc: {jnp.mean(Eloc)}")

        Eloc = Eloc.real.reshape(-1)
        # Implements the formula above:
        Egrad_h =  2*jnp.real(jnp.mean(jnp.conj(O_k_h)*(Eloc-jnp.mean(Eloc))))
        Egrad_phi =  2*jnp.real(jnp.mean(jnp.conj(O_k_phi)*(Eloc-jnp.mean(Eloc))))
        
        logger.info(f"Calculated gradients - Egrad_h: {Egrad_h}, Egrad_phi: {Egrad_phi}")
        Egrad = {"h": Egrad_h, "phi": Egrad_phi}
        
 
        return Egrad


    def update_params(params, Egrads, learning_rate_h, learning_rate_phi):

        new_params = {}
        
        new_params["h"] = params["h"] - learning_rate_h * Egrads["h"]
        new_params["phi"] = params["phi"] - learning_rate_phi * Egrads["phi"]
        
        return new_params

    
    ##############################################################################################################################################################

    # Initial guess for parameters
    params = {"h": jnp.array(0.06, dtype=jnp.float64),
              "phi": jnp.array(0.1, dtype=jnp.float64)}
    
    learning_rate = 0.001
    learning_rate_h = learning_rate
    learning_rate_phi = learning_rate*100
    n_iterations = 2000 # Increased for better convergence
    n_samples = 1024
    dtype = jnp.complex128
    
    # Define a simple model and vstate for Eloc calculation
    gutz_model = GutzwillerWaveFunction(lattice=lattice, dtype=dtype)
    hi = nk.hilbert.Spin(s=1/2, N=lattice.n_nodes, total_sz=0)
    sampler = nk.sampler.MetropolisLocal(hilbert=hi)
    vstate = nk.vqs.MCState(sampler, gutz_model, n_samples=n_samples)

    # Lists to store history for plotting
    energy_history = []
    h_history = []
    phi_history = []

    logger.info(f"Running Gutzwiller optimization for {n_iterations} iterations.")
    for i in range(n_iterations):
        vstate.parameters = params
        vstate.reset()

        vstate.sample()
        # Reshape samples to (n_samples, n_sites)
        x = vstate.samples.reshape(-1, vstate.samples.shape[-1])
        logger.info(f"Iteration {i}: Samples = {x}")
        
        # Calculate real local energy
        Eloc = vstate.local_estimators(ha) 
        logger.info(f"Iteration {i}: Local Energy = {Eloc}")
        
        e_mean = jnp.mean(Eloc)
        e_var = jnp.var(Eloc)
        logger.info(f"Iteration {i}: Mean Energy = {e_mean}, Variance = {e_var}")

        Egrads = energy_gradient(params, x, Eloc, lattice, dtype)
        logger.info(f"Iteration {i}: Gradients = {Egrads}")
        
        params = update_params(params, Egrads, learning_rate_h, learning_rate_phi)
        logger.info(f"Iteration {i}: Updated params = {params}")

        # Store values for plotting
        energy_history.append(jnp.real(e_mean))
        h_history.append(params["h"])
        phi_history.append(params["phi"])


    logger.info(f"Gutzwiller optimization finished. Final params: {params}")

    if output_folder:
        os.makedirs(output_folder, exist_ok=True)
        
        plot_gutzwiller_optimization_history(range(n_iterations), energy_history, h_history, phi_history, output_folder)
            
    return params

"""