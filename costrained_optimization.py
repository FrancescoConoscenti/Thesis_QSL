import netket as nk
import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn



# 1. Define the System
L = 10
graph = nk.graph.Hypercube(length=L, n_dim=1, pbc=True)
hilbert = nk.hilbert.Spin(s=1/2, N=graph.n_nodes)
hamiltonian = nk.operator.Heisenberg(hilbert, graph=graph)


# 1. Compute Exact Ground State
E_gs, psi_exact = nk.exact.lanczos_ed(hamiltonian, compute_eigenvectors=True)

# User Correction: Select the first eigenvector
psi_exact = psi_exact[:, 0] 

# Precompute log amplitudes
exact_log_amps = jnp.log(np.abs(psi_exact) + 0j)
#exact_log_amps = jnp.log(psi_exact + 0j)
exact_log_amps = jax.device_put(exact_log_amps) # Move to GPU/TPU if available

# 2. Define the Model
class PhaseOnlyModel(nn.Module):
    
    @nn.compact
    def __call__(self, x):
        # --- 1. LEARNABLE PART (Phase) ---
        # A simple dense network for the phase
        phase = nn.Dense(features=32, dtype=float)(x)
        phase = nn.relu(phase)
        phase = nn.Dense(features=1, dtype=float)(phase)
        phase = phase.squeeze(-1)

        # --- 2. FIXED PART (Amplitude) ---
        # We access 'exact_log_amps' directly from the outer scope.
        # JAX handles this capture automatically.
        
        # Convert spins (-1, 1) to bits (0, 1)
        x_bits = (x + 1) / 2
        
        # Convert to integer indices
        # CAUTION: This assumes standard basis ordering and no constraints
        powers = 2**jnp.arange(x.shape[-1] - 1, -1, -1)
        indices = jnp.sum(x_bits * powers, axis=-1).astype(int)
        
        # Lookup in the GLOBAL array captured by closure
        log_amp = exact_log_amps[indices]

        # --- 3. COMBINE ---
        return log_amp + 1j * phase


class ComplexOverrideModel(nn.Module):
    
    @nn.compact
    def __call__(self, x):
        # --- 1. THE COMPLEX NETWORK ---
        # We define a standard network with complex weights.
        # Note: dtype=complex allows the weights to be complex numbers
        y = nn.Dense(features=32, dtype=complex)(x)
        y = nk.nn.log_cosh(y)
        
        # Final layer outputs a single complex number per sample
        network_out = nn.Dense(features=1, dtype=complex)(y)
        network_out = network_out.squeeze(-1) # Shape: (batch,)
        jax.debug.print("{x}",x=network_out.imag)
        
        # --- 2. THE EXACT LOOKUP ---
        # Standard bit conversion logic
        x_bits = (x + 1) / 2
        powers = 2**jnp.arange(x.shape[-1] - 1, -1, -1)
        indices = jnp.sum(x_bits * powers, axis=-1).astype(int)
        
        # Lookup the exact Real part (Log Amplitude)
        exact_real_part = exact_log_amps[indices]

        # --- 3. THE HYBRID COMBINATION ---
        # We take the Exact Real part
        # We take the Network's Imaginary part
        # We discard the Network's Real part
        return exact_real_part + 1j * network_out.imag
    


# 3. Setup Optimization
#model = PhaseOnlyModel() # No arguments needed now
model = ComplexOverrideModel()

# Use MetropolisLocal
sampler = nk.sampler.MetropolisLocal(hilbert)

vstate = nk.vqs.MCState(sampler, model, n_samples=1000)

optimizer = nk.optimizer.Sgd(learning_rate=0.05)
sr = nk.optimizer.SR(diag_shift=0.01)

gs = nk.driver.VMC(hamiltonian, optimizer, variational_state=vstate, preconditioner=sr)

print("Starting optimization...")
gs.run(n_iter=100)