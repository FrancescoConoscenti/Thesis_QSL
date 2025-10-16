"""def _marshal_sign_single_full_hilbert(sigma, vstate):
    N_sites = sigma.shape[-1]
    #M_A = 0.5 * jnp.sum(sigma[..., ::2], axis=-1)
    M_A = jnp.array([0.5*sum(sample[::2]) for sample in sigma]) #jnp.sum(0.5 * sigma_test[A_sites]) # Magn on A sublattice
    S_A = 0.5 * (N_sites // 2)
    psi = jnp.exp(vstate.log_value(sigma))
    sign = jnp.real((psi * ((-1.0) ** (S_A - M_A))) / jnp.abs(psi))

    return  sign


#########################################################################################

def Marshall_Sign(vstate, folder_path, n_samples, L):
    
    number_models = len([name for name in os.listdir(f"{folder_path}/models") if os.path.isfile(os.path.join(f"{folder_path}/models", name))])
    sign = np.zeros(number_models)
    vstate.n_samples = n_samples

    for i in range(0, number_models):
        with open(folder_path + f"/models/model_{i} .mpack", "rb") as f:
            vstate.variables = flax.serialization.from_bytes(vstate.variables, f.read())

        # Compute expectation value full Hilbert space
        configs = balanced_combinations_numpy(L*L)
        logpsi = vstate.log_value(configs)
        psi = jnp.exp(logpsi)
        weights = jnp.abs(psi) ** 2

        N_sites = configs.shape[-1]
        M_A = jnp.array([0.5*sum(sample[::2]) for sample in configs]) #jnp.sum(0.5 * sigma_test[A_sites]) # Magn on A sublattice
        S_A = 0.5 * (N_sites // 2)
        psi = jnp.exp(vstate.log_value(configs))
        signs = jnp.real((psi * ((-1.0) ** (S_A - M_A))) / jnp.abs(psi)) 
        sign[i] = jnp.sum(weights * signs) / jnp.sum(weights)
        
    return sign1"""

"""
import netket as nk
import jax
import jax.numpy as jnp

import netket as nk
import jax.numpy as jnp

L = 4
n_dim = 2
J2 = 0.2

# Define 2D square lattice with up to 2nd-neighbor bonds
lattice = nk.graph.Hypercube(length=L, n_dim=n_dim, pbc=True, max_neighbor_order=2)
hi = nk.hilbert.Spin(s=1/2, N=lattice.n_nodes, total_sz=0)

# Unfrustrated Heisenberg model (no built-in Marshall sign rule)
ha = nk.operator.Heisenberg(hilbert=hi, graph=lattice, J=[1.0, J2], sign_rule=[False, False]).to_jax_operator()

E_gs, ket_gs = nk.exact.lanczos_ed(ha, compute_eigenvectors=True)

# ---- Marshall sign evaluation ----
N_sites = hi.size
configs = hi.all_states()  # shape (2**N_sites, N_sites)
configs_05 = 0.5 * configs

# Checkerboard A/B sublattice definition
#A_sites = jnp.arange(0, N_sites, 2) # A sublattice
A_sites = []
for idx in range(N_sites):
    x = idx % L
    y = idx // L
    if (x + y) % 2 == 0:
        A_sites.append(idx)
A_sites = jnp.array(A_sites)
print(A_sites)

B_sites = []
for idx in range(N_sites):
    x = idx % L
    y = idx // L
    if (x + y) % 2 == 1:
        B_sites.append(idx)
B_sites = jnp.array(B_sites)
print(B_sites)

M_A = jnp.sum(configs_05[:, A_sites], axis=1)
print(M_A)

M_B = jnp.sum(configs_05[:, B_sites], axis=1)
print(M_B)

S_A = 0.5 * len(A_sites)
print(S_A)

marshall_phase = (-1.0) ** (S_A - M_A)
print("Marshall phase",marshall_phase)
print("Marshall phase shape",marshall_phase.shape)

psi = ket_gs[:,0]
print("psi", psi)
print("psi shape", psi.shape)

signs = jnp.real((psi * marshall_phase) / jnp.abs(psi))
print("Signs",signs)
print("Signs shape",signs.shape)

weights = jnp.abs(psi) ** 2
marshall_sign_expect = jnp.sum(weights * signs) / jnp.sum(weights)

print("⟨Marshall sign gs⟩ =", float(jnp.real(marshall_sign_expect)))

"""
#%%
import numpy as np
import matplotlib.pyplot as plt

# Example data
steps = np.arange(20)
values1 = np.random.randint(0, 2, size=20)
values2 = np.random.randint(0, 2, size=20)
line_data = np.random.rand(20)  # continuous variable

# Convert 0→'-' and 1→'+'
symbols1 = np.where(values1 == 1, '+', '-')
symbols2 = np.where(values2 == 1, '+', '-')

# Create figure
fig, ax = plt.subplots(figsize=(10, 4))

# --- 1️⃣ Plot the line ---
ax.plot(steps, line_data, color='cyan', lw=2, label='Continuous variable')

# --- 2️⃣ Add binary sequences as text ---
offset1 = -0.2
offset2 = -0.4

for i, sym in enumerate(symbols1):
    ax.text(steps[i], offset1, sym, ha='center', va='center', fontsize=14, color='red')

for i, sym in enumerate(symbols2):
    ax.text(steps[i], offset2, sym, ha='center', va='center', fontsize=14, color='blue')

# --- 3️⃣ Style ---
ax.set_xlabel("Step")
ax.set_ylabel("Value")
ax.set_title("Binary sequences (+/−) and continuous line")

ax.set_xlim(-0.5, len(steps) - 0.5)

# Adjust vertical limits to fit the line and symbols
ymin = min(line_data.min(), offset2 - 0.5)
ymax = max(line_data.max(), 0.5)
ax.set_ylim(ymin, ymax)

ax.legend()

plt.show()



# %%
