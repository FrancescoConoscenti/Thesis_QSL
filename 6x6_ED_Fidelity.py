"""
Unified script: J1-J2 Heisenberg model on a 6×6 square lattice.

Pipeline
--------
1. Build the symmetry-reduced Hilbert space with lattice-symmetries.
2. Diagonalise the Hamiltonian via Lanczos (ARPACK) → exact GS ψ_ED.
3. Define the same Hamiltonian in NetKet and attach an RBM variational state.
4. Train the RBM with VMC + SR (a few hundred steps, enough to get close).
5. Compute the fidelity  F = |⟨ψ_ED | ψ_RBM⟩|²  by expanding both states
   over the same full-basis spin configurations obtained from LS.

Representation bridge
---------------------
LS works in the symmetry-reduced basis; NetKet's MCState works in the full
computational basis.  To compare them we:
  • enumerate all representative configurations from LS  (basis.states)
  • convert each uint64 bit-string → NetKet ±1 spin config
  • query  log ψ_RBM(σ)  for every representative via vs.log_value()
  • weight each amplitude by  basis.norms[i]  (encodes the orbit size and
    character sum, so that  Σ_i |norm_i · ψ_RBM(σ_i)|²  approximates the
    full projected norm)
  • take the inner product with ψ_ED in the reduced basis

Dependencies
------------
    conda install -c twesterhout lattice-symmetries
    pip install netket flax optax
"""

import numpy as np
import scipy.sparse.linalg
import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
import netket as nk
import lattice_symmetries as ls

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 0. Hyper-parameters
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Lx, Ly   = 6, 6
N        = Lx * Ly       # 36 sites
J1       = 1.0           # NN  coupling (AFM)
J2       = 0.5           # NNN coupling (frustrated)

# RBM
ALPHA    = 2             # hidden-unit density  M = alpha * N
N_CHAINS = 16            # independent Markov chains
N_SAMPLES= 2048          # MC samples per VMC step
N_STEPS  = 300           # VMC optimisation steps
LR       = 1e-3          # Adam / SR learning rate

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. Lattice geometry helpers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def site(x, y):
    return (x % Lx) + Lx * (y % Ly)

nn_bonds, nnn_bonds = [], []
for y in range(Ly):
    for x in range(Lx):
        i = site(x, y)
        nn_bonds.append( (i, site(x+1, y  )) )
        nn_bonds.append( (i, site(x,   y+1)) )
        nnn_bonds.append((i, site(x+1, y+1)) )
        nnn_bonds.append((i, site(x+1, y-1)) )

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. Exact diagonalisation with lattice-symmetries
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("=" * 60)
print("  STEP 1 – Exact diagonalisation (lattice-symmetries)")
print("=" * 60)

# --- symmetry generators ---
def _perm(fn):
    return np.array([fn(x, y) for y in range(Ly) for x in range(Lx)],
                    dtype=np.int32)

sym_Tx  = ls.Symmetry(_perm(lambda x,y: site(x+1, y  )), sector=0)
sym_Ty  = ls.Symmetry(_perm(lambda x,y: site(x,   y+1)), sector=0)
sym_Rx  = ls.Symmetry(_perm(lambda x,y: site(Lx-1-x, y)), sector=0)
# spin inversion (maps every site to itself, flips all spins)
sym_SI  = ls.Symmetry(np.arange(N, dtype=np.int32), sector=0,
                      is_spin_inversion=True)

symmetries = ls.SymmetryGroup([sym_Tx, sym_Ty, sym_Rx, sym_SI])

basis = ls.SpinBasis(
    symmetries,
    number_spins  = N,
    hamming_weight= N // 2,   # Sz = 0
    spin_inversion= 1,        # positive parity → singlet sector
)
basis.build()

dim = basis.number_states
print(f"  Lattice : {Lx}×{Ly},  N={N} sites")
print(f"  J1={J1}, J2={J2}  (J2/J1={J2/J1:.2f})")
print(f"  Symmetry-reduced dim = {dim:,}")

# --- Hamiltonian ---
def heisenberg_terms(bonds, J):
    terms = []
    for (i, j) in bonds:
        terms += [("+-", 0.5*J, [i, j]),
                  ("-+", 0.5*J, [i, j]),
                  ("zz",     J, [i, j])]
    return terms

H_ls = ls.Operator(basis, heisenberg_terms(nn_bonds,  J1)
                        + heisenberg_terms(nnn_bonds, J2))

class _LinOp(scipy.sparse.linalg.LinearOperator):
    def __init__(self, H, dim):
        super().__init__(dtype=np.float64, shape=(dim, dim))
        self._H = H
    def _matvec(self, v):
        out = np.zeros_like(v)
        self._H.apply(v, out)
        return out

print("  Running Lanczos … ", end="", flush=True)
evals, evecs = scipy.sparse.linalg.eigsh(
    _LinOp(H_ls, dim), k=2, which="SA", tol=1e-10, maxiter=10_000)
E0_ED  = evals[0]
psi_ED = evecs[:, 0]           # real, normalised, in the LS reduced basis
print("done.")
print(f"  E0      = {E0_ED:.8f}")
print(f"  E0/N    = {E0_ED/N:.8f}")
print(f"  Gap     = {evals[1]-evals[0]:.6f}")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. NetKet setup – Hilbert space & Hamiltonian
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print()
print("=" * 60)
print("  STEP 2 – NetKet RBM variational state + VMC")
print("=" * 60)

# NetKet Hilbert space: spin-1/2, Sz=0 sector
hi = nk.hilbert.Spin(s=0.5, N=N, total_sz=0)

# Build the Hamiltonian as a NetKet LocalOperator
H_nk = nk.operator.LocalOperator(hi, dtype=complex)

# Pauli matrices (site-local)
Sp = np.array([[0, 1], [0, 0]], dtype=complex)   # S+ = |↑⟩⟨↓|
Sm = np.array([[0, 0], [1, 0]], dtype=complex)   # S- = |↓⟩⟨↑|
Sz = np.array([[0.5, 0], [0, -0.5]], dtype=complex)

def add_heisenberg_bond(H, i, j, J):
    H += J * 0.5 * nk.operator.LocalOperator(hi, [Sp, Sm], [[i],[j]])
    H += J * 0.5 * nk.operator.LocalOperator(hi, [Sm, Sp], [[i],[j]])
    H += J       * nk.operator.LocalOperator(hi, [Sz, Sz], [[i],[j]])

for (i, j) in nn_bonds:
    add_heisenberg_bond(H_nk, i, j, J1)
for (i, j) in nnn_bonds:
    add_heisenberg_bond(H_nk, i, j, J2)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 4. RBM ansatz
#    We use NetKet's built-in RBM with a Marshall-sign rule applied via a
#    separate sign network, giving  log ψ(σ) = RBM_mag(σ) + i·π·Marshall(σ).
#    For the frustrated J1-J2 model the Marshall sign is not exact, but it
#    gives a useful initialisation bias.  You can set use_marshall=False to
#    use a fully complex RBM instead.
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
USE_MARSHALL = False   # set True to bias toward Néel sign structure

if USE_MARSHALL:
    # Complex RBM with fixed Marshall-sign pre-factor
    model = nk.models.RBMModPhase(
        alpha      = ALPHA,
        use_hidden_bias = True,
        use_visible_bias= True,
    )
else:
    # Fully complex RBM (handles frustrated sign structure freely)
    model = nk.models.RBM(
        alpha      = ALPHA,
        use_hidden_bias = True,
        use_visible_bias= True,
        dtype       = complex,
    )

# Sampler: local spin-flip moves within the Sz=0 sector
sampler = nk.sampler.MetropolisExchange(
    hi,
    n_chains = N_CHAINS,
    graph    = nk.graph.Grid([Lx, Ly], pbc=True),
)

# Variational state
vs = nk.vqs.MCState(
    sampler,
    model,
    n_samples    = N_SAMPLES,
    n_discard_per_chain = 64,
)

print(f"  RBM parameters : {vs.n_parameters:,}")
print(f"  MC samples/step: {N_SAMPLES}")
print(f"  VMC steps      : {N_STEPS}")

# Optimiser: Adam with Stochastic Reconfiguration (SR)
optimiser = optax.adam(LR)
sr        = nk.optimizer.SR(diag_shift=1e-3)

gs_driver = nk.VMC(H_nk, optimiser, variational_state=vs, preconditioner=sr)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 5. VMC optimisation loop
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print()
energies = []
for step in range(N_STEPS):
    gs_driver.advance(1)
    e = float(vs.expect(H_nk).mean.real)
    energies.append(e)
    if step % 50 == 0 or step == N_STEPS - 1:
        print(f"  step {step:4d}  E/N = {e/N:.6f}   (E0/N_ED = {E0_ED/N:.6f})")

E0_RBM = energies[-1]
print(f"\n  Final VMC energy : {E0_RBM:.8f}  (E0/N = {E0_RBM/N:.8f})")
print(f"  Relative error   : {abs(E0_RBM - E0_ED) / abs(E0_ED) * 100:.4f} %")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 6. Fidelity  F = |⟨ψ_ED | ψ_RBM⟩|²
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print()
print("=" * 60)
print("  STEP 3 – Fidelity calculation")
print("=" * 60)

# --- 6a. Extract representative spin configurations from LS ---
# basis.states : uint64 array of shape (dim,)
#   bit k of states[i]  = 1 if site k is spin-up in the i-th representative
rep_ints = basis.states   # shape (dim,)

def ls_int_to_nk_config(state_int, N):
    """uint64 LS bit-string  →  NetKet ±1 array (shape N,)."""
    bits = np.array([(int(state_int) >> k) & 1 for k in range(N)],
                    dtype=np.float64)
    return 2.0 * bits - 1.0          # 0 → -1 (down),  1 → +1 (up)

print(f"  Converting {dim:,} representative states … ", end="", flush=True)
configs_nk = np.stack([ls_int_to_nk_config(s, N) for s in rep_ints])
# shape: (dim, N)
print("done.")

# --- 6b. Evaluate log ψ_RBM on every representative configuration ---
# vs.log_value accepts a 2-D array of shape (n_configs, N)
print(f"  Evaluating RBM amplitudes on {dim:,} configs … ", end="", flush=True)
log_psi_rbm = np.array(vs.log_value(configs_nk))   # shape (dim,), complex
psi_rbm_rep = np.exp(log_psi_rbm)                  # ψ_RBM(σ_rep), unnorm.
print("done.")

# --- 6c. Weight by LS orbit norms ---
# basis.norms[i] = sqrt(|orbit_i| · |Σ_g χ(g)|²) / |G|
# Multiplying ψ_RBM(σ_rep) by norms[i] gives the contribution of the i-th
# representative to the full projected inner product:
#   ⟨ψ_ED | ψ_RBM⟩ = Σ_i ψ_ED[i]* · norm[i] · ψ_RBM(σ_i)
ls_norms = basis.norms               # shape (dim,), float64
psi_rbm_proj = ls_norms * psi_rbm_rep

# --- 6d. Numerator: ⟨ψ_ED | ψ_RBM^proj⟩ ---
overlap_num = np.dot(psi_ED.conj(), psi_rbm_proj)   # complex scalar

# --- 6e. Denominator: ||ψ_ED|| · ||ψ_RBM^proj|| ---
# ψ_ED is already normalised by eigsh; verify just in case
norm_ED  = np.linalg.norm(psi_ED)
norm_RBM = np.linalg.norm(psi_rbm_proj)

overlap  = overlap_num / (norm_ED * norm_RBM)
fidelity = float(np.abs(overlap) ** 2)

print()
print(f"  ||ψ_ED||            = {norm_ED:.8f}  (should be 1.0)")
print(f"  ||ψ_RBM_proj||      = {norm_RBM:.6e}")
print(f"  |⟨ψ_ED|ψ_RBM⟩|     = {np.abs(overlap):.8f}")
print(f"  Fidelity            = {fidelity:.8f}")
print(f"  Infidelity (1−F)    = {1.0 - fidelity:.4e}")
print()
print("=" * 60)
print("  Summary")
print("=" * 60)
print(f"  E0 (ED)  = {E0_ED:.8f}   (E0/N = {E0_ED/N:.8f})")
print(f"  E0 (RBM) = {E0_RBM:.8f}   (E0/N = {E0_RBM/N:.8f})")
print(f"  Rel. ΔE  = {abs(E0_RBM-E0_ED)/abs(E0_ED)*100:.4f} %")
print(f"  Fidelity = {fidelity:.8f}")
print(f"  1 − F    = {1.0-fidelity:.4e}")