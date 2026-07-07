"""
J1-J2 Heisenberg model on an L×L square lattice.

Pipeline
--------
1. QuSpin ED with four symmetry generators: Tx, Ty (translations), Rx
   (x-reflection), and Z (spin inversion) in the k=(0,0), parity = +1 sector.
2. NetKet RBM (optionally translation-symmetrised via RBMSymm) + VMC/SR.
3. Fidelity  F = |⟨ψ_ED | ψ_RBM⟩|²  via orbit-norm weighted inner product.

Orbit-norm projection (Step 3)
------------------------------
For the k=(0,0) sector with all group characters = +1, the projected ED state
in the full Fock basis is:

    ψ_full(σ) = c_i / √n_i      for every σ in orbit i of the representatives

where  n_i = basis.n[i]  is the orbit size under the full symmetry group.
Therefore:

    ⟨ψ_ED | ψ_RBM⟩  ≈  Σ_i  c_i*  √n_i  ψ_RBM(σ_i)
    ‖ψ_RBM‖_full²   ≈  Σ_i  n_i  |ψ_RBM(σ_i)|²

These are exact if ψ_RBM respects the lattice symmetries; otherwise they
approximate the true overlap from above by the Cauchy–Schwarz inequality.

Scaling
-------
    L=4  →  full-Sz=0 dim 12 870,   ~4 symmetry sectors → a few hundred states
    L=6  →  full-Sz=0 dim ~9.1 B,   symmetry-reduced dim ~10–60 M
             (build: minutes; Lanczos: tens of minutes on a single core)
"""

import re
import time
import numpy as np
import netket as nk
import optax
import flax.serialization
from quspin.basis import spin_basis_general
from quspin.operators import hamiltonian
from Observables import load_vstate, setup_model, parse_model_path



# ─── Parameters ───────────────────────────────────────────────────────────────
Lx, Ly    = 6,6
N         = Lx * Ly
J1        = 1.0           # NN coupling (AFM)
J2        = 0.0           # NNN coupling (frustrated)

ALPHA     = 2             # RBM hidden-unit density  M = α N
N_CHAINS  = 16
N_SAMPLES = 2048          # MC samples per VMC step
N_STEPS   = 200            # VMC optimisation steps
LR        = 5e-2          # SGD learning rate for VMC_SR

N_SAMPLES_FIDELITY = 4096*2*2*2  # MC samples for fidelity estimate (more → smaller error bar)

# Pre-trained model to load instead of running VMC (set to None to train from scratch).
# Supported architectures detected automatically from the path name:
#   ViT  → path must contain the tag  layers<N>_d<D>_heads<H>_patch<P>_...parity<B>_rot<B>
#   RBM  → any other path (or None → train RBM from scratch)
MODEL_PATH = None

# ─── Lattice helpers ──────────────────────────────────────────────────────────
def site(x, y):
    return (x % Lx) + Lx * (y % Ly)

nn_pairs   = [(site(x, y), site(x+1, y  )) for y in range(Ly) for x in range(Lx)]
nn_pairs  += [(site(x, y), site(x,   y+1)) for y in range(Ly) for x in range(Lx)]
nnn_pairs  = [(site(x, y), site(x+1, y+1)) for y in range(Ly) for x in range(Lx)]
nnn_pairs += [(site(x, y), site(x+1, y-1)) for y in range(Ly) for x in range(Lx)]

# ─── 1. QuSpin exact diagonalisation ──────────────────────────────────────────
print("=" * 60)
print("  STEP 1 — Exact diagonalisation (QuSpin + symmetries)")
print("=" * 60)

try:
    basis = np.load(f"basis_{J2}.npy")
    psi_ED = np.load(f"psi_ED_{J2}.npy")
except:

    # Permutation arrays for each symmetry generator.
    # site enumeration: row-major  i = x + Lx*y
    T_x  = np.array([site(x+1, y)     for y in range(Ly) for x in range(Lx)], dtype=np.int32)
    T_y  = np.array([site(x, y+1)     for y in range(Ly) for x in range(Lx)], dtype=np.int32)
    R_x  = np.array([site(Lx-1-x, y) for y in range(Ly) for x in range(Lx)], dtype=np.int32)
    R_y  = np.array([site(x, Ly-1-y) for y in range(Ly) for x in range(Lx)], dtype=np.int32)
    P_id = np.arange(N, dtype=np.int32)  # identity on sites; zblock flips all spins

    # Diagonal (transpose) reflection — only valid for square lattices (Lx == Ly).
    # Together with R_x this generates the full D4 point group (order 8), giving a
    # 4× extra reduction vs R_x alone, crucial for making 6×6 ED tractable.
    _extra_symm = {}
    if Lx == Ly:
        R_xy = np.array([site(y, x) for y in range(Ly) for x in range(Lx)], dtype=np.int32)
        _extra_symm = dict(pyblock=(R_y, 0), pxyblock=(R_xy, 0))

    # Build the symmetry-reduced basis.
    # The GS of the AFM / frustrated Heisenberg model on a square lattice lives in
    # the k=(0,0) sector with all parities = +1.
    # If you want to check other sectors, change the sector indices (0 → 1, etc.).
    basis = spin_basis_general(
        N, Nup=N // 2,
        kxblock=(T_x,  0),   # k_x = 0  (eigenvalue exp(2πi·0/Lx) = 1)
        kyblock=(T_y,  0),   # k_y = 0
        pxblock=(R_x,  0),   # x-reflection parity +1
        zblock =(P_id, 0),   # spin-inversion parity +1
        **_extra_symm,       # pyblock + pxyblock for square lattices (full D4)
    )
    np.save(f"basis_{J2}.npy", basis)
    print(f"  {Lx}×{Ly} lattice,  J1={J1},  J2/J1={J2/J1:.2f}")
    print(f"  Symmetry-reduced dim = {basis.Ns:,}   ()")

    # Hamiltonian — QuSpin format: [[coupling, i, j], ...]
    def _bond_terms(pairs, J):
        pm = [[J * 0.5, i, j] for (i, j) in pairs]
        mp = [[J * 0.5, i, j] for (i, j) in pairs]
        zz = [[J,       i, j] for (i, j) in pairs]
        return pm, mp, zz

    pm1, mp1, zz1 = _bond_terms(nn_pairs,  J1)
    pm2, mp2, zz2 = _bond_terms(nnn_pairs, J2)

    static = [
        ["+-", pm1 + pm2],
        ["-+", mp1 + mp2],
        ["zz", zz1 + zz2],
    ]
    H_qs = hamiltonian(static, [], basis=basis, dtype=np.float64,
                    check_symm=False, check_herm=False)

    print("  Running Lanczos ...", end=" ", flush=True)
    evals, evecs = H_qs.eigsh(k=2, which="SA", tol=1e-10, maxiter=10_000)
    E0_ED  = evals[0]
    psi_ED = evecs[:, 0].real   # real for k=(0,0) with real Hamiltonian
    np.save(f"psi_ED_{J2}.npy", psi_ED)

    print(f"  E0   = {E0_ED:.8f}")
    print(f"  E0/N = {E0_ED / N:.8f}")
    print(f"  Gap  = {evals[1] - evals[0]:.6f}")

# ─── 2. NetKet RBM + VMC ──────────────────────────────────────────────────────
print()
print("=" * 60)
print("  STEP 2 — NetKet RBM + VMC")
print("=" * 60)

lattice = nk.graph.Hypercube(length=Lx, n_dim=2, pbc=True, max_neighbor_order=2)
hilbert = nk.hilbert.Spin(s=1 / 2, N=lattice.n_nodes, total_sz=0)

# Heisenberg bond operator as a 4×4 matrix (acting on two sites)
hamiltonian = nk.operator.Heisenberg(
    hilbert=hilbert, graph=lattice, J=[1.0, J2], sign_rule=[False, False]
).to_jax_operator()  # No Marshall sign rule


# ── Model ────────────────────────────────────────────────────────────────────

#load a model
folder = "/scratch/f/F.Conoscenti/Thesis_QSL/ViT_Heisenberg/plot/6x6/layers2_d60_heads10_patch2_sample4096_lr0.0075_iter3000_parityTrue_rotTrue_latest_model/J=0.5/seed_1/models/model_20.mpack"

params = parse_model_path(folder)
model = setup_model(params, hilbert, Lx)

sampler = nk.sampler.MetropolisExchange(
        hilbert=hilbert,
        graph=lattice,
        d_max=2,
        n_chains=1024,
        sweep_size=lattice.n_nodes,
    )

vstate = load_vstate(folder, sampler, model)


#train a model
if model is None:
    # Plain complex RBM — simpler, no symmetry enforcement
    model = nk.models.RBM(
            alpha=ALPHA,
            use_hidden_bias=True,
            use_visible_bias=True,
            param_dtype=complex,
        )

    sampler = nk.sampler.MetropolisExchange(
        hilbert=hilbert,
        graph=lattice,
        d_max=2,
        n_chains=64,
        sweep_size=lattice.n_nodes,
    )

    optimizer = nk.optimizer.Sgd(learning_rate=0.01)

    vstate = nk.vqs.MCState(
        sampler=sampler,
        model=model,
        n_samples=N_SAMPLES,
        n_discard_per_chain=128,
        chunk_size=128,
    )

    from netket.driver import VMC_SR

    vmc = VMC_SR(
        hamiltonian=hamiltonian,
        optimizer=optimizer,
        diag_shift=1e-4,
        variational_state=vstate,
        mode="complex",
    )

    log = nk.logging.RuntimeLog()
    vmc.run(n_iter=N_STEPS, out=log)



# ─── 3. Fidelity via importance sampling ─────────────────────────────────────

def fidelity_sampling(psi_ED, quspin_basis, vs, N, n_samples=100_000, n_boot=200):
    """
    Estimate F = |⟨ψ_ED | ψ_NQS⟩|² by importance sampling from |ψ_NQS|².

    Estimator
    ---------
    For each sample σ drawn from the NQS Markov chain (∝ |ψ_NQS(σ)|²):

        w(σ) = ψ_ED(σ) / ψ_NQS(σ)*
        F    = |E[w]|² / E[|w|²]

    This follows from inserting a resolution of identity:
        |⟨ψ_ED|ψ_NQS⟩|² / ‖ψ_NQS‖² = |E_{|ψ_NQS|²}[ψ_ED/ψ_NQS*]|²
        ‖ψ_ED‖²           / ‖ψ_NQS‖² = E_{|ψ_NQS|²}[|ψ_ED/ψ_NQS|²]

    ψ_ED(σ) is computed on-the-fly via basis.get_amp(), which applies the
    symmetry projection internally.  Configurations outside the k=(0,0)
    sector return ψ_ED = 0, contributing 0 to the overlap (correct physics).

    Scales to any L: only n_samples NQS evaluations are needed, no full-basis
    enumeration. Accuracy improves with n_samples.

    Parameters
    ----------
    psi_ED        : 1-D real array, shape (Ns,), normalised eigenvector
    quspin_basis  : spin_basis_general used for ED (with symmetry blocks)
    vs            : nk.vqs.MCState, trained variational state
    N             : number of sites
    n_samples     : number of MC samples (default 100 000)
    n_boot        : bootstrap resamples for error bar (default 200)

    Returns
    -------
    F     : float  — fidelity estimate
    std_F : float  — bootstrap standard deviation
    """
    # ── Draw samples from |ψ_NQS|² ───────────────────────────────────────────
    old_n = vs.n_samples
    vs.n_samples = n_samples
    vs.reset()
    samples_nk = np.array(vs.samples).reshape(-1, N)   # (n, N), values ±1
    vs.n_samples = old_n

    # ── log ψ_NQS for each sample ─────────────────────────────────────────────
    log_psi_nqs = np.array(vs.log_value(samples_nk))   # (n,), complex

    # ── Convert NetKet ±1 → QuSpin uint64 state integers ─────────────────────
    # +1 (up) → bit = 1,   −1 (down) → bit = 0
    bits  = ((samples_nk + 1) // 2).astype(np.uint64)
    pow2  = np.array([np.uint64(1) << np.uint64(k) for k in range(N)],
                     dtype=np.uint64)
    state_ints = bits @ pow2                            # (n,) uint64

    # ── ψ_ED(σ) via QuSpin symmetry projection ────────────────────────────────
    # get_amp(states, mode='full_basis') returns C such that ψ_full(σ) = C×ψ_sym(rep(σ)).
    # It does NOT modify the input array; we need representative() separately.
    # basis.states is sorted DESCENDING, so searchsorted must operate on the
    # flipped array and the resulting index must be mirrored.
    C    = quspin_basis.get_amp(state_ints.copy(), mode='full_basis')  # (n,) complex
    reps = quspin_basis.representative(state_ints)                     # (n,) uint64

    flipped = quspin_basis.states[::-1]                                # ascending view
    idx_asc = np.searchsorted(flipped, reps.astype(quspin_basis.states.dtype))
    rep_idx = len(quspin_basis.states) - 1 - idx_asc
    rep_idx = np.clip(rep_idx, 0, len(psi_ED) - 1)                    # guard for C=0
    psi_ED_vals = psi_ED[rep_idx] * C                                  # ψ_ED(σ) for each sample

    # ── Ratio w(σ) = ψ_ED(σ) / ψ_NQS(σ)* ────────────────────────────────────
    psi_nqs = np.exp(log_psi_nqs)
    w = psi_ED_vals.astype(complex) / np.conj(psi_nqs)

    valid = np.isfinite(np.abs(w))
    n_valid = valid.sum()
    w = w[valid]
    print(f"    {n_valid}/{n_samples} samples in k=(0,0) sector "
          f"({100 * n_valid / n_samples:.1f}%)")

    # ── Point estimate ────────────────────────────────────────────────────────
    mean_w  = np.mean(w)
    mean_w2 = np.mean(np.abs(w) ** 2)
    F = float(np.abs(mean_w) ** 2 / mean_w2)

    # ── Bootstrap error ───────────────────────────────────────────────────────
    rng = np.random.default_rng(0)
    F_boot = np.empty(n_boot)
    for b in range(n_boot):
        idx      = rng.integers(0, len(w), len(w))
        wb       = w[idx]
        F_boot[b] = float(np.abs(np.mean(wb)) ** 2 / np.mean(np.abs(wb) ** 2))
    std_F = float(np.std(F_boot))

    return F, std_F

def fidelity_exact(psi_ED, quspin_basis, vs, N):
    """
    Exact fidelity F = |⟨ψ_ED | ψ_NQS⟩|² by full enumeration of Sz=0 configs.

    Only feasible for small L (N ≤ 20 or so). For L=4: 12 870 configs;
    for L=6: ~9.1 B configs — use fidelity_sampling instead.

    Steps
    -----
    1. Expand psi_ED from the symmetry-reduced basis to the full 2^N Hilbert
       space via quspin_basis.get_vec().
    2. Build the Sz=0 sector (no symmetry) and index into the 2^N vector to
       get ψ_ED on every Sz=0 Fock state.
    3. Evaluate the NQS on all Sz=0 configs, normalise, and take the overlap.

    Parameters
    ----------
    psi_ED       : 1-D real array, shape (Ns,), normalised in the sym. basis
    quspin_basis : spin_basis_general used for ED (with symmetry blocks)
    vs           : nk.vqs.MCState
    N            : number of sites

    Returns
    -------
    F : float — exact fidelity
    """
    # Expand symmetry-sector eigenvector → full 2^N vector (zeros outside Sz=0)
    psi_full_2N = quspin_basis.get_vec(psi_ED, sparse=False)   # length 2^N

    # Full Sz=0 basis (no symmetry reduction) to enumerate all Sz=0 configs
    basis_full  = spin_basis_general(N, Nup=N // 2)
    psi_ED_full = psi_full_2N[basis_full.states.astype(int)]   # (Ns_full,)

    # Convert QuSpin state integers → NetKet ±1 configs
    # QuSpin: bit k = 1 → spin-up at site k;  NetKet: +1 = up, -1 = down
    state_ints = basis_full.states.astype(np.uint64)           # (Ns_full,)
    bits = ((state_ints[:, None] >> np.arange(N, dtype=state_ints.dtype)) & 1).astype(np.float64)
    configs_nk = bits * 2.0 - 1.0                              # (Ns_full, N), values ±1

    # Evaluate NQS amplitudes on all Sz=0 configs and normalise
    log_psi_nqs  = np.array(vs.log_value(configs_nk))          # (Ns_full,), complex
    psi_nqs      = np.exp(log_psi_nqs)
    psi_nqs_norm = psi_nqs / np.linalg.norm(psi_nqs)

    overlap = np.dot(psi_ED_full.conj(), psi_nqs_norm)
    return float(np.abs(overlap) ** 2)



print()
print("=" * 60)
print("  STEP 3 — Fidelity F = |⟨ψ_ED | ψ_NQS⟩|²")
print("=" * 60)

if N<=20:
    print("  Computing exact fidelity (full enumeration) ...")
    F_exact = fidelity_exact(psi_ED, basis, vstate, N)
    print(f"  Exact fidelity   = {F_exact:.6f}")

print(f"  Drawing {N_SAMPLES_FIDELITY:,} samples from |ψ_NQS|² ...")
F_sampl, std_F = fidelity_sampling(psi_ED, basis, vstate, N,
                              n_samples=N_SAMPLES_FIDELITY, n_boot=200)

print()
print(f"  Sampling fidelity= {F_sampl:.6f}  ±  {std_F:.1e}  (bootstrap)")
print(f"  Infidelity 1−F   = {1.0 - F_sampl:.4e}")
print()
