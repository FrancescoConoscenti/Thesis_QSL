"""
VMC optimization of the Gutzwiller-projected BCS mean-field state
for the J1-J2 Heisenberg model on the square lattice.

Method:    NetKet MCState + Stochastic Reconfiguration (SR / natural gradient)
Ansatz:    |Psi> = P_G |Phi_BCS>  with  log Psi(σ) = log det[ f(R_i↑ - R_j↓) ]
Reference: Ferrari & Becca, PRB 102, 014417 (2020)
           Hu, Becca, Parola, Sorella, PRB 88, 060402(R) (2013)

Energy convention note:
  NetKet Heisenberg uses Pauli matrices (σ·σ, not S·S = σ·σ/4).
  All energies are divided by 4 to match the physical S=1/2 convention:
  H_phys = J Σ S_i·S_j  with  S = σ/2.

Usage:
    python vmc_optimize_gutzwiller.py [--Lx 4] [--Ly 4] [--J2 0.5]
    python vmc_optimize_gutzwiller.py --scan
"""

import os
os.environ["JAX_ENABLE_X64"] = "1"

import argparse, json, time
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import flax.linen as nn
import optax
import netket as nk


# ---------------------------------------------------------------------------
# BCS pair wave function
# ---------------------------------------------------------------------------

def build_fmat(d1, d4, d5, Lx, Ly):
    """
    N×N pair wave function f_{ij} = f(r_i - r_j) from BCS parameters.

    Pairing structure (Hu et al. 2013, t ≡ 1 gauge):
      ε(k) = -2(cos kx + cos ky)
      Δ(k) = d1*(cos kx - cos ky)                       [d_{x²-y²} 1st-nn]
            + d4*(cos 2kx - cos 2ky)                     [d_{x²-y²} 4th-nn]
            + d5*(sin(kx+2ky)+sin(2kx+ky)
                  -sin(kx-2ky)-sin(2kx-ky))              [d_{xy} 5th-nn]
      f(r) = (1/N) Σ_k (v_k/u_k) e^{ik·r}
    """
    N = Lx * Ly
    kx_vals = 2.0 * jnp.pi * jnp.arange(Lx) / Lx
    ky_vals = 2.0 * jnp.pi * jnp.arange(Ly) / Ly
    kx, ky  = jnp.meshgrid(kx_vals, ky_vals, indexing="ij")
    kx, ky  = kx.reshape(-1), ky.reshape(-1)

    eps   = -2.0 * (jnp.cos(kx) + jnp.cos(ky))
    Delta = (d1 * (jnp.cos(kx) - jnp.cos(ky))
             + d4 * (jnp.cos(2*kx) - jnp.cos(2*ky))
             + d5 * (jnp.sin(kx+2*ky) + jnp.sin(2*kx+ky)
                     - jnp.sin(kx-2*ky) - jnp.sin(2*kx-ky)))

    E     = jnp.sqrt(eps**2 + Delta**2 + 1e-14)
    u     = jnp.sqrt(jnp.clip((1 + eps/E)/2, 0.0, 1.0))
    v_mag = jnp.sqrt(jnp.clip((1 - eps/E)/2, 0.0, 1.0))
    v     = jnp.where(Delta >= 0, v_mag, -v_mag)
    ratio = jnp.where(jnp.abs(u) > 1e-10, v/u, 0.0).astype(jnp.complex128)

    xs = jnp.tile(jnp.arange(Lx), Ly).astype(float)
    ys = jnp.repeat(jnp.arange(Ly), Lx).astype(float)
    f_r = jnp.dot(ratio, jnp.exp(1j*(jnp.outer(kx,xs)+jnp.outer(ky,ys)))) / N

    xi = jnp.tile(jnp.arange(Lx), Ly)
    yi = jnp.repeat(jnp.arange(Ly), Lx)
    return f_r[(xi[:,None]-xi[None,:])%Lx * Ly + (yi[:,None]-yi[None,:])%Ly]


# ---------------------------------------------------------------------------
# Flax model
# ---------------------------------------------------------------------------

class GutzwillerBCS(nn.Module):
    """log Ψ(σ) = log det[f(R_i↑ - R_j↓)]  with P_G enforced by sampler."""
    Lx: int
    Ly: int

    @nn.compact
    def __call__(self, sigma):
        # sigma: (batch, N) with values ±1  (NetKet Pauli convention)
        N, Nup = self.Lx * self.Ly, self.Lx * self.Ly // 2
        d1 = self.param("d1", nn.initializers.constant( 0.8),  ())
        d4 = self.param("d4", nn.initializers.constant( 0.1),  ())
        d5 = self.param("d5", nn.initializers.constant( 0.05), ())
        f  = build_fmat(d1, d4, d5, self.Lx, self.Ly)

        def log_amp(s):
            up = jnp.where(s >  0, size=Nup, fill_value=0)[0]
            dn = jnp.where(s <= 0, size=Nup, fill_value=0)[0]
            sub = f[up][:, dn]
            sgn, loga = jnp.linalg.slogdet(sub)
            return loga + jnp.log(sgn.astype(jnp.complex128) + 0j)

        return jax.vmap(log_amp)(sigma)


# ---------------------------------------------------------------------------
# Hamiltonian
# ---------------------------------------------------------------------------

def make_hamiltonian(hi, Lx, Ly, J1, J2):
    graph_nn  = nk.graph.Square(Lx, pbc=True)
    H_J1      = nk.operator.Heisenberg(hilbert=hi, graph=graph_nn, J=J1)

    seen, edges_nnn = set(), []
    for x in range(Lx):
        for y in range(Ly):
            i = x * Ly + y
            for dx, dy in [(1,1),(1,-1),(-1,1),(-1,-1)]:
                j    = ((x+dx)%Lx)*Ly + ((y+dy)%Ly)
                bond = tuple(sorted((i,j)))
                if bond not in seen:
                    seen.add(bond); edges_nnn.append(bond)

    H_J2 = nk.operator.Heisenberg(
        hilbert=hi, graph=nk.graph.Graph(edges=edges_nnn, n_nodes=Lx*Ly), J=J2
    )
    return H_J1 + H_J2


# ---------------------------------------------------------------------------
# VMC loop
# ---------------------------------------------------------------------------

def run_vmc(Lx=4, Ly=4, J1=1.0, J2=0.5,
            n_samples=2048, n_iter=300,
            sr_diag_shift=0.02, sr_decay=0.98,
            lr=0.02, seed=42):

    N    = Lx * Ly
    CONV = 4.0   # NetKet Pauli → physical S=1/2

    print(f"\n{'='*65}")
    print(f"Gutzwiller-BCS VMC | {Lx}×{Ly} | J2/J1={J2:.3f} | N={N}")
    print(f"n_samples={n_samples}, n_iter={n_iter}, lr={lr}, sr_shift={sr_diag_shift}")
    print(f"{'='*65}")

    hi      = nk.hilbert.Spin(s=0.5, N=N, total_sz=0)
    H       = make_hamiltonian(hi, Lx, Ly, J1, J2)
    model   = GutzwillerBCS(Lx=Lx, Ly=Ly)
    sampler = nk.sampler.MetropolisExchange(
        hi, n_chains=16, graph=nk.graph.Square(Lx, pbc=True)
    )
    vs = nk.vqs.MCState(
        sampler, model, n_samples=n_samples, n_discard_per_chain=32, seed=seed
    )

    print(f"\nInitial params (t≡1 fixed):  ", end="")
    for k, v in vs.parameters.items():
        print(f"{k}={float(v):.3f}", end="  ")
    print()

    optimizer = optax.sgd(lr)
    sr        = nk.optimizer.SR(
        qgt        = nk.optimizer.qgt.QGTJacobianDense(holomorphic=False),
        solver     = nk.optimizer.solver.cholesky,
        diag_shift = sr_diag_shift,
    )
    driver = nk.driver.VMC(H, optimizer, variational_state=vs, preconditioner=sr)

    history = {"energy": [], "energy_err": [], "params": [], "time": []}
    best_E, best_params = np.inf, None
    t0 = time.time()

    print(f"\n{'Step':>5}  {'E/N (phys)':>12}  {'±σ':>8}  {'time':>7}  params")
    print("-" * 70)

    for step in range(n_iter):
        sr.diag_shift = max(sr_diag_shift * (sr_decay ** step), 5e-5)
        driver.run(n_iter=1)
        e       = vs.expect(H)
        E       = e.mean.real / (N * CONV)
        E_err   = e.error_of_mean.real / (N * CONV)
        elapsed = time.time() - t0
        params  = {k: float(v) for k, v in vs.parameters.items()}

        history["energy"].append(E)
        history["energy_err"].append(E_err)
        history["params"].append(params)
        history["time"].append(elapsed)

        if E < best_E:
            best_E, best_params = E, dict(params)

        if step % 10 == 0 or step < 5:
            p = params
            print(f"{step:>5d}  {E:>12.6f}  {E_err:>8.5f}  {elapsed:>6.1f}s"
                  f"  d1={p['d1']:+.4f} d4={p['d4']:+.4f} d5={p['d5']:+.4f}")

    print(f"\n{'='*65}")
    print(f"Best E/N (physical, S=1/2 convention) = {best_E:.6f}")
    print(f"Best params: d1={best_params['d1']:+.5f}  "
          f"d4={best_params['d4']:+.5f}  d5={best_params['d5']:+.5f}")
    print(f"\nLiterature (J2/J1=0.5, Gutzwiller-VMC, TDL):")
    print(f"  E/N ≈ -0.4966  [Capriotti et al. 2001]")
    print(f"  E/N ≈ -0.4940  [Hu et al. 2013]")
    print(f"  (4×4 finite-size corrections add ~+0.02–0.05)")

    return history, best_params, vs


# ---------------------------------------------------------------------------
# J2 scan
# ---------------------------------------------------------------------------

def scan_j2(Lx=4, Ly=4, J2_values=None, n_samples=2048, n_iter=200, seed=42):
    if J2_values is None:
        J2_values = [0.0, 0.3, 0.4, 0.45, 0.5, 0.55, 0.6]
    results = {}
    for J2 in J2_values:
        history, best_params, _ = run_vmc(
            Lx=Lx, Ly=Ly, J1=1.0, J2=J2,
            n_samples=n_samples, n_iter=n_iter, seed=seed,
        )
        results[J2] = {"E_min": min(history["energy"]), "best_params": best_params}

    print("\n\n=== Phase diagram ===")
    print(f"{'J2/J1':>8}  {'E_min/N':>12}  {'d1':>8}  {'d4':>8}  {'d5':>8}")
    print("-" * 55)
    for J2, res in sorted(results.items()):
        p = res["best_params"]
        print(f"{J2:>8.3f}  {res['E_min']:>12.6f}  "
              f"{p['d1']:>+8.4f}  {p['d4']:>+8.4f}  {p['d5']:>+8.4f}")
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--Lx",        type=int,   default=4)
    ap.add_argument("--Ly",        type=int,   default=4)
    ap.add_argument("--J1",        type=float, default=1.0)
    ap.add_argument("--J2",        type=float, default=0.5)
    ap.add_argument("--n_samples", type=int,   default=2048)
    ap.add_argument("--n_iter",    type=int,   default=300)
    ap.add_argument("--lr",        type=float, default=0.02)
    ap.add_argument("--sr_shift",  type=float, default=0.02)
    ap.add_argument("--sr_decay",  type=float, default=0.98)
    ap.add_argument("--seed",      type=int,   default=42)
    ap.add_argument("--scan",      action="store_true")
    ap.add_argument("--save",      type=str,   default=None)
    args = ap.parse_args()

    if args.scan:
        results = scan_j2(
            Lx=args.Lx, Ly=args.Ly,
            J2_values=[0.0, 0.3, 0.4, 0.45, 0.5, 0.55, 0.6],
            n_samples=args.n_samples, n_iter=args.n_iter, seed=args.seed,
        )
        if args.save:
            with open(args.save, "w") as f:
                json.dump({str(k): v for k, v in results.items()}, f, indent=2)
    else:
        history, best_params, vs = run_vmc(
            Lx=args.Lx, Ly=args.Ly, J1=args.J1, J2=args.J2,
            n_samples=args.n_samples, n_iter=args.n_iter,
            sr_diag_shift=args.sr_shift, sr_decay=args.sr_decay,
            lr=args.lr, seed=args.seed,
        )
        if args.save:
            with open(args.save, "w") as f:
                json.dump({"history": history, "best_params": best_params}, f, indent=2)
