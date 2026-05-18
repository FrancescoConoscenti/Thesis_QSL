"""
tJExchangeRule — drop-in replacement for exchange_new_sym.tJExchangeRule,
compatible with JAX 0.10.0 and NetKet 3.21.

Three things the original did that our SpinExchangeRule missed:

1. random_state generates STRICTLY SINGLE-OCCUPANCY initial states
   by random permutation of site indices, without calling the machine.
   (Original called apply_machine inside a sharding context → pvary crash.)

2. transition only proposes swaps between sites with DIFFERENT combined
   spin state (anti-parallel pairs). Swapping parallel pairs does nothing,
   so they are masked out. This keeps the Markov chain efficient.

3. Hastings correction log(n_conn / n_conn_proposed) accounts for the
   fact that the number of hoppable clusters changes after a swap, making
   the proposal distribution asymmetric.

Together 1+2 guarantee the sampler NEVER leaves the D=0 (no double
occupancy) subspace, so the −1e14 penalty in the wavefunction is never
triggered and the Sz saturation is recovered correctly.
"""

import jax
import jax.numpy as jnp
from netket.sampler.rules.base import MetropolisRule
from netket.utils import struct


@jax.jit
def _hoppable_mask(clusters, sigma):
    """
    Returns bool (n_chains, n_clusters): True where sites i and j differ
    in their combined (up + dn) spin state, i.e. the swap would change sigma.

    sigma shape: (n_chains, 2N)
      sigma[..., :N]  = up-spin occupations  ∈ {0,1}
      sigma[..., N:]  = dn-spin occupations  ∈ {0,1}
    """
    N  = sigma.shape[-1] // 2
    si = clusters[:, 0]
    sj = clusters[:, 1]
    same_up = jnp.isclose(sigma[..., si],     sigma[..., sj])
    same_dn = jnp.isclose(sigma[..., N + si], sigma[..., N + sj])
    return ~(same_up & same_dn)


class tJExchangeRule(MetropolisRule):
    """
    Single-occupancy-conserving exchange rule for Gutzwiller VMC.

    Usage:
        rule = tJExchangeRule(graph=g)          # NN edges from NetKet graph
        rule = tJExchangeRule(clusters=arr)     # custom (n_edges, 2) array
        sa   = nk.sampler.MetropolisSampler(hi, n_chains=512, rule=rule)
    """

    clusters: jnp.ndarray = struct.field(pytree_node=True)

    def __init__(self, graph=None, clusters=None):
        if graph is not None:
            self.clusters = jnp.array(list(graph.edges()))
        elif clusters is not None:
            self.clusters = jnp.array(clusters)
        else:
            raise ValueError("Provide either graph= or clusters=.")

    # ── Initial state ─────────────────────────────────────────────────────────

    def random_state(self, sampler, machine, params, state, rng):
        """
        Generate batch_size strictly single-occupancy states.

        Each state: random permutation of site indices [0, N).
          - First N_up indices  → up-spin modes   [0,   N)
          - Next  N_dn indices  → dn-spin modes   [N,  2N)
        No double occupancies, no empty sites (at half-filling).
        Does NOT call the machine → no sharding/pvary issue.
        """
        hi          = sampler.hilbert
        batch_size  = sampler.n_batches
        n_orbitals  = hi.n_orbitals          # N spatial sites
        n_fermions  = hi.n_fermions          # N_up + N_dn
        N_up        = hi.n_fermions_per_spin[0]

        keys = jax.random.split(rng, batch_size)

        def make_one(key):
            perm     = jax.random.permutation(key, jnp.arange(n_orbitals))
            up_sites = perm[:N_up]
            dn_sites = perm[N_up:n_fermions] + n_orbitals   # offset into dn sector
            occupied = jnp.concatenate([up_sites, dn_sites])
            s        = jnp.zeros(hi.size, dtype=jnp.int8)
            return s.at[occupied].set(1)

        return jax.vmap(make_one)(keys)

    # ── Transition kernel ─────────────────────────────────────────────────────

    def transition(self, sampler, machine, parameters, state, key, sigma):
        """
        Propose swapping the full spin state at two neighbouring sites.

        Only hoppable (anti-parallel) pairs are considered. Applies the
        Hastings correction for the asymmetric proposal distribution.
        Single occupancy is preserved by construction (swapping full site
        states cannot create double occupancies).

        Returns: (sigma_proposed, log_prob_correction)
        """
        n_chains = sigma.shape[0]
        N        = sigma.shape[1] // 2
        cl       = self.clusters

        hoppable = _hoppable_mask(cl, sigma)   # (n_chains, n_clusters) bool
        keys     = jax.random.split(key, n_chains)

        @jax.vmap
        def _update(k, s, hop):
            n_conn = hop.sum()

            # Pick a random hoppable cluster
            cluster = jax.random.choice(
                k,
                a=jnp.arange(cl.shape[0]),
                p=hop / (n_conn + 1e-30),
                replace=True,
            )

            si = cl[cluster, 0]
            sj = cl[cluster, 1]

            # Swap full spin state at si ↔ sj (preserves single occupancy)
            sp = s.at[si].set(s[sj]).at[sj].set(s[si])
            sp = sp.at[N + si].set(s[N + sj]).at[N + sj].set(s[N + si])

            # Hastings correction: proposal is not symmetric because n_conn
            # may differ before and after the swap
            hop_p          = _hoppable_mask(cl, sp)
            n_conn_p       = hop_p.sum()
            log_prob_corr  = jnp.log(n_conn + 1e-30) - jnp.log(n_conn_p + 1e-30)

            return sp, log_prob_corr

        return _update(keys, sigma, hoppable)
