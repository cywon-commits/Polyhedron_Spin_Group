"""Parallel tempering (replica exchange) and the hysteresis scan.

All replicas live in one array ``spins[R, N, n, 3]``; replica ``r`` owns an xoshiro state, so the
sweeps run in parallel (numba prange) and give bit-identical results for any thread count.  The
exchange step uses its own xoshiro state, hence a run is exactly reproducible from its seed.

Conventions: ``T_index`` runs from the coldest (0) to the hottest (R-1) temperature;
``rep_at[t]`` is the replica currently at temperature t and ``temp_of[r]`` the inverse map.
"""
import numpy as np
from numba import njit, prange

from . import moves
from .model import Params, energy_total, random_spins
from .rng import make_state, rand


# ------------------------------------------------------------------ kernels
@njit(parallel=True, cache=True)
def run_sweeps(spins, nbr, p, betas, deltas, delta_rs, kappa, use_single, use_rot, use_overlap,
               use_perm, states, counters, nsweeps, energies):
    """Advance every replica ``nsweeps`` sweeps at its own (fixed) beta/delta; write the total energy of
    each replica to ``energies``.  counters (R,3,2) is zeroed on entry."""
    R = spins.shape[0]
    for r in prange(R):
        counters[r, :, :] = 0
        for _ in range(nsweeps):
            moves.mc_sweep(spins[r], nbr, p, betas[r], deltas[r], delta_rs[r], kappa, use_single,
                           use_rot, use_overlap, use_perm, states[r], counters[r])
        energies[r] = energy_total(spins[r], nbr, p)


@njit(cache=True)
def swap_pass(energies_by_T, betas, rep_at, temp_of, parity, state, att, acc):
    """One exchange pass over neighbouring temperature pairs (t, t+1) with t % 2 == parity.
    energies_by_T[t] is the energy of the replica at temperature t."""
    R = len(betas)
    for t in range(parity, R - 1, 2):
        d = (betas[t] - betas[t + 1]) * (energies_by_T[t] - energies_by_T[t + 1])
        att[t] += 1
        if d >= 0.0 or rand(state) < np.exp(d):
            a, b = rep_at[t], rep_at[t + 1]
            rep_at[t], rep_at[t + 1] = b, a
            temp_of[b], temp_of[a] = t, t + 1
            tmp = energies_by_T[t]
            energies_by_T[t] = energies_by_T[t + 1]
            energies_by_T[t + 1] = tmp
            acc[t] += 1


# ------------------------------------------------------------------ engine
class ParallelTempering:
    def __init__(self, lattice, params: Params, temps, seed=0, init="random", ref_state=None,
                 kappa=20.0, use_overlap=True, use_perm=False, delta0=None):
        self.lat, self.P = lattice, params
        self.p = params.array()
        self.temps = np.asarray(temps, float)
        if np.any(np.diff(self.temps) <= 0):
            raise ValueError("temperatures must be strictly increasing")
        self.betas = 1.0 / self.temps
        R, N, n = len(self.temps), lattice.N, params.n
        self.R = R
        self.kappa, self.use_overlap, self.use_perm = float(kappa), bool(use_overlap), bool(use_perm)
        rng = np.random.default_rng([seed, 7919])
        if init == "random":
            self.spins = random_spins((R, N, n), rng)
        elif init == "ordered":
            if ref_state is None:
                raise ValueError("init='ordered' needs ref_state (n,3)")
            self.spins = np.tile(np.asarray(ref_state, float), (R, N, 1, 1)).copy()
        else:
            raise ValueError(init)
        self.states = np.stack([make_state(seed, r + 1) for r in range(R)])
        self.swap_state = make_state(seed, 0)
        self.rep_at = np.arange(R, dtype=np.int64)
        self.temp_of = np.arange(R, dtype=np.int64)
        self.nbr = lattice.nbr
        d0 = np.sqrt(self.temps) if delta0 is None else np.full(R, delta0)
        self.delta = np.minimum(d0 * 1.5, 1.5)     # per temperature index
        self.delta_r = np.minimum(d0, 0.8)
        self.energies = np.zeros(R)                # per replica
        self.counters = np.zeros((R, moves.N_COUNTERS, 2), dtype=np.int64)
        self.acc_T = np.zeros((R, moves.N_COUNTERS, 2), dtype=np.int64)   # accumulated per temperature
        self.swap_att = np.zeros(R, dtype=np.int64)
        self.swap_acc = np.zeros(R, dtype=np.int64)
        self.parity = 0
        self.sweeps_done = 0
        # replica flow / round trips: label 1 = last visited coldest, 2 = last visited hottest
        self.label = np.zeros(R, dtype=np.int64)
        self.round_trips = 0
        self.up_fraction_sum = np.zeros(R)
        self.up_fraction_n = np.zeros(R)

    # ------------------------------------------------------------- stepping
    def _betas_by_rep(self):
        return self.betas[self.temp_of]

    def sweep_block(self, nsweeps):
        """``nsweeps`` sweeps for every replica; refreshes self.energies and the per-T counters."""
        run_sweeps(self.spins, self.nbr, self.p, self._betas_by_rep(), self.delta[self.temp_of],
                   self.delta_r[self.temp_of], self.kappa, True, True, self.use_overlap, self.use_perm,
                   self.states, self.counters, nsweeps, self.energies)
        self.acc_T[self.temp_of] += self.counters
        self.sweeps_done += nsweeps

    def exchange(self):
        e_by_T = self.energies[self.rep_at].copy()
        swap_pass(e_by_T, self.betas, self.rep_at, self.temp_of, self.parity, self.swap_state,
                  self.swap_att, self.swap_acc)
        self.parity ^= 1
        # round-trip bookkeeping
        cold, hot = self.rep_at[0], self.rep_at[-1]
        if self.label[cold] == 2:
            self.round_trips += 1
        self.label[cold] = 1
        self.label[hot] = 2
        lab = self.label[self.rep_at]
        self.up_fraction_sum += (lab == 1)
        self.up_fraction_n += (lab != 0)

    def swap_rates(self):
        return self.swap_acc[:-1] / np.maximum(self.swap_att[:-1], 1)

    def reset_swap_stats(self):
        self.swap_att[:] = 0
        self.swap_acc[:] = 0

    def acceptance(self):
        """(R,3) acceptance rates (single, rotation, overlap) per temperature since last reset."""
        a = self.acc_T
        return a[:, :, 0] / np.maximum(a[:, :, 1], 1)

    def reset_acceptance(self):
        self.acc_T[:] = 0

    def tune(self, target=(0.3, 0.5), gain=1.5, max_delta=2.0, max_delta_r=1.0):
        """Adjust per-temperature step sizes from the accumulated acceptance (equilibration only)."""
        acc = self.acceptance()
        mid = 0.5 * (target[0] + target[1])
        for t in range(self.R):
            a = acc[t]
            if self.acc_T[t, 0, 1] and not (target[0] <= a[0] <= target[1]):
                self.delta[t] = min(self.delta[t] * np.exp(gain * (a[0] - mid)), max_delta)
            if self.acc_T[t, 1, 1] and not (target[0] <= a[1] <= target[1]):
                self.delta_r[t] = min(self.delta_r[t] * np.exp(gain * (a[1] - mid)), max_delta_r)
        self.reset_acceptance()

    # ------------------------------------------------------------- accessors
    def spins_at_T(self, t):
        return self.spins[self.rep_at[t]]

    def energy_by_T(self):
        return self.energies[self.rep_at]

    def full_energy_check(self):
        """Debug: total energies recomputed from scratch (should equal self.energies)."""
        return np.array([energy_total(self.spins[r], self.nbr, self.p) for r in range(self.R)])

    # ------------------------------------------------------------- checkpoint
    def state_dict(self):
        return dict(spins=self.spins, states=self.states, swap_state=self.swap_state, rep_at=self.rep_at,
                    temp_of=self.temp_of, delta=self.delta, delta_r=self.delta_r, energies=self.energies,
                    swap_att=self.swap_att, swap_acc=self.swap_acc, parity=np.array(self.parity),
                    sweeps_done=np.array(self.sweeps_done), label=self.label,
                    round_trips=np.array(self.round_trips), acc_T=self.acc_T,
                    up_fraction_sum=self.up_fraction_sum, up_fraction_n=self.up_fraction_n)

    def load_state_dict(self, d):
        for k in ("spins", "states", "swap_state", "rep_at", "temp_of", "delta", "delta_r", "energies",
                  "swap_att", "swap_acc", "label", "acc_T", "up_fraction_sum", "up_fraction_n"):
            getattr(self, k)[...] = d[k]
        self.parity = int(d["parity"])
        self.sweeps_done = int(d["sweeps_done"])
        self.round_trips = int(d["round_trips"])


# ------------------------------------------------------------------ hysteresis scan
def hysteresis_scan(lattice, make_params, values, T, start_spins, n_equil, n_meas, seed=0, kappa=20.0,
                    delta=None, delta_r=None, observe=None):
    """Quasi-static parameter sweep at fixed temperature T.

    ``make_params(v)`` returns the Params for scan value v (e.g. a zG).  For every value the previous
    final configuration is the starting point (no re-initialisation): n_equil sweeps, then n_meas
    sweeps during which ``observe(spins, params) -> dict`` is evaluated after each sweep and averaged.
    Step sizes are tuned once at the first point and then frozen.  Returns a list of dicts
    (value, energy per cluster, observables..., final spins).
    """
    from .tuning import tune_steps

    spins = np.array(start_spins, float).copy()
    state = make_state(seed, 0)
    nbr = lattice.nbr
    out = []
    beta = 1.0 / T
    first = True
    for v in values:
        P = make_params(v)
        p = P.array()
        if first:
            if delta is None:
                delta, delta_r = tune_steps(spins, nbr, p, beta, state, delta0=np.sqrt(T),
                                            use_overlap=True)
            first = False
        c = np.zeros((moves.N_COUNTERS, 2), dtype=np.int64)
        for _ in range(n_equil):
            moves.mc_sweep(spins, nbr, p, beta, delta, delta_r, kappa, True, True, True, False, state, c)
        acc, es = {}, []
        for _ in range(n_meas):
            moves.mc_sweep(spins, nbr, p, beta, delta, delta_r, kappa, True, True, True, False, state, c)
            es.append(energy_total(spins, nbr, p) / lattice.N)
            if observe is not None:
                for k, x in observe(spins, P).items():
                    acc.setdefault(k, []).append(x)
        rec = dict(value=v, e=float(np.mean(es)), e_std=float(np.std(es)),
                   final=spins.copy(), **{k: float(np.mean(x)) for k, x in acc.items()})
        out.append(rec)
    return out
