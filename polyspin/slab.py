"""Slab coexistence for the C-F boundary (H6): half the lattice in C, half in F (same axis), run at fixed
T and zG, and follow the fraction of clusters of each net moment.  The zG at which the interface stands
still is the boundary at that temperature.

Cluster classes by |M_a|/n: C (0), M2 (0.5, one spin reversed), F (1) for n = 4.
"""
import numpy as np

from . import moves
from .model import Params, energy_total
from .rng import make_state
from .tuning import tune_steps


def make_slab(lattice, S_left, S_right, axis=0, frac=0.5):
    """Spins (N,n,3): clusters with coordinate along ``axis`` < frac*L get S_left, the rest S_right."""
    L = lattice.size
    left = lattice.coords[:, axis] < int(round(frac * L))
    spins = np.empty((lattice.N,) + S_left.shape)
    spins[left] = S_left
    spins[~left] = S_right
    return spins


def moment_fractions(spins, lo=0.25, hi=0.75):
    """Fractions of clusters with |M|/n < lo (C), in [lo, hi) (M2), >= hi (F)."""
    M = np.linalg.norm(spins.sum(1), axis=1) / spins.shape[1]
    return np.array([(M < lo).mean(), ((M >= lo) & (M < hi)).mean(), (M >= hi).mean()])


def run_slab(lattice, params: Params, T, S_C, S_F, n_sweeps, record_every=50, seed=0, kappa=20.0,
             n_tune=4000):
    """Single-temperature MC of a slab configuration; returns (sweeps, fractions (k,3), energies (k,))."""
    spins = make_slab(lattice, S_C, S_F)
    p, nbr, beta = params.array(), lattice.nbr, 1.0 / T
    s = make_state(seed, 0)
    d, dr = tune_steps(spins, nbr, p, beta, s, delta0=np.sqrt(T), n_rounds=15, block=20, use_overlap=True)
    c = np.zeros((moves.N_COUNTERS, 2), dtype=np.int64)
    ts, fr, es = [], [], []
    for sw in range(0, n_sweeps, record_every):
        for _ in range(record_every):
            moves.mc_sweep(spins, nbr, p, beta, d, dr, kappa, True, True, True, False, s, c)
        ts.append(sw + record_every)
        fr.append(moment_fractions(spins))
        es.append(energy_total(spins, nbr, p) / lattice.N)
    return np.array(ts), np.array(fr), np.array(es), spins


def interface_velocity(ts, frac_F, skip=0.3):
    """Least-squares slope of the F fraction over the later part of the run (fraction per sweep) and its
    standard error."""
    n = len(ts)
    i0 = int(skip * n)
    x, y = np.asarray(ts[i0:], float), np.asarray(frac_F[i0:], float)
    A = np.stack([x, np.ones_like(x)], 1)
    coef, res, *_ = np.linalg.lstsq(A, y, rcond=None)
    dof = max(len(x) - 2, 1)
    s2 = (np.sum((y - A @ coef) ** 2) / dof) if len(x) > 2 else 0.0
    se = np.sqrt(s2 / np.sum((x - x.mean()) ** 2)) if len(x) > 2 else np.nan
    return float(coef[0]), float(se)
