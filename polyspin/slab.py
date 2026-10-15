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


def make_slab(lattice, S_left, S_right, axis=0, frac=0.5, kind="axis"):
    """Spins (N,n,3): clusters with coordinate along ``axis`` < frac*L get S_left, the rest S_right.
    kind="diag": the interface runs along a lattice diagonal ((sum of coordinates) mod L < frac*L)."""
    L = lattice.size
    if kind == "diag":
        left = (lattice.coords.sum(1) % L) < int(round(frac * L))
    else:
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
             n_tune=4000, kind="axis"):
    """Single-temperature MC of a slab configuration; returns (sweeps, fractions (k,3), energies (k,))."""
    spins = make_slab(lattice, S_C, S_F, kind=kind)
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


def cubic_energy_per_cluster(spins, nbr, params: Params):
    """E3/N: the G x^3 inter-cluster term (all other couplings switched off).  The Hamiltonian is linear in G, so
    d H / d zG = E3 / zG exactly (used in the Clausius-Clapeyron slope of the C|F boundary)."""
    p3 = np.array([0.0, 0.0, 0.0, params.G, 0.0, 0.0])
    return energy_total(spins, nbr, p3) / len(spins)


def run_branch(lattice, params: Params, T, S0, n_equil, n_meas, seed=0, kappa=20.0, record_every=20, n_blocks=10):
    """Single-temperature chain started in the ordered state S0 (one branch, no exchange): returns dict with <E>/N and
    <dH/dzG>/N = <E3>/(N zG) with block errors, and the moment-class fractions (C, M2, F) averaged over the run (a
    branch that stays in its class has fraction ~1 in its own class)."""
    spins = np.empty((lattice.N,) + S0.shape)
    spins[:] = S0
    p, nbr, beta = params.array(), lattice.nbr, 1.0 / T
    s = make_state(seed, 0)
    d, dr = tune_steps(spins, nbr, p, beta, s, delta0=np.sqrt(T), n_rounds=15, block=20, use_overlap=True)
    c = np.zeros((moves.N_COUNTERS, 2), dtype=np.int64)
    for _ in range(n_equil):
        moves.mc_sweep(spins, nbr, p, beta, d, dr, kappa, True, True, True, False, s, c)
    E, E3, fr = [], [], []
    for _ in range(0, n_meas, record_every):
        for _ in range(record_every):
            moves.mc_sweep(spins, nbr, p, beta, d, dr, kappa, True, True, True, False, s, c)
        E.append(energy_total(spins, nbr, p) / lattice.N)
        E3.append(cubic_energy_per_cluster(spins, nbr, params))
        fr.append(moment_fractions(spins))
    E, E3, fr = np.array(E), np.array(E3), np.array(fr)
    dH = E3 / params.zG
    def blk(x):
        b = len(x) // n_blocks
        m = x[: b * n_blocks].reshape(n_blocks, b).mean(1)
        return float(m.mean()), float(m.std(ddof=1) / np.sqrt(n_blocks))
    out = dict(T=T, zG=params.zG)
    out["E"], out["E_err"] = blk(E)
    out["dH"], out["dH_err"] = blk(dH)
    out["frac"] = fr.mean(0).tolist()
    out["frac_min_own"] = float(fr.min(0).max())
    return out


# ---------------------------------------------------------------------------------------------
# ordered | disordered slab (first-order nematic transition of cut B; decision Oct 13: slab tool also applied to B 3D zH4 = 1.0)
def layer_nematic(spins, lattice, axis=0):
    """Nematic order of every layer perpendicular to ``axis``: the largest eigenvalue of the layer-averaged
    traceless tensor Q = < (3 s s^T - 1) / 2 >  (all spins of all clusters).  ~1 for aligned collinear clusters, ~1/sqrt(layer spins) for a
    disordered layer."""
    L = lattice.size
    Qc = (1.5 * np.einsum("cni,cnj->cij", spins, spins) - 0.5 * spins.shape[1] * np.eye(3)) / spins.shape[1]       # per cluster
    out = np.empty(L)
    layer = lattice.coords[:, axis]
    for x in range(L):
        Qb = Qc[layer == x].mean(0)
        out[x] = np.linalg.eigvalsh(Qb)[-1]
    return out


def make_ordered_disordered(lattice, S_ordered, axis=0, frac=0.5, seed=0):
    """Clusters with coordinate along ``axis`` < frac L ordered (S_ordered), the others uniformly random spins."""
    rng = np.random.default_rng(seed)
    L = lattice.size
    left = lattice.coords[:, axis] < int(round(frac * L))
    spins = rng.normal(size=(lattice.N,) + S_ordered.shape)
    spins /= np.linalg.norm(spins, axis=2, keepdims=True)
    spins[left] = S_ordered
    return spins


def run_slab_order(lattice, params: Params, T, S_ordered, n_sweeps, record_every=50, seed=0, kappa=20.0, axis=0,
                   thr=0.45, n_tune=4000):
    """Ordered|disordered slab at fixed T.  Returns (sweeps, ordered fraction of layers (S_layer > thr), layer-resolved
    nematic order (k, L), energies per cluster (k,))."""
    spins = make_ordered_disordered(lattice, S_ordered, axis=axis, seed=seed)
    p, nbr, beta = params.array(), lattice.nbr, 1.0 / T
    s = make_state(seed, 0)
    d, dr = tune_steps(spins, nbr, p, beta, s, delta0=np.sqrt(T), n_rounds=15, block=20, use_overlap=True)
    c = np.zeros((moves.N_COUNTERS, 2), dtype=np.int64)
    ts, frac, lay, es = [], [], [], []
    for sw in range(0, n_sweeps, record_every):
        for _ in range(record_every):
            moves.mc_sweep(spins, nbr, p, beta, d, dr, kappa, True, True, True, False, s, c)
        q = layer_nematic(spins, lattice, axis)
        ts.append(sw + record_every); lay.append(q); frac.append((q > thr).mean())
        es.append(energy_total(spins, nbr, p) / lattice.N)
    return np.array(ts), np.array(frac), np.array(lay), np.array(es), spins
