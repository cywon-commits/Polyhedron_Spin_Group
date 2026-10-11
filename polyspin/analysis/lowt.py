"""Low-temperature equipartition slope: <E>/N - E0 = a T + b T^2 (a = 4 for 8 harmonic
transverse degrees of freedom per cluster, minus finite-size zero modes)."""
import numpy as np

from .. import moves
from ..groundstate import phase_of_gs
from ..lattice import Lattice
from ..model import Params, energy_total
from ..rng import make_state
from ..tuning import tune_steps


def block_mean(x, nb=20):
    x = np.asarray(x)
    m = len(x) // nb
    b = x[: m * nb].reshape(nb, m).mean(1)
    return b.mean(), b.std(ddof=1) / np.sqrt(nb)


def energy_above_gs(S0, E0, p, lat, T, neq, nmeas, seed):
    spins = np.tile(S0, (lat.N, 1, 1)).copy()
    s = make_state(seed, int(round(T * 1e5)))
    d, dr = tune_steps(spins, lat.nbr, p, 1 / T, s, delta0=np.sqrt(T))
    c = np.zeros((moves.N_COUNTERS, 2), dtype=np.int64)
    for _ in range(neq):
        moves.mc_sweep(spins, lat.nbr, p, 1 / T, d, dr, 20.0, True, True, False, False, s, c)
    e = np.empty(nmeas)
    for k in range(nmeas):
        moves.mc_sweep(spins, lat.nbr, p, 1 / T, d, dr, 20.0, True, True, False, False, s, c)
        e[k] = energy_total(spins, lat.nbr, p) / lat.N
    return block_mean(e - E0)


def measure_slope(zG, zH, K, Ts=(0.002, 0.004, 0.006, 0.008, 0.01), L=8, dim=2, S0=None,
                  neq=4000, nmeas=40000, seed=1):
    """Weighted fit of <e> - E0 = a T + b T^2.  Returns dict(a, a_err, b, T, y, err, phase)."""
    lat = Lattice(dim, L)
    P = Params.from_z(zG, zH, z=lat.z, K=K)
    p = P.array()
    ph, E0, Sgs = phase_of_gs(P)
    if S0 is None:
        S0 = Sgs
    Ts = np.asarray(Ts, float)
    ys, errs = zip(*(energy_above_gs(S0, E0, p, lat, T, neq, nmeas, seed) for T in Ts))
    ys, errs = np.array(ys), np.array(errs)
    A = np.stack([Ts, Ts ** 2], 1) / errs[:, None]
    coef, *_ = np.linalg.lstsq(A, ys / errs, rcond=None)
    cov = np.linalg.inv(A.T @ A)
    return dict(a=coef[0], a_err=float(np.sqrt(cov[0, 0])), b=coef[1], T=Ts, y=ys, err=errs,
                phase=ph, N=lat.N)
