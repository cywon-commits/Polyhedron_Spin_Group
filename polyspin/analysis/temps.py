"""Temperature-ladder design from measured e(T) and specific heat.

For adjacent temperatures T1 < T2 the exchange exponent Delta = (b1 - b2)(E1 - E2) is Gaussian with
mean -mu, mu = (b1 - b2)(<E2> - <E1>), and variance s^2 = (b1 - b2)^2 (var E1 + var E2), where the energy
variances follow from the specific heat, var E = N T^2 c(T).  The acceptance is then
    A = Phi(-mu/s) + exp(-mu + s^2/2) Phi((mu - s^2)/s).
Since mu and s^2 are extensive, a ladder measured on a small system can be rescaled to a larger N.
"""
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.stats import norm


def swap_acceptance(T1, T2, e, c, N):
    """Predicted acceptance; e(T) energy per cluster, c(T) specific heat per cluster (callables)."""
    b1, b2 = 1.0 / T1, 1.0 / T2
    mu = (b1 - b2) * N * (e(T2) - e(T1))
    var = (b1 - b2) ** 2 * N * (T1 ** 2 * c(T1) + T2 ** 2 * c(T2))
    s = np.sqrt(max(var, 1e-300))
    mu = max(mu, 0.0)
    # second term in the log domain (large s overflows exp otherwise)
    return float(norm.cdf(-mu / s) + np.exp(-mu + s * s / 2 + norm.logcdf((mu - s * s) / s)))


def interpolators(T, e, c):
    """Monotone interpolants of measured per-cluster energy and specific heat (positive floor)."""
    ef = PchipInterpolator(T, e, extrapolate=True)
    cf = PchipInterpolator(T, np.maximum(c, 1e-3), extrapolate=True)
    return ef, cf


def place_temperatures(Tmin, Tmax, e, c, N, target=0.3, max_ratio=1.3, min_ratio=1.002):
    """Greedy ladder: each next temperature is the largest one whose predicted acceptance with the
    previous one is >= target (bisection), capped by max_ratio."""
    Ts = [Tmin]
    while Ts[-1] < Tmax * (1 - 1e-9):
        T1 = Ts[-1]
        lo, hi = T1 * min_ratio, min(T1 * max_ratio, Tmax)
        if swap_acceptance(T1, hi, e, c, N) >= target:
            nxt = hi
        else:
            for _ in range(50):
                mid = 0.5 * (lo + hi)
                if swap_acceptance(T1, mid, e, c, N) >= target:
                    lo = mid
                else:
                    hi = mid
            nxt = lo
        Ts.append(min(nxt, Tmax))
        if len(Ts) > 2000:
            raise RuntimeError("ladder explodes: target acceptance unreachable")
    return np.array(Ts)
