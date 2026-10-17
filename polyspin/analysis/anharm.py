"""Anharmonic (T^2) correction to the harmonic + label free energies (decision Oct 17 (d); rule in docs/CD_rules.md).
e(T) = e0 + a T + c T^2  =>  f/T = beta e0 + a ln beta - c T + const, so the free-energy correction of a branch is -c T^2 per cluster."""
import numpy as np
from .cc_fit import wls


def fit_curvature(T, e, err, e0, order=2):
    """Weighted fit of e - e0 = a T + c T^2 (+ d T^3 for order 3).  Returns dict(a, sa, c, sc)."""
    T, y, s = (np.asarray(x, float) for x in (T, np.asarray(e, float) - e0, err))
    X = np.stack([T ** k for k in range(1, order + 1)], 1)
    co, cov = wls(X, y, np.maximum(s, 1e-12))
    return dict(a=float(co[0]), sa=float(np.sqrt(cov[0, 0])), c=float(co[1]), sc=float(np.sqrt(cov[1, 1])))


def solve_T(A, d, dc):
    """Positive root of A/T + d - dc T = 0 (A > 0: energy gain per cluster x(n^2-M^2)/2, d = delta (negative), dc = curvature difference)."""
    if abs(dc) < 1e-12:
        return -A / d
    disc = d * d + 4 * dc * A
    return float((d + np.sqrt(disc)) / (2 * dc)) if disc >= 0 else float("nan")
