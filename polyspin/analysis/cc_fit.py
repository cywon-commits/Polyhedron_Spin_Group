"""H6 combined fit (decision Oct 13/15): coexistence line zG*(T) = J + a T + b T^2 from slab runs, together with the
Clausius-Clapeyron slope s(T) = dzG*/dT = a + 2 b T measured on single-branch chains.

Rule (ii) for the verdict (applied to the joint fit with b free, the PRIMARY result): the 1-sigma interval of a contains exactly one of the
predictions -> that prediction is supported; two or more -> undetermined; none -> all rejected.  Reported alongside
(auxiliary, not used for the verdict): slab-only with b = 0 and slab-only with b free.
Rule (iii) (pinning), fixed here as my reading of the approved text: let w_p = span of zG values whose interface did not move
(|v| <= 3 se) between the nearest moved runs below/above the crossing.  A temperature is EXCLUDED if w_p > 3 sigma_boot (the fit
would claim a precision finer than the region where nothing moves); otherwise sigma_zG*^2 = sigma_boot^2 + w_p^2 / 12."""
import numpy as np

PRED = {"harmonic + labels": 0.382, "labels only": 0.224, "harmonic only": 0.158}


def crossing_samples(zs, vs, ses, nboot=2000, seed=0):
    """Zero crossings of a linear fit of v(zG) over moved runs (|v| > 3 se), bootstrapped by v ~ N(v, se).
    Returns (samples (nboot,), pinned_width, n_moved)."""
    zs, vs, ses = (np.asarray(x, float) for x in (zs, vs, ses))
    rng = np.random.default_rng(seed)
    moved = np.abs(vs) > 3 * np.maximum(ses, 1e-12)
    if moved.sum() < 2:
        return np.array([]), float("nan"), int(moved.sum())
    out = []
    for _ in range(nboot):
        v = rng.normal(vs[moved], np.maximum(ses[moved], 1e-12))
        A = np.stack([zs[moved], np.ones(moved.sum())], 1)
        (m, c), *_ = np.linalg.lstsq(A, v, rcond=None)
        if m != 0:
            out.append(-c / m)
    out = np.array(out)
    z0 = np.median(out)
    below = zs[moved & (vs < 0) & (zs < z0)] if (vs[moved] < 0).any() else np.array([])
    above = zs[moved & (vs > 0) & (zs > z0)] if (vs[moved] > 0).any() else np.array([])
    # sign convention: v = dF_F/dt; below/above the crossing the sign flips; pinned zG = unmoved ones near the crossing
    lo = zs[moved & (zs < z0)].max() if (moved & (zs < z0)).any() else zs.min()
    hi = zs[moved & (zs > z0)].min() if (moved & (zs > z0)).any() else zs.max()
    pinned = zs[(~moved) & (zs > lo) & (zs < hi)]
    w_p = float(pinned.max() - pinned.min() + (zs[1] - zs[0] if len(zs) > 1 else 0)) if len(pinned) else 0.0
    return out, w_p, int(moved.sum())


def wls(X, y, sig):
    """Weighted least squares; returns (coef, covariance)."""
    W = 1.0 / np.asarray(sig, float) ** 2
    A = X.T @ (X * W[:, None])
    cov = np.linalg.inv(A)
    return cov @ (X.T @ (W * y)), cov


def fit_a(T, zstar, s_zstar, s=None, s_sigma=None, J=1.0, b_free=True):
    """a (and b) from zG*(T) - J = a T + b T^2 (+ slopes s = a + 2 b T).  Returns dict(a, sa, b, sb)."""
    T, y, sz = (np.asarray(x, float) for x in (T, np.asarray(zstar, float) - J, s_zstar))
    if b_free:
        X = np.stack([T, T ** 2], 1)
        if s is not None:
            X = np.vstack([X, np.stack([np.ones_like(T), 2 * T], 1)])
            y = np.concatenate([y, np.asarray(s, float)]); sz = np.concatenate([sz, np.asarray(s_sigma, float)])
        c, cov = wls(X, y, sz)
        return dict(a=float(c[0]), sa=float(np.sqrt(cov[0, 0])), b=float(c[1]), sb=float(np.sqrt(cov[1, 1])))
    X = T[:, None]
    c, cov = wls(X, y, sz)
    return dict(a=float(c[0]), sa=float(np.sqrt(cov[0, 0])), b=0.0, sb=0.0)


def verdict(a, sa, pred=PRED):
    inside = [k for k, v in pred.items() if abs(a - v) <= sa]
    if len(inside) == 1:
        return f"supports '{inside[0]}' ({pred[inside[0]]})"
    if len(inside) >= 2:
        return "undetermined (1-sigma interval contains " + " and ".join(inside) + ")"
    return "all predictions rejected at 1 sigma"
