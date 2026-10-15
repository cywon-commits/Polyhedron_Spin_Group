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


LOCAL_W = 0.2


def crossing_samples(zs, vs, ses, nboot=2000, seed=0, w=LOCAL_W):
    """Zero crossings of v(zG) by a LOCAL linear fit, bootstrapped by v ~ N(v, se) (all runs, seeds included).
    The velocity is not linear over the whole scan (front mobility differs on the two sides: slopes differ by ~2 at T = 0.10), so a global
    fit is biased; the local window keeps the runs with |v| <= w max|v| (w = 0.2, fixed after seeing T = 0.10 only), needs >= 3 distinct zG
    and falls back to the bracketing interpolation otherwise.  Returns (samples, pinned_width, n_moved)."""
    zs, vs, ses = (np.asarray(x, float) for x in (zs, vs, ses))
    rng = np.random.default_rng(seed)
    se_ = np.maximum(ses, 1e-12)
    moved = np.abs(vs) > 3 * se_
    if moved.sum() < 2:
        return np.array([]), float("nan"), int(moved.sum())
    out = []
    for _ in range(nboot):
        v = rng.normal(vs, se_)
        sel = np.abs(v) <= w * np.abs(v).max()
        if len(np.unique(zs[sel])) >= 3:
            m, c = np.polyfit(zs[sel], v[sel], 1)
            if m != 0:
                out.append(-c / m)
                continue
        order = np.argsort(zs)
        vv, zz = v[order], zs[order]
        k = np.where(np.diff(np.sign(vv)) != 0)[0]
        if len(k):
            k = k[0]
            out.append(zz[k] - vv[k] * (zz[k + 1] - zz[k]) / (vv[k + 1] - vv[k]))
    out = np.array(out)
    if len(out) == 0:
        return out, float("nan"), int(moved.sum())
    z0 = np.median(out)
    lo = zs[moved & (zs < z0)].max() if (moved & (zs < z0)).any() else zs.min()
    hi = zs[moved & (zs > z0)].min() if (moved & (zs > z0)).any() else zs.max()
    pinned = zs[(~moved) & (zs > lo) & (zs < hi)]
    step = float(np.min(np.diff(np.unique(zs)))) if len(np.unique(zs)) > 1 else 0.0
    w_p = float(pinned.max() - pinned.min() + step) if len(pinned) else 0.0
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
