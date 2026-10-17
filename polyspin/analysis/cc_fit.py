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
    or a joint local fit is biased; the window keeps the runs with |v| <= w max|v| (w = 0.2, fixed after seeing T = 0.10 only), each side needs >= 2
    distinct zG, otherwise the bracketing interpolation is used.  Returns (samples, pinned_width, n_moved)."""
    zs, vs, ses = (np.asarray(x, float) for x in (zs, vs, ses))
    rng = np.random.default_rng(seed)
    se_ = np.maximum(ses, 1e-12)
    moved = np.abs(vs) > 3 * se_
    if moved.sum() < 2:
        return np.array([]), float("nan"), int(moved.sum())
    out = []
    n = len(zs)
    for _ in range(nboot):
        idx = rng.integers(0, n, n)                                    # run-level resampling (seed variation) ...
        zb, v = zs[idx], rng.normal(vs[idx], se_[idx])                  # ... plus measurement noise
        sel = np.abs(v) <= w * np.abs(v).max()
        zeros = []
        for side in (v < 0, v > 0):                                     # one-sided fits: the two slopes differ, so a joint fit is biased
            ss = sel & side
            if len(np.unique(zb[ss])) >= 2:
                m, c = np.polyfit(zb[ss], v[ss], 1)
                if m != 0:
                    zeros.append(-c / m)
        if len(zeros) == 2:
            out.append(float(np.mean(zeros)))
            continue
        order = np.argsort(zb)
        vv, zz = v[order], zb[order]
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


def crossing_with_systematics(zs, vs, ses, windows=(0.1, 0.2, 0.3), nboot=2000):
    """(zG*, sigma_total, sigma_boot, systematic, pinned_width, n_moved): the central value is the w = LOCAL_W bootstrap median; the
    window-choice systematic is half the range of the medians over ``windows`` (rule fixed before T = 0.125-0.20 were analysed);
    sigma_total^2 = sigma_boot^2 + systematic^2 + w_p^2 / 12."""
    res = {w: crossing_samples(zs, vs, ses, nboot=nboot, w=w) for w in windows}
    smp, wp, nm = res[LOCAL_W]
    if len(smp) == 0:
        return float("nan"), float("nan"), float("nan"), float("nan"), wp, nm
    med = [np.median(res[w][0]) for w in windows if len(res[w][0])]
    sysm = 0.5 * (max(med) - min(med))
    sb = float(smp.std())
    return float(np.median(smp)), float(np.sqrt(sb ** 2 + sysm ** 2 + wp ** 2 / 12.0)), sb, float(sysm), wp, nm


def crossing_robust(zs, vs, ses, windows=(0.1, 0.2, 0.3), nboot=2000):
    """New principle (decision Oct 16): the PRIMARY zero-crossing error is the half-width of the 16-84% percentile interval of the bootstrap
    samples (robust to the rare two-point extrapolation resamples that inflate the standard deviation), not the standard deviation.
    Returns dict(zero, sigma_total, sigma_pct, sigma_std, systematic, pinned_width, n_moved): sigma_total^2 = sigma_pct^2 + systematic^2 + w_p^2/12
    (same window systematic and pinning term as crossing_with_systematics); sigma_std is reported alongside.  Not applied retroactively to results
    obtained before this principle (H6, the 2D M2 wedge)."""
    res = {w: crossing_samples(zs, vs, ses, nboot=nboot, w=w) for w in windows}
    smp, wp, nm = res[LOCAL_W]
    if len(smp) == 0:
        return dict(zero=float("nan"), sigma_total=float("nan"), sigma_pct=float("nan"), sigma_std=float("nan"), systematic=float("nan"), pinned_width=wp, n_moved=nm)
    lo, hi = np.percentile(smp, [16, 84])
    spct = 0.5 * float(hi - lo)
    med = [np.median(res[w][0]) for w in windows if len(res[w][0])]
    sysm = 0.5 * (max(med) - min(med))
    return dict(zero=float(np.median(smp)), sigma_total=float(np.sqrt(spct ** 2 + sysm ** 2 + wp ** 2 / 12.0)), sigma_pct=spct,
                sigma_std=float(smp.std()), systematic=float(sysm), pinned_width=wp, n_moved=nm)


def wedge_verdict(T1, s1, T2, s2, T1_pred, T2_pred, tol=0.03):
    """M2-wedge rule (docs/CD_rules.md): 'confirmed' iff T1 < T2 and T2 - T1 > 2 sqrt(s1^2 + s2^2); 'location consistent' iff both within tol of the
    predictions.  Returns dict(confirmed, location)."""
    return dict(confirmed=bool(T1 < T2 and (T2 - T1) > 2 * np.hypot(s1, s2)),
                location=bool(abs(T1 - T1_pred) <= tol and abs(T2 - T2_pred) <= tol))


SM_WIDTHS = (0.2, 0.3, 0.4, 0.5, 0.75, 1.0)


def crossing_seedmean(Ts, vs, nboot=4000, seed=0, widths=SM_WIDTHS):
    """Seed-mean estimator (decision Oct 17 (a); rule in docs/CD_rules.md), for scans with several seeds per temperature.
    1. Average v over the seeds at each T (the seed mean is the unit of the curve).
    2. Window from the full-data seed-mean curve: the smallest w in ``widths`` such that the temperatures with |mean| <= w max|mean| are at
       least 4 distinct values and include both signs; the window is widened until it does.  NEVER falls back to a first-sign-change interpolation.
    3. Bootstrap = resample the seeds within each T, take means, one linear fit over the window temperatures, zero = -c/k.
    4. sigma_pct = half of the 16-84% range (primary), sigma_std reported; systematic = half the range of the full-data zeros at the chosen and the
       next two widths; sigma_total^2 = sigma_pct^2 + systematic^2.
    Returns dict(zero, sigma_total, sigma_pct, sigma_std, systematic, width, n_window)."""
    Ts, vs = np.asarray(Ts, float), np.asarray(vs, float)
    keys = np.unique(np.round(Ts, 6))
    groups = [vs[np.isclose(Ts, k)] for k in keys]
    means = np.array([g.mean() for g in groups])
    amax = np.abs(means).max()

    def window(w):
        return np.where(np.abs(means) <= w * amax)[0]

    def ok(ix):
        return len(ix) >= 4 and (means[ix] > 0).any() and (means[ix] < 0).any()

    chosen = None
    for j, w in enumerate(widths):
        if ok(window(w)):
            chosen = j; break
    if chosen is None:
        return dict(zero=float("nan"), sigma_total=float("nan"), sigma_pct=float("nan"), sigma_std=float("nan"), systematic=float("nan"), width=float("nan"), n_window=0)

    def zero_of(m, ix):
        k, c = np.polyfit(keys[ix], m[ix], 1)
        return -c / k

    ix = window(widths[chosen])
    rng = np.random.default_rng(seed)
    z = np.array([zero_of(np.array([rng.choice(g, len(g)).mean() for g in groups]), ix) for _ in range(nboot)])
    z = z[np.isfinite(z)]
    lo, med, hi = np.percentile(z, [16, 50, 84])
    spct = 0.5 * float(hi - lo)
    full = [zero_of(means, window(widths[j])) for j in range(chosen, min(chosen + 3, len(widths))) if ok(window(widths[j]))]
    syst = 0.5 * (max(full) - min(full))
    return dict(zero=float(zero_of(means, ix)), sigma_total=float(np.hypot(spct, syst)), sigma_pct=spct, sigma_std=float(z.std()), systematic=float(syst),
                width=float(widths[chosen]), n_window=int(len(ix)))
