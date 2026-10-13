"""Finite-size analysis helpers for H1/H3 (first-order tests) and H2 (scale dependence)."""
import numpy as np


def loglog_slope(N, y):
    """Exponent a in y ~ N^a with its standard error (least squares in log-log)."""
    x, v = np.log(np.asarray(N, float)), np.log(np.asarray(y, float))
    if len(x) < 2:
        return float("nan"), float("nan")
    A = np.stack([x, np.ones_like(x)], 1)
    coef, res, *_ = np.linalg.lstsq(A, v, rcond=None)
    dof = len(x) - 2
    se = float(np.sqrt(res[0] / dof / np.sum((x - x.mean()) ** 2))) if dof > 0 and len(res) else float("nan")
    return float(coef[0]), se


def energy_cumulant(e):
    """Energy cumulant V = 1 - <e^4> / (3 <e^2>^2) of the raw energy per cluster (Challa-Landau-Binder):
    V -> 2/3 away from a first-order transition and has a minimum below 2/3 at it."""
    e = np.asarray(e, float)
    return 1.0 - np.mean(e ** 4) / (3.0 * np.mean(e ** 2) ** 2)


def two_gaussian_fit(x, iters=300, tau=0.5):
    """EM fit of a two-component 1D Gaussian mixture.  Returns dict(w, mu, sd, loglik) of the fit and the
    BIC difference BIC(1) - BIC(2) (> 10: bimodal), the peak separation |mu1 - mu2| and the valley/peak
    ratio of the fitted density.  ``dBIC`` uses the effective sample size N_eff = N / (2 tau) (tau: integrated
    autocorrelation time in samples, 0.5 = uncorrelated): the log-likelihoods are scaled by N_eff / N and the
    BIC penalty uses ln N_eff.  ``dBIC_raw`` is the uncorrected value for comparison."""
    x = np.asarray(x, float)
    n = len(x)
    mu = np.array([np.percentile(x, 25), np.percentile(x, 75)])
    sd = np.array([x.std() / 2 + 1e-12] * 2)
    w = np.array([0.5, 0.5])
    for _ in range(iters):
        p = np.stack([w[k] * np.exp(-0.5 * ((x - mu[k]) / sd[k]) ** 2) / (sd[k] * np.sqrt(2 * np.pi)) for k in range(2)], 1)
        s = p.sum(1, keepdims=True) + 1e-300
        r = p / s
        nk = r.sum(0) + 1e-12
        w = nk / n
        mu = (r * x[:, None]).sum(0) / nk
        sd = np.sqrt((r * (x[:, None] - mu) ** 2).sum(0) / nk) + 1e-9
    ll2 = np.log(s[:, 0]).sum()
    ll1 = n * (-0.5 * np.log(2 * np.pi * x.var()) - 0.5) if x.var() > 0 else -np.inf       # n * (per-sample max loglik)
    bic1, bic2 = -2 * ll1 + 2 * np.log(n), -2 * ll2 + 5 * np.log(n)
    f = 1.0 / (2.0 * max(tau, 0.5))
    n_eff = max(n * f, 2.0)
    bic1c, bic2c = -2 * ll1 * f + 2 * np.log(n_eff), -2 * ll2 * f + 5 * np.log(n_eff)
    grid = np.linspace(mu.min(), mu.max(), 200)
    dens = sum(w[k] * np.exp(-0.5 * ((grid - mu[k]) / sd[k]) ** 2) / sd[k] for k in range(2))
    ratio = float(dens.min() / min(dens[0], dens[-1])) if mu.min() != mu.max() else 1.0
    return dict(w=w, mu=mu, sd=sd, dBIC=float(bic1c - bic2c), dBIC_raw=float(bic1 - bic2), n_eff=float(n_eff), separation=float(abs(mu[0] - mu[1])), valley_ratio=ratio)


def first_order_signature(e_series, tau=0.5):
    """Bimodality summary of an energy time series at the specific-heat peak (per-run part of the rule).
    Pre-registered rule (M4 tier-3 request, decision 3): 'bimodal' needs ALL of (1) autocorrelation-corrected
    dBIC > 10, (2) valley ratio < 0.9 and (3) the minimum over T of the energy cumulant V moving away from
    2/3 as L grows (a property of a set of sizes: see ``cumulant_trend``).  This function returns
    conditions (1) and (2) as ``bimodal_run``; the old amplitude test is kept as ``legacy_flag``."""
    f = two_gaussian_fit(e_series, tau=tau)
    legacy = f["dBIC_raw"] > 10 and f["separation"] > 2 * max(f["sd"]) and min(f["w"]) > 0.05
    run = f["dBIC"] > 10 and f["valley_ratio"] < 0.9
    return dict(bimodal=bool(run), bimodal_run=bool(run), legacy_flag=bool(legacy), **f)


def cumulant_trend(sizes, vmin):
    """Condition (3): |2/3 - V_min| must increase with L.  Needs >= 2 sizes; returns (holds, deviations)."""
    dev = np.abs(2.0 / 3.0 - np.asarray(vmin, float))
    order = np.argsort(sizes)
    dev = dev[order]
    return bool(len(dev) >= 2 and np.all(np.diff(dev) > 0)), dev.tolist()
