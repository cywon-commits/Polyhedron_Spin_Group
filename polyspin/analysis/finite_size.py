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


def two_gaussian_fit(x, iters=300):
    """EM fit of a two-component 1D Gaussian mixture.  Returns dict(w, mu, sd, loglik) of the fit and the
    BIC difference BIC(1) - BIC(2) (> 10: bimodal), the peak separation |mu1 - mu2| and the valley/peak
    ratio of the fitted density."""
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
    ll1 = (-0.5 * np.log(2 * np.pi * x.var()) - 0.5).sum() if x.var() > 0 else -np.inf
    bic1, bic2 = -2 * ll1 + 2 * np.log(n), -2 * ll2 + 5 * np.log(n)
    grid = np.linspace(mu.min(), mu.max(), 200)
    dens = sum(w[k] * np.exp(-0.5 * ((grid - mu[k]) / sd[k]) ** 2) / sd[k] for k in range(2))
    ratio = float(dens.min() / min(dens[0], dens[-1])) if mu.min() != mu.max() else 1.0
    return dict(w=w, mu=mu, sd=sd, dBIC=float(bic1 - bic2), separation=float(abs(mu[0] - mu[1])), valley_ratio=ratio)


def first_order_signature(e_series):
    """Bimodality summary of an energy time series at the specific-heat peak."""
    f = two_gaussian_fit(e_series)
    bimodal = f["dBIC"] > 10 and f["separation"] > 2 * max(f["sd"]) and min(f["w"]) > 0.05
    return dict(bimodal=bool(bimodal), **f)
