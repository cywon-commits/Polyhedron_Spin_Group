import numpy as np

from polyspin.analysis.finite_size import first_order_signature, loglog_slope, two_gaussian_fit


def test_loglog_slope():
    N = np.array([64, 100, 144.0])
    a, se = loglog_slope(N, 0.3 * N ** 0.8)
    assert abs(a - 0.8) < 1e-10 and se < 1e-8


def test_bimodality_detection():
    rng = np.random.default_rng(0)
    uni = rng.normal(-1.0, 0.05, 6000)
    bi = np.concatenate([rng.normal(-1.0, 0.05, 3000), rng.normal(-0.7, 0.05, 3000)])
    assert not first_order_signature(uni)["bimodal"]
    s = first_order_signature(bi)
    assert s["bimodal"] and abs(s["separation"] - 0.3) < 0.01 and s["valley_ratio"] < 0.1
    assert two_gaussian_fit(bi)["dBIC"] > 100


def test_autocorrelation_correction_kills_false_bimodality():
    """A strongly autocorrelated unimodal series fools the uncorrected BIC; the corrected rule must not fire,
    while a real bimodal series with tau = 5 still does."""
    rng = np.random.default_rng(1)
    n, rho = 20000, 0.99
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = rho * x[i - 1] + rng.normal() * np.sqrt(1 - rho ** 2)
    x = np.exp(0.6 * x)               # skewed, unimodal
    tau = 0.5 * (1 + rho) / (1 - rho)
    s = first_order_signature(x, tau=tau)
    assert s["dBIC_raw"] > 10 and not s["bimodal_run"]
    bi = np.concatenate([rng.normal(-1.0, 0.05, 4000), rng.normal(-0.7, 0.05, 4000)])
    assert first_order_signature(bi, tau=5.0)["bimodal_run"]


def test_cumulant_trend():
    from polyspin.analysis.finite_size import cumulant_trend
    assert cumulant_trend([8, 10, 12], [0.6660, 0.6650, 0.6600])[0]
    assert not cumulant_trend([8, 10, 12], [0.6660, 0.6666, 0.6650])[0]


def test_rb_rule():
    from polyspin.analysis.defect_threshold import rb_verdict, rb_verdict_one
    T = np.linspace(0.5, 1.5, 21)
    low = np.full_like(T, 0.2); free = np.where(T < 1.0, 0.2, 0.9)
    ud = np.full_like(T, 0.02)
    assert rb_verdict_one(T, free, ud, 1.0)["verdict"] == "free"
    assert rb_verdict_one(T, low, ud, 1.0)["verdict"] == "none"
    assert rb_verdict_one(T, np.full_like(T, 0.6), ud, 1.0)["verdict"] == "undetermined"
    bad = np.where(T > 1.05, 0.5, 0.02)                       # f_ud >= 0.1 at 1.1 Tc -> value not usable
    assert rb_verdict_one(T, low, bad, 1.0)["verdict"] == "undetermined"
    assert rb_verdict({24: dict(verdict="free"), 32: dict(verdict="none")}) == "undetermined"
    assert rb_verdict({24: dict(verdict="none"), 32: dict(verdict="none")}) == "none"
