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
