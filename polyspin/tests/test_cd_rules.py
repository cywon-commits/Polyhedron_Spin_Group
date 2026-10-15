import numpy as np

from polyspin.analysis.cd_rules import h5_intervals, h5_verdict, reentrance_T, reentrance_verdict


def test_h5_interval_rule():
    T = np.linspace(0.05, 0.4, 15)
    s = np.where((T > 0.1) & (T < 0.25), 0.7, 0.2)
    m = np.where((T > 0.1) & (T < 0.25), 0.25, 0.5)
    iv = h5_intervals(T, s, m)
    assert len(iv) == 1 and iv[0][2] >= 3
    assert h5_intervals(T, s, np.full_like(T, 0.5)) == []                    # m1 outside [0.2, 0.3]
    assert h5_intervals(T, np.full_like(T, 0.4), m) == []                    # {211} fraction <= 0.5
    two = h5_intervals(T[:2], np.array([0.7, 0.7]), np.array([0.25, 0.25]))
    assert two == []                                                          # fewer than 3 points is not an interval
    assert h5_verdict({1.0: iv, 1.2: iv}) == "supported" and h5_verdict({1.0: iv, 1.2: []}) == "not supported"


def test_reentrance_rule():
    T = np.linspace(0.1, 0.6, 26)
    m1 = np.where(T < 0.23, 0.95, 0.03); m2 = np.full_like(T, 0.9)
    f, t = reentrance_T(T, m1, m2)
    assert f and 0.2 < t < 0.26
    assert reentrance_verdict((f, t), (f, t)).startswith("supported")
    assert reentrance_verdict((f, t), (False, float("nan"))) == "F -> C seen in one ladder only"
    assert reentrance_verdict((f, 0.40), (f, 0.41)).startswith("F -> C seen in both ladders, T_re outside")
    assert reentrance_T(T, np.full_like(T, 0.95), m2)[0] is False            # stays F: no re-entrance
    assert reentrance_T(T, m1, np.full_like(T, 0.3))[0] is False             # high-T side is not nematic C
    assert reentrance_verdict((False, float("nan")), (False, float("nan"))) == "not seen"
