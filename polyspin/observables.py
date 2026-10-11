"""Label-permutation-invariant order parameters and cluster-shape diagnostics."""
from math import factorial, pi, sqrt

import numpy as np
from numba import njit
from numpy.polynomial import Legendre, Polynomial

from .groundstate import classify_cluster, partition  # noqa: F401
from .model import channel_A, legendre  # noqa: F401


def _build_table():
    L_, M_, SC_, C_, P_ = [], [], [], [], []
    for l in range(1, 5):
        pl = Legendre.basis(l).convert(kind=Polynomial)
        for m in range(0, l + 1):
            coef = np.zeros(5)
            d = pl.deriv(m).coef
            coef[: len(d)] = d
            norm = sqrt((2 * l + 1) / (4 * pi) * factorial(l - m) / factorial(l + m))
            for sc in ((0,) if m == 0 else (0, 1)):  # 0: Re (x+iy)^m, 1: Im
                L_.append(l)
                M_.append(m)
                SC_.append(sc)
                C_.append(norm * (sqrt(2.0) if m else 1.0))
                P_.append(coef)
    return (np.array(L_), np.array(M_), np.array(SC_), np.array(C_), np.array(P_))


_YL, _YM, _YSC, _YC, _YP = _build_table()  # 24 real harmonics l=1..4


@njit(cache=True)
def _harmonics(x, y, z, yl, ym, ysc, yc, yp, out):
    cm = np.empty(5)
    sm = np.empty(5)
    cm[0], sm[0] = 1.0, 0.0
    for m in range(1, 5):
        cm[m] = cm[m - 1] * x - sm[m - 1] * y
        sm[m] = sm[m - 1] * x + cm[m - 1] * y
    for k in range(yl.shape[0]):
        poly = 0.0
        for c in range(4, -1, -1):
            poly = poly * z + yp[k, c]
        m = ym[k]
        trig = sm[m] if ysc[k] == 1 else cm[m]
        out[k] = yc[k] * poly * trig


@njit(cache=True)
def multipole_sum(spins, yl, ym, ysc, yc, yp):
    """Sum over all clusters/spins of Y_lm(S): returns the (24,) vector."""
    N, n, _ = spins.shape
    tot = np.zeros(yl.shape[0])
    tmp = np.empty(yl.shape[0])
    for a in range(N):
        for i in range(n):
            _harmonics(spins[a, i, 0], spins[a, i, 1], spins[a, i, 2], yl, ym, ysc, yc, yp, tmp)
            tot += tmp
    return tot


def harmonics(v):
    """Real Y_lm(v), l=1..4 (24 values) for one unit vector."""
    out = np.empty(len(_YL))
    _harmonics(v[0], v[1], v[2], _YL, _YM, _YSC, _YC, _YP, out)
    return out


def m_l(spins):
    """Ferro-type order parameters m_1..m_4 (see the request document)."""
    N, n, _ = spins.shape
    tot = multipole_sum(spins, _YL, _YM, _YSC, _YC, _YP)
    res = np.empty(4)
    for l in range(1, 5):
        q = tot[_YL == l]
        res[l - 1] = np.linalg.norm(q) * sqrt(4 * pi / (2 * l + 1)) / (N * n)
    return res


def cluster_shapes(spins, thr=0.95):
    """Per-cluster labels (T/U/D/C/F/X) and split-type counts."""
    labels = [classify_cluster(S, thr) for S in spins]
    parts = [partition(S, thr) for S in spins]
    return labels, parts


def majority_phase(spins, thr=0.95):
    labels, _ = cluster_shapes(spins, thr)
    vals, counts = np.unique(labels, return_counts=True)
    return vals[np.argmax(counts)], dict(zip(vals, counts / len(labels)))
