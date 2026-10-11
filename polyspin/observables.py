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
    # powers of (x + i y) by recurrence in scalars (no allocation in the hot loop)
    c1, s1 = x, y
    c2, s2 = c1 * x - s1 * y, s1 * x + c1 * y
    c3, s3 = c2 * x - s2 * y, s2 * x + c2 * y
    c4, s4 = c3 * x - s3 * y, s3 * x + c3 * y
    for k in range(yl.shape[0]):
        poly = 0.0
        for c in range(4, -1, -1):
            poly = poly * z + yp[k, c]
        m = ym[k]
        if m == 0:
            trig = 1.0
        elif m == 1:
            trig = s1 if ysc[k] == 1 else c1
        elif m == 2:
            trig = s2 if ysc[k] == 1 else c2
        elif m == 3:
            trig = s3 if ysc[k] == 1 else c3
        else:
            trig = s4 if ysc[k] == 1 else c4
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


def global_class(m, m_thr):
    """Coarse phase label from the global order parameters m_1..m_4 only (no per-cluster
    thresholds, which are unstable near continuous transitions such as C-D).

    m_thr should lie above the finite-size noise floor ~ 1/sqrt(N n).
      F  : m1 ~ 1   (m1 > 0.9)
      M  : m1 > m_thr, not F  (U or D: net moment; U vs D needs the shape diagnostics)
      C  : m1, m3 < m_thr and m2 > m_thr   (nematic, odd channels vanish)
      T  : m1, m2 < m_thr and (m3 or m4) > m_thr
      0  : everything below m_thr (disordered)
    """
    m1, m2, m3, m4 = m
    if m1 > 0.9:
        return "F"
    if m1 > m_thr:
        return "M"
    if m2 > m_thr and m3 < m_thr:
        return "C"
    if m2 < m_thr and (m3 > m_thr or m4 > m_thr):
        return "T"
    return "0"


# ----------------------------------------------------------- per-cluster shape diagnostics
@njit(cache=True)
def _legendre_all(x):
    return (x, 0.5 * (3 * x * x - 1), 0.5 * (5 * x ** 3 - 3 * x), (35 * x ** 4 - 30 * x * x + 3) / 8)


@njit(cache=True)
def cluster_shape_arrays(spins, thr):
    """Per cluster: number of groups and size of the largest group (spins with S_i.S_j > thr are
    joined), A_l(a) = sum_ij P_l(S_i.S_j) for l = 1..4, |M_a|/n and the collinearity
    lambda_max(Q_a) / (2n/3) of the quadrupole tensor Q = sum_i (S S^T - I/3).  For n = 4:
    (groups, largest) = (4,1) {1111}, (3,2) {211}, (2,2) {22}, (2,3) {31}, (1,4) {4}."""
    N, n, _ = spins.shape
    ngroups = np.empty(N, np.int64)
    largest = np.empty(N, np.int64)
    A = np.zeros((N, 4))
    Mabs = np.empty(N)
    lam = np.empty(N)
    parent = np.empty(n, np.int64)
    size = np.empty(n, np.int64)
    Q = np.empty((3, 3))
    for a in range(N):
        for i in range(n):
            parent[i] = i
        mx = my = mz = 0.0
        for i in range(n):
            mx += spins[a, i, 0]
            my += spins[a, i, 1]
            mz += spins[a, i, 2]
            for j in range(n):
                x = (spins[a, i, 0] * spins[a, j, 0] + spins[a, i, 1] * spins[a, j, 1]
                     + spins[a, i, 2] * spins[a, j, 2])
                p1, p2, p3, p4 = _legendre_all(x)
                A[a, 0] += p1
                A[a, 1] += p2
                A[a, 2] += p3
                A[a, 3] += p4
                if j > i and x > thr:
                    ri = i
                    while parent[ri] != ri:
                        ri = parent[ri]
                    rj = j
                    while parent[rj] != rj:
                        rj = parent[rj]
                    if ri != rj:
                        parent[ri] = rj
        for i in range(n):
            size[i] = 0
        for i in range(n):
            r = i
            while parent[r] != r:
                r = parent[r]
            size[r] += 1
        ng, big = 0, 0
        for i in range(n):
            if size[i] > 0:
                ng += 1
                if size[i] > big:
                    big = size[i]
        ngroups[a], largest[a] = ng, big
        Mabs[a] = np.sqrt(mx * mx + my * my + mz * mz) / n
        for k in range(3):
            for l in range(3):
                Q[k, l] = 0.0
        for i in range(n):
            for k in range(3):
                for l in range(3):
                    Q[k, l] += spins[a, i, k] * spins[a, i, l]
        for k in range(3):
            Q[k, k] -= n / 3.0
        w = np.linalg.eigvalsh(Q)
        lam[a] = w[2] / (2.0 * n / 3.0)
    return ngroups, largest, A, Mabs, lam


SPLIT_NAMES = ("1111", "211", "22", "31", "4")


def split_histogram(ngroups, largest):
    """Fractions of clusters of type {1111},{211},{22},{31},{4} (n = 4)."""
    code = {(4, 1): 0, (3, 2): 1, (2, 2): 2, (2, 3): 3, (1, 4): 4}
    out = np.zeros(5)
    for g, b in zip(ngroups, largest):
        out[code.get((int(g), int(b)), 2)] += 1
    return out / len(ngroups)


@njit(cache=True)
def multipoles_per_cluster(spins, yl, ym, ysc, yc, yp):
    """Sum over the n spins of each cluster of Y_lm: (N, 24)."""
    N, n, _ = spins.shape
    out = np.zeros((N, yl.shape[0]))
    tmp = np.empty(yl.shape[0])
    for a in range(N):
        for i in range(n):
            _harmonics(spins[a, i, 0], spins[a, i, 1], spins[a, i, 2], yl, ym, ysc, yc, yp, tmp)
            out[a] += tmp
    return out


def correlators(spins, lattice):
    """Spatial correlation G_l(r) = < q_0^(l) . q_r^(l) > / (n^2 (2l+1)/(4 pi)), l = 1..4, averaged over
    the coordinate axes, r = 0..L/2.  Normalised so that G_l(0) <= 1 and a perfectly ordered state
    gives G_l(r) = 1."""
    q = multipoles_per_cluster(spins, _YL, _YM, _YSC, _YC, _YP)
    L, dim, N = lattice.size, lattice.dim, lattice.N
    n = spins.shape[1]
    out = np.zeros((4, L // 2 + 1))
    for l in range(1, 5):
        cols = np.flatnonzero(_YL == l)
        grid = np.zeros((L,) * dim + (len(cols),))
        grid[tuple(lattice.coords.T)] = q[:, cols]
        f = np.fft.fftn(grid, axes=tuple(range(dim)))
        c = np.fft.ifftn((np.abs(f) ** 2).sum(-1), axes=tuple(range(dim))).real / N
        norm = n * n * (2 * l + 1) / (4 * np.pi)
        for r in range(L // 2 + 1):
            vals = []
            for ax in range(dim):
                idx = [0] * dim
                idx[ax] = r
                vals.append(c[tuple(idx)])
            out[l - 1, r] = np.mean(vals) / norm
    return out
