"""T=0 single-cluster minimisation (valid for the whole lattice by the uniform
state theorem when all kappa-channel couplings g_l >= 0)."""
import numpy as np
from scipy.optimize import minimize

from .model import (Params, channel_A, channel_coeffs, legendre, legendre_deriv,
                    tetrahedron)


def _energy_grad(v, kappa, const, n):
    V = v.reshape(n, 3)
    nr = np.linalg.norm(V, axis=1, keepdims=True)
    S = V / nr
    X = np.clip(S @ S.T, -1.0, 1.0)
    e = const
    W = np.zeros_like(X)  # W_ij = sum_l kappa_l P_l'(x_ij)
    for l in (1, 2, 3, 4):
        e += 0.5 * kappa[l - 1] * legendre(l, X).sum()
        W += kappa[l - 1] * legendre_deriv(l, X)
    np.fill_diagonal(W, 0.0)
    gS = W @ S
    gV = (gS - np.sum(gS * S, axis=1, keepdims=True) * S) / nr
    return e, gV.ravel()


def minimize_cluster(P: Params, S0):
    kappa, const = channel_coeffs(P)
    n = P.n
    res = minimize(_energy_grad, np.asarray(S0, float).ravel(), args=(kappa, const, n),
                   jac=True, method="L-BFGS-B",
                   options=dict(maxiter=5000, ftol=1e-15, gtol=1e-11, maxcor=30))
    V = res.x.reshape(n, 3)
    S = V / np.linalg.norm(V, axis=1, keepdims=True)
    return float(res.fun), S


def _rand_unit(rng, k=1):
    v = rng.normal(size=(k, 3))
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def structured_starts(rng, noise=0.05):
    """Starting points near {4},{31},{22},{211},{1111}; >= 20 in total."""
    out = []
    for _ in range(3):  # {4}
        d = _rand_unit(rng)[0]
        out.append(np.tile(d, (4, 1)))
    for _ in range(4):  # {31}: three together, one anywhere / opposite
        d = _rand_unit(rng)[0]
        out.append(np.vstack([d, d, d, _rand_unit(rng)[0] if _ % 2 else -d]))
    for ang in (np.pi, 2.0, 1.5, 1.0, 0.6):  # {22}, bent by angle ang
        d = _rand_unit(rng)[0]
        t = np.cross(d, _rand_unit(rng)[0])
        t /= np.linalg.norm(t)
        e = np.cos(ang) * d + np.sin(ang) * t
        out.append(np.vstack([d, d, e, e]))
    for _ in range(5):  # {211}
        d = _rand_unit(rng)[0]
        r = _rand_unit(rng, 2)
        out.append(np.vstack([d, d, r[0], r[1]]))
    T = tetrahedron()
    for _ in range(5):  # near-tetrahedron, random orientation
        Q, _r = np.linalg.qr(rng.normal(size=(3, 3)))
        if np.linalg.det(Q) < 0:
            Q[:, 0] *= -1
        out.append(T @ Q.T)
    out = [S + noise * rng.normal(size=S.shape) for S in out]
    return [S / np.linalg.norm(S, axis=1, keepdims=True) for S in out]


def find_ground_state(P: Params, n_random=12, seed=0, return_all=False):
    """Global minimum over random + structured starts (>= 12 + 20)."""
    rng = np.random.default_rng(seed)
    starts = [_rand_unit(rng, P.n) for _ in range(n_random)] + structured_starts(rng)
    results = [minimize_cluster(P, S0) for S0 in starts]
    results.sort(key=lambda r: r[0])
    return results if return_all else results[0]


# ------------------------------------------------------------ classification
def partition(S, thr=0.95):
    """Group spins with S_i.S_j > thr; returns sorted tuple of group sizes."""
    n = len(S)
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    X = S @ S.T
    for i in range(n):
        for j in range(i + 1, n):
            if X[i, j] > thr:
                parent[find(i)] = find(j)
    sizes = {}
    for i in range(n):
        sizes[find(i)] = sizes.get(find(i), 0) + 1
    return tuple(sorted(sizes.values(), reverse=True))


def classify_cluster(S, thr=0.95, m_tol=0.15):
    """Label one cluster as T, U, D, C, F or X (anything else).

    {1111} with A1~A2~0 -> T;  {211} -> U;  {22}: antiparallel dimers -> C, else D;
    {4} -> F.  X marks {31}, non-tetrahedral {1111}, etc.
    """
    part = partition(S, thr)
    M = np.linalg.norm(S.sum(axis=0)) / len(S)
    A = channel_A(S)
    if part == (4,):
        return "F"
    if part == (2, 2):
        return "C" if M < m_tol else "D"
    if part == (2, 1, 1):
        return "U"
    if part == (1, 1, 1, 1) and abs(A[0]) < 0.5 and abs(A[1]) < 0.5:
        return "T"
    return "X"


def phase_of_gs(P: Params, **kw):
    E, S = find_ground_state(P, **kw)
    return classify_cluster(S, thr=0.999), E, S


# ----------------------------------------------------------- analytic bounds
def boundary_TC(J, K, zG):
    """zH4 on the T-C boundary."""
    return 0.9 * K + 0.3 * zG


def boundary_TF(J, K, zH4):
    """zG on the T-F boundary: (7/9) zG + (20/27) zH4 = J + 2K/3."""
    return (J + 2 * K / 3 - 20 * zH4 / 27) * 9 / 7


def boundary_CF(J):
    return J
