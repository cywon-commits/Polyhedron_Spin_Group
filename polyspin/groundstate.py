"""T=0 single-cluster minimisation (valid for the whole lattice by the uniform
state theorem when all kappa-channel couplings g_l >= 0).

Single-spin anisotropy V = sum_i [a2 P2(z.S_i) + a4 P4(z.S_i)] is a site term, so the uniform-state
reduction still holds and V is simply added to the cluster energy (``aniso = (a2, a4)``).
``minimize_cluster_at_orientation`` relaxes the cluster shape at a FIXED body orientation (Eckart
condition sum_i S0_i x S_i = 0); it gives the orientation dependence E(n) of docs Gamma formula.
"""
import numpy as np
from scipy.optimize import minimize

from .model import (Params, channel_A, channel_coeffs, legendre, legendre_deriv,
                    tetrahedron)


def _energy_grad(v, kappa, const, n, aniso=(0.0, 0.0)):
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
    a2, a4 = aniso
    if a2 or a4:
        cz = S[:, 2]
        e += float((a2 * legendre(2, cz) + a4 * legendre(4, cz)).sum())
        gS = gS.copy()
        gS[:, 2] += a2 * legendre_deriv(2, cz) + a4 * legendre_deriv(4, cz)
    gV = (gS - np.sum(gS * S, axis=1, keepdims=True) * S) / nr
    return e, gV.ravel()


def anisotropy(P: Params):
    return (float(getattr(P, "a2", 0.0)), float(getattr(P, "a4", 0.0)))


def minimize_kappa(kappa, const, S0):
    """Minimise E = const + 1/2 sum_l kappa_l A_l directly from channel couplings."""
    n = len(S0)
    res = minimize(_energy_grad, np.asarray(S0, float).ravel(), args=(np.asarray(kappa), const, n),
                   jac=True, method="L-BFGS-B",
                   options=dict(maxiter=5000, ftol=1e-15, gtol=1e-11, maxcor=30))
    V = res.x.reshape(n, 3)
    return float(res.fun), V / np.linalg.norm(V, axis=1, keepdims=True)


def minimize_cluster(P: Params, S0):
    kappa, const = channel_coeffs(P)
    n = P.n
    res = minimize(_energy_grad, np.asarray(S0, float).ravel(), args=(kappa, const, n, anisotropy(P)),
                   jac=True, method="L-BFGS-B",
                   options=dict(maxiter=5000, ftol=1e-15, gtol=1e-11, maxcor=30))
    V = res.x.reshape(n, 3)
    S = V / np.linalg.norm(V, axis=1, keepdims=True)
    return float(res.fun), S


def minimize_cluster_at_orientation(P: Params, S0, tol=1e-14):
    """Relax the cluster shape with its orientation held at that of S0 (unit rows): minimise the
    uniform-state cluster energy + anisotropy subject to the Eckart condition sum_i S0_i x S_i = 0,
    which removes the three rigid rotations.  Returns (E, S)."""
    from scipy.optimize import minimize as _min
    kappa, const = channel_coeffs(P)
    n = P.n; S0 = np.asarray(S0, float); an = anisotropy(P)
    f = lambda v: _energy_grad(v, kappa, const, n, an)

    def cons(v):
        V = v.reshape(n, 3); S = V / np.linalg.norm(V, axis=1, keepdims=True)
        return np.cross(S0, S).sum(0)

    res = _min(f, S0.ravel(), jac=True, method="SLSQP", constraints=[dict(type="eq", fun=cons)],
               options=dict(maxiter=1000, ftol=tol))
    V = res.x.reshape(n, 3)
    return float(res.fun), V / np.linalg.norm(V, axis=1, keepdims=True)


def gamma_formula(P: Params):
    """Gamma = (35/9) a4 + 2 a2^2 (3/lam_E - 1/lam_T2) with Phi(x) = sum_l kappa_l P_l(x) (uniform
    state): lam_E = (8/3) Phi''(-1/3), lam_T2 = (8/9) [3 Phi'(-1/3) + 2 Phi''(-1/3)].
    E_eff(n) = -Gamma sum_alpha n_alpha^4 + const, n = lab z in the body (S4-axis) frame."""
    kappa, _ = channel_coeffs(P)
    x = -1.0 / 3.0
    d1 = sum(k * float(legendre_deriv(l, x)) for l, k in zip((1, 2, 3, 4), kappa))
    d2 = sum(k * _legendre_dd(l, x) for l, k in zip((1, 2, 3, 4), kappa))
    lam_E = 8.0 / 3.0 * d2
    lam_T2 = 8.0 / 9.0 * (3 * d1 + 2 * d2)
    a2, a4 = anisotropy(P)
    return 35.0 / 9.0 * a4 + 2 * a2 ** 2 * (3 / lam_E - 1 / lam_T2), lam_E, lam_T2


def _legendre_dd(l, x):
    return {1: 0.0, 2: 3.0, 3: 15.0 * x, 4: 0.5 * (105.0 * x * x - 15.0)}[l]


def oriented_tetrahedron(nhat, spin=0.0):
    """Tetrahedron (body S4 axes = lab axes before rotation) rotated so that the body-frame unit
    vector ``nhat`` points along lab z; ``spin`` rotates further about lab z."""
    from scipy.spatial.transform import Rotation as Rot
    nhat = np.asarray(nhat, float) / np.linalg.norm(nhat)
    R1, _ = Rot.align_vectors([[0, 0, 1]], [nhat])
    R = Rot.from_rotvec([0, 0, spin]) * R1
    return R.apply(tetrahedron())


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


def find_ground_state_kappa(kappa, const=0.0, n=4, n_random=12, seed=0):
    rng = np.random.default_rng(seed)
    starts = [_rand_unit(rng, n) for _ in range(n_random)] + structured_starts(rng)
    return min((minimize_kappa(kappa, const, S0) for S0 in starts), key=lambda r: r[0])


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


def phase_of_gs(P: Params, m_tol=1e-3, **kw):
    """Ground-state phase.  C is the exactly collinear {22} state (M = 0): with the default
    m_tol a {22} state with any net moment is D (D bends continuously out of C)."""
    E, S = find_ground_state(P, **kw)
    return classify_cluster(S, thr=0.999, m_tol=m_tol), E, S


# ----------------------------------------------------------- analytic bounds
def boundary_TC(J, K, zG):
    """zH4 on the T-C boundary."""
    return 0.9 * K + 0.3 * zG


def boundary_TF(J, K, zH4):
    """zG on the T-F boundary: (7/9) zG + (20/27) zH4 = J + 2K/3."""
    return (J + 2 * K / 3 - 20 * zH4 / 27) * 9 / 7


def boundary_CF(J):
    return J


def phi_prime1(kappa):
    """Phi'(1) = sum_l kappa_l l(l+1)/2: slope of the pair function at full overlap.
    The all-parallel (F) state is locally stable against continuous splitting iff < 0."""
    return float(sum(k * l * (l + 1) / 2 for l, k in zip((1, 2, 3, 4), kappa)))


def boundary_CD(J, K, zH4):
    """zG on the continuous C-D boundary, from Phi'(-1) = k1 - 3k2 + 6k3 - 10k4 = 0
    (collinear dimers become unstable to bending): zG = (J - 2K + 4 zH4)/3."""
    return (J - 2 * K + 4 * zH4) / 3


def boundary_DF(J, K):
    """zG on the first-order D-F boundary at zH4 = 0 (kappa4 = 0): root of
    k1 + 3/4 k2 + 3/8 k3 - 9 k2^2/(40 k3) = 0 with k1 = J - 3g/5, k2 = 2K/3, k3 = -2g/5.
    Multiplying out gives a quadratic in g; returns the root > 0 that lies above C-D."""
    from scipy.optimize import brentq

    def h(g):
        k1, k2, k3 = J - 0.6 * g, 2 * K / 3, -0.4 * g
        return k1 + 0.75 * k2 + 0.375 * k3 - 9 * k2 ** 2 / (40 * k3)

    return brentq(h, boundary_CD(J, K, 0.0) + 1e-6, 3.0 * J)
