"""Harmonic (spin-wave) analysis around a uniform state.

For a uniform state S0 (n,3) the energy to second order in tangent displacements u_{a,i} in R^2
is block translation invariant:  H(q) = A + C s(q) + C^T conj(s(q)),  s(q) = sum_d exp(i q_d),
a Hermitian 2n x 2n matrix per wave vector.  The classical harmonic free energy per cluster is
    F_harm / N = const + (T/2) * mean_q sum_k ln lambda_k(q)      (zero modes excluded),
so a *softer* state has the lower free energy and is selected by thermal fluctuations
(order by disorder).  n is a parameter (n = 4 here, n = 6 for the octahedral systems).
"""
import numpy as np

from .model import Params


# ---------------------------------------------------------------- pair derivatives
def f1(x, P):
    return P.J + 2 * P.K * x + 4 * P.L * x ** 3


def f2(x, P):
    return 2 * P.K + 12 * P.L * x ** 2


def g1(x, P):
    return 3 * P.G * x ** 2 + 4 * P.H4 * x ** 3 + P.G1


def g2(x, P):
    return 6 * P.G * x + 12 * P.H4 * x ** 2


def tangent_basis(S0):
    """(n,3,2): orthonormal tangent vectors for every spin."""
    T = np.empty((len(S0), 3, 2))
    for i, s in enumerate(S0):
        a = np.array([1.0, 0, 0]) if abs(s[0]) < 0.9 else np.array([0, 1.0, 0])
        t1 = np.cross(s, a)
        t1 /= np.linalg.norm(t1)
        T[i, :, 0], T[i, :, 1] = t1, np.cross(s, t1)
    return T


def gradient_lagrange(S0, P: Params):
    """Euclidean gradient of the energy per spin, G_i, and lambda_i = S_i . G_i.
    Stationarity requires G_i to be parallel to S_i."""
    n, z = len(S0), P.z
    X = S0 @ S0.T
    G = np.zeros_like(S0)
    for i in range(n):
        for j in range(n):
            if j != i:
                G[i] += f1(X[i, j], P) * S0[j]
            G[i] -= z * g1(X[i, j], P) * S0[j]
    lam = np.einsum("ik,ik->i", S0, G)
    return G, lam


def blocks(S0, P: Params):
    """A, C (2n x 2n) tangent-space on-site and bond blocks, and the stationarity residual."""
    n, z = len(S0), P.z
    X = S0 @ S0.T
    T = tangent_basis(S0)
    G, lam = gradient_lagrange(S0, P)
    resid = float(np.abs(G - lam[:, None] * S0).max())
    A = np.zeros((2 * n, 2 * n))
    C = np.zeros((2 * n, 2 * n))
    for i in range(n):
        for j in range(n):
            x = X[i, j]
            Sj_Si = np.outer(S0[j], S0[i])          # [l,m] = S_j,l S_i,m
            if i != j:                              # intra-cluster pair (i,j)
                blk = f1(x, P) * np.eye(3) + f2(x, P) * Sj_Si
                A[2 * i:2 * i + 2, 2 * j:2 * j + 2] += T[i].T @ blk @ T[j]
                A[2 * i:2 * i + 2, 2 * i:2 * i + 2] += T[i].T @ (f2(x, P) * np.outer(S0[j], S0[j])) @ T[i]
            # inter-site: on-site part of the z bonds, and the bond-coupling block
            A[2 * i:2 * i + 2, 2 * i:2 * i + 2] -= z * g2(x, P) * (T[i].T @ np.outer(S0[j], S0[j]) @ T[i])
            blkC = -(g1(x, P) * np.eye(3) + g2(x, P) * Sj_Si)
            C[2 * i:2 * i + 2, 2 * j:2 * j + 2] += T[i].T @ blkC @ T[j]
    for i in range(n):                              # curvature of the sphere
        A[2 * i:2 * i + 2, 2 * i:2 * i + 2] -= lam[i] * np.eye(2)
    return A, C, resid


def hessian_q(A, C, q):
    """Hermitian 2n x 2n matrix H(q) for q (dim,) in units of the lattice spacing."""
    s = np.exp(1j * np.asarray(q)).sum()
    return A + C * s + C.T * np.conj(s)


def spectrum(S0, P: Params, qs):
    """Eigenvalues (len(qs), 2n) of H(q)."""
    A, C, _ = blocks(S0, P)
    return np.array([np.linalg.eigvalsh(hessian_q(A, C, q)) for q in qs])


def q_grid(dim, Lq):
    k = 2 * np.pi * np.arange(Lq) / Lq
    return np.stack(np.meshgrid(*([k] * dim), indexing="ij"), -1).reshape(-1, dim)


def log_det_per_cluster(S0, P: Params, dim, Lq, zero_tol=1e-8):
    """(1/N) sum_q sum_k ln lambda_k(q) over the Lq^dim grid, dropping zero modes (the global
    rotation modes at q = 0).  Returns (value, number of zero modes, min nonzero eigenvalue)."""
    A, C, resid = blocks(S0, P)
    qs = q_grid(dim, Lq)
    # H(q) depends on q only through s(q): evaluate unique s values once
    s = np.exp(1j * qs).sum(1)
    ev = np.array([np.linalg.eigvalsh(A + C * sv + C.T * np.conj(sv)) for sv in s])
    zero = np.abs(ev) < zero_tol
    if (ev < -zero_tol).any():
        raise ValueError("state is not a local minimum (negative Hessian eigenvalue %.3g)" % ev.min())
    logs = np.where(zero, 0.0, np.log(np.where(zero, 1.0, ev)))
    return float(logs.sum() / len(qs)), int(zero.sum()), float(ev[~zero].min())


def collinear_state(n, n_down):
    """Collinear state with n_down spins reversed (net moment n - 2 n_down)."""
    S = np.zeros((n, 3))
    S[:, 2] = 1.0
    S[:n_down, 2] = -1.0
    return S


def selection(P: Params, dim, Lq, states=None):
    """Free-energy comparison of the collinear states on the C-F degenerate line (zG = J).

    For every number of reversed spins k (net moment M = n - 2k) returns
      logdet_per_cluster, zero_modes,
      dF_harm_over_T  = (1/2)(logdet_k - logdet_F)          harmonic (spin-wave) part,
      ln_multiplicity = ln C(n,k)                           labelled-spin degeneracy,
      dF_over_T       = dF_harm_over_T - ln C(n,k)          total, relative to F.
    The multiplicity term is exact and extensive: every cluster independently chooses which k of
    its n labelled spins are reversed, and the MC (like the partition function) samples labelled
    spins.  The expression in the request contains only the harmonic part.
    """
    from math import comb, log

    n = P.n
    ks = states if states is not None else range(0, n // 2 + 1)
    out = {k: log_det_per_cluster(collinear_state(n, k), P, dim, Lq) for k in ks}
    ref = out[0][0]
    res = {}
    for k, v in out.items():
        harm = 0.5 * (v[0] - ref)
        mult = log(comb(n, k))
        res[k] = dict(logdet=v[0], zero_modes=v[1], dF_harm_over_T=harm, ln_multiplicity=mult,
                      dF_over_T=harm - mult, moment=n - 2 * k)
    return res


def collinear_energy_offset(P: Params, k):
    """E_k - E_F per cluster at fixed (J - zG) for collinear states (L = G1 = 0, uniform):
    E(M) = (J - zG) M^2 / 2 + const, M = n - 2k."""
    n = P.n
    M = n - 2 * k
    return (P.J - P.zG) * (M ** 2 - n ** 2) / 2


def CF_boundary_slope(P: Params, dim, Lq):
    """Low-T shift of the C-F boundary, d(zG*)/dT, from F_C - F_F = 0:
    -(J - zG)(n^2)/2 + T * dF_over_T(C) = 0  =>  zG* = J + T * dF_over_T(C) / (n^2/2)
    (n = 4: zG* = J + T dF/8).  Negative dF => zG* > J : C survives beyond zG = J."""
    n = P.n
    r = selection(P, dim, Lq, states=[0, n // 2])
    return r[n // 2]["dF_over_T"] * 2 / n ** 2 * (-1)
