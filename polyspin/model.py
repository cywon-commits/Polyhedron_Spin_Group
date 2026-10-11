"""Hamiltonian, local energy differences and the Legendre-channel form.

H = sum_a sum_{i<j} [J x + K x^2 + L x^4]_{aa}
    - sum_<ab> sum_{i,j} [G x^3 + H4 x^4 + G1 x]_{ab},   x = S_ai . S_bj

Spins are stored as an array of shape (N, n, 3).  Numba kernels receive the
couplings as the float array ``p = [J, K, L, G, H4, G1]``.
"""
from dataclasses import dataclass

import numpy as np
from numba import njit


@dataclass(frozen=True)
class Params:
    J: float = 1.0
    K: float = 0.5
    L: float = 0.0
    G: float = 0.0
    H4: float = 0.0
    G1: float = 0.0
    z: int = 4
    n: int = 4

    @classmethod
    def from_z(cls, zG=0.0, zH4=0.0, z=4, **kw):
        """Build from the products zG, zH4 used in the T=0 phase diagram."""
        return cls(G=zG / z, H4=zH4 / z, z=z, **kw)

    @property
    def zG(self):
        return self.z * self.G

    @property
    def zH4(self):
        return self.z * self.H4

    def array(self):
        return np.array([self.J, self.K, self.L, self.G, self.H4, self.G1], dtype=np.float64)


# ----------------------------------------------------------------- pair terms
@njit(cache=True, inline="always")
def f_pair(x, p):
    x2 = x * x
    return p[0] * x + p[1] * x2 + p[2] * x2 * x2


@njit(cache=True, inline="always")
def g_pair(x, p):
    x2 = x * x
    return p[3] * x2 * x + p[4] * x2 * x2 + p[5] * x


# --------------------------------------------------------------- total energy
@njit(cache=True)
def energy_total(spins, nbr, p):
    """Full direct spin sum.  Each bond is counted once (first z/2 columns)."""
    N, n, _ = spins.shape
    e = 0.0
    for a in range(N):
        for i in range(n):
            for j in range(i + 1, n):
                x = (spins[a, i, 0] * spins[a, j, 0] + spins[a, i, 1] * spins[a, j, 1]
                     + spins[a, i, 2] * spins[a, j, 2])
                e += f_pair(x, p)
        for k in range(nbr.shape[1] // 2):
            b = nbr[a, k]
            for i in range(n):
                for j in range(n):
                    x = (spins[a, i, 0] * spins[b, j, 0] + spins[a, i, 1] * spins[b, j, 1]
                         + spins[a, i, 2] * spins[b, j, 2])
                    e -= g_pair(x, p)
    return e


# ------------------------------------------------------- local energy helpers
@njit(cache=True)
def spin_energy(spins, nbr, p, a, i, sx, sy, sz):
    """Energy terms that involve spin (a,i) if it pointed along (sx,sy,sz):
    n-1 intra-cluster pairs plus n pairs per neighbouring cluster."""
    n = spins.shape[1]
    e = 0.0
    for j in range(n):
        if j != i:
            x = sx * spins[a, j, 0] + sy * spins[a, j, 1] + sz * spins[a, j, 2]
            e += f_pair(x, p)
    for k in range(nbr.shape[1]):
        b = nbr[a, k]
        for j in range(n):
            x = sx * spins[b, j, 0] + sy * spins[b, j, 1] + sz * spins[b, j, 2]
            e -= g_pair(x, p)
    return e


@njit(cache=True)
def cluster_inter_energy(spins, nbr, p, a, rows):
    """Inter-site energy between cluster ``a`` (taken to have spins ``rows``,
    shape (n,3)) and its neighbours.  Intra-cluster energy is not included."""
    n = spins.shape[1]
    e = 0.0
    for k in range(nbr.shape[1]):
        b = nbr[a, k]
        for i in range(n):
            for j in range(n):
                x = (rows[i, 0] * spins[b, j, 0] + rows[i, 1] * spins[b, j, 1]
                     + rows[i, 2] * spins[b, j, 2])
                e -= g_pair(x, p)
    return e


# --------------------------------------------------------- Legendre channels
def legendre(l, x):
    x = np.asarray(x, dtype=np.float64)
    if l == 1:
        return x
    if l == 2:
        return 0.5 * (3 * x ** 2 - 1)
    if l == 3:
        return 0.5 * (5 * x ** 3 - 3 * x)
    if l == 4:
        return (35 * x ** 4 - 30 * x ** 2 + 3) / 8
    raise ValueError(l)


def legendre_deriv(l, x):
    x = np.asarray(x, dtype=np.float64)
    if l == 1:
        return np.ones_like(x)
    if l == 2:
        return 3 * x
    if l == 3:
        return 0.5 * (15 * x ** 2 - 3)
    if l == 4:
        return 0.5 * (35 * x ** 3 - 15 * x)
    raise ValueError(l)


def channel_coeffs(P: Params):
    """Return (kappa[1..4], const) with E_cluster = 1/2 sum_l kappa_l A_l + const
    for a uniform state; A_l = sum_{i,j} P_l(S_i.S_j) including i=j."""
    J, K, L, G, H, G1, z, n = P.J, P.K, P.L, P.G, P.H4, P.G1, P.z, P.n
    f = np.array([J, 2 * K / 3 + 4 * L / 7, 0.0, 8 * L / 35])
    g = np.array([3 * G / 5 + G1, 4 * H / 7, 2 * G / 5, 8 * H / 35])
    kappa = f - z * g
    cf, cg = K / 3 + L / 5, H / 5
    const = n * (n - 1) / 2 * cf - z * n * n / 2 * cg - n / 2 * f.sum()
    return kappa, const


def channel_A(S):
    """A_l = sum_{i,j} P_l(S_i.S_j), l=1..4, for one cluster S of shape (n,3)."""
    X = S @ S.T
    return np.array([legendre(l, X).sum() for l in (1, 2, 3, 4)])


def channel_energy(S, P: Params):
    """Energy per cluster of the uniform state with cluster configuration S."""
    kappa, const = channel_coeffs(P)
    return 0.5 * float(kappa @ channel_A(S)) + const


# ------------------------------------------------------------ analytic values
def analytic_energies(P: Params):
    """Exact T=0 energies per cluster (n=4, L=G1=0) of the T, C, F states."""
    J, K, G, H, z = P.J, P.K, P.G, P.H4, P.z
    return {
        "T": -2 * J + 2 * K / 3 - z * (16 * G / 9 + 56 * H / 27),
        "C": -2 * J + 6 * K - 8 * z * H,
        "F": 6 * J + 6 * K - 8 * z * (G + H),
    }


def tetrahedron():
    return np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], float) / np.sqrt(3)


def random_spins(shape, rng):
    v = rng.normal(size=tuple(shape) + (3,))
    return v / np.linalg.norm(v, axis=-1, keepdims=True)
