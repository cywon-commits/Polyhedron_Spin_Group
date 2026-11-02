"""Helicity modulus (stiffness) of the spin bodies under a twist about an axis (docs/CD_rules.md 'Oct/Nov stiffness').

Convention: the bodies at lattice coordinate x are rotated rigidly by theta*x about the axis ``a`` (theta = twist per lattice spacing =
total twist / L).  Then only the inter-body terms of the x-bonds depend on theta: for the bond (a -> b = a + x) the pair argument becomes
x_ij(phi) = S_ai . R_phi S_bj with phi = theta, R_phi the rotation by phi about ``a``.  With g(x) = G x^3 + H4 x^4 + G1 x (H_inter = -sum g):
    d x/d phi (0) = S_ai . (a x S_bj),   d^2 x/d phi^2 (0) = -S_ai . S_bj + (a . S_ai)(a . S_bj),
    d h/d phi = -sum_ij g'(x) x',   d^2 h/d phi^2 = -sum_ij [g''(x) x'^2 + g'(x) x''].
The stiffness (= rho_s for the per-spacing convention) is  Y = L^-d [ <S2> - beta <S1^2> ]  with S1 = sum_bonds dh/dphi, S2 = sum_bonds d^2h/dphi^2
(Y has the dimension of an energy; L^-d divides by the number of bodies).
"""
import numpy as np
from numba import njit, prange


@njit(cache=True)
def _bond_sums(spins, nbr, G, H4, G1, axis, col):
    """(S1, S2) summed over all bonds of direction ``col`` (0 = x, 1 = y) of one configuration spins (N, n, 3) for the twist axis ``axis``."""
    N, n, _ = spins.shape
    S1 = 0.0
    S2 = 0.0
    for a in range(N):
        b = nbr[a, col]
        for i in range(n):
            sa = spins[a, i]
            sa_ax = axis[0] * sa[0] + axis[1] * sa[1] + axis[2] * sa[2]
            for j in range(n):
                sb = spins[b, j]
                x = sa[0] * sb[0] + sa[1] * sb[1] + sa[2] * sb[2]
                cx = axis[1] * sb[2] - axis[2] * sb[1]
                cy = axis[2] * sb[0] - axis[0] * sb[2]
                cz = axis[0] * sb[1] - axis[1] * sb[0]
                xp = sa[0] * cx + sa[1] * cy + sa[2] * cz
                sb_ax = axis[0] * sb[0] + axis[1] * sb[1] + axis[2] * sb[2]
                xpp = -x + sa_ax * sb_ax
                g1 = 3.0 * G * x * x + 4.0 * H4 * x * x * x + G1
                g2 = 6.0 * G * x + 12.0 * H4 * x * x
                S1 -= g1 * xp
                S2 -= g2 * xp * xp + g1 * xpp
    return S1, S2


@njit(cache=True)
def director(spins):
    """Principal axis (unit vector) of Q = sum over all spins of (3 s s^T - 1)/2: the instantaneous global director."""
    N, n, _ = spins.shape
    Q = np.zeros((3, 3))
    for a in range(N):
        for i in range(n):
            s = spins[a, i]
            for u in range(3):
                for v in range(3):
                    Q[u, v] += 1.5 * s[u] * s[v]
    for u in range(3):
        Q[u, u] -= 0.5 * N * n
    w, V = np.linalg.eigh(Q)
    return V[:, 2].copy()


@njit(cache=True)
def terms(spins, nbr, p, col):
    """(S1, S2) for the lab axes x, y, z and the instantaneous director: arrays (4,) each; p = [J, K, L, G, H4, G1]."""
    out1 = np.zeros(4)
    out2 = np.zeros(4)
    for k in range(3):
        ax = np.zeros(3)
        ax[k] = 1.0
        out1[k], out2[k] = _bond_sums(spins, nbr, p[3], p[4], p[5], ax, col)
    d = director(spins)
    out1[3], out2[3] = _bond_sums(spins, nbr, p[3], p[4], p[5], d, col)
    return out1, out2


@njit(cache=True)
def twisted_inter_energy(spins, nbr, p, axis, phi, col):
    """Direct (finite-phi) sum of the x- (or y-) bond inter-body energies with the relative rotation phi about ``axis`` across every bond:
    used to test the analytic derivatives and the fluctuation formula."""
    N, n, _ = spins.shape
    c, s = np.cos(phi), np.sin(phi)
    E = 0.0
    for a in range(N):
        b = nbr[a, col]
        for j in range(n):
            sb = spins[b, j]
            ad = axis[0] * sb[0] + axis[1] * sb[1] + axis[2] * sb[2]
            rx = sb[0] * c + (axis[1] * sb[2] - axis[2] * sb[1]) * s + axis[0] * ad * (1 - c)
            ry = sb[1] * c + (axis[2] * sb[0] - axis[0] * sb[2]) * s + axis[1] * ad * (1 - c)
            rz = sb[2] * c + (axis[0] * sb[1] - axis[1] * sb[0]) * s + axis[2] * ad * (1 - c)
            for i in range(n):
                sa = spins[a, i]
                x = sa[0] * rx + sa[1] * ry + sa[2] * rz
                E -= p[3] * x * x * x + p[4] * x * x * x * x + p[5] * x
    return E


@njit(parallel=True, cache=True)
def terms_all(spins_all, nbr, p, col):
    """(R, 4) arrays S1, S2 for every replica."""
    R = spins_all.shape[0]
    o1 = np.zeros((R, 4))
    o2 = np.zeros((R, 4))
    for r in prange(R):
        a, b = terms(spins_all[r], nbr, p, col)
        o1[r] = a
        o2[r] = b
    return o1, o2


def stiffness(S1, S2, beta, n_bodies):
    """Y = [<S2> - beta <S1^2>] / n_bodies from samples S1, S2 (1-D arrays)."""
    S1, S2 = np.asarray(S1, float), np.asarray(S2, float)
    return (S2.mean() - beta * np.mean(S1 ** 2)) / n_bodies
