"""Monte Carlo moves.  All kernels take a per-replica xoshiro state ``s``.

Move set (one sweep = every spin tried once + every cluster rotated once):
  * single-spin Gaussian step
  * rigid SO(3) rotation of a whole cluster (intra-cluster energy unchanged)
  * overlap/split: independence proposal that either re-draws a spin uniformly
    on the sphere or places it near another spin of the same cluster
    (von Mises-Fisher, concentration kappa).  Acceptance uses the exact
    proposal-density ratio, so detailed balance holds.
  * label permutation (optional, always accepted; diagnostic only)

counters layout (acceptance, attempts): [single, rotation, overlap].
"""
import numpy as np
from numba import njit

from .model import cluster_inter_energy, f_pair, g_pair, spin_energy  # noqa: F401
from .rng import rand, randint, randn

FOUR_PI = 4.0 * np.pi
N_COUNTERS = 3


@njit(cache=True)
def single_spin_sweep(spins, nbr, p, beta, delta, s, counters):
    N, n, _ = spins.shape
    for a in range(N):
        for i in range(n):
            ox, oy, oz = spins[a, i, 0], spins[a, i, 1], spins[a, i, 2]
            nx = ox + delta * randn(s)
            ny = oy + delta * randn(s)
            nz = oz + delta * randn(s)
            r = np.sqrt(nx * nx + ny * ny + nz * nz)
            nx, ny, nz = nx / r, ny / r, nz / r
            de = (spin_energy(spins, nbr, p, a, i, nx, ny, nz)
                  - spin_energy(spins, nbr, p, a, i, ox, oy, oz))
            counters[0, 1] += 1
            if de <= 0.0 or rand(s) < np.exp(-beta * de):
                spins[a, i, 0], spins[a, i, 1], spins[a, i, 2] = nx, ny, nz
                counters[0, 0] += 1


@njit(cache=True)
def _rotate_rows(rows_in, wx, wy, wz, rows_out):
    th = np.sqrt(wx * wx + wy * wy + wz * wz)
    if th < 1e-14:
        rows_out[:] = rows_in
        return
    kx, ky, kz = wx / th, wy / th, wz / th
    c, sn = np.cos(th), np.sin(th)
    for i in range(rows_in.shape[0]):
        vx, vy, vz = rows_in[i, 0], rows_in[i, 1], rows_in[i, 2]
        kv = kx * vx + ky * vy + kz * vz
        rows_out[i, 0] = vx * c + (ky * vz - kz * vy) * sn + kx * kv * (1 - c)
        rows_out[i, 1] = vy * c + (kz * vx - kx * vz) * sn + ky * kv * (1 - c)
        rows_out[i, 2] = vz * c + (kx * vy - ky * vx) * sn + kz * kv * (1 - c)


@njit(cache=True)
def rotation_sweep(spins, nbr, p, beta, delta_r, s, counters):
    N, n, _ = spins.shape
    new = np.empty((n, 3))
    for a in range(N):
        wx, wy, wz = delta_r * randn(s), delta_r * randn(s), delta_r * randn(s)
        _rotate_rows(spins[a], wx, wy, wz, new)
        de = (cluster_inter_energy(spins, nbr, p, a, new)
              - cluster_inter_energy(spins, nbr, p, a, spins[a]))
        counters[1, 1] += 1
        if de <= 0.0 or rand(s) < np.exp(-beta * de):
            spins[a] = new
            counters[1, 0] += 1


@njit(cache=True)
def _vmf_density(x, kappa):
    """von Mises-Fisher density on S^2 as a function of x = y.mu."""
    return kappa / (2.0 * np.pi * (1.0 - np.exp(-2.0 * kappa))) * np.exp(kappa * (x - 1.0))


@njit(cache=True)
def _proposal_density(spins, a, i, vx, vy, vz, kappa):
    n = spins.shape[1]
    acc = 0.0
    for j in range(n):
        if j != i:
            x = vx * spins[a, j, 0] + vy * spins[a, j, 1] + vz * spins[a, j, 2]
            acc += _vmf_density(x, kappa)
    return 0.5 / FOUR_PI + 0.5 * acc / (n - 1)


@njit(cache=True)
def overlap_sweep(spins, nbr, p, beta, kappa, s, counters):
    N, n, _ = spins.shape
    for a in range(N):
        i = randint(s, n)
        ox, oy, oz = spins[a, i, 0], spins[a, i, 1], spins[a, i, 2]
        if rand(s) < 0.5:  # uniform re-draw
            nx, ny, nz = randn(s), randn(s), randn(s)
            r = np.sqrt(nx * nx + ny * ny + nz * nz)
            nx, ny, nz = nx / r, ny / r, nz / r
        else:  # attach near another spin
            j = randint(s, n - 1)
            if j >= i:
                j += 1
            mx, my, mz = spins[a, j, 0], spins[a, j, 1], spins[a, j, 2]
            u = rand(s)
            w = 1.0 + np.log(u + (1.0 - u) * np.exp(-2.0 * kappa)) / kappa
            sin_t = np.sqrt(max(0.0, 1.0 - w * w))
            phi = 2.0 * np.pi * rand(s)
            # orthonormal frame around mu
            if abs(mx) < 0.9:
                tx, ty, tz = 1.0, 0.0, 0.0
            else:
                tx, ty, tz = 0.0, 1.0, 0.0
            e1x = ty * mz - tz * my
            e1y = tz * mx - tx * mz
            e1z = tx * my - ty * mx
            r1 = np.sqrt(e1x * e1x + e1y * e1y + e1z * e1z)
            e1x, e1y, e1z = e1x / r1, e1y / r1, e1z / r1
            e2x = my * e1z - mz * e1y
            e2y = mz * e1x - mx * e1z
            e2z = mx * e1y - my * e1x
            cp, sp = np.cos(phi), np.sin(phi)
            nx = w * mx + sin_t * (cp * e1x + sp * e2x)
            ny = w * my + sin_t * (cp * e1y + sp * e2y)
            nz = w * mz + sin_t * (cp * e1z + sp * e2z)
            r = np.sqrt(nx * nx + ny * ny + nz * nz)
            nx, ny, nz = nx / r, ny / r, nz / r
        de = (spin_energy(spins, nbr, p, a, i, nx, ny, nz)
              - spin_energy(spins, nbr, p, a, i, ox, oy, oz))
        q_new = _proposal_density(spins, a, i, nx, ny, nz, kappa)
        q_old = _proposal_density(spins, a, i, ox, oy, oz, kappa)
        counters[2, 1] += 1
        if rand(s) < np.exp(-beta * de) * q_old / q_new:
            spins[a, i, 0], spins[a, i, 1], spins[a, i, 2] = nx, ny, nz
            counters[2, 0] += 1


@njit(cache=True)
def permutation_sweep(spins, s):
    N, n, _ = spins.shape
    for a in range(N):
        for i in range(n - 1, 0, -1):
            j = randint(s, i + 1)
            for c in range(3):
                t = spins[a, i, c]
                spins[a, i, c] = spins[a, j, c]
                spins[a, j, c] = t


@njit(cache=True)
def mc_sweep(spins, nbr, p, beta, delta, delta_r, kappa, use_single, use_rot,
             use_overlap, use_perm, s, counters):
    """One sweep.  ``counters`` has shape (N_COUNTERS, 2)."""
    if use_single:
        single_spin_sweep(spins, nbr, p, beta, delta, s, counters)
    if use_rot:
        rotation_sweep(spins, nbr, p, beta, delta_r, s, counters)
    if use_overlap:
        overlap_sweep(spins, nbr, p, beta, kappa, s, counters)
    if use_perm:
        permutation_sweep(spins, s)
