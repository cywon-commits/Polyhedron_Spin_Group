"""Wedge ladder of collinear states for n spins per cluster (theory section, no MC): harmonic + label free energies of the states with k reversed
spins (M = n - 2k) in the same model family (K = 0.5, zH4 = 1.5).  Delta F_k / T = a_k u + delta_k with u = x/T, x = zG - J, a_k = (n^2 - M^2)/2,
delta_k = (logdet_k - logdet_F)/2 - ln C(n, k).  State k is present in the equilibrium ladder iff it lies on the lower envelope of these lines.
Usage: python tools/n6_ladder.py [n]"""
import sys
from math import comb, log
import numpy as np
from polyspin.harmonic import collinear_state, log_det_per_cluster
from polyspin.model import Params


def ladder(n, dim, zG, zH4=1.5, K=0.5, Lq=None):
    Lq = Lq or {2: 96, 3: 24}[dim]
    z = {2: 4, 3: 6}[dim]
    P = Params.from_z(zG, zH4, z=z, K=K, n=n)
    ld = {k: log_det_per_cluster(collinear_state(n, k), P, dim, Lq)[0] for k in range(0, n // 2 + 1)}
    a = {k: (n ** 2 - (n - 2 * k) ** 2) / 2 for k in ld}
    d = {k: 0.5 * (ld[k] - ld[0]) - log(comb(n, k)) for k in ld}
    # lower envelope of lines a_k u + d_k over u > 0: start at u -> infinity (F), decrease u
    ks, u, seq = [0], np.inf, []
    cur = 0
    while True:
        best = None
        for k in ld:
            if a[k] > a[cur]:
                ucross = (d[k] - d[cur]) / (a[cur] - a[k]) * -1 if False else (d[cur] - d[k]) / (a[k] - a[cur])   # a_cur u + d_cur = a_k u + d_k
                if ucross > 0 and (best is None or ucross > best[0]) and ucross < u:
                    best = (ucross, k)
        if best is None:
            break
        seq.append((cur, best[1], best[0])); u, cur = best[0], best[1]
    return d, a, seq


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    for dim in (2, 3):
        for zG in (1.0, 1.1):
            d, a, seq = ladder(n, dim, zG)
            names = {k: f"M{n - 2 * k}" for k in d}
            print(f"n = {n}, dim {dim}, zG = {zG}: delta_k = " + ", ".join(f"{names[k]} {d[k]:.3f}" for k in d))
            print("   ladder (u = x/T boundaries, going down): " + " ; ".join(f"{names[i]} -> {names[j]} at x/T = {u:.4f}" for i, j, u in seq)
                  + f"   states in ladder: {['F'] + [names[j] for _, j, _ in seq]}")
            x = zG - 1.0
            if x > 0:
                print("   temperatures at x = %.2f: " % x + " ; ".join(f"{names[i]} -> {names[j]} at T = {x / u:.4f}" for i, j, u in seq))
