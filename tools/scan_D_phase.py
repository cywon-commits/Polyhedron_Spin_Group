"""Search for the D phase (bent dimer pair, {22} with M != 0) in extended planes.

Physical plane: J=1, G1=0, G,H4,G2 >= 0 (uniform-state theorem needs g_l >= 0), any K, L.
  kappa1 = J - 3zG/5,  kappa2 = 2K/3 + 4L/7 - z(4H4/7 + 2 G2/3),
  kappa3 = -2zG/5,     kappa4 = 8L/35 - 8 z H4/35
G2 is an inter-site quadrupole coupling  -G2 * sum_ij x_ij^2  (not in the current Hamiltonian).
Only (zG, kappa2, kappa4) matter, so scan that 3D space directly and count phases.
"""
import itertools
import sys

import numpy as np

from polyspin.groundstate import classify_cluster, find_ground_state_kappa

z = 4
rng = np.random.default_rng(0)
counts, examples = {}, {}
grid_zG = np.linspace(0.0, 2.5, 26)
grid_k2 = np.linspace(-6, 6, 25)
grid_k4 = np.linspace(-3, 3, 13)
total = 0
for zG, k2, k4 in itertools.product(grid_zG, grid_k2, grid_k4):
    kappa = np.array([1 - 0.6 * zG, k2, -0.4 * zG, k4])
    # fewer starts for the coarse scan
    E, S = find_ground_state_kappa(kappa, 0.0, n_random=4, seed=total)
    ph = classify_cluster(S, thr=0.999)
    counts[ph] = counts.get(ph, 0) + 1
    examples.setdefault(ph, []).append((zG, k2, k4, E))
    total += 1
print("scanned", total, "points; phase counts:", counts)
for ph in ("D", "X"):
    if ph in examples:
        print(ph, "examples:", examples[ph][:8])
