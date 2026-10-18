"""Step 2 of the n = 6 check (decision Oct 18): the T = 0 competitors at K = 0.5, zH4 = 1.5 near zG = J for six spins per cluster.
Global minimisation of the uniform-state cluster energy (uniform state theorem: g_l >= 0) from many random starts plus the collinear starts;
distinct minima are listed with their spin-group partition and net moment.  Usage: python tools/n6_groundstate.py"""
import numpy as np
from polyspin.groundstate import minimize_cluster, partition
from polyspin.harmonic import collinear_state
from polyspin.model import Params

n = 6
rng = np.random.default_rng(7)
for zG in (0.95, 1.0, 1.05, 1.1, 1.2):
    P = Params.from_z(zG, 1.5, z=4, K=0.5, n=n)
    starts = [collinear_state(n, k) for k in range(4)]
    for _ in range(300):
        v = rng.normal(size=(n, 3)); starts.append(v / np.linalg.norm(v, axis=1, keepdims=True))
    for _ in range(100):                                 # noisy collinear and grouped starts
        k = rng.integers(0, 4); S = collinear_state(n, k) + 0.3 * rng.normal(size=(n, 3)); starts.append(S / np.linalg.norm(S, axis=1, keepdims=True))
    res = sorted((minimize_cluster(P, S0) for S0 in starts), key=lambda r: r[0])
    seen = []
    for E, S in res:
        part = partition(S, 0.999); M = float(np.linalg.norm(S.sum(0)))
        key = (round(E, 5), part)
        if key not in [s[0] for s in seen]:
            seen.append((key, E, part, M))
    print(f"zG = {zG}: lowest distinct minima (energy per cluster, group sizes, |net moment|, collinear?):")
    for key, E, part, M in seen[:7]:
        coll = (len(part) == 2 and sum(part) == n)
        print(f"   E = {E:.5f}  partition {part}  |M| = {M:.3f}  {'collinear {a,b}' if coll else ''}")
