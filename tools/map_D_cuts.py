"""Phases along zG at fixed (K, zH4) in the physical plane (L = G1 = 0, J = 1, z = 4)."""
import sys
from multiprocessing import Pool

import numpy as np

from polyspin.groundstate import classify_cluster, find_ground_state
from polyspin.model import Params

CUTS = [(-1.5, 0.0), (-0.75, 0.0), (-0.5, 0.0), (-0.25, 0.0), (0.0, 0.0), (-0.25, 0.25), (0.0, 0.25), (0.0, 0.5), (0.5, 1.5)]
ZG = np.arange(0.5, 1.6001, 0.02)


def work(a):
    K, zH, zG = a
    E, S = find_ground_state(Params.from_z(zG, zH, K=K), n_random=8, seed=1)
    return classify_cluster(S, 0.999, m_tol=1e-3), np.linalg.norm(S.sum(0)) / 4


if __name__ == "__main__":
    with Pool(4) as pool:
        for K, zH in CUTS:
            res = pool.map(work, [(K, zH, z) for z in ZG])
            segs, cur = [], None
            for z, (ph, M) in zip(ZG, res):
                if cur is None or cur[0] != ph:
                    cur = [ph, z, z, M, M]
                    segs.append(cur)
                cur[2] = z
                cur[4] = M
            print(f"K={K}, zH4={zH}:", "  ".join(f"{p}[{a:.2f}-{b:.2f}, M/n {m0:.2f}->{m1:.2f}]"
                                                  for p, a, b, m0, m1 in segs), flush=True)
