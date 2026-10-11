"""Delineate the D phase in the (kappa2, kappa4) plane at fixed zG and map it to the physical
parameters (J=1, G1=0, zH4=0): kappa4 = 8L/35, kappa2 = 2K/3 + 4L/7, kappa1 = 1 - 3zG/5,
kappa3 = -2zG/5.  With zH4 = 0 and G >= 0 all g_l >= 0, so the uniform-state theorem applies.
Usage: python tools/scan_D_region.py out.npz
"""
import sys
from multiprocessing import Pool

import numpy as np

from polyspin.groundstate import classify_cluster, find_ground_state_kappa, partition

ZG = (0.0, 0.5, 0.9)
K2 = np.linspace(-10.0, 2.0, 49)
K4 = np.linspace(0.0, 3.0, 25)


def work(args):
    zG, k2, k4, seed = args
    kappa = np.array([1 - 0.6 * zG, k2, -0.4 * zG, k4])
    E, S = find_ground_state_kappa(kappa, 0.0, n_random=6, seed=seed)
    ph = classify_cluster(S, thr=0.999)
    M = np.linalg.norm(S.sum(0)) / 4
    return ph, E, M, partition(S, 0.999)


if __name__ == "__main__":
    jobs = [(zG, k2, k4, i) for i, (zG, k2, k4) in enumerate(
        (a, b, c) for a in ZG for b in K2 for c in K4)]
    res = []
    with Pool(4) as pool:
        for i, r in enumerate(pool.imap(work, jobs, chunksize=10)):
            res.append(r)
            if i % 300 == 0:
                print("progress", i, "/", len(jobs), flush=True)
    ph = np.array([r[0] for r in res]).reshape(len(ZG), len(K2), len(K4))
    M = np.array([r[2] for r in res]).reshape(ph.shape)
    parts = np.array([str(r[3]) for r in res]).reshape(ph.shape)
    np.savez(sys.argv[1], ph=ph, M=M, parts=parts, ZG=ZG, K2=K2, K4=K4)
    for i, zG in enumerate(ZG):
        u, c = np.unique(ph[i], return_counts=True)
        print("zG", zG, dict(zip(u, c)))
