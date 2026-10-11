"""D-phase search directly in the physical plane L = G1 = 0 (J = 1, z = 4) including K < 0.
Grid over (K, zG, zH4); each point minimised with 6 random + 20 structured starts.
Usage: python tools/scan_D_physical.py out.npz
"""
import sys
from multiprocessing import Pool

import numpy as np

from polyspin.groundstate import classify_cluster, find_ground_state, partition
from polyspin.model import Params

KS = np.linspace(-3.0, 1.0, 17)
ZG = np.linspace(0.0, 2.5, 26)
ZH = np.linspace(0.0, 1.5, 7)


def work(args):
    K, zG, zH, seed = args
    P = Params.from_z(zG, zH, z=4, K=K)
    E, S = find_ground_state(P, n_random=6, seed=seed)
    return classify_cluster(S, thr=0.999), E, np.linalg.norm(S.sum(0)) / 4, str(partition(S, 0.999))


if __name__ == "__main__":
    jobs = [(K, zG, zH, i) for i, (K, zG, zH) in enumerate(
        (a, b, c) for a in KS for b in ZG for c in ZH)]
    res = []
    with Pool(4) as pool:
        for i, r in enumerate(pool.imap(work, jobs, chunksize=10)):
            res.append(r)
            if i % 500 == 0:
                print("progress", i, "/", len(jobs), flush=True)
    ph = np.array([r[0] for r in res]).reshape(len(KS), len(ZG), len(ZH))
    M = np.array([r[2] for r in res]).reshape(ph.shape)
    np.savez(sys.argv[1], ph=ph, M=M, KS=KS, ZG=ZG, ZH=ZH)
    u, c = np.unique(ph, return_counts=True)
    print("counts", dict(zip(u.tolist(), c.tolist())))
    for k, a, b in zip(*np.where(ph == "D")):
        print("D at K=%.2f zG=%.2f zH4=%.2f  M/n=%.3f" % (KS[k], ZG[a], ZH[b], M[k, a, b]))
