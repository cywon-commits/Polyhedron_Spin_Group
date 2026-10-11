"""T=0 phase map in the (K, zG) plane of the physical model (L = G1 = 0, J = 1, z = 4).
Usage: python tools/map_D_plane.py out.png"""
import sys
from multiprocessing import Pool

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from polyspin.groundstate import classify_cluster, find_ground_state
from polyspin.model import Params

KS = np.linspace(-1.6, 0.6, 45)
ZG = np.linspace(0.4, 1.6, 61)
ZHS = (0.0, 0.25)
NAMES = ["T", "U", "D", "C", "F", "X"]


def work(a):
    K, zG, zH = a
    E, S = find_ground_state(Params.from_z(zG, zH, K=K), n_random=6, seed=3)
    return NAMES.index(classify_cluster(S, 0.999, m_tol=1e-3)), np.linalg.norm(S.sum(0)) / 4


if __name__ == "__main__":
    with Pool(4) as pool:
        res = {zH: np.array(pool.map(work, [(K, z, zH) for K in KS for z in ZG], chunksize=20))
               for zH in ZHS}
    cmap = ListedColormap(["#4c78a8", "#59a14f", "#e15759", "#f2cf5b", "#b07aa1", "#bbbbbb"])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), constrained_layout=True)
    for ax, zH in zip(axes, ZHS):
        ph = res[zH][:, 0].reshape(len(KS), len(ZG))
        ax.pcolormesh(ZG, KS, ph, cmap=cmap, vmin=-0.5, vmax=5.5, shading="nearest")
        for i, nm in enumerate(NAMES):
            if (ph == i).any():
                yy, xx = np.argwhere(ph == i).mean(0)
                ax.text(ZG[int(round(xx))], KS[int(round(yy))], nm, ha="center", va="center", fontsize=13)
        ax.axvline(1.0, color="k", ls=":", lw=1)
        ax.set_xlabel("zG / J")
        ax.set_ylabel("K / J")
        ax.set_title(f"T = 0 phases, zH4 = {zH}, L = G1 = 0")
    fig.suptitle("D (bent dimers) exists in the physical plane for K <~ 0   (dotted: zG = J)")
    fig.savefig(sys.argv[1], dpi=130)
