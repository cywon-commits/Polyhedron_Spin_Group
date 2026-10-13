"""T_c/zG versus zG for the A cut (2D and 3D, all sizes) and the mean-field-type normalisation
T_c / (zG <m3>^2) with <m3> the octupole order parameter of the coldest chain (T = 0.02).
T_c = temperature of the specific-heat maximum of the PT ladder (this is MY definition of the mean-field
comparison: the request only says T_c ~ zG <q3>^2).  Usage: python tools/tc_over_zg.py --out docs/figures/stage1/tc_over_zg.png"""
import argparse, glob, re
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser(); ap.add_argument("--out", default="docs/figures/stage1/tc_over_zg.png"); a = ap.parse_args()
d = {}
for p in glob.glob("data/stage1/A_*_pt.npz"):
    dim, L, z = re.match(r"A_(\d)d_L(\d+)_zG([\d.]+)_pt", p.split("/")[-1]).groups()
    z = np.load(p, allow_pickle=False); T, c = z["T"], z["th_c"]
    cold = np.load(p.replace("_pt", "_cold"), allow_pickle=False) if glob.glob(p.replace("_pt", "_cold")) else None
    zg = float(re.search(r"zG([\d.]+)_pt", p).group(1))
    m3 = float(cold["th_m"][np.argmin(cold["T"]), 2]) if cold is not None else np.nan
    d[(int(dim), int(L), zg)] = (float(T[int(np.argmax(c))]), m3)
for k, (Tc, m3) in list(d.items()):                 # no cold chains at tier 3 (3D L=12): use the other size of the same point
    if not np.isfinite(m3):
        oth = [v[1] for kk, v in d.items() if kk[0] == k[0] and kk[2] == k[2] and np.isfinite(v[1])]
        d[k] = (Tc, float(np.mean(oth)) if oth else np.nan)
fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
cols = {8: "#0072B2", 10: "#D55E00", 12: "#009E73", 16: "#0072B2", 24: "#D55E00", 32: "#009E73", 14: "#CC79A7"}
print("| dim | L | zG | T_c | T_c/zG | <m3>(T=0.02) | T_c/(zG <m3>^2) |\n|---|---|---|---|---|---|---|")
for (dim, L, zg), (Tc, m3) in sorted(d.items()):
    print(f"| {dim}D | {L} | {zg:g} | {Tc:.4f} | {Tc/zg:.4f} | {m3:.3f} | {Tc/(zg*m3**2):.4f} |")
for dim, mk in ((2, "o"), (3, "s")):
    for L in sorted({k[1] for k in d if k[0] == dim}):
        pts = sorted((k[2], v) for k, v in d.items() if k[0] == dim and k[1] == L)
        x = [p[0] for p in pts]
        ax[0].plot(x, [p[1][0] / p[0] for p in pts], mk + "-", color=cols[L], mfc="none" if dim == 2 else cols[L], label=f"{dim}D L={L}")
        ax[1].plot(x, [p[1][0] / (p[0] * p[1][1] ** 2) for p in pts], mk + "-", color=cols[L], mfc="none" if dim == 2 else cols[L])
ax[0].set(xlabel="zG", ylabel="T_c / zG"); ax[1].set(xlabel="zG", ylabel="T_c / (zG <m3>^2), <m3> at T=0.02")
ax[0].legend(fontsize=7); [x.grid(alpha=.3) for x in ax]
fig.tight_layout(); fig.savefig(a.out, dpi=150)
