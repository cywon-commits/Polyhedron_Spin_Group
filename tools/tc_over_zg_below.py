"""Prediction (user, Oct 13): T_c / (zG <m3(T_c^-)>^2) is nearly constant in zG, because at the transition the
fluctuation of the cluster shapes reduces the effective octupole.  <m3(T_c^-)> is the PT-ladder m3 interpolated
at 0.95 T_c (fixed before looking; 0.90 and 0.98 are shown as sensitivity), T_c = specific-heat peak.
Existing data only.  Usage: python tools/tc_over_zg_below.py"""
import glob, re
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

rows = {}
for p in sorted(glob.glob("data/stage1/A_*_pt.npz")):
    dim, L, zg = re.match(r"A_(\d)d_L(\d+)_zG([\d.]+)_pt", p.split("/")[-1]).groups()
    z = np.load(p, allow_pickle=False); T = z["T"]; Tc = float(T[int(np.argmax(z["th_c"]))])
    m3 = lambda f: float(np.interp(f * Tc, T, z["th_m"][:, 2]))
    rows[(int(dim), int(L), float(zg))] = (Tc, m3(0.90), m3(0.95), m3(0.98))
print("| dim | L | zG | T_c | m3(0.90 T_c) | m3(0.95 T_c) | m3(0.98 T_c) | T_c/zG | T_c/(zG m3(0.95 T_c)^2) | (0.90) | (0.98) |")
print("|---|---|---|---|---|---|---|---|---|---|---|")
for (dim, L, zg), (Tc, a, b, c) in sorted(rows.items()):
    print(f"| {dim}D | {L} | {zg:g} | {Tc:.4f} | {a:.3f} | {b:.3f} | {c:.3f} | {Tc/zg:.4f} | {Tc/(zg*b**2):.4f} | {Tc/(zg*a**2):.4f} | {Tc/(zg*c**2):.4f} |")
print()
for dim in (2, 3):
    Lm = max(L for L in {k[1] for k in rows if k[0] == dim} if all((dim, L, z) in rows for z in (0.5, 1.0, 1.5)))
    for name, idx in (("0.95", 2), ("0.90", 1), ("0.98", 3)):
        r = [rows[(dim, Lm, z)][0] / (z * rows[(dim, Lm, z)][idx] ** 2) for z in (0.5, 1.0, 1.5)]
        print(f"- {dim}D L={Lm}, m3 at {name} T_c: ratio over zG = 0.5, 1.0, 1.5: {np.round(r, 3).tolist()}; max/min = {max(r)/min(r):.2f}; "
              f"T_c/zG alone: max/min = {max(rows[(dim, Lm, z)][0]/z for z in (0.5,1,1.5))/min(rows[(dim, Lm, z)][0]/z for z in (0.5,1,1.5)):.2f}")
fig, ax = plt.subplots(1, 2, figsize=(9, 3.6)); cols = {8: "#0072B2", 10: "#D55E00", 12: "#009E73", 16: "#0072B2", 24: "#D55E00", 32: "#009E73", 14: "#CC79A7"}
for dim, mk in ((2, "o"), (3, "s")):
    for L in sorted({k[1] for k in rows if k[0] == dim}):
        zs = sorted(k[2] for k in rows if k[0] == dim and k[1] == L)
        ax[0].plot(zs, [rows[(dim, L, z)][0] / z for z in zs], mk + "-", color=cols[L], mfc="none" if dim == 2 else cols[L], label=f"{dim}D L={L}")
        ax[1].plot(zs, [rows[(dim, L, z)][0] / (z * rows[(dim, L, z)][2] ** 2) for z in zs], mk + "-", color=cols[L], mfc="none" if dim == 2 else cols[L])
ax[0].set(xlabel="zG", ylabel="T_c / zG"); ax[1].set(xlabel="zG", ylabel="T_c / (zG m3(0.95 T_c)^2)")
ax[0].legend(fontsize=7); [x.grid(alpha=.3) for x in ax]; fig.tight_layout(); fig.savefig("docs/figures/stage1/tc_over_zg_below.png", dpi=150)
