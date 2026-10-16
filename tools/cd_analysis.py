"""C and D cuts (2D L = 16): H5 (U phase) and the F -> C re-entrance checks with the rules of docs/CD_rules.md.
Usage: python tools/cd_analysis.py [--size 16]"""
import argparse, glob
import h5py, numpy as np
from polyspin.analysis.cd_rules import (h5_intervals, h5_verdict, reentrance_T, reentrance_verdict, RE_T_PRED, RE_T_RANGE)

ap = argparse.ArgumentParser(); ap.add_argument("--size", type=int, default=16); a = ap.parse_args()

def series(path):
    f = h5py.File(path, "r"); T = f["T"][:]
    m1, m2 = f["series/m1"][:].mean(1), f["series/m2"][:].mean(1)
    sp = f["diag/split"][:] / np.maximum(f["diag/n_diag"][:][:, None], 1)
    return T, m1, m2, sp[:, 1], f

print("## H5 (U phase, cut C)\n")
print("| zG | T range of the interval(s) with {211} > 0.5 and m1 in [0.2, 0.3] (>= 3 consecutive temperatures) | points | m1 range (heat ladder) | {211} at the lowest T |")
print("|---|---|---|---|---|")
per = {}
for zg in (1.0, 1.2):
    T, m1, m2, s211, f = series(f"runs/stage2/C_2d_L{a.size}_zG{zg:g}_heat.h5")
    iv = h5_intervals(T, s211, m1); per[zg] = iv
    print(f"| {zg:g} | {'; '.join(f'{i[0]:.3f}-{i[1]:.3f}' for i in iv) or 'none'} | {sum(i[2] for i in iv)} | {m1.min():.2f}-{m1.max():.2f} | {s211[np.argmin(T)]:.2f} |")
print(f"\n**H5: {h5_verdict(per)}**\n")

print("## Re-entrance F -> C (cut D, zG = 1.1)\n")
try:
    Th, m1h, m2h, _, fh = series(f"runs/stage2/D_2d_L{a.size}_zG1.1_heat.h5")
    Tc_, m1c, m2c, _, fc = series(f"runs/stage2/D_2d_L{a.size}_zG1.1_cool.h5")
    hh, hc = reentrance_T(Th, m1h, m2h), reentrance_T(Tc_, m1c, m2c)
    print(f"| ladder | F seen at low T (m1 >= 0.8) | found | T_re | in [{RE_T_RANGE[0]}, {RE_T_RANGE[1]}] |\n|---|---|---|---|---|")
    print(f"| heating (ordered start) | {m1h.max():.2f} | {hh[0]} | {hh[1]:.3f} | {RE_T_RANGE[0] <= hh[1] <= RE_T_RANGE[1] if hh[0] else 'n/a'} |")
    print(f"| cooling (random start) | {m1c.max():.2f} | {hc[0]} | {hc[1]:.3f} | {RE_T_RANGE[0] <= hc[1] <= RE_T_RANGE[1] if hc[0] else 'n/a'} |")
    print(f"\n**Re-entrance: {reentrance_verdict(hh, hc)}** (prediction from the H6 fit: T_re ~ {RE_T_PRED}, accepted {RE_T_RANGE})")
    print(f"\nheating ladder m1(T): " + " ".join(f"{t:.2f}:{x:.2f}" for t, x in zip(Th[::4], m1h[::4])))
    print(f"cooling ladder m1(T): " + " ".join(f"{t:.2f}:{x:.2f}" for t, x in zip(Tc_[::4], m1c[::4])))
except OSError:
    print("D zG = 1.1 runs not finished yet")
