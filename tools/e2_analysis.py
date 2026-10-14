"""E' first range: T1 (chi_1 peak), T2 (chi_2 peak; heat/cool, lower used) and the H7 / H12 verdicts of docs/E2_rules.md.
Usage: python tools/e2_analysis.py [--dim 2]"""
import argparse, glob, re
import h5py, numpy as np
from polyspin.analysis.e2_rules import peak_sigma, h7_verdict, h12_verdict, T1_WINDOW
from polyspin.analysis.finite_size import first_order_signature, energy_cumulant, cumulant_trend

ap = argparse.ArgumentParser(); ap.add_argument("--dim", type=int, default=None); a = ap.parse_args()
runs = {}
for p in sorted(glob.glob("runs/stage2/E2_*_heat.h5")):
    dim, L, zg = re.match(r"E2_(\d)d_L(\d+)_zG([\d.]+)_heat", p.split("/")[-1]).groups()
    if a.dim and int(dim) != a.dim: continue
    runs.setdefault((int(dim), float(zg)), {})[int(L)] = (p, p.replace("_heat", "_cool"))

def interior(T, y):
    i = int(np.argmax(y)); return 0 < i < len(T) - 1

print("| dim | zG | L | T1 (chi1) | sigma1 | T2 heat | s | T2 cool (interior?) | s | T2 used | gap | 3 sigma | c-peak(heat, near T2) |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for (dim, zg), sizes in sorted(runs.items()):
    per = {}
    for L, (ph, pc) in sorted(sizes.items()):
        fh = h5py.File(ph, "r"); N = L ** dim; T = fh["T"][:]
        t1, s1 = peak_sigma(T, fh["series/m1"][:], N)
        t2h, s2h = peak_sigma(T, fh["series/m2"][:], N)
        t2c = s2c = None; ok_c = False
        try:
            fc = h5py.File(pc, "r"); Tc_ = fc["T"][:]
            chi2c = N * (fc["series/m2"][:].var(1)) / Tc_
            ok_c = interior(Tc_, chi2c)
            if ok_c: t2c, s2c = peak_sigma(Tc_, fc["series/m2"][:], N)
        except Exception:
            pass
        if ok_c and t2c < t2h: t2, s2 = t2c, s2c
        else: t2, s2 = t2h, s2h
        per[L] = dict(T1=t1, s1=s1, T2=t2, s2=s2)
        e = fh["series/e"][:]; c = e.var(1) * N / T ** 2
        print(f"| {dim}D | {zg:g} | {L} | {t1:.4f} | {s1:.4f} | {t2h:.4f} | {s2h:.4f} | " + (f"{t2c:.4f} ({'yes'})" if ok_c else "n/a (edge max)") + f" | {s2c if ok_c else float('nan'):.4f} | {t2:.4f} | {t2 - t1:.4f} | {3 * np.hypot(s1, s2):.4f} | {c[np.argmin(abs(T - t2h))]:.1f} |")
    v, d = h7_verdict(per)
    print(f"\n**H7, {dim}D zG = {zg:g}: {v}** (gap by L: {{{', '.join(f'{L}: {g:.3f}' for L, g in d['gap'].items())}}}; all > 3 sigma: {d['all_3sigma']}; gap kept >= 70%: {d['gap_kept']})\n")
