"""Tier-2 tables from data/stage1/*.npz: size dependence of c_peak / T_c / Binder / PT health (A, B pt),
heat-cool windows (B first-order points), defect statistics at T_c (coarse detector, plaquette) and scale ratio R_b.
Usage: python tools/tier2_tables.py"""
import glob, json, re
from collections import defaultdict
from pathlib import Path
import numpy as np
from polyspin.analysis.finite_size import loglog_slope

def load(p):
    z = np.load(p, allow_pickle=False); d = {k: z[k] for k in z.files}
    d["attrs"] = json.loads(str(z["attrs"])); return d

runs = defaultdict(dict)
for p in sorted(glob.glob("data/stage1/*.npz")):
    cut, dim, L, par, val, tag = re.match(r"(\w+?)_(\d)d_L(\d+)_(zG|zH4)([\d.]+)_(\w+)\.npz", Path(p).name).groups()
    runs[(cut, int(dim), float(val))][(int(L), tag)] = load(p)

def peak(d):
    i = int(np.argmax(d["th_c"])); return i
print("## PT points: size dependence")
print("| point | L | N | T_peak | c_peak | Binder min(m1..) | swap min | trips | rho_coarse(T_c) | f_ud,coarse(T_c) | f_ud,plaq(T_c) | R_b |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|")
exps = {}
for (cut, dim, val), g in sorted(runs.items()):
    Ls = sorted(L for (L, t) in g if t == "pt")
    if not Ls: continue
    Ns, cp, Tp = [], [], []
    for L in Ls:
        d = g[(L, "pt")]; i = peak(d)
        key = d['dm_coarse_und_frac'] if 'dm_coarse_und_frac' in d else None
        j = i
        if key is not None and not np.isfinite(key[i]):
            ok = np.where(np.isfinite(key))[0]; j = ok[np.argmin(abs(d['T'][ok] - d['T'][i]))] if len(ok) else i
        cd = d['dm_coarse_density'][j] if 'dm_coarse_density' in d else np.full(7, np.nan); dens = cd[:4].sum() if cd.ndim else cd  # all reportable classes approx
        names = json.loads(str(d.get('class_names', '[]'))) if False else ['1','6+','3+','4','6-','3-','2']
        rep = [k for k, nm in enumerate(names) if nm in ('4','6+','6-')]
        rho = cd[rep].sum()
        rb = d['dm_scale_ratio_R4'][j] if 'dm_scale_ratio_R4' in d else float('nan')
        print(f"| {cut} {dim}D {val} | {L} | {d['attrs']['N']} | {d['T'][i]:.4f} | {d['th_c'][i]:.1f} | {np.nanmin(d['th_binder'][:,1]):.3f} | {d['swap'].min():.2f} | {int(d['round_trips'])} | {rho:.2e} | {d['dm_coarse_und_frac'][j] if 'dm_coarse_und_frac' in d else float('nan'):.2f} | {d['dm_plaq_und_frac'][j]:.2f} | {rb:.2f} |")
        Ns.append(d["attrs"]["N"]); cp.append(d["th_c"][i]); Tp.append(d["T"][i])
    if len(Ls) >= 2:
        a, ea = loglog_slope(np.array(Ns, float), np.array(cp))
        exps[(cut, dim, val)] = (a, ea, Ls)
print("\n## c_peak ~ N^a (a=1: first order; 2D/3D continuous: alpha/(d nu))")
for k, (a, ea, Ls) in exps.items():
    print(f"- {k[0]} {k[1]}D {k[2]}: L={Ls}: a = {a:.2f} ± {ea:.2f}" if np.isfinite(ea) else f"- {k[0]} {k[1]}D {k[2]}: L={Ls}: a = {a:.2f}")
print("\n## First-order points: heat vs cool, nematic m2 half-point and energy jump")
print("| point | L | T_up | T_down | dT | c_peak(heat) | max de (heat) |")
print("|---|---|---|---|---|---|---|")
def half(d, l=1):
    T, m = d["T"], d["th_m"][:, l]; o = np.argsort(T); T, m = T[o], m[o]
    lo, hi = np.nanmin(m), np.nanmax(m); h = 0.5 * (lo + hi)
    for i in range(len(T) - 1):
        if (m[i] - h) * (m[i + 1] - h) <= 0 and m[i] != m[i + 1]:
            return T[i] + (h - m[i]) * (T[i + 1] - T[i]) / (m[i + 1] - m[i])
    return np.nan
for (cut, dim, val), g in sorted(runs.items()):
    for L in sorted({L for (L, t) in g if t == "heat"}):
        h, c = g[(L, "heat")], g[(L, "cool")]
        o = np.argsort(h["T"]); e = h["th_e"][o]
        print(f"| B {dim}D zH4={val} | {L} | {half(h):.4f} | {half(c):.4f} | {half(h)-half(c):+.4f} | {h['th_c'].max():.0f} | {np.nanmax(np.diff(e)):.2f} |")
