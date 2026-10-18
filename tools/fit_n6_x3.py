"""n = 6 M2 interval, 3x-sweep remeasure verdict (rule: docs/CD_rules.md 'Oct 18 ... B').  Usage: python tools/fit_n6_x3.py"""
import json, collections, numpy as np
from polyspin.analysis import cc_fit as cc
def load(f):
    r = json.loads(str(np.load(f"data/stage2/{f}.npz")["results"])); return [x["T"] for x in r], [x["v"] for x in r]
def show(name, T, v):
    e = cc.crossing_seedmean(T, v)
    print(f"{name:34s} zero {e['zero']:.4f} +- {e['sigma_total']:.4f} (pct {e['sigma_pct']:.4f}, std {e['sigma_std']:.4f}, w {e['width']}, n_win {e['n_window']:.0f})"); return e
Tx, vx = load("slab_n6_CM2_zG1.1_x3")
d = collections.defaultdict(list)
for t, x in zip(Tx, vx): d[round(t, 3)].append(x)
print("seed-mean v(1e-4):", " ".join(f"{t:.3f}:{np.mean(w) * 1e4:+.1f}" for t, w in sorted(d.items())))
print("signs:", " ".join(f"{t:.3f}:{''.join('+' if x > 0 else '-' for x in w)}" for t, w in sorted(d.items())))
tc = show("Tc x3 (rule)", Tx, vx)
Ta = show("Ta orig", *load("slab_n6_M4F_zG1.1")); Tb = show("Tb orig", *load("slab_n6_M2M4_zG1.1"))
show("Tc orig (info)", *load("slab_n6_CM2_zG1.1")); to, vo = load("slab_n6_CM2_zG1.1"); ts, vs = load("slab_n6_CM2_zG1.1_supp")
show("Tc supp (info)", ts, vs)
g = tc["zero"] - Tb["zero"]; tol = 2 * np.hypot(tc["sigma_total"], Tb["sigma_total"])
print(f"Tc - Tb = {g:.4f} vs 2 sigma {tol:.4f}; order Ta<Tb<Tc: {Ta['zero'] < Tb['zero'] < tc['zero']}")
print("FINAL VERDICT (M2 interval):", "M2 confirmed" if g > tol and Ta["zero"] < Tb["zero"] < tc["zero"] else "M2 NOT resolved (final)")
print(f"TI prediction of the M2 width: 0.0184 (sigma 0.0080); measured {g:.4f}")
