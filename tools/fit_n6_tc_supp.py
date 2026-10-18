"""n = 6 T_c refinement verdict (rule: docs/CD_rules.md 'n = 6 T_c 정밀화').  Usage: python tools/fit_n6_tc_supp.py"""
import json, collections, numpy as np
from polyspin.analysis import cc_fit as cc
def load(f):
    r = json.loads(str(np.load(f"data/stage2/{f}.npz")["results"])); return [x["T"] for x in r], [x["v"] for x in r]
def show(name, T, v):
    e = cc.crossing_seedmean(T, v)
    print(f"{name:30s} zero {e['zero']:.4f} +- {e['sigma_total']:.4f} (pct {e['sigma_pct']:.4f}, std {e['sigma_std']:.4f}, w {e['width']}, n_win {e['n_window']:.0f})"); return e
Ts, vs = load("slab_n6_CM2_zG1.1_supp")
d = collections.defaultdict(list)
for t, x in zip(Ts, vs): d[round(t, 3)].append(x)
print("seed-mean v(1e-4):", " ".join(f"{t:.3f}:{np.mean(w) * 1e4:+.1f}" for t, w in sorted(d.items())))
print("signs:", " ".join(f"{t:.3f}:{''.join('+' if x > 0 else '-' for x in w)}" for t, w in sorted(d.items())))
tc = show("Tc supp (rule)", Ts, vs)
To, vo = load("slab_n6_CM2_zG1.1"); show("Tc orig (info)", To, vo)
show("Tc pooled (info)", Ts + To, vs + vo)
Ta = show("Ta orig", *load("slab_n6_M4F_zG1.1")); Tb = show("Tb orig", *load("slab_n6_M2M4_zG1.1"))
g = tc["zero"] - Tb["zero"]; tol = 2 * np.hypot(tc["sigma_total"], Tb["sigma_total"])
print(f"Tc - Tb = {g:.4f} vs 2 sigma {tol:.4f}; Tb - Ta = {Tb['zero'] - Ta['zero']:.4f} vs {2 * np.hypot(Tb['sigma_total'], Ta['sigma_total']):.4f}")
pred = (0.3300, 0.3674, 0.3832); loc = all(abs(z - p) <= 0.03 for z, p in zip((Ta["zero"], Tb["zero"], tc["zero"]), pred))
print("order:", Ta["zero"] < Tb["zero"] < tc["zero"], "; location within 0.03:", loc)
print("SUPPLEMENTARY VERDICT:", "M2 phase confirmed (ladder complete)" if g > tol and Ta["zero"] < Tb["zero"] else "M2 phase NOT resolved", "; location", "agrees" if loc else "does NOT agree")
print(f"widths: M4 {Tb['zero'] - Ta['zero']:.4f} (pred 0.0374), M2 {g:.4f} (pred 0.0158)")
