"""n = 6 wedge ladder verdict (rule: docs/CD_rules.md 'n = 6 ...').  Usage: python tools/fit_n6_ladder.py"""
import json, collections, numpy as np
from polyspin.analysis import cc_fit as cc
PRED = {"Ta (M6->M4)": 0.3300, "Tb (M4->M2)": 0.3674, "Tc (M2->M0)": 0.3832}
files = [("Ta (M6->M4)", "slab_n6_M4F_zG1.1"), ("Tb (M4->M2)", "slab_n6_M2M4_zG1.1"), ("Tc (M2->M0)", "slab_n6_CM2_zG1.1")]
est = {}
for name, f in files:
    r = json.loads(str(np.load(f"data/stage2/{f}.npz")["results"]))
    T, v = [x["T"] for x in r], [x["v"] for x in r]
    e = cc.crossing_seedmean(T, v); est[name] = e
    d = collections.defaultdict(list)
    for t, x in zip(T, v): d[round(t, 3)].append(x)
    print(f"{name}: zero {e['zero']:.4f} +- {e['sigma_total']:.4f} (pct {e['sigma_pct']:.4f}, std {e['sigma_std']:.4f}, w {e['width']}, n_win {e['n_window']:.0f}); predicted {PRED[name]:.4f}; diff {e['zero'] - PRED[name]:+.4f}")
    print("   seed-mean v(1e-4):", " ".join(f"{t:.3f}:{np.mean(w) * 1e4:+.1f}" for t, w in sorted(d.items())))
n = list(est)
ok_order = est[n[0]]["zero"] < est[n[1]]["zero"] < est[n[2]]["zero"]
gaps = []
for a, b in ((n[0], n[1]), (n[1], n[2])):
    g = est[b]["zero"] - est[a]["zero"]; tol = 2 * np.hypot(est[a]["sigma_total"], est[b]["sigma_total"]); gaps.append((g, tol, g > tol))
    print(f"gap {a} -> {b}: {g:.4f} vs 2 sigma {tol:.4f}: {'resolved' if g > tol else 'NOT resolved'}")
loc = all(abs(est[k]["zero"] - PRED[k]) <= 0.03 for k in n)
print(f"order Ta<Tb<Tc: {ok_order}; both gaps resolved: {all(g[2] for g in gaps)}; location within 0.03 for all: {loc}")
print("VERDICT:", "ladder confirmed" if ok_order and all(g[2] for g in gaps) else "ladder NOT (fully) confirmed", "; location", "agrees" if loc else "does NOT agree")
