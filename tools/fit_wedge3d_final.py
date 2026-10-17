"""3D M2 wedge FINAL verdict with the seed-mean estimator (docs/CD_rules.md 'Oct 17 final rule').  Usage: python tools/fit_wedge3d_final.py"""
import json, numpy as np
from polyspin.analysis import cc_fit as cc
def load(f):
    r = json.loads(str(np.load(f)["results"]))
    return np.array([x["T"] for x in r]), np.array([x["v"] for x in r])
def est(name, f):
    T, v = load(f); r = cc.crossing_seedmean(T, v); print(f"{name:34s}", {k: round(float(x), 5) for k, x in r.items()}); return r
t1 = est("T1 supp (final rule)", "data/stage2/slab_M2F_3D_L12_supp.npz")
t2 = est("T2 remeasure (final rule)", "data/stage2/slab_CM2_3D_L12_remeasure.npz")
est("T1 orig (info)", "data/stage2/slab_M2F_3D_L12.npz")
est("T2 orig (info)", "data/stage2/slab_CM2_3D_L12.npz")
Ta, va = load("data/stage2/slab_M2F_3D_L12.npz"); Tb, vb = load("data/stage2/slab_M2F_3D_L12_supp.npz")
r = cc.crossing_seedmean(np.concatenate([Ta, Tb]), np.concatenate([va, vb])); print("T1 pooled (info)", {k: round(float(x), 5) for k, x in r.items()})
w = cc.wedge_verdict(t1["zero"], t1["sigma_total"], t2["zero"], t2["sigma_total"], 0.2440, 0.2665)
print("FINAL:", w, "T2-T1 =", round(t2["zero"] - t1["zero"], 5), "2*comb =", round(2 * np.hypot(t1["sigma_total"], t2["sigma_total"]), 5), "width pred 0.0225")
