"""3D M2 wedge verdict (rule: docs/CD_rules.md, 'Oct 16' 3D section).  Usage: python tools/fit_wedge3d.py"""
import json, numpy as np
from polyspin.analysis import cc_fit as cc
def load(f):
    r = json.loads(str(np.load(f)["results"]))
    return np.array([x["T"] for x in r]), np.array([x["v"] for x in r]), np.array([x["se"] for x in r])
out = {}
for k, f in (("T1", "slab_M2F_3D_L12"), ("T2", "slab_CM2_3D_L12")):
    T, v, se = load(f"data/stage2/{f}.npz")
    np.random.seed(1)
    r = cc.crossing_robust(T, v, se); out[k] = r
    print(k, {a: round(float(b), 5) for a, b in r.items()})
a, b = out["T1"], out["T2"]
w = cc.wedge_verdict(a["zero"], a["sigma_total"], b["zero"], b["sigma_total"], 0.2440, 0.2665)
print("pct rule:", w, "diff", b["zero"] - a["zero"], "2*comb", 2 * np.hypot(a["sigma_total"], b["sigma_total"]))
w2 = cc.wedge_verdict(a["zero"], np.hypot(a["sigma_std"], a["systematic"]), b["zero"], np.hypot(b["sigma_std"], b["systematic"]), 0.2440, 0.2665)
print("std-based:", w2)
