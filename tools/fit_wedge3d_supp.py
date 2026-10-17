"""3D M2 wedge T1 supplementary scan verdict (rule: docs/CD_rules.md 'T1 보완 측정').  Usage: python tools/fit_wedge3d_supp.py"""
import json, collections, numpy as np
from polyspin.analysis import cc_fit as cc
def load(f):
    r = json.loads(str(np.load(f)["results"]))
    return np.array([x["T"] for x in r]), np.array([x["v"] for x in r]), np.array([x["se"] for x in r])
def show(name, T, v, se):
    np.random.seed(1); r = cc.crossing_robust(T, v, se)
    print(name, {a: round(float(b), 5) for a, b in r.items()}); return r
Ts, vs, ss = load("data/stage2/slab_M2F_3D_L12_supp.npz")
d = collections.defaultdict(list)
for t, v in zip(Ts, vs): d[round(t, 3)].append(v)
print("signs/mean(1e-4):", [(t, "".join("+" if x > 0 else "-" for x in w), round(np.mean(w) * 1e4, 1)) for t, w in sorted(d.items())])
a = show("T1 supp", Ts, vs, ss)
To, vo, so = load("data/stage2/slab_M2F_3D_L12.npz")
show("T1 orig", To, vo, so)
sel = (To >= 0.2 - 1e-9)
b = show("T1 pooled(info)", np.concatenate([To, Ts]), np.concatenate([vo, vs]), np.concatenate([so, ss]))
T2, s2 = 0.25273, 0.00164
print("supp rule (pct):", cc.wedge_verdict(a["zero"], a["sigma_total"], T2, s2, 0.2440, 0.2665), "diff", T2 - a["zero"], "2*comb", 2 * np.hypot(a["sigma_total"], s2))
print("supp rule (std):", cc.wedge_verdict(a["zero"], np.hypot(a["sigma_std"], a["systematic"]), T2, s2, 0.2440, 0.2665))

# --- diagnostic (NOT the rule): the rule's estimator falls back to "first sign change" when a side has < 2 distinct T inside the window;
# with noisy single seeds that picks a noisy early sign flip (zero lands on a grid point).  Cross-check with seed-resampled linear fit
# of the per-T seed means over T = 0.232-0.242.
def diag(Ts_, vs_, lo=0.232, hi=0.242, nb=4000, seed=3):
    rng = np.random.default_rng(seed); g = collections.defaultdict(list)
    for t, v in zip(Ts_, vs_):
        if lo - 1e-9 <= t <= hi + 1e-9: g[round(t, 3)].append(v)
    ts = np.array(sorted(g)); z = []
    for _ in range(nb):
        m = np.array([np.mean(rng.choice(g[t], len(g[t]))) for t in ts]); k, c = np.polyfit(ts, m, 1); z.append(-c / k)
    z = np.array(z); p = np.percentile(z, [16, 50, 84]); return p[1], 0.5 * (p[2] - p[0]), z.std()
for name, (t_, v_) in {"supp": (Ts, vs), "pooled": (np.concatenate([To, Ts]), np.concatenate([vo, vs]))}.items():
    z, sp, ss_ = diag(t_, v_)
    print(f"diagnostic seed-mean linear fit [{name}]: T1 = {z:.4f} +- {sp:.4f} (std {ss_:.4f});  T2 - T1 = {T2 - z:.4f}, 2*comb = {2*np.hypot(sp, s2):.4f}")
