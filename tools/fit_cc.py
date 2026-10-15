"""H6 joint fit: slab scan (run_slab_scan.py npz, diagonal interface) + Clausius-Clapeyron chains (run_cc_chains.py npz).
Usage: python tools/fit_cc.py chains.npz scan_T0.10.npz scan_T0.125.npz ... [--kind diag]"""
import argparse, json
import numpy as np
from polyspin.analysis.cc_fit import PRED, crossing_samples, fit_a, verdict

ap = argparse.ArgumentParser(); ap.add_argument("chains"); ap.add_argument("scan", nargs="+"); ap.add_argument("--kind", default="diag"); ap.add_argument("--J", type=float, default=1.0)
a = ap.parse_args()
scan = [r for p in a.scan for r in json.loads(str(np.load(p, allow_pickle=False)["results"]))]; ch = json.loads(str(np.load(a.chains, allow_pickle=False)["results"]))
rng = np.random.default_rng(5)
Ts = sorted({r["T"] for r in scan})
rows = []
print("| T | n moved | zG* (bootstrap median) | sigma_boot | pinned width | sigma_total | used | s(T) = dzG*/dT | sigma_s |\n|---|---|---|---|---|---|---|---|---|")
for T in Ts:
    rr = sorted([r for r in scan if r["T"] == T and r["kind"] == a.kind], key=lambda r: r["zG"])
    smp, wp, nm = crossing_samples([r["zG"] for r in rr], [r["v"] for r in rr], [r["se"] for r in rr])
    if len(smp) == 0:
        print(f"| {T} | {nm} | n/a | | | | no (fewer than 2 moved runs) | | |"); continue
    zst, sb = float(np.median(smp)), float(smp.std())
    st = float(np.hypot(sb, wp / np.sqrt(12)))
    used = wp <= 3 * sb or wp == 0.0
    # Clausius-Clapeyron slope at zG*: interpolate E_C - E_F and dH_C - dH_F over the chain zG values (quadratic through 3 points)
    cc = [r for r in ch if abs(r["T"] - T) < 1e-9]
    zg = sorted({r["zG"] for r in cc})
    def get(b, key, z):
        return [next(r for r in cc if r["zG"] == zz and r["branch"] == b)[key] for zz in z]
    s_draw = []
    for _ in range(2000):
        zs_ = rng.choice(smp)
        dE = np.array(get("C", "E", zg)) - np.array(get("F", "E", zg)) + rng.normal(0, 1, len(zg)) * np.hypot(get("C", "E_err", zg), get("F", "E_err", zg))
        dD = np.array(get("C", "dH", zg)) - np.array(get("F", "dH", zg)) + rng.normal(0, 1, len(zg)) * np.hypot(get("C", "dH_err", zg), get("F", "dH_err", zg))
        deg = min(2, len(zg) - 1)
        e_at = np.polyval(np.polyfit(zg, dE, deg), zs_); d_at = np.polyval(np.polyfit(zg, dD, deg), zs_)
        s_draw.append((e_at / T) / d_at)
    s_val, s_sig = float(np.median(s_draw)), float(np.std(s_draw))
    extrap = zst < min(zg) or zst > max(zg)
    print(f"| {T} | {nm} | {zst:.4f} | {sb:.4f} | {wp:.3f} | {st:.4f} | {'yes' if used else 'NO (pinned)'} | {s_val:.3f}{' (extrapolated)' if extrap else ''} | {s_sig:.3f} |")
    if used:
        rows.append((T, zst, st, s_val, s_sig))
rows = np.array(rows)
if len(rows) >= 2:
    T, z, sz, s, ss = rows.T
    for name, kw in (("PRIMARY: joint fit (slab + CC slope), b free", dict(s=s, s_sigma=ss)), ("aux: slab only, b free", {}), ):
        f = fit_a(T, z, sz, J=a.J, **kw)
        print(f"\n**{name}**: a = {f['a']:.3f} +- {f['sa']:.3f}, b = {f['b']:.2f} +- {f['sb']:.2f}  -> {verdict(f['a'], f['sa'])}")
    f0 = fit_a(T, z, sz, J=a.J, b_free=False)
    print(f"\naux: slab only, b = 0: a = {f0['a']:.3f} +- {f0['sa']:.3f}  -> {verdict(f0['a'], f0['sa'])}")
    print("\npredictions:", PRED)
