"""Energy-distribution test at the specific-heat peak of every PT run (tier-3 request, decision 3).
Rule (all three needed for 'bimodal'): (1) dBIC corrected for autocorrelation (N_eff = N / 2 tau_E) > 10,
(2) valley ratio < 0.9, (3) min_T V(L) moves away from 2/3 as L grows.
Usage: python tools/bimodality_table.py [--stage 1] > docs/tier2_bimodality.md"""
import argparse, glob, re
from collections import defaultdict
from pathlib import Path
import h5py, numpy as np, yaml
from polyspin.analysis.finite_size import first_order_signature, energy_cumulant, cumulant_trend

ap = argparse.ArgumentParser(); ap.add_argument("--stage", type=int, default=1); a = ap.parse_args()
rows, pts = [], defaultdict(list)
for p in sorted(glob.glob(f"runs/stage{a.stage}/[AB]_*_pt.h5")):
    cut, dim, L, par, val = re.match(r"(\w)_(\d)d_L(\d+)_(zG|zH4)([\d.]+)_pt", Path(p).stem).groups()
    f = h5py.File(p, "r")
    T = np.array(yaml.safe_load(f.attrs["config_yaml"])["temperatures"]["list"]); N = int(f.attrs["N"])
    e = f["series/e"][:]; tau = f["series/tau_int"][:, 0]
    c = e.var(1) * N / T ** 2; i = int(np.argmax(c))
    s = first_order_signature(e[i], tau=float(tau[i]))
    V = np.array([energy_cumulant(x) for x in e]); j = int(np.argmin(V))
    w = np.where((T >= 0.9 * T[i]) & (T <= 1.1 * T[i]))[0]; jw = w[int(np.argmin(V[w]))]
    rows.append((cut, int(dim), float(val), int(L), T[i], tau[i], s, V[j], T[j], V[jw], T[jw]))
    pts[(cut, int(dim), float(val))].append((int(L), V[j], V[jw]))
print("| point | L | T_peak | tau_E [samples] | N_eff | dBIC (uncorrected) | dBIC (corrected) | valley ratio | run test (1,2) | V_min over ladder (at T) | V_min within 0.9-1.1 T_peak (at T) |")
print("|---|---|---|---|---|---|---|---|---|---|---|")
for cut, dim, val, L, Tp, tau, s, vm, Tv, vw, Tw in sorted(rows, key=lambda r: (r[0], r[1], r[2], r[3])):
    print(f"| {cut} {dim}D {val:g} | {L} | {Tp:.4f} | {tau:.1f} | {s['n_eff']:.0f} | {s['dBIC_raw']:.0f} | {s['dBIC']:.1f} | {s['valley_ratio']:.2f} | {'bimodal' if s['bimodal_run'] else 'no'} | {vm:.5f} ({Tv:.3f}) | {vw:.5f} ({Tw:.3f}) |")
print("\n### Condition (3): |2/3 - V_min| versus L")
for k, v in sorted(pts.items()):
    if len(v) < 2: continue
    for name, col in (("over whole ladder (literal)", 1), ("within 0.9-1.1 T_peak", 2)):
        ok, dev = cumulant_trend([x[0] for x in v], [x[col] for x in v])
        print(f"- {k[0]} {k[1]}D {k[2]:g}, V_min {name}: L = {sorted(x[0] for x in v)}, |2/3 - V_min| = {[f'{d:.2e}' for d in dev]} -> {'moves away from 2/3' if ok else 'does not move away'}")
