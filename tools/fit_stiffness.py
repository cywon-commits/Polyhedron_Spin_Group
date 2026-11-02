"""Stiffness analysis: Y(T, L) with block-bootstrap 16-84% half-widths, the two decision rules, paper/data/stiffness.json.
Usage: python tools/fit_stiffness.py [--out paper/data/stiffness.json]"""
import argparse, json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from polyspin.analysis.stiffness_rules import verdict_A, verdict_B, LS

ap = argparse.ArgumentParser(); ap.add_argument("--out", default=str(ROOT / "paper/data/stiffness.json")); a = ap.parse_args()
NB, NBOOT = 20, 2000
rng = np.random.default_rng(0)
TPEAK_A = 0.21425
GRID = {"A": [f * TPEAK_A for f in (0.6, 0.7, 0.8, 0.9, 1.0)], "B": [round(0.10 + 0.01 * i, 2) for i in range(17)]}


def ups_blocks(z, t, col="x", axes=(0, 1, 2)):
    """per-block sums for temperature index t: returns arrays (NB, 4) of mean S2 and mean S1^2."""
    S1, S2 = z["S1" + col][:, t, :], z["S2" + col][:, t, :]
    idx = np.array_split(np.arange(len(S1)), NB)
    return np.array([S2[i].mean(0) for i in idx]), np.array([(S1[i] ** 2).mean(0) for i in idx])


def boot(z, t, which, col="x"):
    """Y (central), percentile half-width, std for 'axis' (lab-axis average) or 'dir' (director)."""
    T = float(z["T"][t]); beta = 1.0 / T; N = int(z["N"])
    m2, s1s = ups_blocks(z, t, col)
    def Y(b2, b1):
        y = (b2.mean(0) - beta * b1.mean(0)) / N
        return y[:3].mean() if which == "axis" else y[3]
    y0 = Y(m2, s1s)
    draws = np.array([Y(m2[ix], s1s[ix]) for ix in (rng.integers(0, NB, NB) for _ in range(NBOOT))])
    lo, hi = np.percentile(draws, [16, 84])
    return float(y0), float(0.5 * (hi - lo)), float(draws.std())


out = dict(estimator="block bootstrap (20 blocks, 2000 draws) 16-84% half-width (primary), std alongside", rule_doc="docs/CD_rules.md (Nov 2 stiffness)",
           convention="Y = L^-d [<d2H/dtheta2> - beta <(dH/dtheta)^2>] with theta = twist per lattice spacing (= rho_s); x bonds (primary), y bonds as a check")
for point, which in (("A", "axis"), ("B", "dir")):
    zs = {}
    for L in LS:
        p = ROOT / f"data/stage2/stiff_{point}_L{L}.npz"
        if p.exists(): zs[L] = np.load(p)
    if len(zs) < 3:
        print(f"point {point}: runs missing for L = {[L for L in LS if L not in zs]}"); continue
    T = zs[16]["T"]
    assert all(np.allclose(zs[L]["T"], T) for L in LS)
    grid_idx = [int(np.argmin(np.abs(T - g))) for g in GRID[point]]
    full = {L: [boot(zs[L], t, which) for t in range(len(T))] for L in LS}
    full_y = {L: [boot(zs[L], t, which, "y") for t in range(len(T))] for L in LS}
    ups = {L: np.array([full[L][t][0] for t in grid_idx]) for L in LS}; err = {L: np.array([full[L][t][1] for t in grid_idx]) for L in LS}
    Tg = T[grid_idx]
    verdict, det = verdict_A(ups, err) if point == "A" else verdict_B(Tg, ups, err)
    print(f"point {point}: grid T {np.round(Tg, 4).tolist()}\n  VERDICT: {verdict}")
    for L in LS: print(f"  L={L}: Y = {np.round(ups[L], 3).tolist()}  +- {np.round(err[L], 3).tolist()}")
    if point == "B": print("  crossings (T*, |slope|, sigma):", {L: det['crossings'][L] for L in LS})
    out[point] = dict(which=which, T=T.tolist(), grid_T=Tg.tolist(), grid_index=grid_idx, verdict=verdict,
                      details=json.loads(json.dumps(det, default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x))),
                      L={str(L): dict(Y=[f[0] for f in full[L]], err=[f[1] for f in full[L]], std=[f[2] for f in full[L]], Y_ybond=[f[0] for f in full_y[L]],
                                      seed=int(zs[L]["seed"]), init="ordered", phase=str(zs[L]["phase"]), n_equil=int(zs[L]["n_equil"]), n_meas=int(zs[L]["n_meas"]),
                                      every=int(zs[L]["every"]), replicas=int(len(T)), swap_min=float(np.min(zs[L]["swap_rates"])), round_trips=int(zs[L]["round_trips"]))
                         for L in LS})
Path(a.out).write_text(json.dumps(out, indent=1))
print("written", a.out)
