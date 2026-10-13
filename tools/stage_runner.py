"""Stage-wise production runner (follow-up 6): plans ladders point by point, size by size.

For every (cut, dim, point) the ladder of the next size is designed from the data of the previous size
(or from data/ladder_inputs.json for the smallest), covering only ~0.5 Tc - 1.5 Tc; lower temperatures run as
independent single-temperature chains (no exchange).  Strong first-order points get a heating ladder
(ordered start, 0.5-1.12 Tc) and a cooling ladder (random start, 0.9-1.5 Tc).  Sweeps follow tau.
Usage: python tools/stage_runner.py --stage 1 [--plan-only] [--only A_3d]
"""
import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from make_configs import CUTS, make  # noqa: E402

from polyspin.analysis.summary import export_summary, load_run, thermo  # noqa: E402
from polyspin.analysis.temps import interpolators, place_temperatures, swap_acceptance  # noqa: E402

T_SPIN = {2: 0.74e-6, 3: 0.84e-6}          # s per spin-sweep and thread inside the PT loop
STAGES = {
    1: dict(points=[("A", 0.5, "pt"), ("A", 1.0, "pt"), ("A", 1.5, "pt"),
                    ("B", 0.25, "pt"), ("B", 1.0, "first"), ("B", 2.0, "first")],
            sizes={3: (8, 10, 12), 2: (16, 24, 32)}, dims=(3, 2),
            # follow-up 7: tier 3 is reduced -- no B first-order heating/cooling ladders; A 3D L=12 only zG=1.0
            # (pending the L=10 result) and 2D L=32 only zG=1.0 (H2)
            # H1 rule: no c_peak ~ N scaling or bimodality at L=10 (exponents 0.49, 0.12, 0.17) -> all three A points
            # tier-3 request (Oct 13): 3D L=12 A zG=0.5,1.0,1.5 with a narrow 0.9-1.1 T_peak ladder ('narrow'),
            # 2D L=32 A zG=1.0 with the usual ladder; no B first-order tier 3
            tier3_points={3: [("A", 0.5, "narrow"), ("A", 1.0, "narrow"), ("A", 1.5, "narrow")], 2: [("A", 1.0, "pt")]}),
}


def tau_guess(dim, L):
    """Integrated autocorrelation time estimate [sweeps] from the trial runs (scaled by z = 2)."""
    return 90 * (L / 16) ** 2 if dim == 2 else 55 * (L / 8) ** 2.5


def sweeps_for(dim, L, kind):
    s = max(20000, int(math.ceil(200 * tau_guess(dim, L) / 1000.0)) * 1000)
    return s * (2 if kind == "first" else 1)


def hours(R, dim, L, sweeps, threads=4, overhead=1.2):
    return R * L ** dim * 4 * T_SPIN[dim] * sweeps / threads * overhead / 3600.0


def load_inputs(path):
    return json.load(open(path))


def plan_ladder(entry, N, lo, hi, target=0.3, min_ratio=1.01):
    e, c = interpolators(np.array(entry["T"]), np.array(entry["e"]), np.array(entry["c"]))
    return place_temperatures(lo, hi, e, c, N, target, max_ratio=1.3, min_ratio=min_ratio)


def point_items(cut, dim, value, kind, sizes, db):
    par = CUTS[cut][1]
    key = f"{cut}_{dim}d_{par}{value:g}"
    items = []
    for L in sizes:
        e = db[key]
        Tc = e["Tc"]
        N = L ** dim
        S = sweeps_for(dim, L, kind)
        base = dict(cut=cut, dim=dim, L=L, value=value, key=key, Tc=Tc, sweeps=S)
        if kind == "pt":
            Ts = plan_ladder(e, N, 0.5 * Tc, 1.5 * Tc)
            items.append(dict(base, tag="pt", init="ordered", T=Ts.tolist(), exchange=True))
            cold = np.geomspace(0.02, 0.45 * Tc, 4)
            items.append(dict(base, tag="cold", init="ordered", T=[round(float(t), 5) for t in cold],
                              exchange=False, sweeps=max(10000, S // 2)))
        elif kind == "narrow":
            # H1 size scaling: replicas concentrated in 0.9-1.1 T_peak; swap target 0.35 because the measured
            # minimum at L=10 was 0.05-0.1 below the predicted one (goal: min >= 0.2, >= 10 round trips)
            Ts = plan_ladder(e, N, 0.9 * Tc, 1.1 * Tc, target=0.35, min_ratio=1.001)
            R = len(Ts)
            S = max(S, int(math.ceil(75 * R * R / 2000.0)) * 1000)         # total sweeps ~ 75 R^2 (>= ~12 trips)
            items.append(dict(base, tag="pt", init="ordered", T=Ts.tolist(), exchange=True, sweeps=S))
        else:
            heat = plan_ladder(e, N, 0.5 * Tc, 1.12 * Tc)
            cool = plan_ladder(e, N, 0.9 * Tc, 1.5 * Tc)
            items.append(dict(base, tag="heat", init="ordered", T=heat.tolist(), exchange=True))
            items.append(dict(base, tag="cool", init="random", T=cool.tolist(), exchange=True))
            cold = np.geomspace(0.02, 0.45 * Tc, 4)
            items.append(dict(base, tag="cold", init="ordered", T=[round(float(t), 5) for t in cold],
                              exchange=False, sweeps=max(10000, S // 4)))
    return items


def config_of(it, seed=101):
    S = it["sweeps"]
    samp = dict(n_equil=S, n_meas=S, sweeps_per_exchange=1, meas_every=5, diag_every=max(S // 80, 250),
                tune_every=200, checkpoint_minutes=20)
    name0, cfg = make(it["cut"], it["dim"], it["L"], it["value"], dict(list=[round(float(t), 5) for t in it["T"]]),
                      samp, seed, f"_{it['tag']}", it["init"])
    cfg["exchange"] = it["exchange"]
    cfg["diagnostics"] = dict(defects=dict(stride=3, coarse_blocks=[2, 4]))
    cfg["stage"] = 1
    return name0, cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, default=1)
    ap.add_argument("--plan-only", action="store_true")
    ap.add_argument("--only", default=None, help="substring filter on '<cut>_<dim>d'")
    ap.add_argument("--tier", type=int, nargs="*", default=None, help="size tiers to run (1 = smallest)")
    ap.add_argument("--inputs", default=str(ROOT / "data" / "ladder_inputs.json"))
    ap.add_argument("--outdir", default=str(ROOT / "runs"))
    a = ap.parse_args()
    st = STAGES[a.stage]
    db = load_inputs(a.inputs)
    out = Path(a.outdir) / f"stage{a.stage}"
    cfgdir = ROOT / "configs" / f"stage{a.stage}"
    sumdir = ROOT / "data" / f"stage{a.stage}"
    for d in (out, cfgdir, sumdir):
        d.mkdir(parents=True, exist_ok=True)
    tiers = {}
    for tier in range(3):
        if a.tier and tier + 1 not in a.tier:
            continue
        for dim in st["dims"]:
            pts = st["tier3_points"][dim] if tier == 2 and "tier3_points" in st else st["points"]
            for cut, value, kind in pts:
                if a.only and a.only not in f"{cut}_{dim}d":
                    continue
                tiers.setdefault(tier + 1, []).append((dim, cut, value, kind, st["sizes"][dim][tier]))
    total = 0.0
    for tier, lst in sorted(tiers.items()):
        sub = 0.0
        for dim, cut, value, kind, L in lst:
            par = CUTS[cut][1]
            key = f"{cut}_{dim}d_{par}{value:g}"
            for it in point_items(cut, dim, value, kind, [L], db):
                name, cfg = config_of(it)
                h5 = out / f"{name}.h5"
                R = len(it["T"])
                h = hours(R, dim, L, it["sweeps"] * 2)
                sub += h
                print(f"[tier {tier}] {name:40s} R={R:3d} sweeps={2 * it['sweeps']:7d} est {h:5.1f} h", flush=True)
                (cfgdir / f"{name}.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False, default_flow_style=None))
                if a.plan_only or h5.exists():
                    continue
                subprocess.run([sys.executable, "-m", "polyspin.run", str(cfgdir / f"{name}.yaml"), "--out", str(h5),
                                "--quiet"], check=True)
                export_summary(h5, sumdir / f"{name}.npz")
                with open(out / "progress.log", "a") as f:
                    f.write(f"done {name}\n")
                if it["tag"] in ("pt", "heat"):          # refine the seed data for the next size
                    r = load_run(h5)
                    th = thermo(r)
                    ic = int(np.argmax(th["c"]))
                    db[key] = dict(db[key], size=L, N=L ** dim, T=r["T"].tolist(), e=th["e"].tolist(),
                                   c=th["c"].tolist(), Tc=float(r["T"][ic]) if it["tag"] == "pt" else db[key]["Tc"])
        print(f"== tier {tier} subtotal {sub:.1f} h", flush=True)
        total += sub
    print(f"total estimated {total:.0f} h")


if __name__ == "__main__":
    main()
