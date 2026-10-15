"""Stage 2, third priority: cuts C (K=0, zH4=0; zG = 0.6 T, 1.0 U, 1.2 U, 1.6 F) and D (K=0.5, zH4=1.5; zG = 0.9 C, 1.0, 1.1 F),
2D L = 16 first (decision Oct 15).  Per point a heating ladder (ordered start, 0.04 ... 1.12 T_m) and a cooling ladder
(random start, 0.9 T_m ... 1.5 T_m; for D zG = 1.1 it extends down to 0.12 so that the F -> C re-entrance can be seen on cooling).
T_m = specific-heat peak of the L = 8 trial.  Rules: docs/CD_rules.md.  Usage: python tools/stage2_cd.py [--plan-only] [--size 16]"""
import argparse, json, subprocess, sys
from pathlib import Path
import yaml
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import stage_runner as sr
from polyspin.analysis.summary import export_summary

ap = argparse.ArgumentParser(); ap.add_argument("--plan-only", action="store_true"); ap.add_argument("--size", type=int, default=16); a = ap.parse_args()
db = json.load(open(ROOT / "data" / "ladder_inputs_cd.json"))
out, cfgdir, sumdir = ROOT / "runs" / "stage2", ROOT / "configs" / "stage2", ROOT / "data" / "stage2"
for d in (out, cfgdir, sumdir):
    d.mkdir(parents=True, exist_ok=True)
POINTS = [("C", 0.6), ("C", 1.0), ("C", 1.2), ("C", 1.6), ("D", 0.9), ("D", 1.0), ("D", 1.1)]
total, L = 0.0, a.size
for cut, zg in POINTS:
    e = db[f"{cut}_2d_zG{zg:g}"]; N = L * L; Tm = e["Tc"]
    S = sr.sweeps_for(2, L, "first")
    cool_lo = 0.12 if (cut, zg) == ("D", 1.1) else 0.9 * Tm
    heat = sr.plan_ladder(e, N, 0.04, 1.12 * Tm); cool = sr.plan_ladder(e, N, cool_lo, 1.5 * Tm)
    base = dict(cut=cut, dim=2, L=L, value=zg, key=f"{cut}_2d_zG{zg:g}", Tc=Tm, sweeps=S)
    for it in (dict(base, tag="heat", init="ordered", T=heat.tolist(), exchange=True),
               dict(base, tag="cool", init="random", T=cool.tolist(), exchange=True)):
        name, cfg = sr.config_of(it, seed=5001); cfg["stage"] = 2
        R = len(it["T"]); h = sr.hours(R, 2, L, 2 * S); total += h
        print(f"{name:30s} R={R:3d} T {it['T'][0]:.3f}-{it['T'][-1]:.3f} sweeps={2 * S} est {h:4.2f} h", flush=True)
        yml = cfgdir / f"{name}.yaml"
        if not yml.exists():
            yml.write_text(yaml.safe_dump(cfg, sort_keys=False, default_flow_style=None))
        h5 = out / f"{name}.h5"
        if a.plan_only or h5.exists():
            continue
        subprocess.run([sys.executable, "-m", "polyspin.run", str(yml), "--out", str(h5), "--quiet"], check=True)
        export_summary(h5, sumdir / f"{name}.npz")
        with open(out / "progress.log", "a") as f:
            f.write(f"done {name}\n")
print(f"total estimated {total:.1f} h")
