"""Stage 2, first range: cut E' (K=-0.25, zH4=0.25), zG = 0.9, 1.0; 2D L = 16, 24 and 3D L = 8, 10.
Per (dim, L, zG): a heating ladder (ordered start, 0.5 T1 ... 1.12 T2; T1 = polar (m1) and T2 = nematic (m2) transition
estimates from the L=8/6 trials) and a cooling ladder (random start, 0.9 T2 ... 1.5 T2): the nematic melting looks
first-order like cut B (swap rate ~0 across it in the trials), so one ladder cannot cross it.
Usage: python tools/stage2_e2.py [--plan-only] [--only 2d]"""
import argparse, json, subprocess, sys
from pathlib import Path
import yaml
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import stage_runner as sr
from polyspin.analysis.summary import export_summary

ap = argparse.ArgumentParser(); ap.add_argument("--plan-only", action="store_true"); ap.add_argument("--only", default=None); a = ap.parse_args()
db = json.load(open(ROOT / "data" / "ladder_inputs_e2.json"))
out, cfgdir, sumdir = ROOT / "runs" / "stage2", ROOT / "configs" / "stage2", ROOT / "data" / "stage2"
for d in (out, cfgdir, sumdir):
    d.mkdir(parents=True, exist_ok=True)
SIZES = {2: (16, 24), 3: (8, 10)}
total = 0.0
for dim in (2, 3):
    if a.only and a.only != f"{dim}d":
        continue
    for L in SIZES[dim]:
        for zg in (0.9, 1.0):
            e = db[f"E2_{dim}d_zG{zg:g}"]; N = L ** dim; T1, T2 = e["T1"], e["Tc"]
            S = sr.sweeps_for(dim, L, "first")
            base = dict(cut="E2", dim=dim, L=L, value=zg, key=f"E2_{dim}d_zG{zg:g}", Tc=T2, sweeps=S)
            heat = sr.plan_ladder(e, N, 0.5 * T1, 1.12 * T2)
            cool = sr.plan_ladder(e, N, 0.9 * T2, 1.5 * T2)
            for it in (dict(base, tag="heat", init="ordered", T=heat.tolist(), exchange=True),
                       dict(base, tag="cool", init="random", T=cool.tolist(), exchange=True)):
                name, cfg = sr.config_of(it, seed=3001)
                cfg["stage"] = 2
                R = len(it["T"]); h = sr.hours(R, dim, L, 2 * S); total += h
                print(f"{name:32s} R={R:3d} sweeps={2 * S:7d} est {h:4.1f} h", flush=True)
                yml = cfgdir / f"{name}.yaml"
                if yml.exists():                                   # never regenerate: a changed config invalidates checkpoints
                    pass
                else:
                    yml.write_text(yaml.safe_dump(cfg, sort_keys=False, default_flow_style=None))
                h5 = out / f"{name}.h5"
                if a.plan_only or h5.exists():
                    continue
                subprocess.run([sys.executable, "-m", "polyspin.run", str(yml), "--out", str(h5), "--quiet"], check=True)
                export_summary(h5, sumdir / f"{name}.npz")
                with open(out / "progress.log", "a") as f:
                    f.write(f"done {name}\n")
print(f"total estimated {total:.1f} h")
