"""Cut E (K = -0.5, zH4 = 0), 3D L = 8, 10, zG = 0.5, 0.8, 1.0, 1.2, ordered and random start (rule: docs/CD_rules.md 'Oct 18 ... C').
Usage: python tools/stage2_e3d.py [--plan-only]"""
import argparse, subprocess, sys
from pathlib import Path
import numpy as np, yaml
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "tools"))
import stage_runner as sr
from polyspin.analysis.summary import export_summary
ap = argparse.ArgumentParser(); ap.add_argument("--plan-only", action="store_true"); a = ap.parse_args()
out, cfgdir, sumdir = ROOT / "runs" / "stage2", ROOT / "configs" / "stage2", ROOT / "data" / "stage2"
RANGE = {0.5: (0.02, 0.60, 24), 0.8: (0.005, 0.12, 24), 1.0: (0.02, 0.30, 28), 1.2: (0.10, 0.35, 24)}
S = 20000; total = 0.0
for L in (8, 10):
    for zg, (lo, hi, R) in RANGE.items():
        T = [round(float(t), 5) for t in np.geomspace(lo, hi, R)]
        for tag, init in (("ord", "ordered"), ("ran", "random")):
            it = dict(cut="E", dim=3, L=L, value=zg, key=f"E_3d_zG{zg:g}", Tc=hi, sweeps=S, tag=tag, init=init, T=T, exchange=True)
            name, cfg = sr.config_of(it, seed=8001); cfg["stage"] = 2
            h = sr.hours(R, 3, L, 2 * S); total += h
            print(f"{name:28s} R={R} est {h:.2f} h", flush=True)
            yml = cfgdir / f"{name}.yaml"
            if not yml.exists(): yml.write_text(yaml.safe_dump(cfg, sort_keys=False, default_flow_style=None))
            h5 = out / f"{name}.h5"
            if a.plan_only or h5.exists(): continue
            subprocess.run([sys.executable, "-m", "polyspin.run", str(yml), "--out", str(h5), "--quiet"], check=True)
            export_summary(h5, sumdir / f"{name}.npz")
            with open(out / "progress.log", "a") as f: f.write(f"done {name}\n")
print(f"total estimated {total:.2f} h")
