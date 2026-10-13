"""Extra stage-1 runs approved after the tier-3 report (Oct 13):
  l14    : 3D L=14 A zG=0.5, narrow 0.9-1.1 T_peak ladder (H1 rule fixed in docs/H1_L14_rule.md)
  bwin   : B 3D zH4=1.0, L=8 and 10, single-temperature chains (no exchange) in the 0.30-0.37 window,
           ordered and random starts (histogram reweighting of T_c and latent heat)
Usage: python tools/extra_runs.py l14|bwin [--plan-only]"""
import argparse, json, math, subprocess, sys
from pathlib import Path
import numpy as np, yaml
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import stage_runner as sr
from make_configs import make
from polyspin.analysis.summary import export_summary

ap = argparse.ArgumentParser(); ap.add_argument("job"); ap.add_argument("--plan-only", action="store_true"); a = ap.parse_args()
db = json.load(open(ROOT / "data" / "ladder_inputs.json"))
out = ROOT / "runs" / "stage1"; cfgdir = ROOT / "configs" / "stage1"; sumdir = ROOT / "data" / "stage1"
jobs = []
if a.job == "l14":
    L = 14
    it = sr.point_items("A", 3, 0.5, "narrow", [L], db)[0]
    it["sweeps"] = max(it["sweeps"], 60000)          # 12 trips expected at L=14 (k scaling from L=12: 16 trips at 31000 sweeps/phase, R=25)
    jobs.append(it)
elif a.job == "bwin":
    for L in (8, 10):
        for init in ("ordered", "random"):
            T = [round(float(t), 4) for t in np.arange(0.300, 0.3701, 0.005)]
            jobs.append(dict(cut="B", dim=3, L=L, value=1.0, tag=f"win{init[:3]}", init=init, T=T, exchange=False,
                             sweeps=40000, Tc=0.35, key="B_3d_zH41"))
for it in jobs:
    name, cfg = sr.config_of(it, seed=2024)
    R = len(it["T"])
    print(f"{name:34s} R={R:3d} sweeps/phase={it['sweeps']} est {sr.hours(R, 3, it['L'], 2 * it['sweeps']):.1f} h", flush=True)
    (cfgdir / f"{name}.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False, default_flow_style=None))
    h5 = out / f"{name}.h5"
    if a.plan_only or h5.exists():
        continue
    subprocess.run([sys.executable, "-m", "polyspin.run", str(cfgdir / f"{name}.yaml"), "--out", str(h5), "--quiet"], check=True)
    export_summary(h5, sumdir / f"{name}.npz")
    with open(out / "progress.log", "a") as f:
        f.write(f"done {name}\n")
