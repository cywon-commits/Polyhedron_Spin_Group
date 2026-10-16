"""Exploratory point A zG = 0.3 (tricritical search, NOT a verdict): 3D L = 8, 10, ordered start, ladder 0.8-1.2 T_peak (T_peak = 0.1004 from
the L = 6 trial, where the swap rate was 0.00 at the c peak: possibly discontinuous, hence the wider window).  Usage: python tools/stage2_a03.py [--plan-only]"""
import argparse, json, math, subprocess, sys
from pathlib import Path
import yaml
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import stage_runner as sr
from polyspin.analysis.summary import export_summary

ap = argparse.ArgumentParser(); ap.add_argument("--plan-only", action="store_true"); a = ap.parse_args()
e = json.load(open(ROOT / "data" / "ladder_inputs_a03.json"))["A_3d_zG0.3"]; Tp = e["Tc"]
out, cfgdir, sumdir = ROOT / "runs" / "stage2", ROOT / "configs" / "stage2", ROOT / "data" / "stage2"
for L in (8, 10):
    N = L ** 3
    Ts = sr.plan_ladder(e, N, 0.8 * Tp, 1.2 * Tp, target=0.4, min_ratio=1.001)
    R = len(Ts); S = max(sr.sweeps_for(3, L, "pt"), int(math.ceil(75 * R * R / 2000.0)) * 1000)
    it = dict(cut="A", dim=3, L=L, value=0.3, key="A_3d_zG0.3", Tc=Tp, sweeps=S, tag="pt", init="ordered", T=Ts.tolist(), exchange=True)
    name, cfg = sr.config_of(it, seed=6001); cfg["stage"] = 2
    print(f"{name} R={R} T {Ts[0]:.4f}-{Ts[-1]:.4f} sweeps/phase={S} est {sr.hours(R, 3, L, 2 * S):.2f} h", flush=True)
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
