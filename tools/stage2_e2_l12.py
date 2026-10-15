"""E' extra run (decision Oct 15): 3D L=12, zG=1.0, heating ladder (ordered start) restricted to 0.8-1.2 T1 (T1 = 0.1904 from
L=10), below the nematic-melting exchange bottleneck.  Ladder designed from the L=10 e(T), c(T).
Usage: python tools/stage2_e2_l12.py [--plan-only]"""
import argparse, json, math, subprocess, sys
from pathlib import Path
import numpy as np, yaml
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import stage_runner as sr
from polyspin.analysis.summary import export_summary, load_run, thermo

ap = argparse.ArgumentParser(); ap.add_argument("--plan-only", action="store_true"); a = ap.parse_args()
r = load_run(ROOT / "runs/stage2/E2_3d_L10_zG1_heat.h5"); th = thermo(r)
e = dict(T=r["T"].tolist(), e=th["e"].tolist(), c=th["c"].tolist())
L, zg, T1 = 12, 1.0, 0.1904
N = L ** 3
Ts = sr.plan_ladder(e, N, 0.8 * T1, 1.2 * T1, target=0.35, min_ratio=1.001)
R = len(Ts)
S = max(sr.sweeps_for(3, L, "pt"), int(math.ceil(75 * R * R / 2000.0)) * 1000)          # >= ~12 round trips (L=14 experience)
it = dict(cut="E2", dim=3, L=L, value=zg, key="E2_3d_zG1", Tc=T1, sweeps=S, tag="heat", init="ordered", T=Ts.tolist(), exchange=True)
name, cfg = sr.config_of(it, seed=3012); cfg["stage"] = 2
print(f"{name} R={R} sweeps/phase={S} (total {2 * S}) T {Ts[0]:.4f}-{Ts[-1]:.4f} est {sr.hours(R, 3, L, 2 * S):.1f} h", flush=True)
out, cfgdir, sumdir = ROOT / "runs/stage2", ROOT / "configs/stage2", ROOT / "data/stage2"
yml = cfgdir / f"{name}.yaml"
if not yml.exists():
    yml.write_text(yaml.safe_dump(cfg, sort_keys=False, default_flow_style=None))
h5 = out / f"{name}.h5"
if not a.plan_only and not h5.exists():
    subprocess.run([sys.executable, "-m", "polyspin.run", str(yml), "--out", str(h5), "--quiet"], check=True)
    export_summary(h5, sumdir / f"{name}.npz")
    with open(out / "progress.log", "a") as f:
        f.write(f"done {name}\n")
