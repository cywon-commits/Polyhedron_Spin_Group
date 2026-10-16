"""Cut C U points (zG = 1.0, 1.2) in 3D, L = 8 and 10 (decision Oct 16): H5 as a statement about the phase.  Per point a heating ladder
(ordered start, 0.4 ... 1.12 T_peak; T_peak from the L = 6 trial: 0.193, 0.211) plus four cold single-temperature chains (0.02 ... 0.35 T_peak,
no exchange).  The transition looks sharp (swap rate 0.00 at the c peak in the trial), so the ladder is only used on the ordered side.
Rule: docs/CD_rules.md (existing H5 rule: >= 3 consecutive temperatures with {211} > 0.5 and m1 in [0.2, 0.3]; supported iff both points and both sizes).
Usage: python tools/stage2_c3d.py [--plan-only]"""
import argparse, json, subprocess, sys
from pathlib import Path
import numpy as np, yaml
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import stage_runner as sr
from polyspin.analysis.summary import export_summary

ap = argparse.ArgumentParser(); ap.add_argument("--plan-only", action="store_true"); a = ap.parse_args()
db = json.load(open(ROOT / "data" / "ladder_inputs_c3d.json"))
out, cfgdir, sumdir = ROOT / "runs" / "stage2", ROOT / "configs" / "stage2", ROOT / "data" / "stage2"
total = 0.0
for L in (8, 10):
    for zg in (1.0, 1.2):
        e = db[f"C_3d_zG{zg:g}"]; Tp = e["Tc"]; N = L ** 3
        S = sr.sweeps_for(3, L, "first")
        heat = sr.plan_ladder(e, N, 0.4 * Tp, 1.12 * Tp)
        cold = [round(float(t), 5) for t in np.geomspace(0.02, 0.35 * Tp, 4)]
        base = dict(cut="C", dim=3, L=L, value=zg, key=f"C_3d_zG{zg:g}", Tc=Tp, sweeps=S)
        for it in (dict(base, tag="heat", init="ordered", T=heat.tolist(), exchange=True),
                   dict(base, tag="cold", init="ordered", T=cold, exchange=False, sweeps=max(10000, S // 2))):
            name, cfg = sr.config_of(it, seed=7001); cfg["stage"] = 2
            R = len(it["T"]); h = sr.hours(R, 3, L, 2 * it["sweeps"]); total += h
            print(f"{name:28s} R={R:3d} T {it['T'][0]:.3f}-{it['T'][-1]:.3f} sweeps/phase={it['sweeps']} est {h:4.2f} h", flush=True)
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
print(f"total estimated {total:.2f} h")
