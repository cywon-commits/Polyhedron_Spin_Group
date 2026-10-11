"""Trial configs on the smallest production sizes with ladders designed from preliminary runs.
Usage: python tools/make_trial_configs.py <prelim_dir> <outdir> --dim 2 --size 16 --cuts A B
          [--init ordered] [--equil 10000 --meas 20000] [--target 0.3] [--Tmin 0.02 --Tmax 1.5]"""
import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_configs import CUTS, make  # noqa: E402
from replan_ladder import plan  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prelim_dir")
    ap.add_argument("outdir")
    ap.add_argument("--dim", type=int, default=2)
    ap.add_argument("--size", type=int, default=16)
    ap.add_argument("--cuts", nargs="+", default=["A", "B"])
    ap.add_argument("--values", nargs="*", type=float, default=None, help="restrict to these scan values")
    ap.add_argument("--init", default="ordered")
    ap.add_argument("--equil", type=int, default=10000)
    ap.add_argument("--meas", type=int, default=20000)
    ap.add_argument("--target", type=float, default=0.3)
    ap.add_argument("--Tmin", type=float, default=0.02)
    ap.add_argument("--Tmax", type=float, default=1.5)
    ap.add_argument("--diag-every", type=int, default=500)
    ap.add_argument("--suffix", default="_trial")
    ap.add_argument("--prelim-suffix", default="_prelimord")
    ap.add_argument("--prelim-dim", type=int, default=None)
    ap.add_argument("--prelim-size", type=int, default=8)
    ap.add_argument("--seed", type=int, default=11)
    a = ap.parse_args()
    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    pd = a.dim if a.prelim_dim is None else a.prelim_dim
    N = a.size ** a.dim
    S = dict(n_equil=a.equil, n_meas=a.meas, sweeps_per_exchange=1, meas_every=1, diag_every=a.diag_every,
             tune_every=200, checkpoint_minutes=20)
    for cut in a.cuts:
        for v in CUTS[cut][2]:
            if a.values and v not in a.values:
                continue
            par = CUTS[cut][1]
            pname = f"{cut}_{pd}d_L{a.prelim_size}_{par}{v:g}{a.prelim_suffix}.h5"
            Ts, acc = plan(str(Path(a.prelim_dir) / pname), N, a.target, a.Tmin, a.Tmax)
            Tspec = dict(list=[round(float(t), 5) for t in Ts])
            name, cfg = make(cut, a.dim, a.size, v, Tspec, S, a.seed, a.suffix, a.init)
            (out / f"{name}.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False, default_flow_style=None))
            print(name, "R =", len(Ts))


if __name__ == "__main__":
    main()
