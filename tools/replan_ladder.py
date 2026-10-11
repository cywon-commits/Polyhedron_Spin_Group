"""Design a temperature ladder from a measured run and rescale it to another system size.
Usage: python tools/replan_ladder.py run.h5 --N 256 [--target 0.3] [--Tmin 0.02] [--Tmax 1.2]
Prints the number of replicas and a YAML 'temperatures: {list: [...]}' snippet (also --out file)."""
import argparse

import numpy as np
import yaml

from polyspin.analysis.summary import load_run, thermo
from polyspin.analysis.temps import interpolators, place_temperatures, swap_acceptance


def plan(path, N, target=0.3, Tmin=None, Tmax=None, max_ratio=1.3):
    r = load_run(path)
    th = thermo(r)
    T = r["T"]
    e, c = interpolators(T, th["e"], th["c"])            # c is per cluster
    Tmin = T[0] if Tmin is None else Tmin
    Tmax = T[-1] if Tmax is None else Tmax
    Ts = place_temperatures(Tmin, Tmax, e, c, N, target, max_ratio)
    acc = [swap_acceptance(Ts[i], Ts[i + 1], e, c, N) for i in range(len(Ts) - 1)]
    return Ts, np.array(acc)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--N", type=int, required=True)
    ap.add_argument("--target", type=float, default=0.3)
    ap.add_argument("--Tmin", type=float)
    ap.add_argument("--Tmax", type=float)
    ap.add_argument("--max-ratio", type=float, default=1.3)
    ap.add_argument("--out")
    a = ap.parse_args()
    Ts, acc = plan(a.run, a.N, a.target, a.Tmin, a.Tmax, a.max_ratio)
    print(f"R = {len(Ts)} temperatures, predicted acceptance min {acc.min():.2f} mean {acc.mean():.2f}")
    snippet = yaml.safe_dump({"temperatures": {"list": [round(float(t), 5) for t in Ts]}}, default_flow_style=None)
    print(snippet)
    if a.out:
        open(a.out, "w").write(snippet)
