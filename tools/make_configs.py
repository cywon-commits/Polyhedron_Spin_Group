"""Generate one YAML per (cut line, lattice, parameter value).
Usage: python tools/make_configs.py --cuts A B E --dim 2 --size 16 --outdir configs
       [--Tmin 0.02 --Tmax 1.2 --n 48 --equil 20000 --meas 20000 --suffix _trial]
"""
import argparse
from pathlib import Path

import yaml

# name -> (fixed model parameters, scanned parameter, values, T=0 predictions)
CUTS = {
    "A": (dict(K=0.5, zH4=0.0), "zG", [0.5, 1.0, 1.5, 2.0], ["T", "T", "T", "F"]),
    "B": (dict(K=0.5, zG=0.5), "zH4", [0.25, 1.0, 2.0], ["T", "C", "C"]),
    "C": (dict(K=0.0, zH4=0.0), "zG", [0.6, 1.0, 1.2, 1.6], ["T", "U", "U", "F"]),
    "D": (dict(K=0.5, zH4=1.5), "zG", [0.9, 1.0, 1.1], ["C", "degenerate", "F"]),
    "E": (dict(K=-0.5, zH4=0.0), "zG", [0.5, 0.8, 1.0, 1.2], ["C", "D", "D", "F"]),
    # E': nematic-coupled variant (zH4 > 0 couples the collinear C state; at zH4 = 0 it is decoupled)
    "E2": (dict(K=-0.25, zH4=0.25), "zG", [0.6, 0.9, 1.0, 1.1], ["C", "D", "D", "F"]),
}


def make(cut, dim, size, value, Tspec, sampling, seed=1, suffix="", init="random"):
    fixed, par, values, pred = CUTS[cut]
    model = dict(J=1.0, L=0.0, G1=0.0, zG=0.0, zH4=0.0)
    model.update(fixed)
    model[par] = value
    name = f"{cut}_{dim}d_L{size}_{par}{value:g}{suffix}"
    return name, dict(name=name, cut=cut, t0_prediction=pred[values.index(value)],
                      lattice=dict(dim=dim, size=size), model=model, temperatures=Tspec,
                      sampling=sampling, init=init, seed=seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cuts", nargs="+", default=["A", "B", "E"])
    ap.add_argument("--dim", type=int, default=2)
    ap.add_argument("--size", type=int, default=16)
    ap.add_argument("--outdir", default="configs")
    ap.add_argument("--Tmin", type=float, default=0.02)
    ap.add_argument("--Tmax", type=float, default=1.2)
    ap.add_argument("--n", type=int, default=48)
    ap.add_argument("--equil", type=int, default=20000)
    ap.add_argument("--meas", type=int, default=20000)
    ap.add_argument("--diag-every", type=int, default=200)
    ap.add_argument("--suffix", default="")
    ap.add_argument("--init", default="random")
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    T = dict(Tmin=a.Tmin, Tmax=a.Tmax, n=a.n, spacing="log")
    S = dict(n_equil=a.equil, n_meas=a.meas, sweeps_per_exchange=1, meas_every=1,
             diag_every=a.diag_every, tune_every=200, checkpoint_minutes=20)
    for cut in a.cuts:
        for v in CUTS[cut][2]:
            name, cfg = make(cut, a.dim, a.size, v, T, S, a.seed, a.suffix, a.init)
            (out / f"{name}.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
            print(name)


if __name__ == "__main__":
    main()
