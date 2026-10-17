"""Anharmonic correction (decision Oct 17 (d)).  `python tools/anharm.py run out.npz --dim 2 --size 16` runs the branch chains;
`python tools/anharm.py analyze` applies docs/CD_rules.md.  Rule fixed in docs/CD_rules.md (section 'Anharmonic correction')."""
import argparse, json, sys
from multiprocessing import Pool
import numpy as np
from polyspin import slab
from polyspin.harmonic import collinear_state, selection
from polyspin.lattice import Lattice
from polyspin.model import Params, energy_total

TS = [0.08 + 0.02 * i for i in range(9)]
OWN = {0: 2, 1: 1, 2: 0}      # branch k (reversed spins) -> moment-class index (F 2, M2 1, C 0)


def job(a):
    dim, size, zG, k, T, ne, nm, seed = a
    lat = Lattice(dim, size)
    P = Params.from_z(zG, 1.5, z=lat.z, K=0.5)
    S0 = collinear_state(4, k)
    r = slab.run_branch(lat, P, T, S0, ne, nm, seed=seed)
    sp = np.tile(S0, (lat.N, 1, 1)).copy()
    r.update(dim=dim, size=size, k=k, e0=float(energy_total(sp, lat.nbr, P.array()) / lat.N), own=float(r["frac"][OWN[k]]))
    return r


def run(a):
    jobs = []
    for zG in (1.0, 1.1):
        for k in (0, 1, 2):
            for T in TS:
                jobs.append((a.dim, a.size, zG, k, round(T, 3), a.equil, a.meas, a.seed + len(jobs)))
    with Pool(a.procs) as p:
        res = p.map(job, jobs, chunksize=1)
    for r in res:
        print(f"zG={r['zG']} k={r['k']} T={r['T']:.2f} E-e0={r['E'] - r['e0']:.5f}+-{r['E_err']:.5f} own={r['own']:.3f}", flush=True)
    np.savez(a.out, results=json.dumps(res))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sp = ap.add_subparsers(dest="cmd")
    r = sp.add_parser("run"); r.add_argument("out"); r.add_argument("--dim", type=int, default=2); r.add_argument("--size", type=int, default=16)
    r.add_argument("--equil", type=int, default=5000); r.add_argument("--meas", type=int, default=20000)
    r.add_argument("--seed", type=int, default=41000); r.add_argument("--procs", type=int, default=4)
    a = ap.parse_args()
    if a.cmd == "run":
        run(a)
