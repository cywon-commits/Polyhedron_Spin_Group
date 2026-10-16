"""Slab coexistence of two uniform collinear phases (k = number of reversed spins: F = 0, M2 = 1, C = 2) at fixed zG over a T scan:
the velocity of the fraction of the RIGHT phase (class index: F 2, M2 1, C 0) has a zero crossing at the coexistence temperature
(docs/CD_rules.md).  Usage: python tools/run_slab_pair.py out.npz --left 1 --right 0 --measure 2 --zG 1.1 --T 0.20 0.205 ... --seeds 3"""
import argparse, json
from multiprocessing import Pool
import numpy as np
from polyspin import slab
from polyspin.harmonic import collinear_state
from polyspin.lattice import Lattice
from polyspin.model import Params


def job(a):
    size, T, zG, kl, kr, idx, sweeps, seed, K, zH, dim = a
    lat = Lattice(dim, size)
    P = Params.from_z(zG, zH, z=lat.z, K=K)
    ts, fr, e, _ = slab.run_slab(lat, P, T, collinear_state(4, kl), collinear_state(4, kr), sweeps,
                                 record_every=max(sweeps // 800, 5), seed=seed, kind="diag")
    v, se, nu, sat = slab.transient_velocity(ts, fr[:, idx])
    return dict(T=T, zG=zG, v=v, se=se, n_used=nu, saturated=sat, frac_end=float(fr[-1, idx]), frac_start=float(fr[0, idx]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("--size", type=int, default=24); ap.add_argument("--dim", type=int, default=2)
    ap.add_argument("--left", type=int, required=True); ap.add_argument("--right", type=int, required=True); ap.add_argument("--measure", type=int, required=True)
    ap.add_argument("--zG", type=float, default=1.1); ap.add_argument("--T", nargs="+", type=float, required=True)
    ap.add_argument("--seeds", type=int, default=3); ap.add_argument("--sweeps", type=int, default=12000)
    ap.add_argument("--K", type=float, default=0.5); ap.add_argument("--zH4", type=float, default=1.5)
    ap.add_argument("--seed", type=int, default=9000); ap.add_argument("--procs", type=int, default=4)
    a = ap.parse_args()
    jobs = [(a.size, T, a.zG, a.left, a.right, a.measure, a.sweeps, a.seed + i, a.K, a.zH4, a.dim) for i, T in enumerate([T for T in a.T for _ in range(a.seeds)])]
    with Pool(a.procs) as p:
        res = p.map(job, jobs, chunksize=1)
    for r in res:
        print(f"T={r['T']:.3f} v={r['v']:.3e}+-{r['se']:.1e} frac {r['frac_start']:.2f}->{r['frac_end']:.2f} sat={r['saturated']}", flush=True)
    np.savez(a.out, results=json.dumps(res))
