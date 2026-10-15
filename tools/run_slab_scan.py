"""H6 slab-coexistence scan: for each temperature find the zG at which the C|F interface stands still.
For every (T, zG, interface kind) a slab run gives the F-fraction velocity; zG*(T) is the zero crossing of
a linear fit of v(zG).  Usage: python tools/run_slab_scan.py out.npz --size 24 --T 0.1 0.15 0.2 0.25
   --zG 0.98 1.0 1.02 ... --sweeps 100000 [--kinds axis diag]"""
import argparse
import json
from multiprocessing import Pool

import numpy as np

from polyspin import slab
from polyspin.harmonic import collinear_state
from polyspin.lattice import Lattice
from polyspin.model import Params


def job(a):
    size, T, zG, kind, sweeps, seed, K, zH = a
    lat = Lattice(2, size)
    P = Params.from_z(zG, zH, z=4, K=K)
    ts, fr, e, _ = slab.run_slab(lat, P, T, collinear_state(4, 2), collinear_state(4, 0), sweeps,
                                 record_every=max(sweeps // 800, 5), seed=seed, kind=kind)
    v, se, nu, sat = slab.transient_velocity(ts, fr[:, 2])
    return dict(T=T, zG=zG, kind=kind, v=v, se=se, n_used=nu, saturated=sat, fF_end=float(fr[-1, 2]), fC_end=float(fr[-1, 0]))


def crossing(zs, vs, ses):
    """Zero crossing of v(zG): linear least squares over runs that actually moved (|v| > 3 se)."""
    zs, vs, ses = map(np.asarray, (zs, vs, ses))
    moved = np.abs(vs) > 3 * np.maximum(ses, 1e-12)
    if moved.sum() < 2:
        return float("nan"), float("nan")
    A = np.stack([zs[moved], np.ones(moved.sum())], 1)
    (m, c), *_ = np.linalg.lstsq(A, vs[moved], rcond=None)
    return float(-c / m) if m != 0 else float("nan"), float(m)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--size", type=int, default=24)
    ap.add_argument("--T", nargs="+", type=float, default=[0.1, 0.15, 0.2])
    ap.add_argument("--zG", nargs="+", type=float, default=[0.98, 1.0, 1.02, 1.04, 1.06, 1.08, 1.1, 1.14])
    ap.add_argument("--sweeps", type=int, default=100000)
    ap.add_argument("--kinds", nargs="+", default=["axis", "diag"])
    ap.add_argument("--K", type=float, default=0.5)
    ap.add_argument("--zH4", type=float, default=1.5)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--procs", type=int, default=4)
    a = ap.parse_args()
    jobs = [(a.size, T, z, k, a.sweeps, a.seed + i, a.K, a.zH4)
            for i, (T, z, k) in enumerate((T, z, k) for T in a.T for k in a.kinds for z in a.zG)]
    with Pool(a.procs) as p:
        res = p.map(job, jobs, chunksize=1)
    out = {}
    for T in a.T:
        for k in a.kinds:
            r = [x for x in res if x["T"] == T and x["kind"] == k]
            zs, c = crossing([x["zG"] for x in r], [x["v"] for x in r], [x["se"] for x in r])
            out[f"T{T}_{k}"] = dict(zstar=zs, slope=c)
            print(f"T={T} {k}: zG* = {zs:.4f}  (J + 0.382 T = {1 + 0.382 * T:.4f})", flush=True)
    np.savez(a.out, results=json.dumps(res), summary=json.dumps(out))
