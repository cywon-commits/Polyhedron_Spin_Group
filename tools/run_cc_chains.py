"""Clausius-Clapeyron single-branch chains for the C|F boundary (H6; decision Oct 13): at each T and zG, one ordered-start
chain in C and one in F (L = 24, no exchange); <E> and <dH/dzG> give  s(T) = [ (E_C - E_F)/T ] / [ <dH/dzG>_C - <dH/dzG>_F ]
= d zG*/dT on the coexistence line.  Usage: python tools/run_cc_chains.py out.npz --size 24 --T 0.10 0.125 0.15 0.175 0.20
   --zG 1.02 1.05 1.08 --equil 10000 --meas 40000"""
import argparse, json
from multiprocessing import Pool
import numpy as np
from polyspin import slab
from polyspin.harmonic import collinear_state
from polyspin.lattice import Lattice
from polyspin.model import Params


def job(a):
    size, T, zG, branch, ne, nm, seed, K, zH = a
    lat = Lattice(2, size)
    P = Params.from_z(zG, zH, z=4, K=K)
    S0 = collinear_state(4, 2) if branch == "C" else collinear_state(4, 0)
    r = slab.run_branch(lat, P, T, S0, ne, nm, seed=seed)
    r["branch"] = branch
    return r


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("--size", type=int, default=24)
    ap.add_argument("--T", nargs="+", type=float, default=[0.10, 0.125, 0.15, 0.175, 0.20])
    ap.add_argument("--zG", nargs="+", type=float, default=[1.02, 1.05, 1.08])
    ap.add_argument("--equil", type=int, default=10000); ap.add_argument("--meas", type=int, default=40000)
    ap.add_argument("--K", type=float, default=0.5); ap.add_argument("--zH4", type=float, default=1.5)
    ap.add_argument("--seed", type=int, default=11); ap.add_argument("--procs", type=int, default=4)
    a = ap.parse_args()
    jobs = [(a.size, T, z, b, a.equil, a.meas, a.seed + i, a.K, a.zH4)
            for i, (T, z, b) in enumerate((T, z, b) for T in a.T for z in a.zG for b in ("C", "F"))]
    with Pool(a.procs) as p:
        res = p.map(job, jobs, chunksize=1)
    for r in res:
        print(f"T={r['T']:.3f} zG={r['zG']:.3f} {r['branch']}: E={r['E']:.4f}+-{r['E_err']:.4f} dH={r['dH']:.4f}+-{r['dH_err']:.4f} frac(C,M2,F)={np.round(r['frac'], 3).tolist()}", flush=True)
    np.savez(a.out, results=json.dumps(res))
