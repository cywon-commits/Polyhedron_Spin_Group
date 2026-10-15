"""B 3D zH4 = 1.0: T_c and latent heat of the first-order nematic transition by ordered|disordered slab coexistence.
Per temperature the ordered fraction of layers is followed; the zero crossing of its velocity v(T) is T_c (bootstrap error,
pinning treated like docs/E2_rules.md rule (iii)).  Latent heat = e_disordered(T_c) - e_ordered(T_c) from the metastable branch
energies of the window chains (L = 10).  Usage: python tools/run_slab_B.py out.npz --size 12 --sweeps 80000"""
import argparse, json
from multiprocessing import Pool
import h5py, numpy as np
from polyspin import slab
from polyspin.analysis.cc_fit import crossing_samples
from polyspin.harmonic import collinear_state
from polyspin.lattice import Lattice
from polyspin.model import Params


def job(a):
    size, T, sweeps, seed, zG, zH = a
    lat = Lattice(3, size)
    P = Params.from_z(zG, zH, z=lat.z, K=0.5)
    ts, fr, lay, es, _ = slab.run_slab_order(lat, P, T, collinear_state(4, 2), sweeps, record_every=max(sweeps // 400, 10), seed=seed)
    v, se = slab.interface_velocity(ts, fr, skip=0.2)
    return dict(T=T, v=v, se=se, frac_end=float(fr[-1]), frac_start=float(fr[0]), e_end=float(es[-1]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("--size", type=int, default=12)
    ap.add_argument("--T", nargs="+", type=float, default=[round(0.310 + 0.005 * i, 3) for i in range(12)])
    ap.add_argument("--sweeps", type=int, default=80000); ap.add_argument("--zG", type=float, default=0.5); ap.add_argument("--zH4", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=21); ap.add_argument("--procs", type=int, default=4)
    a = ap.parse_args()
    with Pool(a.procs) as p:
        res = p.map(job, [(a.size, T, a.sweeps, a.seed + i, a.zG, a.zH4) for i, T in enumerate(a.T)], chunksize=1)
    for r in res:
        print(f"T={r['T']:.3f} v={r['v']:.3e}+-{r['se']:.1e} ordered fraction {r['frac_start']:.2f}->{r['frac_end']:.2f}", flush=True)
    smp, wp, nm = crossing_samples([r["T"] for r in res], [r["v"] for r in res], [r["se"] for r in res])
    if len(smp):
        Tc, sb = float(np.median(smp)), float(smp.std())
        print(f"\nT_c = {Tc:.4f} +- {sb:.4f} (bootstrap; pinned width {wp:.3f}; {nm} moved runs)")
        try:
            eo = {}
            for tag in ("winord", "winran"):
                f = h5py.File(f"runs/stage1/B_3d_L10_zH41_{tag}.h5", "r"); eo[tag] = (f["T"][:], f["series/e"][:].mean(1))
            To, Eo = eo["winord"]; Td, Ed = eo["winran"]
            m = Td >= 0.31
            lat_heat = float(np.interp(Tc, Td[m], Ed[m]) - np.interp(Tc, To[To <= 0.365], Eo[To <= 0.365]))
            print(f"latent heat e_dis(T_c) - e_ord(T_c) = {lat_heat:.3f} per cluster (branch energies of the L=10 window chains)")
        except Exception as ex:
            print("latent heat: not available", ex)
    np.savez(a.out, results=json.dumps(res))
