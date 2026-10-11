"""Wall-clock benchmark of the production loop on an idle machine.
Usage: python tools/benchmark.py --dim 2 --size 16 --R 90 [--sweeps 300] [--meas-every 5] [--stride 3]
Reports ms per parallel-tempering sweep for: MC only, + magnetisation measurements, + diagnostics."""
import argparse
import time

import numpy as np

from polyspin import measure
from polyspin.lattice import Lattice
from polyspin.model import Params, tetrahedron
from polyspin.pt import ParallelTempering


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", type=int, default=2)
    ap.add_argument("--size", type=int, default=16)
    ap.add_argument("--R", type=int, default=90)
    ap.add_argument("--sweeps", type=int, default=300)
    ap.add_argument("--meas-every", type=int, default=5)
    ap.add_argument("--diag-every", type=int, default=500)
    ap.add_argument("--stride", type=int, default=3)
    a = ap.parse_args()
    lat = Lattice(a.dim, a.size)
    P = Params.from_z(0.5, 0.0, z=lat.z, K=0.5)
    pt = ParallelTempering(lat, P, np.geomspace(0.02, 1.5, a.R), seed=1, init="ordered", ref_state=tetrahedron())
    pt.sweep_block(3)
    measure.measure_m(pt.spins)
    t = time.time()
    for _ in range(a.sweeps):
        pt.sweep_block(1)
        pt.exchange()
    t_mc = (time.time() - t) / a.sweeps
    t = time.time()
    for _ in range(a.sweeps // a.meas_every):
        measure.measure_m(pt.spins)
    t_m = (time.time() - t) / a.sweeps
    dg = measure.Diagnostics(lat, P, a.R, dict(defects=dict(holonomy_ref=tetrahedron().tolist(), stride=a.stride)))
    t = time.time()
    for tt in range(a.R):
        dg.add(tt, pt.spins_at_T(tt))
    t_d = (time.time() - t) / a.diag_every
    print(f"dim={a.dim} L={a.size} R={a.R} N={lat.N}: MC+exchange {t_mc * 1e3:.1f} ms/sweep, "
          f"measure_m (every {a.meas_every}) {t_m * 1e3:.2f}, diagnostics (every {a.diag_every}, defect stride "
          f"{a.stride}) {t_d * 1e3:.2f}  ->  total {(t_mc + t_m + t_d) * 1e3:.1f} ms per PT sweep")


if __name__ == "__main__":
    main()
