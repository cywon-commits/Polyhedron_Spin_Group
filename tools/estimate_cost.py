"""Compute-time estimate for the production runs from measured sweep speed and ladder size.

wall time per PT sweep  = R * N * n * t_spin / n_threads          (t_spin = measured seconds per spin-sweep)
number of replicas R    = measured ladder size scaled with sqrt(N)  (swap acceptance of an extensive system)
Usage: python tools/estimate_cost.py [--threads 4] [--sweeps 200000]"""
import argparse

# measured on this machine (single thread, mc_sweep with single+rotation+overlap moves)
# effective seconds per spin-sweep and thread inside the PT loop (benchmark on an idle 4-core machine:
# 2D L=16 R=90 17.1 ms, 3D L=8 R=121 52.3 ms per PT sweep for MC + exchange)
T_SPIN = {2: 0.74e-6, 3: 0.84e-6}
# ladder sizes for target acceptance 0.3 over T in [0.01, 1.5] (2D) / [0.02, 1.5] (3D), from the L=8 trial runs
R_REF = {2: (8, 52), 3: (8, 121)}      # (L_ref, R at L_ref) -> R ~ L^(dim/2)  (R ~ sqrt(N))
SIZES = {2: (16, 24, 32, 48, 64), 3: (8, 10, 12, 16)}
POINTS = {"A": 4, "A'": 0, "B": 3, "C": 4, "D": 3, "E": 4}


def replicas(dim, L):
    Lr, Rr = R_REF[dim]
    return Rr * (L / Lr) ** (dim / 2.0)


def hours(dim, L, sweeps, threads, overhead=1.2):
    N = L ** dim
    R = replicas(dim, L)
    return R * N * 4 * T_SPIN[dim] * sweeps / threads * overhead / 3600.0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--sweeps", type=int, default=200000, help="equilibration + measurement sweeps per run")
    a = ap.parse_args()
    print(f"({a.sweeps} sweeps per run, {a.threads} threads, 20% measurement/diagnostics overhead (meas_every 5, defect stride 3))")
    print("| dim | L | N | replicas R | ms per PT sweep | hours per point | hours per cut line (A,B,C,D,E = 18 points) |")
    print("| --- | --- | --- | --- | --- | --- | --- |")
    for dim in (2, 3):
        for L in SIZES[dim]:
            R = replicas(dim, L)
            h = hours(dim, L, a.sweeps, a.threads)
            print(f"| {dim} | {L} | {L ** dim} | {R:.0f} | {h * 3600e3 / a.sweeps:.0f} | {h:.1f} | {18 * h:.0f} |")
