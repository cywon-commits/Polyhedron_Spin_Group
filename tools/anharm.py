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
            for T in (a.T if a.T else TS):
                jobs.append((a.dim, a.size, zG, k, round(T, 3), a.equil, a.meas, a.seed + len(jobs)))
    with Pool(a.procs) as p:
        res = p.map(job, jobs, chunksize=1)
    for r in res:
        print(f"zG={r['zG']} k={r['k']} T={r['T']:.2f} E-e0={r['E'] - r['e0']:.5f}+-{r['E_err']:.5f} own={r['own']:.3f}", flush=True)
    np.savez(a.out, results=json.dumps(res))


def curv(res, dim, zG, k, order=2, tmax=None):
    from polyspin.analysis.anharm import fit_curvature
    rows = sorted((r for r in res if r["dim"] == dim and r["zG"] == zG and r["k"] == k and r["own"] >= 0.9 and (tmax is None or r["T"] <= tmax + 1e-9)),
                  key=lambda r: r["T"])
    T = [r["T"] for r in rows]; e = [r["E"] for r in rows]; er = [r["E_err"] for r in rows]
    f = fit_curvature(T, e, er, rows[0]["e0"], order)
    f["n_T"] = len(T); f["Tmax"] = max(T); return f


def analyze():
    from polyspin.analysis.anharm import solve_T
    from polyspin.analysis import cc_fit as cc
    lq = {2: 96, 3: 24}; z = {2: 4, 3: 6}
    meas = {}
    for dim, fa, fb in ((2, "slab_M2F_zG1.1", "slab_CM2_zG1.1"), (3, "slab_M2F_3D_L12_supp", "slab_CM2_3D_L12_remeasure")):
        out = []
        for f in (fa, fb):
            r = json.loads(str(np.load(f"data/stage2/{f}.npz")["results"]))
            e = cc.crossing_seedmean([x["T"] for x in r], [x["v"] for x in r]); out.append((e["zero"], e["sigma_total"]))
        meas[dim] = out
    for dim, size in ((2, 16), (3, 8)):
        try:
            res = json.loads(str(np.load(f"data/stage2/anharm_{dim}d_L{size}.npz")["results"]))
        except OSError:
            print(f"dim {dim}: chains not finished"); continue
        N = size ** dim
        c = {}
        for zG in (1.0, 1.1):
            for k in (0, 1, 2):
                f2, f3, fr = curv(res, dim, zG, k, 2), curv(res, dim, zG, k, 3), curv(res, dim, zG, k, 2, tmax=0.20)
                sc = float(np.hypot(f2["sc"], max(abs(f2["c"] - f3["c"]), abs(f2["c"] - fr["c"]))))
                c[(zG, k)] = (f2["c"], sc, f2["sc"], f3["c"])
                print(f"dim{dim} L={size} zG={zG} k={k}: a={f2['a']:.3f} (harm {4 - 3 / (2 * N):.3f}) c={f2['c']:.3f}+-{f2['sc']:.3f} cubic-fit c={f3['c']:.3f} Tmax<=0.20 c={fr['c']:.3f} nT={f2['n_T']} Tmax={f2['Tmax']:.2f} -> sigma_c {sc:.3f}", flush=True)
        for zG in (1.0, 1.1):
            P = Params.from_z(zG, 1.5, z=z[dim], K=0.5)
            sel = selection(P, dim, lq[dim]); d = {k: sel[k]["dF_over_T"] for k in sel}
            x = zG - 1.0
            dcM = c[(zG, 1)][0] - c[(zG, 0)][0]; sM = np.hypot(c[(zG, 1)][1], c[(zG, 0)][1])
            dcC = c[(zG, 2)][0] - c[(zG, 0)][0]; sC = np.hypot(c[(zG, 2)][1], c[(zG, 0)][1])
            print(f"dim{dim} zG={zG}: delta_M2={d[1]:.4f} delta_C={d[2]:.4f} dc_M2={dcM:.3f}+-{sM:.3f} dc_C={dcC:.3f}+-{sC:.3f}")
            print(f"   (fit-error-only sigma of b1: {np.hypot(c[(zG, 1)][2], c[(zG, 0)][2]) / 6:.3f})")
            print(f"   harmonic F|M2 slope a1 = {-d[1] / 6:.4f}; corrected b1 = dc_M2/6 = {dcM / 6:.3f} +- {sM / 6:.3f} (H6 measured b = 0.15 +- 0.05, 2D)")
            if zG == 1.1:
                rng = np.random.default_rng(0)
                T1 = solve_T(6 * x, d[1], dcM); T2 = solve_T(2 * x, d[2] - d[1], dcC - dcM)
                smp = [(solve_T(6 * x, d[1], dcM + sM * rng.normal()), solve_T(2 * x, d[2] - d[1], dcC - dcM + np.hypot(sC, sM) * rng.normal())) for _ in range(2000)]
                s1, s2 = np.nanstd([q[0] for q in smp]), np.nanstd([q[1] for q in smp])
                T1h, T2h = solve_T(6 * x, d[1], 0.0), solve_T(2 * x, d[2] - d[1], 0.0)
                (m1, e1), (m2, e2) = meas[dim]
                sMf, sCf = np.hypot(c[(zG, 1)][2], c[(zG, 0)][2]), np.hypot(c[(zG, 2)][2], c[(zG, 0)][2])
                smf = [(solve_T(6 * x, d[1], dcM + sMf * rng.normal()), solve_T(2 * x, d[2] - d[1], dcC - dcM + np.hypot(sCf, sMf) * rng.normal())) for _ in range(2000)]
                print(f"   (supplementary) fit-error-only sigma_pred: T1 {np.nanstd([q[0] for q in smf]):.4f}, T2 {np.nanstd([q[1] for q in smf]):.4f}; cubic-fit dc_M2 = {c[(zG, 1)][3] - c[(zG, 0)][3]:+.3f}, dc_C = {c[(zG, 2)][3] - c[(zG, 0)][3]:+.3f} (quadratic: {dcM:+.3f}, {dcC:+.3f})")
                for nm, pr, sp_, mv, me, h in (("T1", T1, s1, m1, e1, T1h), ("T2", T2, s2, m2, e2, T2h)):
                    tol = 2 * np.hypot(sp_, me)
                    print(f"   {nm}: harmonic {h:.4f} -> corrected {pr:.4f} +- {sp_:.4f}; measured {mv:.4f} +- {me:.4f}; |diff| {abs(pr - mv):.4f} vs 2 sigma {tol:.4f}: {'reproduced' if abs(pr - mv) <= tol else 'NOT reproduced'}")
                print(f"   width: corrected {T2 - T1:.4f}, harmonic {T2h - T1h:.4f}, measured {m2 - m1:.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sp = ap.add_subparsers(dest="cmd")
    r = sp.add_parser("run"); r.add_argument("out"); r.add_argument("--dim", type=int, default=2); r.add_argument("--size", type=int, default=16)
    r.add_argument("--equil", type=int, default=5000); r.add_argument("--meas", type=int, default=20000)
    r.add_argument("--seed", type=int, default=41000); r.add_argument("--T", nargs="+", type=float, default=None); r.add_argument("--procs", type=int, default=4)
    sp.add_parser("analyze")
    a = ap.parse_args()
    if a.cmd == "run":
        run(a)
    elif a.cmd == "analyze":
        analyze()
