"""n = 6 thermodynamic-integration prediction of the ladder boundaries (rule: docs/CD_rules.md 'Oct 18 ... A').
`python tools/anharm_n6.py run` runs the branch chains (2D L = 16, zG = 1.1, k = 0..3, T = 0.02..0.40); `python tools/anharm_n6.py integrate` predicts."""
import argparse, importlib.util, json, sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from polyspin import slab
from polyspin.harmonic import collinear_state, selection
from polyspin.lattice import Lattice
from polyspin.model import Params, energy_total
from scipy.optimize import brentq

N6, L, ZG = 6, 16, 1.1
TS = [round(0.02 * i, 2) for i in range(1, 21)]
spec = importlib.util.spec_from_file_location("anh", ROOT / "tools/anharm.py"); anh = importlib.util.module_from_spec(spec); spec.loader.exec_module(anh)


def job(a):
    k, T, ne, nm, seed = a
    lat = Lattice(2, L); P = Params.from_z(ZG, 1.5, z=4, K=0.5, n=N6); S0 = collinear_state(N6, k)
    r = slab.run_branch(lat, P, T, S0, ne, nm, seed=seed, frac_fn=slab.moment_class_fractions)
    sp = np.tile(S0, (lat.N, 1, 1)).copy()
    r.update(k=k, size=L, e0=float(energy_total(sp, lat.nbr, P.array()) / lat.N), own=float(r["frac"][N6 // 2 - k]))
    return r


def run(a):
    jobs = [(k, T, a.equil, a.meas, a.seed + i) for i, (k, T) in enumerate((k, T) for k in range(4) for T in TS)]
    with Pool(a.procs) as p:
        res = p.map(job, jobs, chunksize=1)
    for r in res: print(f"k={r['k']} T={r['T']:.2f} E-e0={r['E'] - r['e0']:.5f}+-{r['E_err']:.5f} own={r['own']:.3f}", flush=True)
    np.savez(a.out, results=json.dumps(res))


def integrate(measured=None):
    res = json.loads(str(np.load(ROOT / "data/stage2/anharm_n6_2d_L16.npz")["results"]))
    N = L * L; x = ZG - 1.0
    P = Params.from_z(ZG, 1.5, z=4, K=0.5, n=N6); sel = selection(P, 2, 96); d = {k: sel[k]["dF_over_T"] for k in sel}
    dfs = {}
    for k in range(4):
        df = anh.DF([r for r in res if r["k"] == k], N); df.ah = 6.0 - 1.0 / N; dfs[k] = df
        de = df.delta_e(); print(f"k={k}: T range {df.T[0]:.2f}-{df.T[-1]:.2f}; delta_e/T^2 at low T {[round(float(q), 2) for q in de[:3] / df.T[:3] ** 2]}")
    def roots(sp):
        lim = min(s.x[-1] for s in sp.values())
        g = {"Ta": lambda T: 10 * x / T + d[1] + (sp[1](T) - sp[0](T)) / T, "Tb": lambda T: 6 * x / T + (d[2] - d[1]) + (sp[2](T) - sp[1](T)) / T,
             "Tc": lambda T: 2 * x / T + (d[3] - d[2]) + (sp[3](T) - sp[2](T)) / T}
        out = {}
        for q, f in g.items():
            try: out[q] = brentq(f, 0.2, lim)
            except ValueError: out[q] = float("nan")
        return out
    nom = roots({k: dfs[k].table()[0] for k in dfs})
    rng = np.random.default_rng(0); smp = [roots({k: dfs[k].sample(rng) for k in dfs}) for _ in range(2000)]
    sg = {q: float(np.nanstd([s[q] for s in smp])) for q in nom}
    harm = {"Ta": 0.1 / ((d[1] - 0) / -10) if False else -10 * x / d[1], "Tb": -6 * x / (d[2] - d[1]), "Tc": -2 * x / (d[3] - d[2])}
    out = dict(nominal=nom, sigma_int=sg, harmonic=harm, T_limit={k: float(dfs[k].T[-1]) for k in dfs})
    print("harmonic:", {q: round(v, 4) for q, v in harm.items()}); print("TI corrected:", {q: f"{nom[q]:.4f} +- {sg[q]:.4f}" for q in nom})
    shifts = {q: nom[q] - harm[q] for q in nom}; print("shifts:", {q: round(v, 4) for q, v in shifts.items()}, " width M2 (Tc-Tb): corrected", round(nom['Tc'] - nom['Tb'], 4), "harmonic", round(harm['Tc'] - harm['Tb'], 4))
    trend = all(v < 0 for v in shifts.values()) and abs(shifts["Ta"]) < abs(shifts["Tb"]) < abs(shifts["Tc"])
    print("trend (all shifts < 0 and |shift| increasing a<b<c):", trend)
    meas = measured or {"Ta": (0.3205, 0.0011), "Tb": (0.3544, 0.0021), "Tc": (0.3641, 0.0045)}
    for q in nom:
        mv, me = meas[q]; tol = 2 * np.hypot(me, sg[q]); print(f"  {q}: predicted {nom[q]:.4f} +- {sg[q]:.4f}; measured {mv:.4f} +- {me:.4f}; |diff| {abs(nom[q] - mv):.4f} vs 2 sigma {tol:.4f}: {'reproduced' if abs(nom[q] - mv) <= tol else 'NOT reproduced'}")
    out["trend"] = bool(trend); return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sp = ap.add_subparsers(dest="cmd")
    r = sp.add_parser("run"); r.add_argument("out", nargs="?", default="data/stage2/anharm_n6_2d_L16.npz"); r.add_argument("--equil", type=int, default=5000); r.add_argument("--meas", type=int, default=20000)
    r.add_argument("--seed", type=int, default=71000); r.add_argument("--procs", type=int, default=4)
    sp.add_parser("integrate"); a = ap.parse_args()
    if a.cmd == "run": run(a)
    elif a.cmd == "integrate": integrate()
