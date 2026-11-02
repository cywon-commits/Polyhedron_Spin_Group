"""Stiffness (helicity modulus) of the 2D spin bodies by parallel tempering (rule: docs/CD_rules.md 'Nov 2 ... 강성'; formulas: polyspin/stiffness.py).
Per replica temperature, every ``--every`` sweeps the sums S1 = sum dh/dphi and S2 = sum d^2h/dphi^2 of the x bonds (and, as a check, of the y bonds) are stored
for the lab axes x, y, z and the instantaneous director.  One common ladder for all sizes (designed for L = 32) so that Y(L) is compared at identical temperatures.
Usage: python tools/run_stiffness.py A 16 [--out data/stage2/stiff_A_L16.npz]   (point A: cut A zG = 1.0 T phase; B: cut E' zG = 1.0 C -> D)"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tools"))
import stage_runner as sr
from polyspin import stiffness as st
from polyspin.groundstate import phase_of_gs
from polyspin.lattice import Lattice
from polyspin.model import Params
from polyspin.pt import ParallelTempering

POINTS = {"A": dict(K=0.5, zH4=0.0, zG=1.0, db="data/ladder_inputs.json", key="A_2d_zG1", seed=9300),
          "B": dict(K=-0.25, zH4=0.25, zG=1.0, db="data/ladder_inputs_e2.json", key="E2_2d_zG1", seed=9400)}


def ladder(point):
    d = POINTS[point]; e = json.load(open(ROOT / d["db"]))[d["key"]]
    if point == "A":
        Tp = e["Tc"]; return sr.plan_ladder(e, 1024, 0.6 * Tp, 1.0 * Tp)
    return sr.plan_ladder(e, 1024, 0.10, 0.26)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("point"); ap.add_argument("L", type=int); ap.add_argument("--out", default=None)
    ap.add_argument("--equil", type=int, default=40000); ap.add_argument("--meas", type=int, default=40000); ap.add_argument("--every", type=int, default=5)
    ap.add_argument("--ckpt-minutes", type=float, default=10.0); ap.add_argument("--seed", type=int, default=None); a = ap.parse_args()
    d = POINTS[a.point]; L = a.L; lat = Lattice(2, L); N = lat.N
    P = Params.from_z(d["zG"], d["zH4"], z=4, K=d["K"]); p = P.array()
    temps = ladder(a.point); R = len(temps); seed = (a.seed if a.seed is not None else d["seed"]) + L
    phase, E0, S0 = phase_of_gs(P)
    out = Path(a.out or ROOT / "data/stage2" / f"stiff_{a.point}_L{L}.npz"); ck = Path(str(out) + ".ckpt.npz")
    pt = ParallelTempering(lat, P, temps, seed=seed, init="ordered", ref_state=S0)
    n_eq, n_ms = a.equil, a.meas; n_samp = n_ms // a.every
    S = {k: np.zeros((n_samp, R, 4)) for k in ("S1x", "S2x", "S1y", "S2y")}; E = np.zeros((n_samp, R)); sweep = 0; taken = 0
    if ck.exists():
        z = np.load(ck, allow_pickle=False)
        pt.load_state_dict({k[3:]: z[k] for k in z.files if k.startswith("pt_")})
        for k in S: S[k][...] = z[k]
        E[...] = z["E"]; sweep, taken = int(z["sweep"]), int(z["taken"]); print(f"resumed from checkpoint at sweep {sweep}", flush=True)
    t0 = time.time(); tlast = time.time()
    while sweep < n_eq + n_ms:
        pt.sweep_block(1); pt.exchange(); sweep += 1
        if sweep <= n_eq:
            if sweep % 200 == 0: pt.tune()
            if sweep == n_eq: pt.reset_swap_stats()
        elif (sweep - n_eq) % a.every == 0 and taken < n_samp:
            for col, key in ((0, "x"), (1, "y")):
                o1, o2 = st.terms_all(pt.spins, lat.nbr, p, col)
                S["S1" + key][taken] = o1[pt.rep_at]; S["S2" + key][taken] = o2[pt.rep_at]
            E[taken] = pt.energies[pt.rep_at] / N; taken += 1
        if time.time() - tlast > a.ckpt_minutes * 60:
            tmp = Path(str(ck) + ".tmp.npz")
            np.savez(tmp, **{f"pt_{k}": v for k, v in pt.state_dict().items()}, **S, E=E, sweep=sweep, taken=taken); tmp.replace(ck); tlast = time.time()
            print(f"[{a.point} L={L}] sweep {sweep}/{n_eq + n_ms}  {time.time() - t0:.0f}s", flush=True)
    np.savez(out, T=temps, L=L, N=N, point=a.point, phase=phase, seed=seed, n_equil=n_eq, n_meas=n_ms, every=a.every, swap_rates=pt.swap_rates(),
             round_trips=pt.round_trips, **S, E=E)
    if ck.exists(): ck.unlink()
    print(f"done {out}  R={R}  swap min {pt.swap_rates().min():.2f}  round trips {pt.round_trips}  {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
