"""Markdown table summarising a directory of run files (test 6 at MC level, transitions, PT health).
Usage: python tools/summarize_runs.py <dir-with-h5> [> table.md]"""
import sys
from pathlib import Path

import numpy as np

from polyspin.analysis.summary import diag_means, load_run, thermo
from polyspin.groundstate import phase_of_gs
from polyspin.lattice import Lattice
from polyspin.model import Params, channel_A, energy_total
from polyspin.observables import global_class, m_l


def row(path):
    r = load_run(path)
    th, dm = thermo(r), diag_means(r)
    a, T, N = r["attrs"], r["T"], r["N"]
    lat = Lattice(int(a["dim"]), int(a["size"]))
    P = Params.from_z(a["zG"], a["zH4"], z=lat.z, K=float(eval(a["config_yaml"].split("K:")[1].split("\n")[0])))
    ph, E0, S0 = phase_of_gs(P)
    m0 = m_l(np.tile(S0, (N, 1, 1)))
    mlow = th["m"][0]
    noise = 3.0 / np.sqrt(N * 4)
    lab = global_class(mlow, max(0.15, noise))
    sp = dm["split"][0]
    shape = ["1111", "211", "22", "31", "4"][int(np.argmax(sp))] + f" {sp.max():.2f}"
    ic = int(np.argmax(th["c"]))
    chi_pk = [float(T[int(np.argmax(th["chi"][:, l]))]) for l in range(4)]
    return dict(name=a["name"], gs=ph, lab=lab, m0=np.round(m0, 3), mlow=np.round(mlow, 3),
                eoff=(th["e"][0] - E0) / T[0], Tc=T[ic], cpk=th["c"][ic], chi_pk=chi_pk,
                swap=float(r["swap"].min()), tau_m=float(r["tau"][:, 1:].max()), tau_e=float(r["tau"][:, 0].max()),
                rt=r["round_trips"], shape=shape)


def main(d):
    rows = [row(p) for p in sorted(Path(d).glob("*.h5"))]
    print("| run | T=0 GS | MC lowest-T class | m_l(T0) T=0 analytic | m_l(lowest T) | (e-E0)/T | T at c peak | c peak | swap min | tau_m max | tau_e max | split(lowest T) |")
    print("|" + " --- |" * 12)
    for x in rows:
        print(f"| {x['name'].replace('_prelim','')} | {x['gs']} | {x['lab']} | {x['m0'].tolist()} | {x['mlow'].tolist()} | "
              f"{x['eoff']:.2f} | {x['Tc']:.3f} | {x['cpk']:.1f} | {x['swap']:.3f} | {x['tau_m']:.0f} | {x['tau_e']:.0f} | {x['shape']} |")


if __name__ == "__main__":
    main(sys.argv[1])
