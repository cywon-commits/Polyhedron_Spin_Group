"""Tables for the M3 report from the trial directories.
Usage: python tools/m3_tables.py --prelim-rand DIR --prelim-ord DIR [--trial DIR ...]"""
import argparse
import glob
import os
import re

import numpy as np

from polyspin.analysis.summary import load_run, thermo
from polyspin.groundstate import phase_of_gs
from polyspin.model import Params


def gs_of(run):
    a = run["attrs"]
    K = float(re.search(r"K: ([\-0-9.]+)", a["config_yaml"]).group(1))
    z = 4 if int(a["dim"]) == 2 else 6
    return phase_of_gs(Params.from_z(a["zG"], a["zH4"], z=z, K=K))


def equilibration_table(rand_dir, ord_dir):
    from compare_inits import compare
    print("| run | T=0 GS | not equilibrated (random vs ordered start) | c peak T (ordered) | c peak | tau_max(m) ordered |")
    print("| --- | --- | --- | --- | --- | --- |")
    for p in sorted(glob.glob(os.path.join(ord_dir, "*_prelimord.h5"))):
        n = os.path.basename(p).replace("_prelimord.h5", "")
        q = os.path.join(rand_dir, n + "_prelim.h5")
        if not os.path.exists(q):
            continue
        T, ta, tb, de, bad = compare(q, p)
        ro = load_run(p)
        ph = gs_of(ro)[0]
        ic = int(np.argmax(tb["c"]))
        rng = f"{T[bad].min():.3f}-{T[bad].max():.3f} ({bad.sum()}/{len(T)})" if bad.any() else "none"
        print(f"| {n} | {ph} | {rng} | {T[ic]:.3f} | {tb['c'][ic]:.0f} | {ro['tau'][:, 1:].max():.0f} |")


def trial_table(d):
    print("| run | R | sweeps | swap min | swap median | round trips | tau_e max | tau_m max | c peak T | c peak | wall [min] |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for p in sorted(glob.glob(os.path.join(d, "*.h5"))):
        r = load_run(p)
        th = thermo(r)
        ic = int(np.argmax(th["c"]))
        print(f"| {r['attrs']['name']} | {len(r['T'])} | {int(r['attrs']['sweeps'])} | {r['swap'].min():.3f} | "
              f"{np.median(r['swap']):.2f} | {r['round_trips']} | {r['tau'][:, 0].max() * r['meas_every']:.0f} | {r['tau'][:, 1:].max() * r['meas_every']:.0f} | "
              f"{r['T'][ic]:.3f} | {th['c'][ic]:.0f} | {r['attrs']['wall_seconds'] / 60:.1f} |")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--prelim-rand")
    ap.add_argument("--prelim-ord")
    ap.add_argument("--trial", nargs="*", default=[])
    a = ap.parse_args()
    if a.prelim_rand and a.prelim_ord:
        equilibration_table(a.prelim_rand, a.prelim_ord)
    for d in a.trial:
        print(f"\n{d}")
        trial_table(d)
