"""Low-temperature slope table for the report (follow-up 1.2).
Usage: python tools/lowT_report.py docs/lowT_slopes.md"""
import sys
from multiprocessing import Pool

import numpy as np

from polyspin.analysis.lowt import measure_slope
from polyspin.groundstate import find_ground_state
from polyspin.model import Params

# name, zG, zH4, K, L, ground-state override (None = global minimum)
CASES = [("T", 0.5, 0.0, 0.5, 8, None), ("T", 0.5, 0.0, 0.5, 6, None), ("T", 0.5, 0.0, 0.5, 12, None),
         ("C", 0.5, 1.0, 0.5, 8, None), ("F", 2.0, 0.0, 0.5, 8, None), ("U", 1.0, 0.0, 0.0, 8, None),
         ("D", 0.9, 0.0, -0.5, 8, None), ("D", 0.9, 0.0, -0.5, 12, None),
         ("C (zG=J degenerate)", 1.0, 1.5, 0.5, 8, "C"), ("F (zG=J degenerate)", 1.0, 1.5, 0.5, 8, "F")]


def start(zG, zH, K, which):
    up = np.array([0, 0, 1.0])
    return {"C": np.array([up, up, -up, -up]), "F": np.array([up] * 4)}[which]


def work(c):
    name, zG, zH, K, L, ov = c
    S0 = None if ov is None else start(zG, zH, K, ov)
    r = measure_slope(zG, zH, K, L=L, S0=S0)
    return c, r


if __name__ == "__main__":
    with Pool(4) as pool:
        out = pool.map(work, CASES)
    lines = ["| case | (zG, zH4, K) | L | N | fitted slope a | ± | 4 - 3/(2N) | 4 - 2/(2N) | a/4 - 1 |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for (name, zG, zH, K, L, ov), r in out:
        N = r["N"]
        lines.append(f"| {name} (found {r['phase']}) | ({zG}, {zH}, {K}) | {L} | {N} | {r['a']:.4f} | "
                     f"{r['a_err']:.4f} | {4 - 1.5 / N:.4f} | {4 - 1.0 / N:.4f} | {r['a'] / 4 - 1:+.4f} |")
    open(sys.argv[1], "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
