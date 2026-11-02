"""Compact (committed) form of a stiffness run: per temperature, axis and bond direction the block means of S2 and S1^2 (20 blocks of the samples), which is all the
fluctuation formula needs.  The raw per-sample file (tens of MB) is not committed.  Usage: python tools/stiff_blocks.py data/stage2/stiff_A_L16.npz"""
import sys
from pathlib import Path
import numpy as np
NB = 20


def compact(raw):
    raw = Path(raw); z = np.load(raw)
    n = z["S1x"].shape[0]; idx = np.array_split(np.arange(n), NB)
    out = dict(T=z["T"], L=z["L"], N=z["N"], point=z["point"], phase=z["phase"], seed=z["seed"], n_equil=z["n_equil"], n_meas=z["n_meas"], every=z["every"],
               swap_rates=z["swap_rates"], round_trips=z["round_trips"], n_blocks=NB, n_samples=n)
    for col in ("x", "y"):
        S1, S2 = z["S1" + col], z["S2" + col]
        out["S2" + col] = np.stack([S2[i].mean(0) for i in idx]); out["S1sq" + col] = np.stack([(S1[i] ** 2).mean(0) for i in idx])   # (NB, R, 4)
    out["E"] = np.stack([z["E"][i].mean(0) for i in idx])
    dst = raw.with_name(raw.stem + "_blocks.npz"); np.savez_compressed(dst, **out); return dst


if __name__ == "__main__":
    for p in sys.argv[1:]: print(compact(p))
