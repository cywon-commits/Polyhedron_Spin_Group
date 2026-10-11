"""Equilibration check: ordered-start vs random-start runs of the same point (request, 'size and temperature').
For every temperature the two runs must agree; disagreement beyond 4 combined block errors marks the
temperature as NOT equilibrated.  Usage: python tools/compare_inits.py rand.h5 ord.h5 [out.png]"""
import sys

import numpy as np

from polyspin.analysis.summary import load_run, thermo


def compare(rand_path, ord_path):
    a, b = load_run(rand_path), load_run(ord_path)
    ta, tb = thermo(a), thermo(b)
    err = np.sqrt(ta["e_err"] ** 2 + tb["e_err"] ** 2) + 1e-4
    de = ta["e"] - tb["e"]
    bad = np.abs(de) > 4 * err
    return a["T"], ta, tb, de, bad


def main(rp, op, out=None):
    T, ta, tb, de, bad = compare(rp, op)
    print(f"non-equilibrated temperatures: {bad.sum()} of {len(T)}")
    if bad.any():
        print("T range of disagreement:", f"{T[bad].min():.3f} - {T[bad].max():.3f}",
              " max |de| per cluster = %.3f" % np.abs(de).max())
    if out:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
        ax[0].plot(T, ta["e"], label="random start", color="#D55E00")
        ax[0].plot(T, tb["e"], label="ordered start", color="#0072B2")
        ax[0].scatter(T[bad], tb["e"][bad], s=14, color="k", zorder=3, label="disagree (> 4 sigma)")
        ax[0].set_xscale("log")
        ax[0].set_xlabel("T / J")
        ax[0].set_ylabel("energy per cluster")
        ax[0].legend()
        ax[1].plot(T, ta["m"][:, 2], color="#D55E00", label="m3 random")
        ax[1].plot(T, tb["m"][:, 2], color="#0072B2", label="m3 ordered")
        ax[1].plot(T, ta["m"][:, 0], "--", color="#D55E00", label="m1 random")
        ax[1].plot(T, tb["m"][:, 0], "--", color="#0072B2", label="m1 ordered")
        ax[1].set_xscale("log")
        ax[1].set_xlabel("T / J")
        ax[1].legend()
        fig.savefig(out, dpi=110)
    return bad


if __name__ == "__main__":
    main(*sys.argv[1:4])
