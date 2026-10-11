"""Random-start vs ordered-start comparison for selected points (2D L=8 trials).
Usage: python tools/fig_equilibration.py <rand_dir> <ord_dir> out.png name1 name2 ..."""
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from polyspin.analysis.summary import load_run, thermo

OI = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]


def main(rd, od, out, names):
    fig, ax = plt.subplots(2, len(names), figsize=(4.4 * len(names), 7.2), constrained_layout=True, squeeze=False)
    for j, n in enumerate(names):
        a = thermo(load_run(f"{rd}/{n}_prelim.h5"))
        r = load_run(f"{od}/{n}_prelimord.h5")
        b = thermo(r)
        T = r["T"]
        ax[0, j].plot(T, a["e"], color=OI[1], label="random start")
        ax[0, j].plot(T, b["e"], color=OI[0], label="ordered start")
        ax[0, j].set_xscale("log")
        ax[0, j].set_title(n.replace("_2d_L8", " (2D L=8)"), fontsize=10)
        ax[0, j].set_ylabel("energy per cluster")
        for l, ls in ((0, "-"), (1, "--")):
            ax[1, j].plot(T, a["m"][:, l], ls, color=OI[1], label=f"m{l + 1} random")
            ax[1, j].plot(T, b["m"][:, l], ls, color=OI[0], label=f"m{l + 1} ordered")
        ax[1, j].set_xscale("log")
        ax[1, j].set_xlabel("T / J")
        if j == 0:
            ax[0, j].legend(fontsize=8)
            ax[1, j].legend(fontsize=7, ncol=2)
    fig.suptitle("Equilibration check: the random-start parallel tempering run is trapped for ordered "
                 "F / U / D ground states", fontsize=11)
    fig.savefig(out, dpi=110)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:])
