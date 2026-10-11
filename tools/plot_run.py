"""One-page overview of a run: energy and specific heat, m_l, Binder, split fractions, defects, PT health.
Usage: python tools/plot_run.py run.h5 out.png"""
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from polyspin.analysis.summary import diag_means, load_run, thermo

OI = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#999999", "#000000"]


def main(path, out):
    r = load_run(path)
    th, dm = thermo(r), diag_means(r)
    T, a = r["T"], r["attrs"]
    fig, ax = plt.subplots(3, 3, figsize=(15, 11), constrained_layout=True)
    ax = ax.ravel()
    ax[0].errorbar(T, th["e"], th["e_err"], color=OI[0], marker=".", ms=4, lw=1)
    ax[0].set_ylabel("energy per cluster")
    ax0b = ax[0].twinx()
    ax0b.plot(T, th["c"], color=OI[1], lw=1.2)
    ax0b.set_ylabel("specific heat per cluster", color=OI[1])
    for l in range(4):
        ax[1].errorbar(T, th["m"][:, l], th["m_err"][:, l], color=OI[l], marker=".", ms=3, lw=1, label=f"m{l + 1}")
        ax[2].plot(T, th["binder"][:, l], color=OI[l], lw=1.2, label=f"U{l + 1}")
    ax[1].set_ylabel("<m_l>")
    ax[1].legend(ncol=4, fontsize=8)
    ax[2].set_ylabel("Binder U_l")
    ax[2].legend(ncol=4, fontsize=8)
    for l in range(4):
        ax[3].plot(T, th["chi"][:, l], color=OI[l], lw=1.2, label=f"chi{l + 1}")
    ax[3].set_yscale("log")
    ax[3].set_ylabel("susceptibility chi_l")
    ax[3].legend(ncol=4, fontsize=8)
    for k, nm in enumerate(["{1111}", "{211}", "{22}", "{31}", "{4}"]):
        ax[4].plot(T, dm["split"][:, k], color=OI[k], lw=1.4, label=nm)
    ax[4].set_ylabel("cluster split fractions")
    ax[4].legend(ncol=5, fontsize=8)
    if "class_names" in dm:
        names = dm["class_names"]
        triv = max(range(len(names)), key=lambda i: dm["plaq_density"][0, i])
        for i, nm in enumerate(names):
            if i != triv:
                ax[5].plot(T, dm["block2_density"][:, i] + 1e-9, lw=1.4, color=OI[i % 8], label=f"block k=2 {nm}")
        ax[5].plot(T, dm["block2_nontrivial_fraction"], "k--", lw=1, label="nontrivial block fraction")
        ax[5].axhline(0.1, color="0.6", lw=0.8)
        ax[5].set_yscale("log")
        ax[5].set_ylabel("defects per site (blocks) / fraction")
        ax[5].legend(fontsize=6, ncol=2)
    elif "z2_density" in dm:
        ax[5].plot(T, dm["z2_density"], color=OI[1], lw=1.4, label="Z2 half-vortices / site")
        ax[5].set_yscale("log")
        ax[5].legend(fontsize=8)
    elif "sky_density" in dm:
        ax[5].plot(T, dm["sky_density"], color=OI[2], lw=1.4, label="|skyrmion charge| density")
        ax[5].set_yscale("log")
        ax[5].legend(fontsize=8)
    ax[6].plot(T[:-1], r["swap"], color=OI[0], marker=".", label="swap acc. (measurement)")
    ax[6].plot(T[:-1], r["swap_equil"], color=OI[1], marker=".", lw=0.8, label="swap acc. (equil.)")
    ax[6].axhline(0.2, color="0.5", lw=0.8)
    ax[6].set_ylabel("exchange acceptance")
    ax[6].legend(fontsize=8)
    for k, nm in enumerate(["e", "m1", "m2", "m3", "m4"]):
        ax[7].plot(T, r["tau"][:, k], lw=1.2, color=OI[k], label=nm)
    ax[7].set_yscale("log")
    ax[7].set_ylabel("tau_int (samples)")
    ax[7].legend(ncol=5, fontsize=8)
    ax[8].plot(T, r["up_fraction"], color=OI[2])
    ax[8].set_ylabel("replica-flow fraction 'up'")
    ax[8].set_ylim(0, 1)
    ax[8].set_title(f"round trips: {r['round_trips']}", fontsize=9)
    for x in ax:
        x.set_xscale("log")
        x.set_xlabel("T / J")
    fig.suptitle(f"{a['name']}   (dim {a['dim']}, L {a['size']}, zG={a['zG']:g}, zH4={a['zH4']:g}, "
                 f"GS {a['gs_phase']}, {a['sweeps']} sweeps, {a['wall_seconds']:.0f}s)")
    fig.savefig(out, dpi=110)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
