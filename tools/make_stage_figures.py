"""Stage figures from data/stageN/*.npz: for every point (cut, dim, value) one figure with size curves of
e, c, m_l, Binder of the order parameter, defect density rho_def and undefined fraction f_ud (H2 logic).
Usage: python tools/make_stage_figures.py --stage 1 --outdir docs/figures/stage1"""
import argparse
import glob
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OI = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#999999", "#000000"]


def load(path):
    z = np.load(path, allow_pickle=False)
    d = {k: z[k] for k in z.files}
    d["attrs"] = json.loads(str(z["attrs"]))
    d["class_names"] = json.loads(str(z["class_names"]))
    return d


def group(stage):
    g = defaultdict(lambda: defaultdict(list))
    for p in sorted(glob.glob(f"data/stage{stage}/*.npz")):
        m = re.match(r"(\w+?)_(\d)d_L(\d+)_(zG|zH4)([\d.]+)_(\w+)\.npz", Path(p).name)
        cut, dim, L, par, val, tag = m.groups()
        g[(cut, int(dim), par, float(val))][int(L)].append((tag, load(p)))
    return g


def merged(parts, keys):
    """Concatenate runs of one size (ladder + cold chains) sorted by temperature."""
    T = np.concatenate([d["T"] for _, d in parts])
    o = np.argsort(T)
    return T[o], {k: np.concatenate([d[k] for _, d in parts])[o] for k in keys}


def main(stage, outdir):
    Path(outdir).mkdir(parents=True, exist_ok=True)
    for (cut, dim, par, val), sizes in group(stage).items():
        fig, ax = plt.subplots(2, 3, figsize=(15, 8.5), constrained_layout=True)
        ax = ax.ravel()
        gs = list(sizes.values())[0][0][1]["attrs"]["gs_phase"]
        order = 1 if gs == "C" else 2            # m3 for the T phase, m2 for the nematic C
        for ci, (L, parts) in enumerate(sorted(sizes.items())):
            col = OI[ci % 8]
            main_parts = [(t, d) for t, d in parts if t in ("pt", "heat", "cold")]
            T, v = merged(main_parts, ["th_e", "th_c", "th_binder"])
            ax[0].plot(T, v["th_e"], color=col, label=f"L={L}")
            ax[1].plot(T, v["th_c"], color=col, label=f"L={L}")
            m = np.concatenate([d["th_m"] for _, d in main_parts])[np.argsort(np.concatenate([d["T"] for _, d in main_parts]))]
            ax[2].plot(T, m[:, order], color=col, label=f"L={L}")
            ax[3].plot(T, v["th_binder"][:, order], color=col, label=f"L={L}")
            for t, d in main_parts:
                if t == "cold":
                    continue
                names = d["class_names"]
                if "dm_block2_density" in d:
                    rep = [i for i, n in enumerate(names) if n in ("4", "6+", "6-", "4+", "4-")]
                    ls = "-" if len(sizes) == 1 else "-"
                    for key, lab, st in (("dm_plaq_density", "plaquette", "o-"), ("dm_coarse_density", "coarse 2x2", "s--"),
                                         ("dm_block2_density", "block k=2", "^:")):
                        if key in d:
                            rho = d[key][:, rep].sum(1)
                            ok = np.isfinite(rho) & (rho > 0)
                            ax[4].plot(d["T"][ok], rho[ok], st, ms=3, color=col, alpha=0.9,
                                       label=f"L={L} {lab}")
                    for key, lab, st in (("dm_plaq_und_frac", "plaquette", "o-"), ("dm_coarse_und_frac", "coarse", "s--"),
                                         ("dm_block2_nontrivial_fraction", "block nontrivial", "^:")):
                        if key in d:
                            y = d[key]
                            ok = np.isfinite(y)
                            ax[5].plot(d["T"][ok], y[ok], st, ms=3, color=col, label=f"L={L} {lab}")
                elif "dm_z2_density" in d:
                    z2 = d["dm_z2_density"]
                    ok = np.isfinite(z2) & (z2 > 0)
                    ax[4].plot(d["T"][ok], z2[ok], "o-", ms=3, color=col, label=f"L={L} Z2")
        for i, tt in enumerate(["energy per cluster", "specific heat per cluster", f"<m{order + 1}>",
                                f"Binder U{order + 1}", "defect density per site (1,4,6+- reliable) / Z2",
                                "undefined fraction f_ud (block: nontrivial fraction)"]):
            ax[i].set_title(tt, fontsize=10)
            ax[i].set_xlabel("T / J")
            ax[i].legend(fontsize=6 if i >= 4 else 8)
        ax[4].set_yscale("log")
        fig.suptitle(f"{cut} {dim}D {par}={val:g}  (T=0 GS {gs})")
        fig.savefig(f"{outdir}/{cut}_{dim}d_{par}{val:g}.png", dpi=100)
        plt.close(fig)
        print("wrote", f"{outdir}/{cut}_{dim}d_{par}{val:g}.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, default=1)
    ap.add_argument("--outdir", default="docs/figures/stage1")
    a = ap.parse_args()
    main(a.stage, a.outdir)
