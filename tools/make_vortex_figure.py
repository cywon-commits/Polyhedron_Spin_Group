"""Figure for M2: artificial vortices of every conjugacy class of 2T, the Z2 half-vortex of the
C phase and a skyrmion of the F phase, with the detector output.
Usage: python tools/make_vortex_figure.py docs/figures/m2_artificial_vortices.png
"""
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from polyspin import defects as D
from polyspin.lattice import Lattice
from polyspin.model import tetrahedron

L, X0, Y0 = 16, 7, 7
CORE = (X0 + 0.5, Y0 + 0.5)
lat = Lattice(2, L)
coords = lat.coords.astype(float)
g = D.FrameGroup(tetrahedron())
order = ["1", "4", "6+", "6-", "3+", "3-", "2"]
colors = {"1": "#9aa0a6", "4": "#1f77b4", "6+": "#d62728", "6-": "#ff7f0e", "3+": "#2ca02c",
          "3-": "#17becf", "2": "#9467bd"}

fig, axes = plt.subplots(2, 5, figsize=(17, 7.2), constrained_layout=True)
axes = axes.ravel()
for ax, name in zip(axes, order):
    c = g.class_names.index(name)
    gi = g.classes[c][0]
    alpha = np.degrees(g.rotation_angle(gi))
    k = 1 if alpha <= 180 + 1e-6 else 2
    q = D.vortex_orientations(coords, CORE, gi, g)
    spins = D.spins_from_quats(q, g.ref)
    out = D.detect(spins, lat.plaquettes, g)
    cls_loop, ov = D.holonomy(out["q"], D.square_loop(L, X0, Y0, k)[None, :], g)
    cls1, ov1 = D.holonomy(out["q"], D.square_loop(L, X0, Y0, 1)[None, :], g)
    th = (np.arctan2(coords[:, 1] - CORE[1], coords[:, 0] - CORE[0]) % (2 * np.pi)) / (2 * np.pi)
    ax.scatter(coords[:, 0], coords[:, 1], c=th, cmap="twilight", s=28, vmin=0, vmax=1)
    ax.plot(*CORE, "k+", ms=14, mew=2)
    # square loop actually used
    lp = D.square_loop(L, X0, Y0, k)
    xy = np.vstack([coords[lp], coords[lp[:1]]])
    ax.plot(xy[:, 0], xy[:, 1], "-", color=colors[g.class_names[cls_loop[0]]], lw=2.2)
    one = g.class_names[cls1[0]] if cls1[0] >= 0 else "undefined"
    ax.set_title(f"inserted {name}  (SU(2) angle {alpha:.0f}°)\n"
                 f"loop k={k}: detected {g.class_names[cls_loop[0]]}\n"
                 f"single plaquette: {one}", fontsize=9.5)
    ax.set_xlim(2.5, 12.5)
    ax.set_ylim(2.5, 12.5)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])

# Z2 half-vortex (C phase)
ax = axes[7]
th = np.arctan2(coords[:, 1] - CORE[1], coords[:, 0] - CORE[0])
n = np.stack([np.cos(th / 2), np.sin(th / 2), np.zeros(len(th))], 1)
spins = np.stack([n, n, -n, -n], 1)
z = D.z2_vortices(spins, lat.plaquettes)
seam = (lat.coords[:, 0] == L - 1) | (lat.coords[:, 1] == L - 1)
ax.quiver(coords[:, 0], coords[:, 1], n[:, 0], n[:, 1], headwidth=0, headlength=0, headaxislength=0,
          pivot="mid", color="0.35", scale=22)
v = np.flatnonzero((z == -1) & ~seam)
for p in v:
    xy = coords[lat.plaquettes[p]].mean(0)
    ax.plot(*xy, "o", color="#d62728", ms=11)
ax.set_title(f"C phase: director half-vortex\ndetected Z2 plaquettes: {len(v)} (red)", fontsize=9.5)
ax.set_xlim(2.5, 12.5); ax.set_ylim(2.5, 12.5); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])

# skyrmion (F phase)
ax = axes[8]
lat2 = Lattice(2, 24)
c2 = lat2.coords.astype(float) - 11.5
r = np.hypot(c2[:, 0], c2[:, 1]); phi = np.arctan2(c2[:, 1], c2[:, 0])
tt = np.pi * np.clip(1 - r / 8.0, 0, 1)
m = np.stack([np.sin(tt) * np.cos(phi), np.sin(tt) * np.sin(phi), np.cos(tt)], 1)
spins = np.repeat(m[:, None, :], 4, axis=1)
q, bad = D.skyrmion_density(spins, lat2.plaquettes)
cen = np.array([lat2.coords[lat2.plaquettes[:, 0]][:, i] + 0.5 for i in range(2)]).T
sc = ax.scatter(cen[:, 0], cen[:, 1], c=q, cmap="viridis", s=26)
ax.set_title(f"F phase: skyrmion\nBerg-Lüscher charge Q = {q.sum():.6f}", fontsize=9.5)
ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
plt.colorbar(sc, ax=ax, shrink=0.8, label="charge / plaquette")
axes[9].axis("off")
axes[9].text(0, 0.5, "dots: cluster orientation\n(colour = polar angle around\nthe core, i.e. the winding)\n"
             "+ : core;  line: loop used\n(colour = detected class)", fontsize=10, va="center")
fig.suptitle("M2: artificial vortices and detector output (T phase, G = 2T, 7 conjugacy classes)", fontsize=12)
fig.savefig(sys.argv[1], dpi=130)
print("saved", sys.argv[1])
