"""Entropic ladder at zG = 1.1 (2D, K = 0.5, zH4 = 1.5) with the measured boundaries (slab coexistence, paper/data/wedge.json) marked as u = (zG - J)/T."""
from style import *; import numpy as np, json
W = json.load(open('../data/wedge.json'))
fig, axs = plt.subplots(2, 1, figsize=(COL, 4.9), sharex=False)
x = 0.1
def panel(ax, n, deltas, labels, umin, umax, title, meas, harm_T=None, ti=None):
    u = np.linspace(umin, umax, 800); lines = {'F': 0 * u}
    for (lab, M), d in zip(labels, deltas): lines[lab] = (n * n - M * M) / 2 * u + d
    names = list(lines); Y = np.array([lines[k] for k in names]); env = Y.argmin(0)
    cols = {'F': C['F'], 'M4': C['M4'], 'M2': C['M2'], 'C': C['C']}
    for i, k in enumerate(names):
        ax.plot(u, Y[i], color=cols[k], lw=0.9, alpha=0.45); m = env == i
        ax.plot(np.where(m, u, np.nan), np.where(m, Y[i], np.nan), color=cols[k], lw=2.6, label=k)
    bnd = [u[j] for j in np.where(np.diff(env) != 0)[0]]
    for b in bnd: ax.axvline(b, color='k', ls=':', lw=0.7)
    ylo, yhi = ax.get_ylim() if False else (None, None)
    return bnd
# n = 4: zG = 1.1 (delta_M2 = -2.4685, delta_C = -3.2245)
b4 = panel(axs[0], 4, [-2.4685, -3.2245], [('M2', 2), ('C', 0)], 0.33, 0.47, r'(a) $n=4$ (2D, $zG=1.1$)', None)
axs[0].set_ylim(-0.9, 1.5)
b6 = panel(axs[1], 6, [-3.030, -4.663, -5.185], [('M4', 4), ('M2', 2), ('C', 0)], 0.22, 0.36, r'(b) $n=6$ (2D, $zG=1.1$)', None)
axs[1].set_ylim(-1.0, 1.7)
def meas(ax, Ts, col_y, mk, fc, ylab):
    for T, s in Ts:
        ax.errorbar([x / T], [col_y], xerr=[x * s / T ** 2], fmt=mk, color='k', ms=5, mfc=fc, capsize=1.5, elinewidth=0.9)
yt = 0.85
meas(axs[0], [(W['d2']['T1']['zero'], W['d2']['T1']['sigma']), (W['d2']['T2']['zero'], W['d2']['T2']['sigma'])], yt, 'o', 'white', '')
ti = W['TI']
for q in ('T1', 'T2'): axs[0].plot([x / ti[f'{q}_2d']['pred']], [yt - 0.2], 'x', color='k', ms=5, mew=1.3)
yt6 = 0.95
meas(axs[1], [(W['n6']['Ta']['zero'], W['n6']['Ta']['sigma']), (W['n6']['Tb']['zero'], W['n6']['Tb']['sigma'])], yt6, 'o', 'white', '')
meas(axs[1], [(W['n6']['Tc']['zero'], W['n6']['Tc']['sigma'])], yt6, 'o', 'white', '')
for q in ('Ta', 'Tb', 'Tc'): axs[1].plot([x / W['TI_n6']['nominal'][q]], [yt6 - 0.3], 'x', color='k', ms=5, mew=1.3)
axs[1].plot([], [], 'o', color='k', mfc='white', ms=5, label='slab'); axs[1].plot([], [], 'x', color='k', ms=5, mew=1.3, label='TI-corrected'); axs[0].plot([], [], 'o', color='k', mfc='white', ms=5, label='slab'); axs[0].plot([], [], 'x', color='k', ms=5, mew=1.3, label='TI-corrected')
for ax in axs:
    ax.set_ylabel(r'$\Delta F_k/T$'); ax.set_xlabel(r'$u=(zG-J)/T$   (low $T$ $\rightarrow$)')
axs[0].set_title(r'(a) $n=4$ (2D, $zG=1.1$)', loc='left', fontsize=9.5); axs[1].set_title(r'(b) $n=6$ (2D, $zG=1.1$)', loc='left', fontsize=9.5)
axs[0].legend(loc='upper left', ncol=4, handlelength=1.2, columnspacing=0.8, fontsize=8, bbox_to_anchor=(0.0, 1.0)); axs[1].legend(loc='upper left', ncol=5, handlelength=1.2, columnspacing=0.8, fontsize=8, bbox_to_anchor=(0.0, 1.0))
fig.tight_layout(h_pad=0.8); fig.savefig('../fig/fig_ladder.pdf'); print(b4, b6)
