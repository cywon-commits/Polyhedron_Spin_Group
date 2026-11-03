"""Stiffness of the 2D spin bodies (paper/data/stiffness.json; rule: docs/CD_rules.md): (a) T phase, axis-averaged Y vs T/T_peak; (b) C -> D line, director stiffness Y_par vs T with the KT line 2T/pi."""
from style import *; import numpy as np, json
D = json.load(open('../data/stiffness.json')); TP = 0.21425
SEQ = {16: '#9ecae1', 24: '#4292c6', 32: '#08306b'}          # one hue, light -> dark with size
fig, axs = plt.subplots(1, 2, figsize=(DCOL, 2.7))
ax = axs[0]; A = D['A']; T = np.array(A['T']); m = (T >= 0.6 * TP - 1e-9) & (T <= TP + 1e-9)
for L in (16, 24, 32):
    y = np.array(A['L'][str(L)]['Y']); e = np.array(A['L'][str(L)]['err'])
    ax.fill_between(T[m] / TP, (y - 2 * e)[m], (y + 2 * e)[m], color=SEQ[L], alpha=0.25, lw=0)
    ax.plot(T[m] / TP, y[m], '-o', color=SEQ[L], ms=2.5, lw=1.2, label=f'$L={L}$')
for g in A['grid_T']: ax.axvline(g / TP, color='0.85', lw=0.6, zorder=0)
ax.set_xlabel(r'$T/T_{\rm peak}$'); ax.set_ylabel(r'$\Upsilon$ (axis average)'); ax.set_xlim(0.58, 1.02)
ax.legend(loc='lower left', fontsize=8.5, handlelength=1.2)
ax.set_title(r'(a) T phase, cut A ($zG=1$)', loc='left', fontsize=9.5)
ax = axs[1]; B = D['B']; T = np.array(B['T']); ax.plot(T, 2 * T / np.pi, 'k--', lw=1.0)
ax.text(0.2305, 0.168, r'$2T/\pi$', fontsize=8.5, ha='left')
for L in (16, 24, 32):
    y = np.array(B['L'][str(L)]['Y']); e = np.array(B['L'][str(L)]['err'])
    ax.fill_between(T, y - 2 * e, y + 2 * e, color=SEQ[L], alpha=0.25, lw=0); ax.plot(T, y, '-', color=SEQ[L], lw=1.2, label=f'$L={L}$')
    c = B['details']['crossings'][str(L)]
    ax.plot([c[0]], [2 * c[0] / np.pi], 'o', color=SEQ[L], ms=4.5, mfc='white', zorder=5)
ax.axhline(0, color='0.7', lw=0.6, zorder=0); ax.set_xlim(0.10, 0.26); ax.set_ylim(-0.08, 0.72)
ax.set_xlabel(r'$T$'); ax.set_ylabel(r'$\Upsilon_\parallel$ (director)')
ax.legend(loc='upper right', fontsize=8.5, handlelength=1.2)
ax.set_title(r"(b) C$\to$D line, cut E$^\prime$ ($zG=1$)", loc='left', fontsize=9.5)
fig.tight_layout(w_pad=1.4); fig.savefig('../fig/fig_stiffness.pdf')
