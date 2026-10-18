"""Phases near the ferro-nematic line: H6 fit and the individual slab zero crossings (paper/data/H6_points.json), the M2 wedge positions at zG = 1.1 from the
final estimators (paper/data/wedge.json) with the thermodynamic-integration predictions."""
from style import *; import numpy as np, json
H = json.load(open('../data/H6_points.json')); W = json.load(open('../data/wedge.json'))
fig, ax = plt.subplots(figsize=(COL, 3.0))
def Tb(s0, s1):
    zg = np.linspace(1.0, 1.13, 300); s = s0 + (s1 - s0) * (zg - 1.0) / 0.1; return zg, (zg - 1.0) / s
zg, t1 = Tb(0.3899, 0.1 / 0.2431); ax.plot(zg, t1, '-', color=C['F'], lw=1.1, label='F|M2 (harm.+label)')
zg, t2 = Tb(0.3592, 0.1 / 0.2646); ax.plot(zg, t2, '-', color=C['C'], lw=1.1, label='M2|C (harm.+label)')
Tf = np.linspace(0, 0.2, 100); zf = 1 + 0.392 * Tf + 0.15 * Tf ** 2
ax.fill_betweenx(Tf, zf - 0.010 * Tf, zf + 0.010 * Tf, color='k', alpha=0.15, lw=0)
ax.plot(zf, Tf, 'k-', lw=1.4, label=r'H6 fit $a=0.392(10)$')
for a, ls in [(0.224, '--'), (0.158, ':')]: ax.plot(1 + a * Tf, Tf, ls, color='0.45', lw=1.0)
ax.text(1.047, 0.205, 'label\nonly', fontsize=7.5, color='0.35', ha='left', va='bottom'); ax.text(1.029, 0.205, 'harm.\nonly', fontsize=7.5, color='0.35', ha='right', va='bottom')
sc = [r for r in H if r['set'] == 'scan' and r['T'] <= 0.15]; rs = [r for r in H if r['set'] == 'rescan']
ax.errorbar([r['zstar'] for r in sc], [r['T'] for r in sc], xerr=[r['sigma'] for r in sc], fmt='o', color='k', ms=4, capsize=1.5, elinewidth=0.8, label='slab zeros (fit input)')
ax.errorbar([r['zstar'] for r in rs], [r['T'] for r in rs], xerr=[r['sigma'] for r in rs], fmt='D', color='k', ms=4, mfc='white', capsize=1.5, elinewidth=0.8, label='slab zeros, rescan (aux.)')
x2, x3 = 1.094, 1.106
def pt(x, T, s, mk, col, fc): ax.errorbar([x], [T], yerr=[s], fmt=mk, color=col, ms=5, mfc=fc, capsize=1.5, elinewidth=0.8)
pt(x2, W['d2']['T1']['zero'], W['d2']['T1']['sigma'], 'o', C['F'], 'white'); pt(x2, W['d2']['T2']['zero'], W['d2']['T2']['sigma'], 'o', C['C'], 'white')
pt(x3, W['d3']['T1']['zero'], W['d3']['T1']['sigma'], 's', C['F'], C['F']); pt(x3, W['d3']['T2']['zero'], W['d3']['T2']['sigma'], 's', C['C'], C['C'])
ti = W['TI']
for x, d in ((x2 + 0.006, '2d'), (x3 + 0.006, '3d')):
    for q, col in (('T1', C['F']), ('T2', C['C'])):
        ax.errorbar([x], [ti[f'{q}_{d}']['pred']], yerr=[ti[f'{q}_{d}']['sigma_int']], fmt='x', color=col, ms=5, mew=1.3, capsize=0)
ax.plot([], [], 'o', color='0.3', ms=5, mfc='white', label='wedge, 2D slab ($zG=1.1$)'); ax.plot([], [], 's', color='0.3', ms=5, label='wedge, 3D slab ($L=12$)'); ax.plot([], [], 'x', color='0.3', ms=5, mew=1.3, label='anharmonic (TI) prediction')
ax.text(1.003, 0.262, 'C', fontsize=11, color=C['C']); ax.text(1.085, 0.10, 'F', fontsize=11, color=C['F']); ax.annotate('M2 wedge', xy=(1.0985, 0.245), xytext=(1.052, 0.268), fontsize=8.5, color='#B8860B', arrowprops=dict(arrowstyle='-', lw=0.6, color='#B8860B'))
ax.set_xlim(1.0, 1.145); ax.set_ylim(0, 0.29)
ax.set_xlabel(r'$zG$ ($J=1$)'); ax.set_ylabel(r'$T$')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.17), ncol=2, fontsize=7.2, handletextpad=0.3, columnspacing=0.9)
fig.savefig('../fig/fig_H6.pdf')
