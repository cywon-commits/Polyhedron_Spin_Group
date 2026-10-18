"""Channel selection (fig. 6): T1, T2 per size and dimension from paper/data/e2_T1T2.json, cut D T2 and cut E (preliminary) from the derived tables."""
from style import *; import numpy as np, json
D = json.load(open('../data/e2_T1T2.json')); CD = json.load(open('../data/cutD_T2.json')); CE = json.load(open('../data/cutE_prelim.json'))['rows']
fig, axs = plt.subplots(1, 2, figsize=(DCOL, 2.8))
ax = axs[0]; zc = (1 + 0.5 + 1.0) / 3
style = {(2, 16): ('o', 'none', -0.006), (2, 24): ('o', None, -0.002), (3, 8): ('s', 'none', 0.002), (3, 10): ('s', None, 0.006), (3, 12): ('D', None, 0.010)}
for r in D:
    mk, fc, dx = style[(r['dim'], r['L'])]
    kw = dict(fmt=mk, ms=5, capsize=1.5, elinewidth=0.8)
    c1, c2 = C['D'], C['C']
    ax.errorbar(r['zG'] + dx, r['T1'], yerr=r['s1'], color=c1, mfc=('white' if fc == 'none' else c1), **kw)
    if not (r['dim'] == 3 and r['L'] == 12):                       # the L = 12 ladder (0.8-1.2 T1) does not reach T2
        ax.errorbar(r['zG'] + dx, r['T2'], yerr=r['s2'], color=c2, mfc=('white' if fc == 'none' else c2), **kw)
g = np.linspace(zc, 1.05, 100)
for dim, ls, norm in ((2, '--', 0.1788), (3, ':', 0.1904)):
    ax.plot(g, norm * g * (g - zc) / (1.0 * (1.0 - zc)), ls, color=C['D'], lw=1.0)
ax.axvline(zc, color='0.5', ls=':', lw=0.8); ax.text(zc + 0.003, 0.05, 'C–D\n($T=0$)', fontsize=8, color='0.35')
ax.text(0.845, 0.262, r'$T_2$ (nematic)', color=C['C'], fontsize=8.5); ax.text(0.845, 0.135, r'$T_1$ (D order)', color=C['D'], fontsize=8.5)
ax.text(1.048, 0.012, r'dashed (2D), dotted (3D):' '\n' r'$T_1\propto G\,M^2$, norm. at $zG=1$', fontsize=7.2, color='0.3', ha='right')
ax.plot([], [], 'o', color='0.3', ms=5, mfc='white', label='2D $L=16$'); ax.plot([], [], 'o', color='0.3', ms=5, label='2D $L=24$')
ax.plot([], [], 's', color='0.3', ms=5, mfc='white', label='3D $L=8$'); ax.plot([], [], 's', color='0.3', ms=5, label='3D $L=10$'); ax.plot([], [], 'D', color='0.3', ms=4.5, label='3D $L=12$ ($T_1$ only)')
ax.legend(loc='upper left', ncol=2, fontsize=7.2, columnspacing=0.8, handletextpad=0.2, bbox_to_anchor=(0.0, 1.02))
ax.set_xlim(0.82, 1.05); ax.set_ylim(0, 0.42); ax.set_xlabel(r'$zG$'); ax.set_ylabel(r'$T$')
ax.set_title(r"(a) cut E$^\prime$ ($K=-0.25$, $zH_4=0.25$)", loc='left', fontsize=9.5)
ax = axs[1]; zc = (1 + 1.0) / 3
e = [r for r in CE if r['zG'] in (0.8, 1.0)]
ax.plot([r['zG'] for r in e], [r['T_chi1_peak'] for r in e], 's', color=C['D'], ms=6, mfc='white', label=r'cut E: $T_1$ of D order (prelim., 2D $L=8$)')
g = np.linspace(zc, 1.05, 100); n1 = [r['T_chi1_peak'] for r in e if r['zG'] == 1.0][0]
ax.plot(g, n1 * g * (g - zc) / (1.0 * (1.0 - zc)), '--', color=C['D'], label=r'$\propto G\,M^2$')
for kind, mk, fc in (('heat', '^', C['C']), ('cool', 'v', C['C']), ('trial', 'o', 'white')):
    rr = [r for r in CD if r['kind'] == kind]
    ax.plot([r['zG'] for r in rr], [r['T2'] for r in rr], mk, color=C['C'], ms=6, mfc=fc,
            label={'heat': r'cut D: $T_2$, heating, 2D $L=16$', 'cool': r'cut D: $T_2$, cooling, 2D $L=16$', 'trial': r'cut D: $T_2$, 2D $L=8$ trial'}[kind])
ax.text(0.665, 0.17, 'cut E: C bodies decouple\n(odd channels only;\n$A_{\\rm odd}=0$ in C)', fontsize=8, color='0.25', va='bottom')
ax.set_xlim(0.65, 1.15); ax.set_ylim(0, 1.3); ax.set_xlabel(r'$zG$'); ax.set_ylabel(r'$T$')
ax.set_title(r'(b) cuts E ($zH_4=0$) and D ($zH_4=1.5$)', loc='left', fontsize=9.5)
ax.legend(loc='upper left', bbox_to_anchor=(0.0, 1.02), fontsize=7.2, handletextpad=0.3)
fig.tight_layout(w_pad=1.6); fig.savefig('../fig/fig_channel.pdf')
