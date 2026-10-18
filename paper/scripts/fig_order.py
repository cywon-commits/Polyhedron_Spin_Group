"""Effective specific-heat exponent per pair of sizes with 16-84% errors (paper/data/order.json) and the size dependence of T_c in the first-order 3D nematic case."""
from style import *; import numpy as np, json
R = json.load(open('../data/order.json')); W = json.load(open('../data/wedge.json'))
fig, axs = plt.subplots(1, 2, figsize=(DCOL, 2.9), gridspec_kw=dict(width_ratios=[2.0, 1]))
ax = axs[0]
ax.axhspan(-0.1, 0.06, color='#4C72B0', alpha=0.12, lw=0); ax.text(0.0, -0.045, r'standard 3D continuous ($\alpha/d\nu\lesssim0.06$)', fontsize=7.5, color='#2c4a7a')
ax.axhline(1.0, color='#C44E52', ls='--', lw=0.9); ax.text(0.0, 1.03, r'first order ($C_{\max}\propto N$)', fontsize=7.5, color='#C44E52')
items = []   # (label, a, lo, hi, marker, color, filled)
for z in ('0.5', '1', '1.5'):
    for q in R[f'A_3d_zG{z}']['pairs']: items.append((f"{z}\n{q['L0']}→{q['L1']}", q['a'], q['lo'], q['hi'], 'o', '#4C72B0', True, '3D'))
for z in ('0.5', '1', '1.5'):
    for q in R[f'A_2d_zG{z}']['pairs']: items.append((f"{z}\n{q['L0']}→{q['L1']}", q['a'], q['lo'], q['hi'], 's', '#4C72B0', False, '2D'))
q = R['B_2d_zH42']['pairs'][0]; items.append(('2.0\n16→24', q['a'], q['lo'], q['hi'], '^', '#C44E52', True, 'l2'))
xs = np.arange(len(items)); gaps = {0: 0, 7: 0.6, 14: 0.6}
xp = []; x0 = 0
for i, it in enumerate(items):
    if i == 7 or i == 11: x0 += 0.8
    xp.append(i + x0)
for xv, it in zip(xp, items):
    lab, a, lo, hi, mk, col, filled, kind = it
    ax.errorbar([xv], [a], yerr=[[a - lo], [hi - a]], fmt=mk, color=col, ms=5.5, mfc=(col if filled else 'white'), capsize=2, elinewidth=0.9)
ax.set_xticks(xp); ax.set_xticklabels([it[0] for it in items], fontsize=6.3)
ax.text(np.mean(xp[:7]), 1.2, r'3D, $l=3$ (cut A)', fontsize=8, ha='center', color='#2c4a7a'); ax.text(np.mean(xp[7:11]), 1.2, r'2D, $l=3$ (cut A)', fontsize=8, ha='center', color='#2c4a7a'); ax.text(xp[-1], 1.2, r'$l=2$', fontsize=8, ha='center', color='#C44E52')
ax.set_xlim(xp[0] - 0.8, xp[-1] + 0.8); ax.set_ylim(-0.4, 1.4)
ax.set_ylabel(r'$a=d\ln C_{\max}/d\ln N$'); ax.set_xlabel(r'$zG$ (cut A) or $zH_4$ ($l=2$), and pair of sizes $L_0\!\to\!L_1$')
ax.set_title('(a) pairwise exponents', loc='left', fontsize=9.5)
ax = axs[1]; B = W['B3D']
ax.errorbar([12, 16], [B['L12']['zero'], B['L16']['zero']], yerr=[B['L12']['sigma'], B['L16']['sigma']], fmt='o', color='#C44E52', ms=6, capsize=2)
ax.set_xlim(10, 18); ax.set_xticks([12, 16]); ax.set_xlabel(r'slab size $L$'); ax.set_ylabel(r'$T_c$ (3D, $zH_4=1$)')
ax.text(0.5, 0.12, f"latent heat {B['latent_heat']['value']:.2f}({int(round(B['latent_heat']['err'] * 100))})\nper body", transform=ax.transAxes, ha='center', fontsize=8)
ax.set_title(r'(b) $l=2$, 3D: slab coexistence', loc='left', fontsize=9.5)
fig.tight_layout(w_pad=1.2); fig.savefig('../fig/fig_order.pdf')
