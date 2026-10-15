"""H12 supplement (docs/E2_rules.md): Jacobian-corrected m1 distribution P~(m) = P(|m|)/m^2 at T1 (chi_1 peak) of each
heating ladder, per size; first-order iff every size is two-peak and the valley deepens with L.  Also prints the
temperature neighbours (T1 +- one ladder step) as a robustness check.  Usage: python tools/e2_h12_m.py"""
import glob, re
import h5py, numpy as np
from polyspin.analysis.e2_rules import peak_location, chi_curve, m_peak_structure, m_first_order

runs = {}
for p in sorted(glob.glob("runs/stage2/E2_*_heat.h5")):
    dim, L, zg = re.match(r"E2_(\d)d_L(\d+)_zG([\d.]+)_heat", p.split("/")[-1]).groups()
    runs.setdefault((int(dim), float(zg)), {})[int(L)] = p
print("| dim | zG | L | T1 | T (ladder) | tau_m1 | two-peak | valley ratio | n peaks (>=25%) | m* | (T-1 step: two-peak, ratio) | (T+1 step: two-peak, ratio) |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|")
for (dim, zg), sizes in sorted(runs.items()):
    per = {}
    for L, p in sorted(sizes.items()):
        f = h5py.File(p, "r"); T = f["T"][:]; m = f["series/m1"][:]; N = L ** dim
        t1 = peak_location(T, chi_curve(m, N, T)); i = int(np.argmin(abs(T - t1))); tau = f["series/tau_int"][i, 1]
        st = m_peak_structure(m[i]); per[L] = st
        nb = [m_peak_structure(m[j]) if 0 <= j < len(T) else None for j in (i - 1, i + 1)]
        fmt = lambda s: "n/a" if s is None else f"{'yes' if s['two_peak'] else 'no'}, {s['ratio']:.2f}"
        print(f"| {dim}D | {zg:g} | {L} | {t1:.4f} | {T[i]:.4f} | {tau:.1f} | {'yes' if st['two_peak'] else 'no'} | {st['ratio']:.2f} | {st['n_peaks']} | {st['m_star']:.3f} | {fmt(nb[0])} | {fmt(nb[1])} |")
    print(f"\n**{dim}D zG = {zg:g}: m1-distribution rule -> {'FIRST ORDER' if m_first_order(per) else 'no first-order condition'}**\n")
