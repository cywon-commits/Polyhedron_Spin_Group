"""H1: c_peak(N) for 3D A points, exponent a (c ~ N^a) with a block-jackknife error on every c_peak, propagated
by Monte Carlo (c_i ~ Normal(c_i, err_i), refit); V_min(L); corrected bimodality is in tools/bimodality_table.py.
Usage: python tools/h1_scaling.py [--dim 3] [--sizes 8 10 12]"""
import argparse
import h5py
import numpy as np
from polyspin.analysis.finite_size import loglog_slope

ap = argparse.ArgumentParser(); ap.add_argument("--dim", type=int, default=3)
ap.add_argument("--sizes", type=int, nargs="+", default=[8, 10, 12]); ap.add_argument("--cuts", default="0.5 1 1.5")
a = ap.parse_args()
rng = np.random.default_rng(0)

def cpeak_jk(e, T, N, tau, nb=20):
    """c = N var(e)/T^2 at the temperature of the maximum; delete-one-block jackknife error."""
    c = e.var(1) * N / T ** 2; i = int(np.argmax(c))
    x = e[i]; n = len(x); b = max(int(np.ceil(20 * tau[i])), 1); k = n // b
    if k < 5:
        k = 5; b = n // 5
    blocks = [x[j * b:(j + 1) * b] for j in range(k)]
    est = []
    for j in range(k):
        y = np.concatenate([blocks[m] for m in range(k) if m != j]); est.append(y.var() * N / T[i] ** 2)
    est = np.array(est)
    return float(c[i]), float(np.sqrt((k - 1) / k * np.sum((est - est.mean()) ** 2))), float(T[i]), k

print("| zG | L | T_peak | c_peak | jackknife error | blocks | round trips | min swap |")
print("|---|---|---|---|---|---|---|---|")
for z in a.cuts.split():
    Ns, cs, es = [], [], []
    for L in a.sizes:
        f = h5py.File(f"runs/stage1/A_{a.dim}d_L{L}_zG{z}_pt.h5", "r"); T = f["T"][:]; e = f["series/e"][:]
        tau = f["series/tau_int"][:, 0]; N = L ** a.dim
        c, err, Tp, k = cpeak_jk(e, T, N, tau)
        print(f"| {z} | {L} | {Tp:.4f} | {c:.1f} | {err:.1f} | {k} | {int(f['pt'].attrs['round_trips'])} | {f['pt/swap_rate'][:].min():.2f} |")
        Ns.append(N); cs.append(c); es.append(err)
    Ns, cs, es = map(np.array, (Ns, cs, es))
    a0, _ = loglog_slope(Ns.astype(float), cs)
    sl = [loglog_slope(Ns.astype(float), np.maximum(rng.normal(cs, es), 1e-3))[0] for _ in range(4000)]
    lo, hi = np.percentile(sl, [16, 84])
    print(f"\n**zG = {z}: a = {a0:.2f}, 68% interval from c_peak errors [{lo:.2f}, {hi:.2f}] (all L = {a.sizes})**; "
          f"pairwise: " + ", ".join(f"L{a.sizes[i]}->L{a.sizes[i+1]}: {np.log(cs[i+1]/cs[i])/np.log(Ns[i+1]/Ns[i]):.2f}" for i in range(len(Ns) - 1)) + "\n")
