"""H2 (2D A zG=1.0): L dependence of the octupole order parameter m3 and the form of the G_3(r) decay
(power law vs exponential, both with periodic images) at fixed T/T_c for L = 16, 24, 32.
No per-sample errors are stored for G_3(r) (only sums), so fits are unweighted in log space; the RSS ratio is
descriptive, not a statistical test.  Usage: python tools/m3_decay.py"""
import numpy as np
from scipy.optimize import least_squares

Ls = (16, 24, 32)
D = {L: np.load(f"data/stage1/A_2d_L{L}_zG1_pt.npz", allow_pickle=False) for L in Ls}
Tc = {L: D[L]["T"][int(np.argmax(D[L]["th_c"]))] for L in Ls}

def at(L, key, tt, col=None):
    T = D[L]["T"]; y = D[L][key] if col is None else D[L][key][:, col]
    return float(np.interp(tt * Tc[L], T, y))

def fit(L, G, kind):
    r = np.arange(1, L // 2 + 1)[: len(G) - 1] if False else np.arange(1, len(G))
    g = G[1:]
    ok = g > 0
    r, g = r[ok], g[ok]
    def model(p):
        A, s = np.exp(p[0]), np.exp(p[1])
        if kind == "pow":
            return A * (r ** -s + (L - r) ** -s)
        return A * (np.exp(-r / s) + np.exp(-(L - r) / s))
    best = None
    for s0 in ((0.1, 0.5, 1.0) if kind == "pow" else (1.0, 3.0, 10.0)):
        res = least_squares(lambda p: np.log(model(p)) - np.log(g), [0.0, np.log(s0)])
        if best is None or res.cost < best.cost:
            best = res
    return float(np.exp(best.x[1])), float(2 * best.cost / len(r)), len(r)

print("| T/T_c | m3 L=16 | m3 L=24 | m3 L=32 | local exponent -d ln m3 / d ln L (24->32) |")
print("|---|---|---|---|---|")
for t in (0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.3):
    m = [at(L, "th_m", t, 2) for L in Ls]
    print(f"| {t} | {m[0]:.4f} | {m[1]:.4f} | {m[2]:.4f} | {-np.log(m[2]/m[1])/np.log(32/24):.3f} |")
print("\n| T/T_c | L | exponent eta (power law) | MSE log power | xi (exponential) | MSE log exp | better |")
print("|---|---|---|---|---|---|---|")
for t in (0.7, 0.8, 0.9, 1.0, 1.1):
    for L in Ls:
        T = D[L]["T"]; i = int(np.argmin(abs(T - t * Tc[L])))
        G = D[L]["dm_corr"][i, 2]
        eta, mp, n = fit(L, G, "pow"); xi, me, _ = fit(L, G, "exp")
        print(f"| {t} (T={T[i]:.4f}) | {L} | {eta:.3f} | {mp:.2e} | {xi:.2f} | {me:.2e} | {'power' if mp < me else 'exp'} |")
