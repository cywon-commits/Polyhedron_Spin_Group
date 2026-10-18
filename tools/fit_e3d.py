"""Cut E 3D production verdicts (rule: docs/CD_rules.md 'Oct 18 ... C').  Usage: python tools/fit_e3d.py [--json out.json]"""
import argparse, json
import h5py, numpy as np
from polyspin.analysis.e2_rules import peak_sigma
ap = argparse.ArgumentParser(); ap.add_argument("--json", default=None); a = ap.parse_args()
out = {}
def load(L, zg, tag):
    f = h5py.File(f"runs/stage2/E_3d_L{L}_zG{zg}_{tag}.h5", "r"); return f["T"][:], f["series/m1"][:], f["series/m2"][:], f["series/e"][:]
print("## C point, zG = 0.5: collective order?")
rows = {}
for L in (8, 10):
    N = L ** 3
    for tag in ("ord", "ran"):
        T, m1, m2, e = load(L, "0.5", tag); c = e.var(1) * N / T ** 2
        i = int(np.argmax(c)); rows[(L, tag)] = dict(m2=float(m2[0].mean()), m2sqrtN=float(m2[0].mean() * np.sqrt(N)), cmax=float(c.max()), Tcmax=float(T[i]), m2_err=float(m2[0].std() / np.sqrt(len(m2[0]) / 20)))
        print(f"L={L} {tag}: T_low {T[0]:.3f}  m2 {rows[(L, tag)]['m2']:.4f}  m2*sqrt(N) {rows[(L, tag)]['m2sqrtN']:.3f}  c_max {c.max():.2f} at T={T[i]:.3f}")
r8, r10 = rows[(8, "ord")], rows[(10, "ord")]
ratio = r10["m2sqrtN"] / r8["m2sqrtN"]; cr = r10["cmax"] / r8["cmax"]
okm = 0.8 <= ratio <= 1.2; okc = cr <= 1.2
print(f"ordered start: m2*sqrt(N) ratio L10/L8 = {ratio:.3f} (within +-20%: {okm}); c_max ratio = {cr:.3f} (<= 1.2: {okc})")
print("VERDICT C point:", "no collective order (coupling 0 confirmed)" if okm and okc else "NOT confirmed (see values)")
out["C"] = dict(rows={f"L{k[0]}_{k[1]}": v for k, v in rows.items()}, ratio_m2=ratio, ratio_c=cr, ok=bool(okm and okc))
print("\n## D points: T1 (chi_1 peak, ordered start, jackknife)")
D = {}
for L in (8, 10):
    N = L ** 3
    for zg in ("0.8", "1"):
        for tag in ("ord", "ran"):
            T, m1, m2, e = load(L, zg, tag); t1, s1 = peak_sigma(T, m1, N)
            D[(L, zg, tag)] = (float(t1), float(s1)); print(f"L={L} zG={zg} {tag}: T1 = {t1:.4f} +- {s1:.4f}   m1(low T) {m1[0].mean():.3f}")
res = {}
for L in (8, 10):
    a8, b = D[(L, "0.8", "ord")], D[(L, "1", "ord")]; rr = a8[0] / b[0]; sr = rr * np.hypot(a8[1] / a8[0], b[1] / b[0])
    res[f"L{L}"] = (float(rr), float(sr)); print(f"T1(0.8)/T1(1.0), L={L}: {rr:.3f} +- {sr:.3f}  (predicted 0.32, no threshold)")
out["D"] = dict(T1={f"L{k[0]}_zG{k[1]}_{k[2]}": v for k, v in D.items()}, ratio=res)
print("\n## F point, zG = 1.2 (no rule)")
F = {}
for L in (8, 10):
    N = L ** 3
    for tag in ("ord", "ran"):
        try:
            T, m1, m2, e = load(L, "1.2", tag); c = e.var(1) * N / T ** 2; t1, s1 = peak_sigma(T, m1, N); i = int(np.argmax(c))
            F[f"L{L}_{tag}"] = dict(T1=float(t1), s1=float(s1), Tc=float(T[i]), cmax=float(c.max())); print(f"L={L} {tag}: chi1 peak {t1:.4f} +- {s1:.4f}; c max {c.max():.1f} at {T[i]:.3f}; m1 low T {m1[0].mean():.3f}")
        except OSError:
            print(f"L={L} {tag}: not finished")
out["F"] = F
if a.json: json.dump(out, open(a.json, "w"), indent=1)
