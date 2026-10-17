"""3D cut C U points (zG = 1.0, 1.2; L = 8, 10): existing H5 rule (cd_rules.h5_intervals) on heat ladder + cold chains merged by temperature.
Usage: python tools/c3d_analysis.py"""
import os, h5py, numpy as np
from polyspin.analysis.cd_rules import h5_intervals, h5_verdict

def series(path):
    f = h5py.File(path, "r"); T = f["T"][:]
    m1 = f["series/m1"][:].mean(1)
    sp = f["diag/split"][:] / np.maximum(f["diag/n_diag"][:][:, None], 1)
    return T, m1, sp[:, 1]
res = {}
for L in (8, 10):
    for zg in (1.0, 1.2):
        base = f"runs/stage2/C_3d_L{L}_zG{zg:g}"
        if not (os.path.exists(base + "_heat.h5") and os.path.exists(base + "_cold.h5")):
            print(f"L={L} zG={zg:g}: not finished"); continue
        parts = [series(base + s + ".h5") for s in ("_heat", "_cold")]
        T, m1, s = (np.concatenate([p[i] for p in parts]) for i in range(3))
        o = np.argsort(T); T, m1, s = T[o], m1[o], s[o]
        iv = h5_intervals(T, s, m1); res[(L, zg)] = iv
        print(f"L={L} zG={zg:g}: {'; '.join(f'{i[0]:.3f}-{i[1]:.3f} ({i[2]} pts)' for i in iv) or 'none'} | m1 {m1.min():.2f}-{m1.max():.2f} | {{211}} at lowest T {s[0]:.2f}")
        print("   heat m1(T):", " ".join(f"{t:.3f}:{x:.2f}/{y:.2f}" for t, x, y in zip(parts[0][0][::5], parts[0][1][::5], parts[0][2][::5])))
ok = all(res.get(k) for k in [(8, 1.0), (8, 1.2), (10, 1.0), (10, 1.2)])
print("H5 (3D):", "supported" if ok and len(res) == 4 else "not (yet) established", "(needs all four points with an interval)")
