"""Derive the numbers behind the paper figures from the raw data of the repository (runs/ is not versioned, so the derived tables are written to
paper/data/*.json and committed).  Usage: python paper/make_data.py [--only e2,cutD,cutE,H6,wedge,order]"""
import argparse, collections, glob, importlib.util, io, contextlib, json, re, sys
from pathlib import Path
import h5py, numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from polyspin.analysis import cc_fit as cc
from polyspin.analysis.e2_rules import peak_sigma
from polyspin.analysis.finite_size import loglog_slope
OUT = ROOT / "paper" / "data"; OUT.mkdir(exist_ok=True)
SCRATCH = "/tmp/claude-0/-home-user-Polyhedron-Spin-Group/cf6a3478-f312-5246-bd3b-9d3e93acc190/scratchpad/trial/outord"
ap = argparse.ArgumentParser(); ap.add_argument("--only", default="e2,cutD,cutE,H6,wedge,order"); a = ap.parse_args(); todo = set(a.only.split(","))
dump = lambda name, obj: (OUT / f"{name}.json").write_text(json.dumps(obj, indent=1))

def load_slab(f):
    r = json.loads(str(np.load(ROOT / "data/stage2" / f"{f}.npz")["results"]))
    return [x["T"] for x in r], [x["v"] for x in r]

# ---------------------------------------------------------------- E' (cut E'): T1 (chi_1 peak), T2 (chi_2 peak, lower of heat/cool)
if "e2" in todo:
    rows = []
    for p in sorted(glob.glob(str(ROOT / "runs/stage2/E2_*_heat.h5"))):
        dim, L, zg = re.match(r"E2_(\d)d_L(\d+)_zG([\d.]+)_heat", Path(p).name).groups(); dim, L, zg = int(dim), int(L), float(zg); N = L ** dim
        fh = h5py.File(p, "r"); T = fh["T"][:]
        t1, s1 = peak_sigma(T, fh["series/m1"][:], N); t2h, s2h = peak_sigma(T, fh["series/m2"][:], N); t2, s2, t2c = t2h, s2h, None
        try:
            fc = h5py.File(p.replace("_heat", "_cool"), "r"); Tc_ = fc["T"][:]; chi2c = N * fc["series/m2"][:].var(1) / Tc_
            if 0 < int(np.argmax(chi2c)) < len(Tc_) - 1:
                t2c, s2c = peak_sigma(Tc_, fc["series/m2"][:], N)
                if t2c < t2h: t2, s2 = t2c, s2c
        except OSError:
            pass
        rows.append(dict(dim=dim, L=L, zG=zg, T1=float(t1), s1=float(s1), T2=float(t2), s2=float(s2), T2_heat=float(t2h), T2_cool=None if t2c is None else float(t2c)))
    dump("e2_T1T2", rows); print("e2:", len(rows), "rows")

# ---------------------------------------------------------------- cut D: nematic melting (specific-heat peak above T = 0.4)
if "cutD" in todo:
    rows = []
    for zg in ("0.9", "1", "1.1"):
        for kind in ("heat", "cool"):
            p = ROOT / f"runs/stage2/D_2d_L16_zG{zg}_{kind}.h5"
            f = h5py.File(p, "r"); T = f["T"][:]; e = f["series/e"][:]; c = e.var(1) * 256 / T ** 2; m = T > 0.4
            rows.append(dict(zG=float(zg), L=16, kind=kind, T2=float(T[m][np.argmax(c[m])]), cmax=float(c[m].max())))
        p = ROOT / f"runs/trial_cd/D_2d_L8_zG{zg}_trial.h5"
        if p.exists():
            f = h5py.File(p, "r"); T = f["T"][:]; e = f["series/e"][:]; c = e.var(1) * 64 / T ** 2; m = T > 0.4
            rows.append(dict(zG=float(zg), L=8, kind="trial", T2=float(T[m][np.argmax(c[m])]), cmax=float(c[m].max())))
    dump("cutD_T2", rows); print("cutD:", [(r["zG"], r["L"], r["kind"], round(r["T2"], 3)) for r in rows])

# ---------------------------------------------------------------- cut E (preliminary M3 trials, 2D L = 8, ordered start; raw files are in the session scratch, not in the repository)
if "cutE" in todo:
    rows = []
    for zg in ("0.5", "0.8", "1", "1.2"):
        p = Path(SCRATCH) / f"E_2d_L8_zG{zg}_prelimord.h5"
        if not p.exists(): continue
        f = h5py.File(p, "r"); T = f["T"][:]; m1 = f["series/m1"][:]; m2 = f["series/m2"][:]
        mm1 = m1.mean(1); mm2 = m2.mean(1); chi1 = 64 * m1.var(1) / T
        half = float(np.interp(0.5 * mm1[0], mm1[::-1], T[::-1])) if mm1[0] > 0.05 else float("nan")
        rows.append(dict(zG=float(zg), m1_lowT=float(mm1[0]), m2_lowT=float(mm2[0]), T_chi1_peak=float(T[np.argmax(chi1)]), T_m1_half=half))
    dump("cutE_prelim", dict(note="preliminary M3 trial, 2D L = 8, ordered start; scratch data", rows=rows)); print("cutE:", rows)

# ---------------------------------------------------------------- H6 slab zero points (zG*(T)) with the primary estimator of docs/H6_result.md
if "H6" in todo:
    rows = []
    for tag, files in (("scan", [f"slab_H6_T{t}" for t in ("0.10", "0.125", "0.15", "0.175", "0.20")]), ("rescan", ["slab_H6_rescan_T0.175", "slab_H6_rescan_T0.20"])):
        for f in files:
            r = json.loads(str(np.load(ROOT / "data/stage2" / f"{f}.npz")["results"])); T = r[0]["T"]
            rr = sorted([x for x in r if x.get("kind", "diag") == "diag"], key=lambda x: x["zG"])
            zst, st, sb, sysm, wp, nm = cc.crossing_with_systematics([x["zG"] for x in rr], [x["v"] for x in rr], [x["se"] for x in rr])
            rows.append(dict(set=tag, T=float(T), zstar=float(zst), sigma=float(st), pinned_width=float(wp), n_moved=int(nm)))
    dump("H6_points", rows); print("H6:", [(r["set"], r["T"], round(r["zstar"], 4), round(r["sigma"], 4)) for r in rows])

# ---------------------------------------------------------------- wedge positions (final estimators) and thermodynamic-integration predictions
if "wedge" in todo:
    def est(*files):
        T, v = [], []
        for f in files:
            t, x = load_slab(f); T += t; v += x
        e = cc.crossing_seedmean(T, v); return dict(zero=e["zero"], sigma=e["sigma_total"])
    W = dict(
        d2=dict(T1=est("slab_M2F_zG1.1"), T2=est("slab_CM2_zG1.1")),
        d3=dict(T1=est("slab_M2F_3D_L12_supp"), T2=est("slab_CM2_3D_L12_remeasure"), T1_orig=est("slab_M2F_3D_L12"), T2_orig=est("slab_CM2_3D_L12")),
        n6=dict(Ta=est("slab_n6_M4F_zG1.1"), Tb=est("slab_n6_M2M4_zG1.1"), Tc=est("slab_n6_CM2_zG1.1"), Tc_supp=est("slab_n6_CM2_zG1.1_supp"), Tc_pooled=est("slab_n6_CM2_zG1.1", "slab_n6_CM2_zG1.1_supp")),
        harmonic=dict(d2=dict(T1=0.2431, T2=0.2646), d3=dict(T1=0.2440, T2=0.2665), n6=dict(Ta=0.3300, Tb=0.3674, Tc=0.3832)))
    spec = importlib.util.spec_from_file_location("anh", ROOT / "tools/anharm.py"); anh = importlib.util.module_from_spec(spec); spec.loader.exec_module(anh)
    with contextlib.redirect_stdout(io.StringIO()): anh.integrate()
    W["TI"] = {f"{q}_{dim}d": v for (q, dim), v in anh.integrate.results.items()}
    W["B3D"] = dict(L12=est("slab_B3D_zH41_L12_v2"), L16=est("slab_B3D_zH41_L16"), latent_heat=dict(value=2.05, err=0.03, note="L = 8, 10 window chains, docs/M4_stage2_followup_report.md"))
    dump("wedge", W); print("wedge:", json.dumps(W, indent=0)[:1500])

# ---------------------------------------------------------------- specific-heat exponents: c_peak(L) with block-jackknife errors
if "order" in todo:
    def cpeak(path, N, nb=20):
        f = h5py.File(path, "r"); T = f["T"][:]; e = f["series/e"][:]; tau = f["series/tau_int"][:, 0]
        c = e.var(1) * N / T ** 2; i = int(np.argmax(c)); x = e[i]; n = len(x); b = max(int(np.ceil(20 * tau[i])), 1); k = n // b
        if k < 5: k = 5; b = n // 5
        bl = [x[j * b:(j + 1) * b] for j in range(k)]
        est = np.array([np.concatenate([bl[m] for m in range(k) if m != j]).var() * N / T[i] ** 2 for j in range(k)])
        return float(c[i]), float(np.sqrt((k - 1) / k * np.sum((est - est.mean()) ** 2))), float(T[i])
    rng = np.random.default_rng(0); R = {}
    for dim, sizes in ((3, [8, 10, 12, 14]), (2, [16, 24, 32])):
        for z in ("0.5", "1", "1.5"):
            pts = []
            for L in sizes:
                p = ROOT / f"runs/stage1/A_{dim}d_L{L}_zG{z}_pt.h5"
                if p.exists():
                    c, err, Tp = cpeak(p, L ** dim); pts.append(dict(L=L, N=L ** dim, c=c, err=err, Tpeak=Tp))
            pairs = []
            for i in range(len(pts) - 1):
                p0, p1 = pts[i], pts[i + 1]
                s = [np.log(max(rng.normal(p1["c"], p1["err"]), 1e-3) / max(rng.normal(p0["c"], p0["err"]), 1e-3)) / np.log(p1["N"] / p0["N"]) for _ in range(4000)]
                pairs.append(dict(L0=p0["L"], L1=p1["L"], a=float(np.log(p1["c"] / p0["c"]) / np.log(p1["N"] / p0["N"])), lo=float(np.percentile(s, 16)), hi=float(np.percentile(s, 84))))
            R[f"A_{dim}d_zG{z}"] = dict(points=pts, pairs=pairs)
    pts = []
    for L in (16, 24):
        p = ROOT / f"runs/stage1/B_2d_L{L}_zH42_heat.h5"
        c, err, Tp = cpeak(p, L * L); pts.append(dict(L=L, N=L * L, c=c, err=err, Tpeak=Tp))
    p0, p1 = pts; s = [np.log(max(rng.normal(p1["c"], p1["err"]), 1e-3) / max(rng.normal(p0["c"], p0["err"]), 1e-3)) / np.log(p1["N"] / p0["N"]) for _ in range(4000)]
    R["B_2d_zH42"] = dict(points=pts, pairs=[dict(L0=16, L1=24, a=float(np.log(p1["c"] / p0["c"]) / np.log(p1["N"] / p0["N"])), lo=float(np.percentile(s, 16)), hi=float(np.percentile(s, 84)))])
    dump("order", R)
    for k, v in R.items(): print(k, [(q["L0"], q["L1"], round(q["a"], 2), round(q["lo"], 2), round(q["hi"], 2)) for q in v["pairs"]])
