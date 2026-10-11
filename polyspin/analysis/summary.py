"""Load a run file and compute the per-temperature summary used by the figures and reports."""
import json

import h5py
import numpy as np

from .stats import binder, block_error, jackknife, tau_int


def load_run(path):
    with h5py.File(path, "r") as f:
        r = dict(attrs=dict(f.attrs), T=f["T"][:], e=f["series/e"][:],
                 m=np.stack([f[f"series/m{k}"][:] for k in range(1, 5)], axis=1),   # (R,4,n)
                 tau=f["series/tau_int"][:], swap=f["pt/swap_rate"][:],
                 swap_equil=f["pt/swap_rate_equil"][:], round_trips=int(f["pt"].attrs["round_trips"]),
                 up_fraction=f["pt/up_fraction"][:], acceptance=f["pt/acceptance"][:],
                 final_spins=f["final_spins"][:], gs_spins=f["gs_spins"][:],
                 diag={k: f["diag"][k][:] for k in f["diag"]},
                 diag_attrs={k: (json.loads(v) if isinstance(v, str) else v) for k, v in f["diag"].attrs.items()},
                 meas_every=int(f["series"].attrs["meas_every_sweeps"]))
    r["N"] = int(r["attrs"]["N"])
    r["beta"] = 1.0 / r["T"]
    return r


def thermo(r, nblocks=20):
    """Per-temperature <e>, c, <m_l>, chi_l, Binder U_l with block/jackknife errors and tau."""
    R, N = len(r["T"]), r["N"]
    out = {k: np.zeros(R) for k in ("e", "e_err", "c", "c_err")}
    for k in ("m", "m_err", "chi", "binder", "binder_err"):
        out[k] = np.zeros((R, 4))
    for t in range(R):
        b = r["beta"][t]
        e = r["e"][t]
        out["e"][t], out["e_err"][t] = block_error(e, nblocks)
        out["c"][t], out["c_err"][t] = jackknife(lambda x: N * b * b * np.var(x), e, nblocks)
        for l in range(4):
            m = r["m"][t, l]
            out["m"][t, l], out["m_err"][t, l] = block_error(m, nblocks)
            out["chi"][t, l] = N * b * (np.mean(m ** 2) - np.mean(m) ** 2)
            out["binder"][t, l], out["binder_err"][t, l] = jackknife(binder, m, nblocks)
    out["tau"] = r["tau"] * r["meas_every"]            # integrated autocorrelation time in sweeps
    return out


def diag_means(r):
    """Normalise the accumulated diagnostics by the number of samples per temperature."""
    d, n = r["diag"], r["diag"]["n_diag"][:, None]
    n_all = n
    n = d["n_def"][:, None] if "n_def" in d else n          # defect diagnostics: samples actually taken
    n = np.where(n == 0, np.nan, n)
    n_all = n_all
    out = dict(split=d["split"] / n_all, A=d["A_sum"] / n_all, lam=d["lam_sum"] / n_all[:, 0],
               M_hist=d["M_hist"] / n_all, corr=d["corr"][:, :, :] / n_all[:, :, None])
    names = r["diag_attrs"].get("class_names")
    if names:
        npl = r["attrs"]["N"] * (1 if r["attrs"]["dim"] == 2 else 3)
        out["class_names"] = names
        out["plaq_density"] = d["h_counts"] / n / r["N"]             # per site, reliable plaquettes
        out["plaq_amb_frac"] = d["h_amb"] / n[:, 0] / npl
        out["plaq_und_frac"] = d["h_und"] / n[:, 0] / npl
        out["link_amb_frac"] = d["h_linkamb"] / n[:, 0]
        out["fit_resid"] = d["h_resid"] / n[:, 0]
        if "h_scan" in d and d["h_scan"].shape[1] > 3:            # (res, overlap, margin) combos
            out["scan_per_plaq"] = d["h_scan"] / n[:, :, None] / npl
        if "h_resid_hist" in d:
            out["resid_hist"] = d["h_resid_hist"] / n
        if "c_counts" in d:
            nco = (r["attrs"]["size"] // 2) ** r["attrs"]["dim"] * (1 if r["attrs"]["dim"] == 2 else 3)
            out["coarse_density"] = d["c_counts"] / n / r["N"]           # per fine site
            out["coarse_und_frac"] = d["c_und"] / n[:, 0] / nco
            out["coarse_amb_frac"] = d["c_amb"] / n[:, 0] / nco
            out["coarse_quality"] = d["c_quality"] / n[:, 0]
        for k in r["diag_attrs"]["block_k"]:
            norm = (2 * k - 1) ** 2 * r["N"]
            out[f"block{k}_density"] = d[f"b{k}_counts"] / n / norm
            out[f"block{k}_nontrivial_fraction"] = d[f"b{k}_frac"] / n[:, 0]
            out[f"block{k}_amb_frac"] = d[f"b{k}_amb"] / n[:, 0] / npl
            out[f"block{k}_und_frac"] = d[f"b{k}_und"] / n[:, 0] / npl
    if "z2_count" in d:
        out["z2_density"] = d["z2_count"] / n[:, 0] / r["N"]
        out["z2_und_frac"] = d["z2_und"] / n[:, 0] / (r["N"] * (1 if r["attrs"]["dim"] == 2 else 3))
    if "sky_Q" in d:
        out["sky_absQ"] = d["sky_Q"][:, 2] / n[:, 0]
        out["sky_density"] = d["sky_abs_density"] / n[:, 0]
    return out


def peak_temperature(T, y):
    i = int(np.argmax(y))
    return float(T[i]), i
