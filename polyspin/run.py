"""Command line driver: one YAML configuration = one parallel-tempering run = one HDF5 file.

    python -m polyspin.run configs/A_2d_L16_zG0.5.yaml [--out out.h5] [--no-resume]

Configuration (all keys optional except lattice/model/temperatures):
    name: A_2d_L16_zG0.5
    lattice:   {dim: 2, size: 16}
    model:     {J: 1.0, K: 0.5, L: 0.0, zG: 0.5, zH4: 0.0, G1: 0.0}      # zG = z*G, zH4 = z*H4
    temperatures: {Tmin: 0.02, Tmax: 1.2, n: 48, spacing: log}             # or list: [..]
    sampling:  {n_equil: 20000, n_meas: 20000, sweeps_per_exchange: 1, meas_every: 1,
                diag_every: 100, tune_every: 200, checkpoint_minutes: 20}
    init: random | ordered
    exchange: true            # false: independent single-temperature chains (cold region, no swaps)
    seed: 1
    overlap_move: true
    diagnostics: {shape_thr: 0.95, correlators: true,
                  defects: {holonomy: auto|tetra|gs|none, z2: auto|true|false, skyrmion: auto|true|false,
                            margin_thr: 0.15, block_k: [2, 3], res_thr: 0.35, match_thr: 0.9}}
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import yaml

from . import measure
from .analysis.stats import tau_int
from .defects import FrameGroup
from .groundstate import phase_of_gs
from .lattice import Lattice
from .model import Params, tetrahedron
from .pt import ParallelTempering

DEFAULTS = dict(
    sampling=dict(n_equil=20000, n_meas=20000, sweeps_per_exchange=1, meas_every=5, diag_every=500,
                  tune_every=200, checkpoint_minutes=20),
    init="random", seed=1, overlap_move=True, kappa=20.0,
    diagnostics=dict(shape_thr=0.95, correlators=True,
                     defects=dict(holonomy="auto", z2="auto", skyrmion="auto", margin_thr=0.15,
                                  block_k=[2, 3], res_thr=0.35, match_thr=0.9, stride=1)),
)


# ------------------------------------------------------------------ config
def _merge(a, b):
    out = dict(a)
    for k, v in b.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_config(path_or_dict):
    if isinstance(path_or_dict, dict):
        raw = path_or_dict
    else:
        raw = yaml.safe_load(Path(path_or_dict).read_text())
    cfg = _merge(DEFAULTS, raw)
    for key in ("lattice", "model", "temperatures"):
        if key not in cfg:
            raise ValueError(f"config needs '{key}'")
    return cfg


def make_temperatures(spec):
    if isinstance(spec, (list, tuple)):
        return np.asarray(spec, float)
    if "list" in spec:
        return np.asarray(spec["list"], float)
    n, lo, hi = int(spec["n"]), float(spec["Tmin"]), float(spec["Tmax"])
    if spec.get("spacing", "log") == "log":
        return np.geomspace(lo, hi, n)
    return np.linspace(lo, hi, n)


def make_params(cfg, lat):
    m = cfg["model"]
    return Params.from_z(zG=m.get("zG", 0.0), zH4=m.get("zH4", 0.0), z=lat.z, J=m.get("J", 1.0),
                         K=m.get("K", 0.5), L=m.get("L", 0.0), G1=m.get("G1", 0.0))


def git_info():
    here = Path(__file__).resolve().parent
    try:
        h = subprocess.run(["git", "rev-parse", "HEAD"], cwd=here, capture_output=True, text=True,
                           check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=here, capture_output=True,
                                    text=True, check=True).stdout.strip())
    except Exception:
        h, dirty = "unknown", False
    return h, dirty


def resolve_detectors(cfg, params, lat):
    """Pick the defect detectors from the T=0 ground-state phase unless the config says otherwise."""
    d = dict(cfg["diagnostics"]["defects"])
    gs_phase, _, gs_spins = phase_of_gs(params)
    hol, z2, sky = d.get("holonomy", "auto"), d.get("z2", "auto"), d.get("skyrmion", "auto")
    ref = None
    if hol == "auto":
        hol = "gs" if gs_phase in ("T", "U", "D") else "none"
    if hol == "tetra":
        ref = tetrahedron()
    elif hol == "gs":
        ref = gs_spins
    if ref is not None:
        try:
            FrameGroup(ref)
        except ValueError:
            ref = None
    d["holonomy_ref"] = None if ref is None else np.asarray(ref).tolist()
    d["z2"] = (gs_phase == "C") if z2 == "auto" else bool(z2)
    d["skyrmion"] = (gs_phase == "F") if sky == "auto" else bool(sky)
    return d, gs_phase, gs_spins


# ------------------------------------------------------------------ run
def run(cfg, out, resume=True, verbose=True, stop_after_blocks=None):
    cfg = load_config(cfg) if not isinstance(cfg, dict) or "sampling" not in cfg else cfg
    import h5py

    lat = Lattice(cfg["lattice"]["dim"], cfg["lattice"]["size"])
    P = make_params(cfg, lat)
    temps = make_temperatures(cfg["temperatures"])
    R = len(temps)
    S = cfg["sampling"]
    spe, every = int(S["sweeps_per_exchange"]), int(S["meas_every"])
    n_eq_blocks = int(S["n_equil"]) // spe
    n_meas_blocks = int(S["n_meas"]) // spe
    meas_stride = max(every // spe, 1)
    n_samples = n_meas_blocks // meas_stride
    diag_stride = max(int(S["diag_every"]) // spe, 1)
    tune_stride = max(int(S["tune_every"]) // spe, 1)

    dcfg, gs_phase, gs_spins = resolve_detectors(cfg, P, lat)
    diag_cfg = dict(cfg["diagnostics"], defects=dcfg)
    cfg_key = hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:16]

    pt = ParallelTempering(lat, P, temps, seed=cfg["seed"], init=cfg["init"], ref_state=gs_spins,
                           kappa=cfg["kappa"], use_overlap=cfg["overlap_move"])
    diag = measure.Diagnostics(lat, P, R, diag_cfg)
    e_series = np.zeros((R, n_samples))
    m_series = np.zeros((R, 4, n_samples))
    block = 0                       # global block counter (equil + measurement)
    n_taken = 0
    swap_equil = np.zeros(R - 1)
    wall_prev = 0.0
    resumes = []                    # blocks at which this run was resumed from a checkpoint

    ckpt = Path(str(out) + ".ckpt.npz")
    if resume and ckpt.exists():
        d = np.load(ckpt, allow_pickle=False)
        if str(d["cfg_key"]) == cfg_key:
            pt.load_state_dict({k[3:]: d[k] for k in d.files if k.startswith("pt_")})
            diag.load_arrays(d)
            e_series[...] = d["e_series"]
            m_series[...] = d["m_series"]
            block, n_taken = int(d["block"]), int(d["n_taken"])
            swap_equil = d["swap_equil"]
            wall_prev = float(d["wall"])
            resumes = [int(b) for b in d["resumes"]] + [block] if "resumes" in d.files else [block]
            print(f"[{cfg.get('name', 'run')}] resumed from checkpoint at block {block}", flush=True)   # also when quiet

    def save_ckpt():
        arrs = {f"pt_{k}": v for k, v in pt.state_dict().items()}
        arrs.update({f"diag_{k}": v for k, v in diag.arrays().items()})
        tmp_ck = Path(str(ckpt) + ".tmp.npz")          # atomic: a kill during the write must not leave a torn checkpoint
        np.savez(tmp_ck, cfg_key=cfg_key, e_series=e_series, m_series=m_series, block=block,
                 n_taken=n_taken, swap_equil=swap_equil, resumes=np.array(resumes, int), wall=wall_prev + time.time() - t0, **arrs)
        os.replace(tmp_ck, ckpt)

    t0 = time.time()
    last_ckpt = t0
    total_blocks = n_eq_blocks + n_meas_blocks
    while block < total_blocks:
        equil = block < n_eq_blocks
        pt.sweep_block(spe)
        if cfg.get("exchange", True):
            pt.exchange()
        block += 1
        if equil:
            if block % tune_stride == 0:
                pt.tune()
            if block == n_eq_blocks:
                swap_equil = pt.swap_rates().copy()
                pt.reset_swap_stats()
                pt.reset_acceptance()
        else:
            mb = block - n_eq_blocks
            if mb % meas_stride == 0 and n_taken < n_samples:
                m = measure.measure_m(pt.spins)
                e_series[:, n_taken] = pt.energies[pt.rep_at] / lat.N
                m_series[:, :, n_taken] = m[pt.rep_at]       # (R_T, 4) by temperature index
                n_taken += 1
            if mb % diag_stride == 0:
                for t in range(R):
                    diag.add(t, pt.spins_at_T(t))
        if verbose and block % max(total_blocks // 20, 1) == 0:
            el = time.time() - t0
            print(f"[{cfg.get('name', 'run')}] block {block}/{total_blocks} "
                  f"({'equil' if equil else 'meas'}) {el:.0f}s, min swap {pt.swap_rates().min():.2f}, "
                  f"round trips {pt.round_trips}", flush=True)
        if time.time() - last_ckpt > 60 * S["checkpoint_minutes"]:
            save_ckpt()
            last_ckpt = time.time()
        if stop_after_blocks is not None and block >= stop_after_blocks and block < total_blocks:
            save_ckpt()
            return None

    wall = wall_prev + time.time() - t0
    # ---------------------------------------------------------------- output
    commit, dirty = git_info()
    series_names = ["e", "m1", "m2", "m3", "m4"]
    all_series = np.concatenate([e_series[:, None, :], m_series], axis=1)       # (R, 5, n)
    taus = np.array([[tau_int(all_series[t, k])[0] for k in range(5)] for t in range(R)])
    tmp = Path(str(out) + ".tmp")
    with h5py.File(tmp, "w") as f:
        f.attrs.update(name=cfg.get("name", ""), seed=cfg["seed"], git_hash=commit, git_dirty=dirty,
                       config_yaml=yaml.safe_dump(cfg), params_json=json.dumps(P.__dict__),
                       zG=P.zG, zH4=P.zH4, dim=lat.dim, size=lat.size, N=lat.N, z=lat.z, n=P.n,
                       gs_phase=gs_phase, wall_seconds=wall, resumed_at_blocks=json.dumps(resumes), sweeps=pt.sweeps_done,
                       sweeps_per_second=pt.sweeps_done / max(wall, 1e-9),
                       numpy=np.__version__, python=sys.version.split()[0])
        f.create_dataset("T", data=temps)
        f.create_dataset("gs_spins", data=gs_spins)
        f.create_dataset("final_spins", data=pt.spins[pt.rep_at], compression="gzip")
        s = f.create_group("series")
        s.create_dataset("e", data=e_series, compression="gzip")
        for k in range(4):
            s.create_dataset(f"m{k + 1}", data=m_series[:, k, :], compression="gzip")
        s.attrs["meas_every_sweeps"] = meas_stride * spe
        s.create_dataset("tau_int", data=taus)
        s["tau_int"].attrs["columns"] = "e m1 m2 m3 m4 (in samples)"
        pg = f.create_group("pt")
        pg.create_dataset("swap_rate", data=pt.swap_rates())
        pg.create_dataset("swap_rate_equil", data=swap_equil)
        pg.attrs["round_trips"] = pt.round_trips
        pg.create_dataset("up_fraction", data=pt.up_fraction_sum / np.maximum(pt.up_fraction_n, 1))
        pg.create_dataset("delta", data=pt.delta)
        pg.create_dataset("delta_r", data=pt.delta_r)
        pg.create_dataset("acceptance", data=pt.acceptance())
        dg = f.create_group("diag")
        for k, v in diag.arrays().items():
            dg.create_dataset(k, data=v, compression="gzip" if v.size > 1000 else None)
        if diag.group is not None:
            dg.attrs["class_names"] = json.dumps(diag.class_names)
            dg.attrs["holonomy_ref"] = json.dumps(dcfg["holonomy_ref"])
            dg.attrs["block_k"] = json.dumps(diag.block_k)
            dg.attrs["margin_scan_thr"] = json.dumps(measure.BLOCK_SCAN_THR)
        dg.attrs["split_names"] = json.dumps(["1111", "211", "22", "31", "4"])
        dg.attrs["pair_edges"] = json.dumps(diag.pair_edges.tolist())
    os.replace(tmp, out)
    if ckpt.exists():
        ckpt.unlink()
    if verbose:
        print(f"wrote {out}  ({wall:.0f}s, {pt.sweeps_done / wall:.1f} sweeps/s)", flush=True)
    return out


# ------------------------------------------------------------------ hysteresis mode
def run_hysteresis(cfg, out, verbose=True):
    """Quasi-static parameter sweep at fixed T (cut lines A' and the B round trip).

    hysteresis: {param: zG|zH4, T: 0.02, start: 0.0, stop: 2.5, step: 0.05, roundtrip: true,
                 n_equil: 2000, n_meas: 2000, start_state: gs|random}
    Each step starts from the final configuration of the previous one."""
    import h5py

    from .measure import measure_m
    from .pt import hysteresis_scan

    cfg = load_config(cfg) if not isinstance(cfg, dict) or "sampling" not in cfg else cfg
    lat = Lattice(cfg["lattice"]["dim"], cfg["lattice"]["size"])
    hc = cfg["hysteresis"]
    base = dict(cfg["model"])
    par = hc["param"]
    up = np.arange(hc["start"], hc["stop"] + 1e-9, hc["step"])
    values = np.concatenate([up, up[-2::-1]]) if hc.get("roundtrip", True) else up
    n_up = len(up)

    def mk(v):
        m = dict(base)
        m[par] = float(v)
        return make_params(dict(model=m), lat)

    P0 = mk(values[0])
    _, _, S0 = phase_of_gs(P0)
    if hc.get("start_state", "gs") == "gs":
        start = np.tile(S0, (lat.N, 1, 1))
    else:
        from .model import random_spins
        start = random_spins((lat.N, P0.n), np.random.default_rng(cfg["seed"]))

    def observe(spins, P):
        m = measure_m(spins[None])[0]
        return dict(m1=m[0], m2=m[1], m3=m[2], m4=m[3])

    t0 = time.time()
    res = hysteresis_scan(lat, mk, values, hc["T"], start, hc["n_equil"], hc["n_meas"], seed=cfg["seed"],
                          kappa=cfg["kappa"], observe=observe)
    commit, dirty = git_info()
    with h5py.File(out, "w") as f:
        f.attrs.update(name=cfg.get("name", ""), mode="hysteresis", seed=cfg["seed"], git_hash=commit,
                       git_dirty=dirty, config_yaml=yaml.safe_dump(cfg), param=par, T=hc["T"], n_up=n_up,
                       dim=lat.dim, size=lat.size, N=lat.N, wall_seconds=time.time() - t0)
        f.create_dataset("values", data=values)
        for k in ("e", "e_std", "m1", "m2", "m3", "m4"):
            f.create_dataset(k, data=np.array([r[k] for r in res]))
        f.create_dataset("final_spins", data=np.array([r["final"] for r in res]), compression="gzip")
    if verbose:
        print(f"wrote {out}", flush=True)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    out = a.out or (Path(a.config).with_suffix(".h5").name)
    if cfg.get("mode") == "hysteresis":
        run_hysteresis(cfg, out, verbose=not a.quiet)
    else:
        run(cfg, out, resume=not a.no_resume, verbose=not a.quiet)


if __name__ == "__main__":
    main()
