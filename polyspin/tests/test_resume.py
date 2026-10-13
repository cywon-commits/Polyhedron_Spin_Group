"""Checkpoint resume is bit-exact: interrupted + resumed runs equal uninterrupted ones in every output
dataset (PT with replica exchange, single-temperature chains, 2D and 3D, production diagnostics including
the coarse-grained detector), also after several interruptions and after a SIGKILL."""
import json
import os
import signal
import subprocess
import sys
import time

import h5py
import numpy as np
import pytest
import yaml

from polyspin import run as runner

SKIP_ATTRS = {"wall_seconds", "sweeps_per_second", "git_dirty", "resumed_at_blocks"}


def cfg_for(kind):
    base = dict(name="r", model=dict(zG=1.0, zH4=0.0, K=0.5), seed=7, init="ordered",
                sampling=dict(n_equil=80, n_meas=120, sweeps_per_exchange=1, meas_every=5, diag_every=16,
                              tune_every=20, checkpoint_minutes=999),
                diagnostics=dict(defects=dict(stride=3, coarse_blocks=[2, 4])))
    if kind == "pt2d":
        base.update(lattice=dict(dim=2, size=8), temperatures=dict(Tmin=0.1, Tmax=0.4, n=7, spacing="linear"))
    elif kind == "blk2d":
        base["diagnostics"] = dict(corr_blocks=4, defects=dict(stride=3, coarse_blocks=[2]))
        base.update(lattice=dict(dim=2, size=8), temperatures=dict(Tmin=0.1, Tmax=0.4, n=5, spacing="linear"))
    elif kind == "cold2d":
        base.update(lattice=dict(dim=2, size=8), temperatures=dict(list=[0.02, 0.05, 0.1, 0.2]), exchange=False)
    elif kind == "pt3d":
        base["diagnostics"] = dict(defects=dict(stride=3, coarse_blocks=[2]))
        base.update(lattice=dict(dim=3, size=4), temperatures=dict(Tmin=0.1, Tmax=0.4, n=5, spacing="linear"))
    return runner.load_config(base)


def dump(path):
    out = {}
    with h5py.File(path, "r") as f:
        for k, v in f.attrs.items():
            if k not in SKIP_ATTRS:
                out["@" + k] = v
        f.visititems(lambda n, o: out.__setitem__(n, o[()]) if isinstance(o, h5py.Dataset) else None)
        for g in ("diag", "pt", "series"):
            for k, v in f[g].attrs.items():
                out[f"{g}@{k}"] = v
    return out


def same(a, b):
    da, db = dump(a), dump(b)
    assert da.keys() == db.keys()
    for k in da:
        assert np.array_equal(np.asarray(da[k]), np.asarray(db[k])), k


@pytest.mark.parametrize("kind,stops", [("pt2d", [(35,), (80,), (140,), (20, 95, 150)]),
                                        ("cold2d", [(60,), (80,), (130,)]), ("pt3d", [(50,), (170,)]), ("blk2d", [(70,), (150,)])])
def test_resume_bit_exact(tmp_path, kind, stops):
    ref = tmp_path / "ref.h5"
    runner.run(cfg_for(kind), ref, verbose=False)
    for i, stop in enumerate(stops):
        out = tmp_path / f"r{i}.h5"
        for s in stop:
            assert runner.run(cfg_for(kind), out, verbose=False, stop_after_blocks=s) is None
        runner.run(cfg_for(kind), out, verbose=False)
        same(ref, out)
        with h5py.File(out) as f:
            assert json.loads(f.attrs["resumed_at_blocks"]) == list(stop)
        with h5py.File(ref) as f:
            assert json.loads(f.attrs["resumed_at_blocks"]) == []
        assert not (tmp_path / f"r{i}.h5.ckpt.npz").exists()


@pytest.mark.slow
def test_resume_after_sigkill(tmp_path):
    """Kill -9 the process after it has written at least one checkpoint (checkpoint_minutes = 0), resume."""
    cfg = cfg_for("pt2d")
    cfg["sampling"].update(n_equil=3000, n_meas=3000, checkpoint_minutes=0.01)
    ref, out = tmp_path / "ref.h5", tmp_path / "out.h5"
    cfgfile = tmp_path / "c.yaml"
    cfgfile.write_text(yaml.safe_dump(cfg))
    code = "import sys, yaml; from polyspin import run; run.run(yaml.safe_load(open(sys.argv[1])), sys.argv[2], verbose=False)"
    p = subprocess.Popen([sys.executable, "-c", code, str(cfgfile), str(out)])
    ck = tmp_path / "out.h5.ckpt.npz"
    t0 = time.time()
    while not ck.exists() and time.time() - t0 < 60:
        time.sleep(0.05)
    time.sleep(1.0)
    p.send_signal(signal.SIGKILL)
    p.wait()
    assert ck.exists() and not out.exists(), "run finished before the kill; lengthen it"
    runner.run(cfg, out, verbose=False)
    runner.run(cfg, ref.with_name("ref2.h5"), verbose=False)
    same(ref.with_name("ref2.h5"), out)


def test_block_correlators_sum_to_total(tmp_path):
    out = tmp_path / "b.h5"
    runner.run(cfg_for("blk2d"), out, verbose=False)
    with h5py.File(out) as f:
        cb, nb, c, n = f["diag/corr_blk"][:], f["diag/corr_blk_n"][:], f["diag/corr"][:], f["diag/n_diag"][:]
        assert cb.shape[1] == 4 and np.allclose(cb.sum(1), c) and np.array_equal(nb.sum(1), n)
        assert (nb > 0).all()                                   # every block holds samples
