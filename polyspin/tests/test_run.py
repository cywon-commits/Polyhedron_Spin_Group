"""End-to-end: config -> PT run -> HDF5; reproducibility, resume, detector selection."""
import json

import h5py
import numpy as np
import pytest

from polyspin import run as runner
from polyspin.analysis.stats import tau_int


def tiny(**over):
    cfg = dict(name="tiny", lattice=dict(dim=2, size=4), model=dict(zG=0.5, zH4=0.0, K=0.5),
               temperatures=dict(Tmin=0.05, Tmax=1.0, n=6, spacing="log"),
               sampling=dict(n_equil=60, n_meas=120, sweeps_per_exchange=2, meas_every=2, diag_every=20,
                             tune_every=20, checkpoint_minutes=999), seed=3)
    cfg.update(over)
    return runner.load_config(cfg)


def test_run_writes_hdf5(tmp_path):
    out = tmp_path / "a.h5"
    runner.run(tiny(), out, verbose=False)
    with h5py.File(out, "r") as f:
        assert f.attrs["seed"] == 3 and len(f.attrs["git_hash"]) >= 7
        assert "config_yaml" in f.attrs and f.attrs["gs_phase"] == "T"
        assert f["T"].shape == (6,)
        assert f["series/e"].shape == (6, 60)
        assert f["series/m3"].shape == (6, 60)
        assert f["final_spins"].shape == (6, 16, 4, 3)
        assert f["diag/n_diag"][0] == 6                # 120 sweeps / 20
        assert f["pt/swap_rate"].shape == (5,)
        # T phase auto-selects the holonomy detector with the 2T classes
        assert sorted(json.loads(f["diag"].attrs["class_names"])) == sorted(["1", "2", "4", "3+", "3-", "6+", "6-"])
        assert "h_counts" in f["diag"] and "b2_counts" in f["diag"] and "z2_count" not in f["diag"]
        # coldest temperature: ordered tetrahedra -> m3 large, m1 ~ 0
        assert f["series/m3"][0].mean() > 0.3 and f["series/m1"][0].mean() < 0.2
        e = f["series/e"][:]
        assert np.all(e[0] < e[-1])                    # energy rises with temperature


def test_same_seed_reproduces_and_resume_matches(tmp_path):
    a, b, c = tmp_path / "a.h5", tmp_path / "b.h5", tmp_path / "c.h5"
    runner.run(tiny(), a, verbose=False)
    runner.run(tiny(), b, verbose=False)
    assert runner.run(tiny(), c, verbose=False, stop_after_blocks=45) is None   # interrupted
    assert (tmp_path / "c.h5.ckpt.npz").exists()
    runner.run(tiny(), c, verbose=False)                                        # resumed
    with h5py.File(a) as fa, h5py.File(b) as fb, h5py.File(c) as fc:
        for key in ("series/e", "series/m2", "final_spins", "diag/split", "diag/h_counts", "diag/corr"):
            assert np.array_equal(fa[key][:], fb[key][:]), key
            assert np.array_equal(fa[key][:], fc[key][:]), key
    assert not (tmp_path / "c.h5.ckpt.npz").exists()


@pytest.mark.parametrize("zG,zH,K,expect", [(0.5, 1.0, 0.5, ("z2",)), (2.0, 0.0, 0.5, ("sky",)),
                                            (0.9, 0.0, -0.5, ("hol", "D"))])
def test_detector_selection(tmp_path, zG, zH, K, expect):
    cfg = tiny(model=dict(zG=zG, zH4=zH, K=K))
    cfg["sampling"].update(n_equil=20, n_meas=40, diag_every=20)
    out = tmp_path / "x.h5"
    runner.run(cfg, out, verbose=False)
    with h5py.File(out, "r") as f:
        keys = set(f["diag"].keys())
        if expect[0] == "z2":
            assert "z2_count" in keys and "h_counts" not in keys
        if expect[0] == "sky":
            assert "sky_Q" in keys and "h_counts" not in keys
        if expect[0] == "hol":
            names = json.loads(f["diag"].attrs["class_names"])
            assert sorted(names) == sorted(["1", "2", "4+", "4-"])
            assert f.attrs["gs_phase"] == "D"


def test_tau_int_on_series(tmp_path):
    out = tmp_path / "t.h5"
    runner.run(tiny(), out, verbose=False)
    with h5py.File(out, "r") as f:
        tau = f["series/tau_int"][:]
    assert tau.shape == (6, 5) and np.all(tau >= 0.5)


def test_hysteresis_mode(tmp_path):
    cfg = tiny()
    cfg.pop("temperatures")
    cfg["temperatures"] = dict(Tmin=0.05, Tmax=0.1, n=2)
    cfg["hysteresis"] = dict(param="zG", T=0.05, start=0.5, stop=0.8, step=0.1, roundtrip=True,
                             n_equil=30, n_meas=30)
    cfg["mode"] = "hysteresis"
    out = tmp_path / "h.h5"
    runner.run_hysteresis(cfg, out, verbose=False)
    with h5py.File(out, "r") as f:
        v = f["values"][:]
        assert np.allclose(v, [0.5, 0.6, 0.7, 0.8, 0.7, 0.6, 0.5])
        assert f["m3"].shape == (7,) and f["final_spins"].shape == (7, 16, 4, 3)
        assert f.attrs["n_up"] == 4 and len(f.attrs["git_hash"]) >= 7
