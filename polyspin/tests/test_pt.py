"""Parallel tempering: exact distributions at every temperature, reproducibility, checkpoints."""
import numpy as np
import numba
import pytest
from scipy.stats import kstest

from polyspin.lattice import Lattice
from polyspin.model import Params, energy_total
from polyspin.pt import ParallelTempering, hysteresis_scan


class SingleCluster:
    """One isolated cluster (no neighbours) -- enough for the PT engine."""
    N, dim, z = 1, 2, 0
    nbr = np.zeros((1, 0), dtype=np.int64)


def test_exchange_detailed_balance_two_spins():
    """Two spins with J coupling: x = S1.S2 must follow exp(-beta J x) on [-1,1] at EVERY
    temperature of the ladder while replicas swap temperatures."""
    temps = np.array([0.3, 0.5, 0.8, 1.3])
    P = Params(J=1.0, K=0.0, z=0, n=2)
    pt = ParallelTempering(SingleCluster(), P, temps, seed=5)
    xs = [[] for _ in temps]
    for _ in range(40000):
        pt.sweep_block(2)
        pt.exchange()
        for t in range(len(temps)):
            S = pt.spins_at_T(t)[0]
            xs[t].append(S[0] @ S[1])
    assert pt.swap_rates().min() > 0.2
    for t, T in enumerate(temps):
        b = 1.0 / T
        cdf = lambda x: (np.exp(b) - np.exp(-b * x)) / (np.exp(b) - np.exp(-b))
        x = np.array(xs[t])[::3]                      # thin: swaps and moves correlate slowly
        assert kstest(x, cdf).statistic < 0.02, (T, kstest(x, cdf).statistic)
    assert pt.round_trips > 20


def _small():
    lat = Lattice(2, 4)
    P = Params.from_z(0.5, 0.0, z=4, K=0.5)
    return lat, P, np.geomspace(0.05, 1.0, 6)


def test_energy_bookkeeping():
    lat, P, temps = _small()
    pt = ParallelTempering(lat, P, temps, seed=1)
    for _ in range(20):
        pt.sweep_block(3)
        pt.exchange()
    assert np.allclose(pt.energies, pt.full_energy_check(), atol=1e-9)
    assert np.allclose(np.linalg.norm(pt.spins, axis=-1), 1.0, atol=1e-12)
    assert sorted(pt.rep_at) == list(range(len(temps)))
    assert np.array_equal(pt.temp_of[pt.rep_at], np.arange(len(temps)))


def _run(seed, nthreads, nblocks=30):
    lat, P, temps = _small()
    numba.set_num_threads(nthreads)
    pt = ParallelTempering(lat, P, temps, seed=seed)
    for _ in range(nblocks):
        pt.sweep_block(2)
        pt.exchange()
    return pt


def test_reproducible_for_any_thread_count():
    a = _run(7, 1)
    b = _run(7, 4)
    c = _run(8, 4)
    numba.set_num_threads(numba.config.NUMBA_NUM_THREADS)
    assert np.array_equal(a.spins, b.spins) and np.array_equal(a.rep_at, b.rep_at)
    assert np.array_equal(a.swap_acc, b.swap_acc)
    assert not np.array_equal(a.spins, c.spins)


def test_checkpoint_roundtrip():
    lat, P, temps = _small()
    full = ParallelTempering(lat, P, temps, seed=3)
    for _ in range(20):
        full.sweep_block(2)
        full.exchange()
    half = ParallelTempering(lat, P, temps, seed=3)
    for _ in range(10):
        half.sweep_block(2)
        half.exchange()
    sd = {k: np.array(v) for k, v in half.state_dict().items()}
    resumed = ParallelTempering(lat, P, temps, seed=99)         # wrong seed on purpose
    resumed.load_state_dict(sd)
    for _ in range(10):
        resumed.sweep_block(2)
        resumed.exchange()
    assert np.array_equal(full.spins, resumed.spins)
    assert np.array_equal(full.rep_at, resumed.rep_at)
    assert full.round_trips == resumed.round_trips


def test_step_tuning_reaches_target():
    lat = Lattice(2, 6)
    P = Params.from_z(0.5, 0.0, z=4, K=0.5)
    pt = ParallelTempering(lat, P, np.geomspace(0.02, 1.0, 8), seed=2)
    for _ in range(40):
        pt.reset_acceptance()
        pt.sweep_block(10)
        pt.exchange()
        pt.tune()
    pt.reset_acceptance()
    for _ in range(10):
        pt.sweep_block(10)
        pt.exchange()
    acc = pt.acceptance()
    # a step size pinned at its cap (nearly free spins at high T) may exceed the band
    capped_s = pt.delta >= 2.0 - 1e-9
    capped_r = pt.delta_r >= 1.0 - 1e-9
    assert (((acc[:, 0] > 0.25) & (acc[:, 0] < 0.55)) | (capped_s & (acc[:, 0] > 0.25))).all(), acc[:, 0]
    assert (((acc[:, 1] > 0.25) & (acc[:, 1] < 0.55)) | (capped_r & (acc[:, 1] > 0.25))).all(), acc[:, 1]


def test_hysteresis_scan_runs_and_reuses_configuration():
    lat = Lattice(2, 4)
    from polyspin.groundstate import phase_of_gs
    mk = lambda zg: Params.from_z(zg, 0.0, z=4, K=0.5)
    S0 = phase_of_gs(mk(0.5))[2]
    start = np.tile(S0, (lat.N, 1, 1))
    res = hysteresis_scan(lat, mk, [0.5, 0.6, 0.7], T=0.05, start_spins=start, n_equil=50, n_meas=50, seed=1)
    assert [r["value"] for r in res] == [0.5, 0.6, 0.7]
    assert all(np.isfinite(r["e"]) for r in res)
    assert res[-1]["final"].shape == start.shape
