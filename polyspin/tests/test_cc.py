"""Clausius-Clapeyron ingredients: dH/dzG = E3/zG is exact (H linear in G); branch chains stay in their class."""
import numpy as np

from polyspin import slab
from polyspin.harmonic import collinear_state
from polyspin.lattice import Lattice
from polyspin.model import Params, energy_total


def test_cubic_term_is_derivative():
    lat = Lattice(2, 4)
    rng = np.random.default_rng(0)
    S = rng.normal(size=(lat.N, 4, 3)); S /= np.linalg.norm(S, axis=2, keepdims=True)
    P = Params.from_z(1.03, 1.5, z=4, K=0.5)
    d = 1e-4
    Pp, Pm = Params.from_z(1.03 + d, 1.5, z=4, K=0.5), Params.from_z(1.03 - d, 1.5, z=4, K=0.5)
    fd = (energy_total(S, lat.nbr, Pp.array()) - energy_total(S, lat.nbr, Pm.array())) / (2 * d) / lat.N
    assert abs(fd - slab.cubic_energy_per_cluster(S, lat.nbr, P) / P.zG) < 1e-7


def test_branch_chain_stays_and_reports():
    lat = Lattice(2, 6)
    P = Params.from_z(1.05, 1.5, z=4, K=0.5)
    rC = slab.run_branch(lat, P, 0.1, collinear_state(4, 2), 200, 400, seed=3)
    rF = slab.run_branch(lat, P, 0.1, collinear_state(4, 0), 200, 400, seed=4)
    assert rC["frac"][0] > 0.95 and rF["frac"][2] > 0.95            # C stays C (|M|~0), F stays F (|M|~n)
    assert rC["E"] > rF["E"] - 1.0 and np.isfinite(rC["dH"]) and np.isfinite(rF["dH_err"])


def test_cc_fit_recovers_a_and_b():
    from polyspin.analysis.cc_fit import crossing_samples, fit_a, verdict
    rng = np.random.default_rng(1)
    a0, b0 = 0.30, 0.20
    T = np.array([0.10, 0.125, 0.15, 0.175, 0.20])
    zst = 1 + a0 * T + b0 * T ** 2 + rng.normal(0, 0.003, len(T))
    s = a0 + 2 * b0 * T + rng.normal(0, 0.02, len(T))
    full = fit_a(T, zst, np.full(5, 0.003), s, np.full(5, 0.02))
    assert abs(full["a"] - a0) < 3 * full["sa"] and abs(full["b"] - b0) < 3 * full["sb"]
    only = fit_a(T, zst, np.full(5, 0.003))                               # slab only, b free: much worse a
    assert only["sa"] > 2 * full["sa"]
    assert verdict(0.38, 0.03).startswith("supports 'harmonic + labels'")
    assert verdict(0.20, 0.05).startswith("undetermined")
    assert verdict(0.5, 0.01).startswith("all predictions rejected")
    zs = np.arange(1.0, 1.101, 0.005); vs = 1e-5 * (zs - 1.05) * 40; ses = np.full_like(zs, 4e-6)
    smp, wp, nm = crossing_samples(zs, vs, ses)
    assert abs(np.median(smp) - 1.05) < 0.003 and nm >= 5


def test_layer_nematic_and_ordered_disordered_slab():
    lat = Lattice(3, 6)
    S0 = collinear_state(4, 2)
    sp = slab.make_ordered_disordered(lat, S0, axis=0, frac=0.5, seed=1)
    q = slab.layer_nematic(sp, lat, axis=0)
    assert q[:3].min() > 0.99 and q[3:].max() < 0.35             # aligned collinear clusters: 1; random layers: small
    P = Params.from_z(0.5, 1.0, z=6, K=0.5)
    ts, fr, lay, es, _ = slab.run_slab_order(lat, P, 0.2, S0, 100, record_every=50, seed=2)
    assert lay.shape == (2, 6) and fr[0] >= 0.4


def test_transient_velocity_resolves_fast_and_slow_fronts():
    ts = np.arange(5, 2005, 5.0)
    slow = np.clip(0.5 + 2e-5 * ts, 0, 1)                      # never saturates
    fast = np.clip(0.5 + 2e-3 * ts, 0, 1)                      # saturates after 250 sweeps, then plateau
    vs, _, ns, sats = slab.transient_velocity(ts, slow)
    vf, _, nf, satf = slab.transient_velocity(ts, fast)
    assert abs(vs - 2e-5) < 1e-6 and not sats and ns == len(ts)
    assert abs(vf - 2e-3) < 2e-4 and satf and nf < 80          # a plateau after saturation must not dilute the slope


def test_slab_tuning_does_not_move_the_interface():
    lat = Lattice(2, 8)
    P = Params.from_z(1.04, 1.5, z=4, K=0.5)
    ts, fr, e, _ = slab.run_slab(lat, P, 0.1, collinear_state(4, 2), collinear_state(4, 0), 5, record_every=5, seed=1, kind="diag")
    assert abs(fr[0, 2] - 0.5) < 0.1                           # first record (5 sweeps after the start) is still ~half F


def test_crossing_is_unbiased_for_asymmetric_velocity():
    from polyspin.analysis.cc_fit import crossing_samples
    zs = np.repeat(np.arange(1.0, 1.1001, 0.005), 3)
    v = np.where(zs < 1.04, 0.011 * (zs - 1.04), 0.022 * (zs - 1.04))        # slopes differ by 2 on the two sides
    ses = np.full_like(zs, 3e-6)
    smp, wp, nm = crossing_samples(zs, v, ses)
    assert abs(np.median(smp) - 1.04) < 0.001                             # a global linear fit would give ~1.0353
    zg = np.arange(1.0, 1.1001, 0.005); A = np.stack([zg, np.ones_like(zg)], 1)
    m, c = np.linalg.lstsq(A, np.where(zg < 1.04, 0.011 * (zg - 1.04), 0.022 * (zg - 1.04)), rcond=None)[0]
    assert abs(-c / m - 1.04) > 0.003


def test_crossing_robust_percentile_error_and_wedge_rule():
    from polyspin.analysis.cc_fit import crossing_robust, crossing_with_systematics, wedge_verdict
    # sparse sign change (like the M2|F slab): few runs near zero, the rest saturated -> rare two-point resamples inflate the std
    Ts = np.repeat(np.arange(0.20, 0.2801, 0.005), 3)
    v = np.where(Ts < 0.2325, 2.5e-4 * (0.2325 - Ts) / 0.0325 * 8, -6e-4 * (Ts - 0.2325) / 0.0475 * 8)
    rng = np.random.default_rng(3)
    v = v + rng.normal(0, 3e-5, len(v)); ses = np.full_like(Ts, 3e-5)
    r = crossing_robust(Ts, v, ses)
    assert abs(r["zero"] - 0.2325) < 0.004 and r["sigma_pct"] <= r["sigma_std"] + 1e-12
    assert r["sigma_total"] < 0.02
    w = wedge_verdict(0.2324, 0.0020, 0.2500, 0.0040, 0.243, 0.265)
    assert w["confirmed"] and w["location"]
    assert not wedge_verdict(0.2324, 0.01, 0.2500, 0.01, 0.243, 0.265)["confirmed"]      # difference below 2 sigma
    assert not wedge_verdict(0.26, 0.001, 0.25, 0.001, 0.243, 0.265)["confirmed"]        # T1 >= T2: no wedge
    assert not wedge_verdict(0.2324, 0.002, 0.2500, 0.004, 0.30, 0.265)["location"]      # T1 prediction off by more than 0.03
