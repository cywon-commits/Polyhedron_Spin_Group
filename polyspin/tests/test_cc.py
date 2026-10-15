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
