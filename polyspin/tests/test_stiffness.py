"""Stiffness (helicity modulus): analytic derivatives, F-phase and disordered limits, the fluctuation formula against reweighting, and the decision rules."""
import numpy as np
import pytest

from polyspin import moves, stiffness as st
from polyspin.analysis.stiffness_rules import kt_crossing, verdict_A, verdict_B
from polyspin.harmonic import collinear_state
from polyspin.lattice import Lattice
from polyspin.model import Params, random_spins
from polyspin.rng import make_state
from polyspin.tuning import tune_steps


def _chain(spins, lat, P, T, ne, nm, seed, every=5, extra=None):
    p, beta, s = P.array(), 1.0 / T, make_state(seed, 0)
    d, dr = tune_steps(spins.copy(), lat.nbr, p, beta, s, delta0=np.sqrt(T), n_rounds=15, block=20, use_overlap=True)
    c = np.zeros((moves.N_COUNTERS, 2), dtype=np.int64)
    for _ in range(ne):
        moves.mc_sweep(spins, lat.nbr, p, beta, d, dr, 20.0, True, True, True, False, s, c)
    S1, S2, ex = [], [], []
    for k in range(nm):
        moves.mc_sweep(spins, lat.nbr, p, beta, d, dr, 20.0, True, True, True, False, s, c)
        if k % every == 0:
            a, b = st.terms(spins, lat.nbr, p, 0)
            S1.append(a); S2.append(b)
            if extra is not None:
                ex.append(extra(spins))
    return np.array(S1), np.array(S2), np.array(ex)


@pytest.mark.parametrize("col", [0, 1])
def test_analytic_derivatives_match_finite_differences(col):
    lat = Lattice(2, 6)
    P = Params.from_z(1.0, 0.5, z=4, K=0.3, n=4, G1=0.1); p = P.array()
    rng = np.random.default_rng(1); sp = random_spins((lat.N, 4), rng)
    for ax in (np.array([0, 0, 1.0]), st.director(sp), rng.normal(size=3)):
        ax = ax / np.linalg.norm(ax)
        S1, S2 = st._bond_sums(sp, lat.nbr, p[3], p[4], p[5], ax, col)
        h = 1e-4
        E = lambda f: st.twisted_inter_energy(sp, lat.nbr, p, ax, f, col)
        assert S1 == pytest.approx((E(h) - E(-h)) / (2 * h), rel=1e-6, abs=1e-6)
        assert S2 == pytest.approx((E(h) - 2 * E(0.0) + E(-h)) / h ** 2, rel=1e-4, abs=1e-4)


def test_ferromagnet_zero_temperature_value_and_low_T_chain():
    """F state: Y = n^2 (3G + 4H4) for an axis perpendicular to the magnetization, 0 for the parallel axis and for the director; the chain at T = 0.005
    stays within 2% (harmonic limit)."""
    lat = Lattice(2, 8); N = lat.N
    P = Params.from_z(1.1, 1.5, z=4, K=0.5, n=4); Y0 = 16 * (3 * P.G + 4 * P.H4)
    sp0 = np.tile(collinear_state(4, 0), (N, 1, 1))
    S1, S2 = st.terms(sp0, lat.nbr, P.array(), 0)
    assert S1[0] == pytest.approx(0, abs=1e-12) and S2[0] / N == pytest.approx(Y0, rel=1e-12) and S2[2] == pytest.approx(0, abs=1e-12) and S2[3] == pytest.approx(0, abs=1e-12)
    T = 0.005
    S1, S2, _ = _chain(sp0.copy(), lat, P, T, 3000, 20000, 3)
    Y = [(S2[:, k].mean() - np.mean(S1[:, k] ** 2) / T) / N for k in range(4)]
    assert Y[0] == pytest.approx(Y0, rel=0.02) and Y[1] == pytest.approx(Y0, rel=0.02)
    assert abs(Y[2]) < 0.02 * Y0 and abs(Y[3]) < 0.02 * Y0


def test_disordered_high_temperature_stiffness_vanishes():
    lat = Lattice(2, 8); N = lat.N
    P = Params.from_z(1.1, 1.5, z=4, K=0.5, n=4); Y0 = 16 * (3 * P.G + 4 * P.H4)
    T = 20.0
    S1, S2, _ = _chain(random_spins((N, 4), np.random.default_rng(2)), lat, P, T, 2000, 20000, 5)
    Y = [(S2[:, k].mean() - np.mean(S1[:, k] ** 2) / T) / N for k in range(4)]
    assert max(abs(y) for y in Y) < 0.05 and max(abs(y) for y in Y) < 0.002 * Y0


def test_fluctuation_formula_matches_reweighted_free_energy():
    """F(phi) + F(-phi) - 2F(0) = Y_tot phi^2 (+ O(phi^4)); F from the untwisted ensemble by reweighting, Y_tot from the fluctuation formula."""
    lat = Lattice(2, 4); N = lat.N
    P = Params.from_z(1.1, 0.5, z=4, K=0.5, n=4); p = P.array()
    T, phi, ax = 0.5, 0.06, np.array([0.0, 0.0, 1.0]); beta = 1 / T
    spins = random_spins((N, 4), np.random.default_rng(3)); s = make_state(7, 0)
    d, dr = tune_steps(spins.copy(), lat.nbr, p, beta, s, delta0=np.sqrt(T), n_rounds=15, block=20, use_overlap=True)
    c = np.zeros((moves.N_COUNTERS, 2), dtype=np.int64)
    for _ in range(3000):
        moves.mc_sweep(spins, lat.nbr, p, beta, d, dr, 20.0, True, True, True, False, s, c)
    S1, S2, dp, dm = [], [], [], []
    for k in range(30000):
        for _ in range(2):
            moves.mc_sweep(spins, lat.nbr, p, beta, d, dr, 20.0, True, True, True, False, s, c)
        a, b = st._bond_sums(spins, lat.nbr, p[3], p[4], p[5], ax, 0); S1.append(a); S2.append(b)
        e0 = st.twisted_inter_energy(spins, lat.nbr, p, ax, 0.0, 0)
        dp.append(st.twisted_inter_energy(spins, lat.nbr, p, ax, phi, 0) - e0); dm.append(st.twisted_inter_energy(spins, lat.nbr, p, ax, -phi, 0) - e0)
    S1, S2, dp, dm = map(np.array, (S1, S2, dp, dm))
    fluct = (S2.mean() - beta * np.mean(S1 ** 2)) * phi ** 2
    rew = -T * np.log(np.mean(np.exp(-beta * dp))) - T * np.log(np.mean(np.exp(-beta * dm)))
    blocks = np.array_split(np.arange(len(S1)), 20)
    err = np.std([(S2[i].mean() - beta * np.mean(S1[i] ** 2)) for i in blocks]) / np.sqrt(20) * phi ** 2
    assert abs(fluct - rew) < 4 * err + 2e-4 and fluct > 0


def test_director_of_rotated_nematic():
    lat = Lattice(2, 4)
    n = np.array([1.0, 2.0, -1.0]); n /= np.linalg.norm(n)
    S0 = np.array([n, n, -n, -n])
    d = st.director(np.tile(S0, (lat.N, 1, 1)))
    assert abs(abs(d @ n) - 1) < 1e-12
    S1, S2 = st.terms(np.tile(S0, (lat.N, 1, 1)), lat.nbr, Params.from_z(1.0, 1.5, z=4, K=0.5, n=4).array(), 0)
    assert abs(S2[3]) < 1e-9                         # twist about the director of a collinear state costs nothing


# ------------------------------------------------------------------ decision rules (synthetic data)
def _mk(f, T, errs=0.01):
    return {L: f(L, T) for L in (16, 24, 32)}, {L: np.full(len(T), errs) for L in (16, 24, 32)}


def test_rule_A():
    T = np.array([0.13, 0.15, 0.17, 0.19, 0.21])
    # asymptotic freedom: Y decreases linearly in ln L at every T
    u, e = _mk(lambda L, T: 1.0 - 0.1 * np.log(L / 16) - 2 * (T - 0.13), T)
    assert verdict_A(u, e)[0] == "asymptotic-freedom crossover supported"
    # plateau: L independent at low T
    u2, e2 = _mk(lambda L, T: np.where(T < 0.18, 1.0 - 2 * (T - 0.13), 1.0 - 2 * (T - 0.13) - 0.12 * np.log(L / 16)), T)
    assert verdict_A(u2, e2)[0] == "quasi-long-range-order candidate"
    # non-monotone mess
    rng = np.random.default_rng(0)
    u3 = {L: 1.0 + rng.normal(0, 0.1, len(T)) for L in (16, 24, 32)}
    assert verdict_A(u3, {L: np.full(len(T), 0.01) for L in (16, 24, 32)})[0] == "undecidable"
    # fewer than 3 temperatures
    assert verdict_A({L: [1, 1] for L in (16, 24, 32)}, {L: [0.1, 0.1] for L in (16, 24, 32)})[0] == "undecidable"
    # decrease only at some temperatures -> not supported
    u4, e4 = _mk(lambda L, T: np.where(T < 0.17, 1.0, 1.0 - 0.1 * np.log(L / 16)) - 2 * (T - 0.13), T)
    assert verdict_A(u4, e4)[0] != "asymptotic-freedom crossover supported"


def test_rule_B():
    T = np.linspace(0.10, 0.22, 13)
    # KT-like: below T_KT the stiffness is L independent, above it drops with a steepness growing with L
    def kt(L, T, Tkt=0.16, A=0.35):
        w = 0.05 * 16 / L
        return A * (1 - np.tanh((T - Tkt) / w)) / 2
    u, e = _mk(kt, T, 0.003)
    v, det = verdict_B(T, u, e)
    assert v == "KT-type crossover consistent" and all(det["crossings"][L] for L in (16, 24, 32))
    # stiffness decreasing with L everywhere, no steepening
    u2, e2 = _mk(lambda L, T: 0.5 - 0.1 * np.log(L / 16) - 2 * (T - 0.1), T, 0.003)
    assert verdict_B(T, u2, e2)[0] == "no KT-type behavior"
    # noise
    rng = np.random.default_rng(1)
    u3 = {L: 0.3 + rng.normal(0, 0.05, len(T)) for L in (16, 24, 32)}
    assert verdict_B(T, u3, {L: np.full(len(T), 0.003) for L in (16, 24, 32)})[0] == "undecidable"
    # crossing helper
    c = kt_crossing(T, 2 * T / np.pi + (0.16 - T), np.full(len(T), 0.001))
    assert c is not None and c[0] == pytest.approx(0.16, abs=1e-9) and c[1] == pytest.approx(1.0, rel=1e-9)
