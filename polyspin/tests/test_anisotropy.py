"""Single-spin uniaxial anisotropy V = sum_i [a2 P2(z.S_i) + a4 P4(z.S_i)] (Paper II, task N3).

  * defaults: a2 = a4 = 0 reproduces the previous Hamiltonian exactly (6-entry p accepted)
  * local energy differences of all three moves = full recomputation (1e-11)
  * Monte Carlo samples the single-spin Boltzmann weight exactly (decoupled spins)
  * 2-design: a rigid tetrahedron has zero quadrupolar (a2) energy in every orientation
  * Gamma formula of the derivation note: fixed-orientation relaxation reproduces
    E(n) = c - Gamma sum n_alpha^4 to 1 % at the production a4, and the deviation is next order
  * sign rule: Gamma > 0 -> S4 axis along z, Gamma < 0 -> C3 axis along z
"""
import numpy as np
import pytest
from scipy.integrate import quad

from polyspin.groundstate import (find_ground_state, gamma_formula, minimize_cluster_at_orientation,
                                  oriented_tetrahedron)
from polyspin.lattice import Lattice
from polyspin.model import (Params, cluster_inter_energy, energy_total, random_spins, spin_energy,
                            tetrahedron)
from polyspin.moves import mc_sweep
from polyspin.rng import make_state
from polyspin.run import make_params

RNG = np.random.default_rng(11)
P_N3 = dict(K=2.0, zG=0.08, zH4=0.80)


def _p(**kw):
    base = dict(J=1.0, K=0.5, L=0.1, G1=0.05)
    base.update(kw)
    zG = base.pop("zG", 0.7); zH4 = base.pop("zH4", 0.4)
    return Params.from_z(zG=zG, zH4=zH4, z=4, **base)


def test_defaults_reproduce_previous_hamiltonian():
    lat = Lattice(2, 4)
    S = random_spins((lat.N, 4), RNG)
    P = _p()
    p8 = P.array(); p6 = p8[:6].copy()
    assert len(p8) == 8 and p8[6] == 0 and p8[7] == 0
    assert energy_total(S, lat.nbr, p8) == energy_total(S, lat.nbr, p6)
    assert spin_energy(S, lat.nbr, p8, 3, 1, *S[3, 1]) == spin_energy(S, lat.nbr, p6, 3, 1, *S[3, 1])


@pytest.mark.parametrize("a2,a4", [(0.3, 0.0), (0.0, -0.4), (-0.2, 0.25)])
def test_local_energy_differences_match_full_recomputation(a2, a4):
    lat = Lattice(2, 4)
    P = _p(a2=a2, a4=a4); p = P.array()
    S = random_spins((lat.N, 4), RNG)
    E0 = energy_total(S, lat.nbr, p)
    for _ in range(40):                                   # single-spin / overlap moves
        a, i = RNG.integers(lat.N), RNG.integers(4)
        v = RNG.normal(size=3); v /= np.linalg.norm(v)
        dE = spin_energy(S, lat.nbr, p, a, i, *v) - spin_energy(S, lat.nbr, p, a, i, *S[a, i])
        S2 = S.copy(); S2[a, i] = v
        assert abs(energy_total(S2, lat.nbr, p) - E0 - dE) < 1e-11
    for _ in range(40):                                   # rigid rotations
        a = RNG.integers(lat.N)
        Q, _r = np.linalg.qr(RNG.normal(size=(3, 3)))
        Q *= np.sign(np.linalg.det(Q))
        new = S[a] @ Q.T
        dE = cluster_inter_energy(S, lat.nbr, p, a, new) - cluster_inter_energy(S, lat.nbr, p, a, S[a])
        S2 = S.copy(); S2[a] = new
        assert abs(energy_total(S2, lat.nbr, p) - E0 - dE) < 1e-11


def test_monte_carlo_samples_single_spin_weight():
    """J = K = L = 0 and no inter-cluster coupling: every spin is independent with weight
    exp(-beta V(cos theta)).  All three moves on; compare <P2>, <P4> with the exact integrals."""
    a2, a4, T = 0.8, -0.6, 0.5
    P = Params(J=0.0, K=0.0, L=0.0, G=0.0, H4=0.0, G1=0.0, a2=a2, a4=a4); p = P.array()
    lat = Lattice(2, 6)
    S = random_spins((lat.N, 4), RNG); st = make_state(5, 1); cnt = np.zeros((3, 2), np.int64)
    V = lambda c: a2 * 0.5 * (3 * c * c - 1) + a4 * (35 * c ** 4 - 30 * c * c + 3) / 8
    Z = quad(lambda c: np.exp(-V(c) / T), -1, 1)[0]
    exact = [quad(lambda c: f(c) * np.exp(-V(c) / T), -1, 1)[0] / Z
             for f in (lambda c: 0.5 * (3 * c * c - 1), lambda c: (35 * c ** 4 - 30 * c * c + 3) / 8)]
    for _ in range(300):
        mc_sweep(S, lat.nbr, p, 1 / T, 0.6, 0.6, 5.0, True, True, True, False, st, cnt)
    samples = []
    for _ in range(3000):
        mc_sweep(S, lat.nbr, p, 1 / T, 0.6, 0.6, 5.0, True, True, True, False, st, cnt)
        c = S[..., 2]
        samples.append([np.mean(0.5 * (3 * c * c - 1)), np.mean((35 * c ** 4 - 30 * c * c + 3) / 8)])
    x = np.array(samples).reshape(30, 100, 2).mean(1)
    m, e = x.mean(0), x.std(0, ddof=1) / np.sqrt(30)
    assert np.all(np.abs(m - exact) < 4.5 * e + 1e-4), (m, exact, e)


def test_two_design_no_quadrupolar_energy_for_rigid_tetrahedron():
    P = Params(J=0.0, K=0.0, a2=1.0, a4=0.0); p = P.array()
    lat = Lattice(2, 4)
    for _ in range(20):
        Q, _r = np.linalg.qr(RNG.normal(size=(3, 3)))
        S = np.tile(tetrahedron() @ Q.T, (lat.N, 1, 1))
        assert abs(energy_total(S, lat.nbr, p)) < 1e-12           # sum_i P2(z.S_i) = 0 exactly


def _gamma_numeric(P):
    """E(C3 along z) - E(S4 along z) = (2/3) Gamma for E = c - Gamma sum n^4."""
    e3, _ = minimize_cluster_at_orientation(P, oriented_tetrahedron([1, 1, 1], 0.3))
    e4, _ = minimize_cluster_at_orientation(P, oriented_tetrahedron([0, 0, 1], 0.3))
    return 1.5 * (e3 - e4)


def test_gamma_formula_a4_at_production_value():
    P = Params.from_z(**P_N3, a4=-0.02)
    g = gamma_formula(P)[0]
    assert abs(_gamma_numeric(P) / g - 1) < 0.01


def test_gamma_formula_a2_squared_term():
    """The a2 contribution is second order (relaxation); the a2^3 correction is removed by
    averaging +a2 and -a2."""
    for K, zG, zH4 in ((2.0, 0.08, 0.80), (0.5, 1.0, 0.0)):
        gp = _gamma_numeric(Params.from_z(K=K, zG=zG, zH4=zH4, a2=0.05))
        gm = _gamma_numeric(Params.from_z(K=K, zG=zG, zH4=zH4, a2=-0.05))
        g = gamma_formula(Params.from_z(K=K, zG=zG, zH4=zH4, a2=0.05))[0]
        assert abs(0.5 * (gp + gm) / g - 1) < 0.01


def test_gamma_deviation_is_next_order():
    devs = [abs(_gamma_numeric(Params.from_z(**P_N3, a4=a)) / (35 / 9 * a) - 1) for a in (-0.02, -0.01, -0.005)]
    assert devs[1] < 0.6 * devs[0] and devs[2] < 0.6 * devs[1]


def test_gamma_full_orientation_dependence():
    """E(n) over random orientations is c - Gamma sum n^4 (fit residual << the anisotropy energy)."""
    P = Params.from_z(**P_N3, a4=-0.01)
    E, X = [], []
    for nh in RNG.normal(size=(10, 3)):
        nh /= np.linalg.norm(nh)
        E.append(minimize_cluster_at_orientation(P, oriented_tetrahedron(nh, RNG.uniform(0, 6)))[0])
        X.append(np.sum(nh ** 4))
    A = np.vstack([np.ones(10), -np.array(X)]).T
    sol, *_ = np.linalg.lstsq(A, np.array(E), rcond=None)
    assert abs(sol[1] / gamma_formula(P)[0] - 1) < 0.01
    assert np.abs(A @ sol - E).max() < 0.01 * abs(sol[1])


def _sum_n4(S):
    """sum_alpha (z . e_alpha)^4 for the S4 axes e_alpha ~ S_0 + S_alpha of a near-tetrahedron."""
    e = [(S[0] + S[k]) / np.linalg.norm(S[0] + S[k]) for k in (1, 2, 3)]
    return sum(v[2] ** 4 for v in e)


@pytest.mark.parametrize("a4,expected", [(-0.02, 1 / 3), (0.02, 1.0)])
def test_ground_state_axis_selection(a4, expected):
    P = Params.from_z(**P_N3, a4=a4)
    E, S = find_ground_state(P)
    assert abs(_sum_n4(S) - expected) < 0.02


def test_run_config_keys():
    class Lat:
        z = 4
    P = make_params(dict(model=dict(zG=0.08, zH4=0.8, K=2.0, a2=0.01, a4=-0.02)), Lat())
    assert P.a2 == 0.01 and P.a4 == -0.02
    assert make_params(dict(model=dict(zG=0.08)), Lat()).a4 == 0.0
