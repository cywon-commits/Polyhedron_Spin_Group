"""M1 validation tests 1-6 (+ ground-state boundary reproduction)."""
import numpy as np
import pytest

from polyspin import moves
from polyspin.groundstate import (boundary_CF, boundary_TC, boundary_TF, classify_cluster,
                                  find_ground_state, phase_of_gs)
from polyspin.lattice import Lattice
from polyspin.model import (Params, analytic_energies, channel_energy, energy_total,
                            cluster_inter_energy, random_spins, spin_energy, tetrahedron)
from polyspin.observables import _YL, harmonics, legendre, m_l, majority_phase
from polyspin.rng import make_state

SEED = 12345


def uniform_lattice_state(lat, S):
    return np.tile(S, (lat.N, 1, 1)).copy()


# ------------------------------------------------------------------ test 1
@pytest.mark.parametrize("dim", [2, 3])
def test_energy_consistency(dim):
    rng = np.random.default_rng(1)
    lat = Lattice(dim, 4)
    for _ in range(100):
        P = Params(J=rng.normal(), K=rng.normal(), L=rng.normal(), G=rng.normal(),
                   H4=rng.normal(), G1=rng.normal(), z=lat.z)
        S = random_spins((4,), rng)
        E_direct = energy_total(uniform_lattice_state(lat, S), lat.nbr, P.array()) / lat.N
        E_chan = channel_energy(S, P)
        assert abs(E_direct - E_chan) <= 1e-12 * max(1.0, abs(E_direct))


def test_analytic_uniform_energies():
    lat = Lattice(2, 4)
    P = Params.from_z(0.7, 0.4, z=4, K=0.5)
    ae = analytic_energies(P)
    up = np.array([0, 0, 1.0])
    cases = {"T": tetrahedron(), "C": np.array([up, up, -up, -up]), "F": np.array([up] * 4)}
    for k, S in cases.items():
        e = energy_total(uniform_lattice_state(lat, S), lat.nbr, P.array()) / lat.N
        assert e == pytest.approx(ae[k], abs=1e-12)


def test_harmonics_addition_theorem():
    rng = np.random.default_rng(2)
    a, b = random_spins((2,), rng)
    Ya, Yb = harmonics(a), harmonics(b)
    for l in range(1, 5):
        sel = _YL == l
        assert Ya[sel] @ Yb[sel] == pytest.approx((2 * l + 1) / (4 * np.pi) * legendre(l, a @ b))


# ------------------------------------------------------------------ test 2
def _random_lattice(dim=3, size=4, seed=3):
    rng = np.random.default_rng(seed)
    lat = Lattice(dim, size)
    P = Params(J=1.0, K=0.5, L=0.3, G=0.4, H4=0.3, G1=0.2, z=lat.z)
    return lat, P, random_spins((lat.N, 4), rng)


def test_local_delta_e():
    lat, P, spins = _random_lattice()
    p = P.array()
    rng = np.random.default_rng(4)
    for _ in range(50):
        a, i = rng.integers(lat.N), rng.integers(4)
        new = random_spins((), rng)
        de = (spin_energy(spins, lat.nbr, p, a, i, *new)
              - spin_energy(spins, lat.nbr, p, a, i, *spins[a, i]))
        e0 = energy_total(spins, lat.nbr, p)
        s2 = spins.copy()
        s2[a, i] = new
        assert energy_total(s2, lat.nbr, p) - e0 == pytest.approx(de, abs=1e-11)


def test_rotation_delta_e():
    lat, P, spins = _random_lattice()
    p = P.array()
    rng = np.random.default_rng(5)
    for _ in range(50):
        a = rng.integers(lat.N)
        Q, _r = np.linalg.qr(rng.normal(size=(3, 3)))
        if np.linalg.det(Q) < 0:
            Q[:, 0] *= -1
        new = spins[a] @ Q.T
        de = (cluster_inter_energy(spins, lat.nbr, p, a, new)
              - cluster_inter_energy(spins, lat.nbr, p, a, spins[a]))
        e0 = energy_total(spins, lat.nbr, p)
        s2 = spins.copy()
        s2[a] = new
        assert energy_total(s2, lat.nbr, p) - e0 == pytest.approx(de, abs=1e-11)


def test_sweep_sanity():
    """Full sweeps keep spins normalised, accept every move type at least once, and are
    reproducible for a fixed seed.  (dE correctness itself: the two tests above.)"""
    lat, P, spins = _random_lattice()
    p = P.array()
    s = make_state(SEED)
    counters = np.zeros((moves.N_COUNTERS, 2), dtype=np.int64)
    for _ in range(20):
        moves.mc_sweep(spins, lat.nbr, p, 2.0, 0.4, 0.2, 30.0, True, True, True, True, s,
                       counters)
    assert np.allclose(np.linalg.norm(spins, axis=-1), 1.0, atol=1e-12)
    assert counters[:, 0].min() > 0  # every move type was accepted at least once
    spins2 = _random_lattice()[2]
    s2 = make_state(SEED)
    c2 = np.zeros_like(counters)
    for _ in range(20):
        moves.mc_sweep(spins2, lat.nbr, p, 2.0, 0.4, 0.2, 30.0, True, True, True, True, s2, c2)
    assert np.array_equal(spins, spins2)


# ------------------------------------------------------------------ test 3
def test_symmetries():
    lat, P, spins = _random_lattice()
    p = P.array()
    rng = np.random.default_rng(6)
    e0, m0 = energy_total(spins, lat.nbr, p), m_l(spins)
    Q, _r = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    rot = spins @ Q.T
    assert energy_total(rot, lat.nbr, p) == pytest.approx(e0, abs=1e-10)
    assert np.allclose(m_l(rot), m0, atol=1e-12)
    perm = spins.copy()
    for a in range(lat.N):
        perm[a] = perm[a][rng.permutation(4)]
    assert energy_total(perm, lat.nbr, p) == pytest.approx(e0, abs=1e-10)
    assert np.allclose(m_l(perm), m0, atol=1e-12)


def test_order_parameter_values():
    lat = Lattice(2, 4)
    up = np.array([0, 0, 1.0])
    F = uniform_lattice_state(lat, np.array([up] * 4))
    assert np.allclose(m_l(F), 1.0)
    C = uniform_lattice_state(lat, np.array([up, up, -up, -up]))
    m = m_l(C)
    assert m[0] < 1e-12 and m[2] < 1e-12 and m[1] > 0.5 and m[3] > 0.5
    T = uniform_lattice_state(lat, tetrahedron())
    m = m_l(T)
    assert m[0] < 1e-12 and m[1] < 1e-12 and m[2] > 0.1 and m[3] > 0.1


# ------------------------------------------------------------------ test 4
def _two_spin_distribution(use, beta=1.5, nsamp=60000, thin=5, seed=7):
    """Two spins in one cluster, J coupling only; sample x = S1.S2."""
    J = 1.0
    p = Params(J=J, K=0.0, L=0.0, G=0.0, H4=0.0, G1=0.0, z=0, n=2).array()
    nbr = np.zeros((1, 0), dtype=np.int64)
    rng = np.random.default_rng(seed)
    spins = random_spins((1, 2), rng)
    s = make_state(seed)
    counters = np.zeros((moves.N_COUNTERS, 2), dtype=np.int64)
    xs = np.empty(nsamp)
    for k in range(nsamp):
        for _ in range(thin):
            moves.mc_sweep(spins, nbr, p, beta, 0.8, 0.5, 20.0, *use, s, counters)
        xs[k] = spins[0, 0] @ spins[0, 1]
    return xs, beta * J


def _ks_against_boltzmann(xs, bJ):
    xs = np.sort(xs)
    cdf_emp = np.arange(1, len(xs) + 1) / len(xs)
    cdf = (np.exp(bJ) - np.exp(-bJ * xs)) / (np.exp(bJ) - np.exp(-bJ))
    return np.max(np.abs(cdf - cdf_emp))


# use = (single, rotation, overlap, permutation)
@pytest.mark.parametrize("use", [
    (True, False, False, False),
    (True, True, False, False),
    (False, False, True, False),
    (True, False, False, True),
    (True, True, True, True),
])
def test_detailed_balance(use):
    xs, bJ = _two_spin_distribution(use)
    # effective sample size is reduced by autocorrelation; 0.02 is ~5 sigma here
    assert _ks_against_boltzmann(xs, bJ) < 0.02


def test_rotation_alone_preserves_x():
    xs, _ = _two_spin_distribution((False, True, False, False), nsamp=200, thin=1)
    assert np.ptp(xs) < 1e-12


# ------------------------------------------------------------------ test 5
@pytest.mark.parametrize("zG,zH,K,expect", [(0.5, 0.0, 0.5, "T"), (0.5, 1.0, 0.5, "C"),
                                            (2.0, 0.0, 0.5, "F")])
def test_low_temperature_limit(zG, zH, K, expect):
    lat = Lattice(2, 6)
    P = Params.from_z(zG, zH, z=4, K=K)
    p = P.array()
    ph, E0, S0 = phase_of_gs(P)
    assert ph == expect
    spins = uniform_lattice_state(lat, S0)
    T = 0.005
    s = make_state(SEED)
    counters = np.zeros((moves.N_COUNTERS, 2), dtype=np.int64)
    for _ in range(3000):
        moves.mc_sweep(spins, lat.nbr, p, 1 / T, 0.05, 0.03, 30.0, True, True, False, False,
                       s, counters)
    es = []
    for _ in range(6000):
        moves.mc_sweep(spins, lat.nbr, p, 1 / T, 0.05, 0.03, 30.0, True, True, False, False,
                       s, counters)
        es.append(energy_total(spins, lat.nbr, p) / lat.N)
    e = np.mean(es)
    # equipartition: 8 transverse dof x T/2 = 4T above the T=0 energy
    assert e - E0 == pytest.approx(4 * T, rel=0.15)


# ------------------------------------------------------------------ ground state
def test_ground_state_boundaries():
    K = 0.5
    # T-C boundary zH = 0.9K + 0.3zG at zG = 0.5
    zG = 0.5
    zHc = boundary_TC(1.0, K, zG)
    assert phase_of_gs(Params.from_z(zG, zHc - 0.02, K=K))[0] == "T"
    assert phase_of_gs(Params.from_z(zG, zHc + 0.02, K=K))[0] == "C"
    # T-F boundary at zH=0
    zGc = boundary_TF(1.0, K, 0.0)
    assert zGc == pytest.approx(1.714, abs=1e-3)
    assert phase_of_gs(Params.from_z(zGc - 0.02, 0.0, K=K))[0] == "T"
    assert phase_of_gs(Params.from_z(zGc + 0.02, 0.0, K=K))[0] == "F"
    # C-F boundary zG = J at zH = 1.5
    assert phase_of_gs(Params.from_z(boundary_CF(1.0) - 0.02, 1.5, K=K))[0] == "C"
    assert phase_of_gs(Params.from_z(boundary_CF(1.0) + 0.02, 1.5, K=K))[0] == "F"


def test_ground_state_quasi_ferro_window():
    """Cut C (K=0, zH=0): T below 0.81, U between 0.81 and 1.33, F above."""
    for zG, expect in [(0.6, "T"), (0.78, "T"), (0.85, "U"), (1.0, "U"), (1.2, "U"),
                       (1.30, "U"), (1.36, "F"), (1.6, "F")]:
        assert phase_of_gs(Params.from_z(zG, 0.0, K=0.0))[0] == expect, zG


def test_ground_state_energies_match_analytic():
    P = Params.from_z(0.5, 0.25, K=0.5)
    E, _ = find_ground_state(P)
    assert E == pytest.approx(analytic_energies(P)["T"], abs=1e-9)


# ------------------------------------------------------------------ test 6
CUTLINE_POINTS = [  # (name, zG, zH, K, expected) -- from the T=0 table
    ("A", 0.5, 0.0, 0.5, "T"), ("A", 1.5, 0.0, 0.5, "T"), ("A", 2.0, 0.0, 0.5, "F"),
    ("B", 0.5, 0.25, 0.5, "T"), ("B", 0.5, 1.0, 0.5, "C"), ("B", 0.5, 2.0, 0.5, "C"),
    ("C", 0.6, 0.0, 0.0, "T"), ("C", 1.0, 0.0, 0.0, "U"), ("C", 1.2, 0.0, 0.0, "U"),
    ("C", 1.6, 0.0, 0.0, "F"), ("D", 0.9, 1.5, 0.5, "C"), ("D", 1.1, 1.5, 0.5, "F"),
]


@pytest.mark.parametrize("name,zG,zH,K,expect", CUTLINE_POINTS)
def test_phase_identification_gs(name, zG, zH, K, expect):
    """Lowest-temperature state (here: T=0 minimiser tiled on the lattice) is
    identified as the predicted phase by the order-parameter / shape classifier."""
    P = Params.from_z(zG, zH, K=K)
    _, E, S0 = phase_of_gs(P)
    lat = Lattice(2, 4)
    spins = uniform_lattice_state(lat, S0)
    phase, frac = majority_phase(spins, thr=0.999)
    assert phase == expect
    m = m_l(spins)
    if expect == "T":
        assert m[0] < 1e-6 and m[1] < 1e-6 and m[2] > 0.05
    if expect == "C":
        assert m[0] < 1e-6 and m[2] < 1e-6 and m[1] > 0.3
    if expect == "F":
        assert np.allclose(m, 1.0, atol=1e-6)
    if expect == "U":
        assert 0.2 < m[0] < 0.8
