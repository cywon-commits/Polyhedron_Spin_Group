"""Follow-up 1 of the request: tests 4b and 5b, and the Phi'(1) diagnostic."""
import numpy as np
import pytest
from scipy.stats import ks_2samp

from polyspin import moves
from polyspin.analysis.lowt import measure_slope
from polyspin.groundstate import find_ground_state, phi_prime1, phase_of_gs
from polyspin.model import Params, channel_coeffs, energy_total, legendre
from polyspin.rng import make_state


# ------------------------------------------------------------------ test 4b
def _cluster_chain(use, T, nsamp, thin, seed, kappa=20.0):
    """One 4-spin cluster, internal J and K only (J=1, K=-1: dimers form at low T)."""
    p = Params(J=1.0, K=-1.0, z=0).array()
    nbr = np.zeros((1, 0), dtype=np.int64)
    up = np.array([0, 0, 1.0])
    spins = np.array([[up, up, -up, -up]], dtype=float)
    s = make_state(seed)
    c = np.zeros((moves.N_COUNTERS, 2), dtype=np.int64)
    out = np.empty((nsamp, 3))
    delta = 2.0 * np.sqrt(T)
    for k in range(nsamp):
        for _ in range(thin):
            moves.mc_sweep(spins, nbr, p, 1 / T, delta, 0.3, kappa, *use, s, c)
        X = spins[0] @ spins[0].T
        out[k] = [energy_total(spins, nbr, p), legendre(2, X).sum(), legendre(3, X).sum()]
    return out


@pytest.mark.parametrize("T", [0.05, 0.3])
def test_four_spin_overlap_vs_single(T):
    """Overlap-only chain and single-spin-only chain must give the same E, A2, A3
    distributions for a 4-spin cluster.  Checks the whole von Mises-Fisher mixture and the
    centre-selection weights.  Mutations (mixture -> one component; 1/(n-1) -> 1/n) give KS
    distances 0.05-0.15 and 0.02-0.05 respectively versus ~0.01 here."""
    a = _cluster_chain((True, False, False, False), T, 60000, 5, seed=11)
    b = _cluster_chain((False, False, True, False), T, 60000, 5, seed=12)
    for j, name in enumerate(("E", "A2", "A3")):
        ks = ks_2samp(a[:, j], b[:, j]).statistic
        assert ks < 0.03, (name, ks)


# ------------------------------------------------------------------ test 5b
@pytest.mark.slow
@pytest.mark.parametrize("zG,zH,K,phase,n_zero", [(0.5, 0.0, 0.5, "T", 3), (0.5, 1.0, 0.5, "C", 2),
                                                  (2.0, 0.0, 0.5, "F", 2), (1.0, 0.0, 0.0, "U", 3)])
def test_low_temperature_slope(zG, zH, K, phase, n_zero):
    """Slope of <E>/N - E0 vs T for T/J = 0.002..0.01 (fit a T + b T^2), L=8:
    4.00 within 1%.  The small deficit is the finite-size zero-mode count: global rotations
    carry no energy, so a = 4 - n_zero/(2N) (3 rotations for T/U, 2 for the collinear
    C and F states whose rotation about the common axis acts trivially)."""
    r = measure_slope(zG, zH, K, L=8)
    assert r["phase"] == phase
    assert r["a"] == pytest.approx(4.0, rel=0.01)
    expected = 4.0 - n_zero / (2 * r["N"])
    assert abs(r["a"] - expected) < max(3 * r["a_err"], 0.004)


# ----------------------------------------------------------------- Phi'(1)
def test_phi_prime_sign_for_F():
    """F ground state => locally stable against continuous splitting => Phi'(1) < 0."""
    for zG, zH, K in [(2.0, 0.0, 0.5), (1.6, 0.0, 0.0), (1.1, 1.5, 0.5), (3.0, 0.5, 0.5)]:
        P = Params.from_z(zG, zH, K=K)
        assert phase_of_gs(P)[0] == "F"
        assert phi_prime1(channel_coeffs(P)[0]) < 0


# --------------------------------------------------------- D phase exists (follow-up 1.3)
@pytest.mark.parametrize("K,zG,zH,angle,M", [(-0.5, 0.9, 0.0, 145.2, 0.299),
                                              (-0.75, 0.9, 0.0, 161.2, 0.163),
                                              (0.0, 1.10, 0.25, 127.6, 0.442)])
def test_D_phase_in_physical_plane(K, zG, zH, angle, M):
    """Contrary to the expectation in the request, bent-dimer D ground states exist in the
    physical plane L = G1 = 0 (K <~ 0, zG ~ 0.7-1.1, zH4 <~ 0.25).  Checks: global minimum with
    300 random starts is {22} with net moment and the stated dimer-dimer angle, and it is a strict
    local minimum (positive tangent Hessian except the 3 global-rotation zero modes)."""
    from polyspin.groundstate import _energy_grad, _rand_unit, minimize_cluster, partition, structured_starts
    P = Params.from_z(zG, zH, K=K)
    rng = np.random.default_rng(7)
    starts = [_rand_unit(rng, 4) for _ in range(300)] + structured_starts(rng)
    E, S = min((minimize_cluster(P, S0) for S0 in starts), key=lambda r: r[0])
    assert phase_of_gs(P)[0] == "D"
    assert partition(S, 0.999) == (2, 2)
    assert np.linalg.norm(S.sum(0)) / 4 == pytest.approx(M, abs=2e-3)
    X = S @ S.T
    i, j = [(a, b) for a in range(4) for b in range(a + 1, 4) if X[a, b] > 0.999][0]
    others = [k for k in range(4) if k not in (i, j)]
    assert np.degrees(np.arccos(X[i, others[0]])) == pytest.approx(angle, abs=0.1)
    kappa, const = channel_coeffs(P)
    v0, n, h = S.ravel().copy(), 12, 1e-6
    H = np.zeros((n, n))
    for a in range(n):
        e = np.zeros(n)
        e[a] = h
        H[:, a] = (_energy_grad(v0 + e, kappa, const, 4)[1] - _energy_grad(v0 - e, kappa, const, 4)[1]) / (2 * h)
    B = []
    for a in range(4):
        t1 = np.cross(S[a], [1, 0, 0])
        if np.linalg.norm(t1) < 0.1:
            t1 = np.cross(S[a], [0, 1, 0])
        t1 /= np.linalg.norm(t1)
        for t in (t1, np.cross(S[a], t1)):
            b = np.zeros(n)
            b[3 * a:3 * a + 3] = t
            B.append(b)
    w = np.linalg.eigvalsh(np.array(B) @ (H + H.T) / 2 @ np.array(B).T)
    assert (w > 1e-3).sum() == 5 and np.all(np.abs(w[:3]) < 1e-5)  # 3 zero modes, rest stiff


def test_C_is_exactly_collinear():
    """C->D is continuous: a {22} state with net moment must not be called C."""
    P = Params.from_z(0.7, 0.0, K=-0.5)  # just inside D (continuous onset near zG = 0.67)
    assert phase_of_gs(P)[0] == "D"
    assert phase_of_gs(Params.from_z(0.6, 0.0, K=-0.5))[0] == "C"


# ------------------------------------------------- follow-up 3: analytic C-D and D-F boundaries
from polyspin.groundstate import boundary_CD, boundary_DF  # noqa: E402


def _moment(P):
    return np.linalg.norm(phase_of_gs(P)[2].sum(0)) / 4


def test_C_D_boundary_is_phi_prime_zero():
    """zG = (J - 2K + 4 zH4)/3: C (M = 0) below, D (M > 0) above; Phi'(-1) = 0 on the line;
    the moment grows like sqrt(zG - zG_c) (mean-field onset)."""
    assert boundary_CD(1.0, -0.5, 0.0) == pytest.approx(2 / 3)
    for K, zH in [(-0.5, 0.0), (-0.25, 0.25), (-0.75, 0.0), (-0.5, 0.1)]:
        zGc = boundary_CD(1.0, K, zH)
        k = channel_coeffs(Params.from_z(zGc, zH, K=K))[0]
        assert abs(k[0] - 3 * k[1] + 6 * k[2] - 10 * k[3]) < 1e-12
        assert _moment(Params.from_z(zGc - 0.01, zH, K=K)) < 1e-6
        m1 = _moment(Params.from_z(zGc + 0.005, zH, K=K))
        m2 = _moment(Params.from_z(zGc + 0.02, zH, K=K))
        assert phase_of_gs(Params.from_z(zGc + 0.005, zH, K=K))[0] == "D"
        assert m2 / m1 == pytest.approx(2.0, rel=0.05)


def test_D_F_boundary():
    """First-order D-F boundary at zH4 = 0: g^2 - g - 1/12 = 0 for K = -0.5 (zG = 1.0774)."""
    assert boundary_DF(1.0, -0.5) == pytest.approx((1 + np.sqrt(1 + 1 / 3)) / 2, abs=1e-12)
    for K in (-0.5, -0.75, -0.25):
        g = boundary_DF(1.0, K)
        assert phase_of_gs(Params.from_z(g - 0.01, 0.0, K=K))[0] == "D"
        assert phase_of_gs(Params.from_z(g + 0.01, 0.0, K=K))[0] == "F"
        # first order: net moment jumps (D -> F)
        assert _moment(Params.from_z(g - 0.01, 0.0, K=K)) < 0.6


def test_global_classification():
    from polyspin.observables import global_class, m_l
    from polyspin.lattice import Lattice
    from polyspin.model import tetrahedron
    lat = Lattice(2, 4)
    up = np.array([0, 0, 1.0])
    for S, exp in [(tetrahedron(), "T"), (np.array([up, up, -up, -up]), "C"), (np.array([up] * 4), "F"),
                   (phase_of_gs(Params.from_z(0.9, 0.0, K=-0.5))[2], "M")]:
        assert global_class(m_l(np.tile(S, (lat.N, 1, 1))), 0.05) == exp
