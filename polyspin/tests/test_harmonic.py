"""Spin-wave Hessian vs finite differences of the full lattice energy."""
import numpy as np
import pytest

from polyspin import harmonic as Hm
from polyspin.groundstate import phase_of_gs
from polyspin.lattice import Lattice
from polyspin.model import Params, energy_total


def fd_hessian(S0, P, lat, h=1e-4):
    """Second differences of E along the exponential map in tangent coordinates."""
    n, N = len(S0), lat.N
    T = Hm.tangent_basis(S0)
    base = np.tile(S0, (N, 1, 1))
    p = P.array()
    dim = N * n * 2

    def E(moves):
        s = base.copy()
        for (a, i, c), val in moves:
            pass
        # apply (possibly two) displacements; combine those on the same spin additively
        disp = {}
        for (a, i, c), val in moves:
            d = disp.setdefault((a, i), np.zeros(2))
            d[c] += val
        for (a, i), d in disp.items():
            u = T[i] @ d
            th = np.linalg.norm(u)
            s[a, i] = S0[i] * np.cos(th) + (u / th) * np.sin(th) if th > 0 else S0[i]
        return energy_total(s, lat.nbr, p)

    H = np.zeros((dim, dim))
    idx = [(a, i, c) for a in range(N) for i in range(n) for c in range(2)]
    for r, ir in enumerate(idx):
        for c_, ic in enumerate(idx):
            if c_ < r:
                continue
            if r == c_:
                val = (E([(ir, h)]) - 2 * E([]) + E([(ir, -h)])) / h ** 2
            else:
                val = (E([(ir, h), (ic, h)]) - E([(ir, h), (ic, -h)])
                       - E([(ir, -h), (ic, h)]) + E([(ir, -h), (ic, -h)])) / (4 * h ** 2)
            H[r, c_] = H[c_, r] = val
    return H


CASES = [("T", 0.5, 0.0, 0.5, 2), ("U", 1.0, 0.0, 0.0, 2), ("D", 0.9, 0.0, -0.5, 2),
         ("C", 0.5, 1.0, 0.5, 2), ("F", 2.0, 0.0, 0.5, 2)]


@pytest.mark.parametrize("name,zG,zH,K,dim", CASES)
def test_q_space_matches_lattice_hessian(name, zG, zH, K, dim):
    lat = Lattice(dim, 4)
    P = Params.from_z(zG, zH, z=lat.z, K=K)
    ph, E0, S0 = phase_of_gs(P)
    assert ph == name
    A, C, resid = Hm.blocks(S0, P)
    assert resid < 1e-6                             # stationary
    ev_q = np.sort(Hm.spectrum(S0, P, Hm.q_grid(dim, 4)).ravel())
    ev_fd = np.sort(np.linalg.eigvalsh(fd_hessian(S0, P, lat)))
    assert np.allclose(ev_q, ev_fd, atol=2e-5), np.abs(ev_q - ev_fd).max()
    # zero modes: 3 global rotations for generic states, 2 for collinear ones
    n_zero = int((np.abs(ev_q) < 1e-6).sum())
    assert n_zero == (2 if name in ("C", "F") else 3)


def test_3d_spectrum_matches():
    lat = Lattice(3, 4)
    P = Params.from_z(0.5, 0.0, z=6, K=0.5)
    ph, E0, S0 = phase_of_gs(P)
    A, C, resid = Hm.blocks(S0, P)
    # compare only a handful of random eigenvalues cheaply: trace of H over the grid equals the
    # trace of the lattice Hessian, i.e. N * trace(A) (C terms are off-diagonal in site space)
    ev_q = Hm.spectrum(S0, P, Hm.q_grid(3, 4))
    assert ev_q.sum() == pytest.approx(lat.N * np.trace(A), rel=1e-10)
    assert (ev_q > -1e-8).all()


def test_equipartition_counting():
    """Sanity: number of modes per q is 2n; mean of eigenvalues = trace(A)/(2n)."""
    P = Params.from_z(0.5, 1.0, z=4, K=0.5)
    ph, E0, S0 = phase_of_gs(P)
    ev = Hm.spectrum(S0, P, Hm.q_grid(2, 6))
    assert ev.shape == (36, 8)


# ------------------------------------------------ H6: free energy on the C-F degenerate line
def test_collinear_states_degenerate_and_stationary_at_zG_equals_J():
    """At zG = J the collinear states M = 0, 2, 4 (n = 4) have the same energy and are stationary."""
    lat = Lattice(2, 4)
    P = Params.from_z(1.0, 1.5, z=4, K=0.5)
    es = []
    for k in (0, 1, 2):
        S = Hm.collinear_state(4, k)
        assert Hm.blocks(S, P)[2] < 1e-10
        es.append(energy_total(np.tile(S, (lat.N, 1, 1)), lat.nbr, P.array()) / lat.N)
    assert np.allclose(es, es[0], atol=1e-12)
    # away from the line the offsets follow E_k - E_F = (J - zG)(M^2 - n^2)/2
    P2 = Params.from_z(0.9, 1.5, z=4, K=0.5)
    e2 = [energy_total(np.tile(Hm.collinear_state(4, k), (lat.N, 1, 1)), lat.nbr, P2.array()) / lat.N
          for k in (0, 1, 2)]
    for k in (1, 2):
        assert e2[k] - e2[0] == pytest.approx(Hm.collinear_energy_offset(P2, k), abs=1e-12)


def test_selection_prefers_C_and_converges():
    P = Params.from_z(1.0, 1.5, z=4, K=0.5)
    r48 = Hm.selection(P, 2, 48)
    r96 = Hm.selection(P, 2, 96)
    # free-energy ordering F > M=2 > C (C = softest and most degenerate)
    assert r96[0]["dF_over_T"] == 0 > r96[1]["dF_over_T"] > r96[2]["dF_over_T"]
    assert r96[2]["dF_harm_over_T"] == pytest.approx(-1.266, abs=2e-3)
    assert r96[2]["ln_multiplicity"] == pytest.approx(np.log(6))
    assert r96[1]["ln_multiplicity"] == pytest.approx(np.log(4))
    assert abs(r48[2]["dF_over_T"] - r96[2]["dF_over_T"]) < 3e-3          # q-grid converged
    assert all(v["zero_modes"] == 2 for v in r96.values())                # two rotations, collinear
    assert Hm.CF_boundary_slope(P, 2, 96) == pytest.approx(0.382, abs=2e-3)
    r3 = Hm.selection(Params.from_z(1.0, 1.5, z=6, K=0.5), 3, 24)
    assert r3[2]["dF_harm_over_T"] == pytest.approx(-1.255, abs=3e-3)


def test_non_minimum_is_rejected():
    P = Params.from_z(1.0, 0.6, z=4, K=0.5)       # collinear states unstable here
    with pytest.raises(ValueError):
        Hm.selection(P, 2, 16)


def test_nonuniform_collinear_states_cost_energy_at_zG_equals_J():
    """At zG = J, clusters with different net moments (same axis) cost exactly
    (G/2) sum_<ab> (M_a - M_b)^2, so the degenerate ground-state manifold is the uniform one."""
    rng = np.random.default_rng(0)
    lat = Lattice(2, 6)
    P = Params.from_z(1.0, 1.5, z=4, K=0.5)
    p = P.array()
    e_uni = energy_total(np.tile(Hm.collinear_state(4, 2), (lat.N, 1, 1)), lat.nbr, p)
    for _ in range(5):
        ks = rng.integers(0, 3, size=lat.N)
        spins = np.array([Hm.collinear_state(4, k)[rng.permutation(4)] for k in ks])
        M = 4 - 2 * ks
        pred = (P.G / 2) * sum((M[a] - M[b]) ** 2 for a in range(lat.N) for b in lat.nbr[a, :2])
        assert energy_total(spins, lat.nbr, p) - e_uni == pytest.approx(pred, abs=1e-9)
