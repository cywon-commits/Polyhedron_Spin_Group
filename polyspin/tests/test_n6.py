"""n = 6 code-path validation (decision Oct 18, step 1): the kernels, the channel energy, the low-T equipartition slope and the harmonic
code must work for six spins per cluster before the n = 6 wedge ladder is tested."""
import numpy as np
import pytest

from polyspin import harmonic as Hm
from polyspin.analysis.lowt import energy_above_gs
from polyspin.lattice import Lattice
from polyspin.model import Params, channel_energy, cluster_inter_energy, energy_total, random_spins, spin_energy

N6 = 6


def _lat_state(lat, S):
    return np.tile(S, (lat.N, 1, 1)).copy()


@pytest.mark.parametrize("dim", [2, 3])
def test_energy_consistency_n6(dim):
    rng = np.random.default_rng(1)
    lat = Lattice(dim, 4)
    for _ in range(50):
        P = Params(J=rng.normal(), K=rng.normal(), L=rng.normal(), G=rng.normal(), H4=rng.normal(), G1=rng.normal(), z=lat.z, n=N6)
        S = random_spins((N6,), rng)
        assert abs(energy_total(_lat_state(lat, S), lat.nbr, P.array()) / lat.N - channel_energy(S, P)) <= 1e-11


def test_local_delta_e_n6():
    rng = np.random.default_rng(4)
    lat = Lattice(2, 4)
    P = Params.from_z(0.7, 0.4, z=4, K=0.5, L=0.1, G1=0.05, n=N6); p = P.array()
    spins = random_spins((lat.N, N6), rng)
    for _ in range(40):
        a, i = rng.integers(lat.N), rng.integers(N6)
        new = random_spins((), rng)
        de = spin_energy(spins, lat.nbr, p, a, i, *new) - spin_energy(spins, lat.nbr, p, a, i, *spins[a, i])
        e0 = energy_total(spins, lat.nbr, p); s2 = spins.copy(); s2[a, i] = new
        assert energy_total(s2, lat.nbr, p) - e0 == pytest.approx(de, abs=1e-10)
    for _ in range(40):                                                    # rigid rotation of a whole cluster
        a = rng.integers(lat.N)
        Q, _r = np.linalg.qr(rng.normal(size=(3, 3)))
        if np.linalg.det(Q) < 0:
            Q[:, 0] *= -1
        new = spins[a] @ Q.T
        de = cluster_inter_energy(spins, lat.nbr, p, a, new) - cluster_inter_energy(spins, lat.nbr, p, a, spins[a])
        e0 = energy_total(spins, lat.nbr, p); s2 = spins.copy(); s2[a] = new
        assert energy_total(s2, lat.nbr, p) - e0 == pytest.approx(de, abs=1e-10)


def test_collinear_offsets_and_stationarity_n6():
    lat = Lattice(2, 4)
    for zG in (1.0, 1.1):
        P = Params.from_z(zG, 1.5, z=4, K=0.5, n=N6)
        e = [energy_total(_lat_state(lat, Hm.collinear_state(N6, k)), lat.nbr, P.array()) / lat.N for k in range(4)]
        for k in range(4):
            assert Hm.blocks(Hm.collinear_state(N6, k), P)[2] < 1e-10                      # stationary
            assert e[k] - e[0] == pytest.approx(Hm.collinear_energy_offset(P, k), abs=1e-11)
            if zG == 1.0:
                assert e[k] == pytest.approx(e[0], abs=1e-12)                              # degenerate at zG = J
        if zG == 1.1:
            assert e[1] - e[0] == pytest.approx(0.1 * (36 - 16) / 2, abs=1e-11)             # E_k - E_F = x (n^2 - M^2) / 2 > 0: F wins at T = 0


def test_collinear_states_are_local_minima_n6():
    for dim, L in ((2, 16), (3, 8)):
        P = Params.from_z(1.1, 1.5, z=4 if dim == 2 else 6, K=0.5, n=N6)
        for k in range(4):
            logdet, zero, lam = Hm.log_det_per_cluster(Hm.collinear_state(N6, k), P, dim, L)    # raises on a negative eigenvalue
            assert zero == 2 and lam > 0 and np.isfinite(logdet)


@pytest.mark.parametrize("k", [0, 2, 3])
def test_low_temperature_slope_n6(k):
    """<e> - e0 = a T with a = 6 - 1/N for n = 6 (12 transverse dofs per cluster, two global zero modes), within 1%."""
    lat = Lattice(2, 8)
    P = Params.from_z(1.0, 1.5, z=4, K=0.5, n=N6); p = P.array()
    S0 = Hm.collinear_state(N6, k)
    E0 = energy_total(_lat_state(lat, S0), lat.nbr, p) / lat.N
    T = 0.01
    m, err = energy_above_gs(S0, E0, p, lat, T, 3000, 20000, 3)
    assert abs(m / T - (6 - 1 / lat.N)) < 0.01 * 6


def test_moment_class_fractions_n6():
    from polyspin import slab
    lat = Lattice(2, 4)
    for n in (4, 6):
        for k in range(n // 2 + 1):
            fr = slab.moment_class_fractions(_lat_state(lat, Hm.collinear_state(n, k)))
            assert len(fr) == n // 2 + 1 and fr[n // 2 - k] == 1.0              # ascending |M|: index n//2 - k
