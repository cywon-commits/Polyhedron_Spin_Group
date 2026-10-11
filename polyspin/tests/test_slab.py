import numpy as np

from polyspin import slab
from polyspin.lattice import Lattice
from polyspin.model import Params, energy_total
from polyspin.harmonic import collinear_state


def test_slab_construction_and_energy_cost():
    lat = Lattice(2, 8)
    S_C, S_F = collinear_state(4, 2), collinear_state(4, 0)
    spins = slab.make_slab(lat, S_C, S_F)
    fr = slab.moment_fractions(spins)
    assert np.allclose(fr, [0.5, 0.0, 0.5])
    # at zG = J the C and F states are degenerate, the interface costs (G/2)(M_a-M_b)^2 per bond:
    # two interfaces x L rows x (G/2)*16 with M = 0 | 4
    P = Params.from_z(1.0, 1.5, z=4, K=0.5)
    e_uni = energy_total(np.tile(S_C, (lat.N, 1, 1)), lat.nbr, P.array())
    cost = energy_total(spins, lat.nbr, P.array()) - e_uni
    assert np.isclose(cost, 2 * lat.size * (P.G / 2) * 16)


def test_interface_velocity_and_short_run():
    ts = np.arange(100, 1100, 100)
    v, se = slab.interface_velocity(ts, 0.5 + 1e-5 * ts)
    assert np.isclose(v, 1e-5) and se < 1e-9
    lat = Lattice(2, 8)
    P = Params.from_z(0.9, 1.5, z=4, K=0.5)
    t, fr, e, spins = slab.run_slab(lat, P, 0.05, collinear_state(4, 2), collinear_state(4, 0), 300, 100, seed=1)
    assert fr.shape == (3, 3) and np.allclose(fr.sum(1), 1.0)
    # at zG = 0.9 < J the C phase is lower in energy: the F fraction must not grow
    assert fr[-1, 2] <= 0.5 + 1e-9
