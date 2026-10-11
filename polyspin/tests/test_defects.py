"""M2 test 7: defect detectors."""
import numpy as np
import pytest

from polyspin import defects as D
from polyspin.groundstate import find_ground_state
from polyspin.lattice import Lattice
from polyspin.model import Params, random_spins, tetrahedron

L = 16
X0 = Y0 = 7
CORE = (X0 + 0.5, Y0 + 0.5)


def reference(name):
    if name == "T":
        return tetrahedron()
    if name == "U":
        return find_ground_state(Params.from_z(1.0, 0.0, K=0.0))[1]
    if name == "D":  # real D ground state: bent dimers, K<0 pocket of the L=G1=0 plane
        return find_ground_state(Params.from_z(0.9, 0.0, K=-0.5))[1]
    raise ValueError(name)


def rand_rot(rng):
    Q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    return Q


def shuffle_labels(spins, rng):
    out = spins.copy()
    for a in range(len(out)):
        out[a] = out[a][rng.permutation(out.shape[1])]
    return out


@pytest.mark.parametrize("name,order_G,n_cls", [("T", 12, 7), ("U", 2, 4), ("D", 2, 4)])
def test_group_structure(name, order_G, n_cls):
    g = D.FrameGroup(reference(name))
    assert g.order_G == order_G
    assert len(g.quats) == 2 * order_G
    assert g.n_classes == n_cls
    if name == "T":
        assert sorted(len(c) for c in g.classes) == [1, 1, 4, 4, 4, 4, 6]
        assert sorted(g.class_names) == sorted(["1", "2", "4", "3+", "3-", "6+", "6-"])
    # quaternion product is a homomorphism onto the rotation matrices
    q = g.quats
    assert np.allclose(D.quat_to_mat(D.qmul(q[1], q[-1])), D.quat_to_mat(q[1]) @ D.quat_to_mat(q[-1]))


@pytest.mark.parametrize("name", ["T", "U", "D"])
@pytest.mark.parametrize("dim", [2, 3])
def test_uniform_is_trivial(name, dim):
    rng = np.random.default_rng(1)
    g = D.FrameGroup(reference(name))
    lat = Lattice(dim, 6)
    spins = np.tile(reference(name) @ rand_rot(rng).T, (lat.N, 1, 1))
    spins = shuffle_labels(spins, rng)
    out = D.detect(spins, lat.plaquettes, g)
    assert not out["undefined"].any()
    assert np.all(out["cls"] == g.class_of[g.identity])


def _loop_size(g, gi):
    """Smallest loop half-width k whose per-link step is below the unambiguous limit:
    plaquette (k=1) for alpha <= 180 deg, 4x4-block boundary (k=2) otherwise."""
    return 1 if np.degrees(g.rotation_angle(gi)) <= 180.0 + 1e-6 else 2


@pytest.mark.parametrize("name", ["T", "U", "D"])
def test_artificial_vortices(name):
    """Every conjugacy class: the loop around the vortex core returns exactly that class,
    for random reference orientation, random label permutations and random q signs."""
    rng = np.random.default_rng(2)
    ref0 = reference(name)
    lat = Lattice(2, L)
    coords = lat.coords.astype(float)
    for trial in range(3):
        Rref = rand_rot(rng)
        ref = ref0 @ Rref.T                         # same physics, rotated reference
        g = D.FrameGroup(ref)
        for c, members in enumerate(g.classes):
            for gi in members:                       # every element, not just a representative
                k = _loop_size(g, gi)
                Q0 = D.mat_to_quat(rand_rot(rng))
                q = D.vortex_orientations(coords, CORE, gi, g, Q0=Q0)
                spins = shuffle_labels(D.spins_from_quats(q, ref), rng)
                R, resid, _ = D.fit_orientations(spins, g)
                assert resid.max() < 1e-8
                qf = D.mat_to_quat(R)
                flip = rng.random(len(qf)) < 0.5
                qf[flip] *= -1                       # arbitrary quaternion signs
                loop = D.square_loop(L, X0, Y0, k)[None, :]
                cls, ov = D.holonomy(qf, loop, g)
                assert cls[0] == c, (name, g.class_names[c], k, ov)


def test_single_plaquette_limit_documented():
    """alpha > ~200 deg classes are not resolved by one plaquette: -1 is mis-read and the
    order-3 classes are flagged undefined.  This pins the limitation noted in the report."""
    g = D.FrameGroup(tetrahedron())
    lat = Lattice(2, L)
    coords = lat.coords.astype(float)
    seen = {}
    for c, members in enumerate(g.classes):
        gi = members[0]
        q = D.vortex_orientations(coords, CORE, gi, g)
        cls, _ = D.holonomy(q, D.square_loop(L, X0, Y0, 1)[None, :], g)
        seen[g.class_names[c]] = g.class_names[cls[0]] if cls[0] >= 0 else "undef"
    assert seen["3+"] == "undef" and seen["3-"] == "undef"
    assert seen["2"] != "2"
    for ok in ("1", "4", "6+", "6-"):
        assert seen[ok] == ok


@pytest.mark.parametrize("name", ["T", "U", "D"])
def test_vortex_through_full_pipeline(name):
    """Core plaquette of the lattice plaquette list carries the class, and it is the only
    non-trivial plaquette (classes with alpha <= 180 deg)."""
    rng = np.random.default_rng(3)
    ref = reference(name)
    g = D.FrameGroup(ref)
    lat = Lattice(2, L)
    coords = lat.coords.astype(float)
    core_plaq = lat.site_index(X0, Y0)               # plaquette lists start at site a
    assert np.all(lat.plaquettes[core_plaq] == [lat.site_index(X0, Y0), lat.site_index(X0 + 1, Y0),
                                                lat.site_index(X0 + 1, Y0 + 1),
                                                lat.site_index(X0, Y0 + 1)])
    for c, members in enumerate(g.classes):
        gi = members[0]
        if g.class_names[c] == "1" or _loop_size(g, gi) != 1:
            continue
        q = D.vortex_orientations(coords, CORE, gi, g)
        spins = shuffle_labels(D.spins_from_quats(q, ref), rng)
        out = D.detect(spins, lat.plaquettes, g)
        assert out["cls"][core_plaq] == c
        # the continuous vortex field is not periodic: ignore plaquettes on the wrap seam
        # (x or y = L-1), where compensating images necessarily sit
        seam = (lat.coords[:, 0] == L - 1) | (lat.coords[:, 1] == L - 1)
        nontriv = np.flatnonzero((out["cls"] != g.class_of[g.identity]) & ~seam)
        assert list(nontriv) == [core_plaq]


def test_noise_marks_undefined():
    rng = np.random.default_rng(4)
    g = D.FrameGroup(tetrahedron())
    lat = Lattice(2, 8)
    spins = random_spins((lat.N, 4), rng)             # infinite temperature
    out = D.detect(spins, lat.plaquettes, g)
    assert out["undefined"].mean() > 0.9


# ------------------------------------------------------------------ Z2, C phase
def _collinear_field(coords, core, winding, rng=None):
    th = np.arctan2(coords[:, 1] - core[1], coords[:, 0] - core[0])
    ang = winding * th / 2 if winding else np.zeros(len(coords))
    n = np.stack([np.cos(ang), np.sin(ang), np.zeros_like(ang)], 1)
    spins = np.stack([n, n, -n, -n], 1)
    if rng is not None:                               # director sign, labels and up/down flips
        sg = np.where(rng.random(len(coords)) < 0.5, -1.0, 1.0)
        spins = spins * sg[:, None, None]
        spins = shuffle_labels(spins, rng)
    return spins


def test_z2_vortex():
    rng = np.random.default_rng(5)
    lat = Lattice(2, L)
    coords = lat.coords.astype(float)
    uni = D.z2_vortices(_collinear_field(coords, CORE, 0, rng), lat.plaquettes)
    assert np.all(uni == 1)
    # a pi-winding of the director: continuous around the core except on the branch cut,
    # where n -> -n is the same physical state (director) -> exactly one half-vortex
    spins = _collinear_field(coords, CORE, 1, rng)
    z = D.z2_vortices(spins, lat.plaquettes)
    seam = (lat.coords[:, 0] == L - 1) | (lat.coords[:, 1] == L - 1)  # non-periodic field
    assert list(np.flatnonzero((z == -1) & ~seam)) == [lat.site_index(X0, Y0)]
    nd, lam = D.directors(spins)
    assert np.allclose(lam, 1.0)


# ----------------------------------------------------------- skyrmion, F phase
def test_skyrmion_charge():
    lat = Lattice(2, 24)
    c = lat.coords.astype(float) - 11.5
    r = np.hypot(c[:, 0], c[:, 1])
    phi = np.arctan2(c[:, 1], c[:, 0])
    theta = np.pi * np.clip(1 - r / 8.0, 0, 1)
    m = np.stack([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)], 1)
    for sign, Q in ((+1, 1), (-1, -1)):
        mm = m.copy()
        mm[:, 2] *= sign                               # flipping m_z flips the charge
        spins = np.repeat(mm[:, None, :], 4, axis=1)
        q, bad = D.skyrmion_density(spins, lat.plaquettes)
        assert not bad.any()
        assert q.sum() == pytest.approx(round(q.sum()), abs=1e-9)
        assert abs(round(q.sum())) == 1
    uni = np.tile(np.array([0, 0, 1.0]), (lat.N, 4, 1))
    assert np.allclose(D.skyrmion_density(uni, lat.plaquettes)[0], 0.0)


# ------------------------------------------------ follow-up 3: ambiguity margin and reporting
def _plaq_loop(k=1):
    return D.square_loop(L, X0, Y0, k)[None, :]


def _vortex_q(g, gi, axis=None):
    lat = Lattice(2, L)
    return D.vortex_orientations(lat.coords.astype(float), CORE, gi, g, axis_hint=axis)


def test_reportable_classes():
    gT = D.FrameGroup(reference("T"))
    assert sorted(gT.class_names[c] for c in D.reportable_classes(gT)) == ["1", "4", "6+", "6-"]
    gD = D.FrameGroup(reference("D"))
    assert sorted(gD.class_names[c] for c in D.reportable_classes(gD)) == ["1", "4+", "4-"]


def test_uniform_has_no_ambiguity():
    g = D.FrameGroup(reference("T"))
    lat = Lattice(3, 4)
    spins = np.tile(reference("T"), (lat.N, 1, 1))
    out = D.detect(spins, lat.plaquettes, g)
    assert out["reliable"].all() and out["link_ambiguous_fraction"] == 0.0
    c = D.census(out, g, lat.N)
    assert c["ambiguous"] == c["undefined"] == 0 and all(v == 0 for v in c["density"].values())


@pytest.mark.parametrize("name", ["T", "D"])
def test_reportable_vortices_are_unambiguous_on_one_plaquette(name):
    """Ideal 1, 4, 6+- vortices: link margins (>= 0.259) stay above the default threshold."""
    g = D.FrameGroup(reference(name))
    for c in D.reportable_classes(g):
        gi = g.classes[c][0]
        r = D.holonomy_full(_vortex_q(g, gi), _plaq_loop(1), g)
        assert r["cls"][0] == c and not r["ambiguous"][0], (g.class_names[c], r["margin"])


def test_2pi_vortex_is_not_reliable_on_one_plaquette():
    """2pi vortex (about 90 deg per link): in D the overlap (0.71) already marks it undefined;
    in T the overlap (0.93) passes the 0.9 cut and the vortex is mis-read as 6-, but the link
    margin (0.22 < 0.25) flags it ambiguous.  Without the margin it would contaminate 6-."""
    for name, expect_defined in (("T", True), ("D", False)):
        g = D.FrameGroup(reference(name))
        gi = [m[0] for m in g.classes if g.quats[m[0], 0] < -0.99][0]
        r = D.holonomy_full(_vortex_q(g, gi), _plaq_loop(1), g)
        assert (r["cls"][0] != D.UNDEFINED) == expect_defined
        assert r["ambiguous"][0] or r["cls"][0] == D.UNDEFINED
    # and the single-plaquette overlap cut alone does NOT catch it in T
    gT = D.FrameGroup(reference("T"))
    gi = [m[0] for m in gT.classes if gT.quats[m[0], 0] < -0.99][0]
    r = D.holonomy_full(_vortex_q(gT, gi), _plaq_loop(1), gT, margin_thr=0.0)
    assert r["cls"][0] != D.UNDEFINED and not r["ambiguous"][0]
    assert gT.class_names[r["cls"][0]] == "6-"


def test_2pi_vortex_axis_dependence_documented():
    """Intrinsic limit: a 2pi vortex winding about an axis near a C3 axis has link margins up
    to 0.259 -- the same as an ideal 6+- vortex -- so no margin threshold can flag it.  With
    the default 0.25, ~95% of random-axis 2pi vortices are flagged; the leak is always read as
    6+-.  Pins the numbers quoted in docs/M3_prep_report.md."""
    g = D.FrameGroup(reference("T"))
    gi = [m[0] for m in g.classes if g.quats[m[0], 0] < -0.99][0]
    lat = Lattice(2, L)
    rng = np.random.default_rng(0)
    axes = rng.normal(size=(1500, 3))
    axes /= np.linalg.norm(axes, axis=1)[:, None]
    flagged, leak = 0, set()
    for ax in axes:
        q = D.vortex_orientations(lat.coords.astype(float), CORE, gi, g, axis_hint=ax)
        r = D.holonomy_full(q, _plaq_loop(1), g)
        if r["ambiguous"][0] or r["cls"][0] == D.UNDEFINED:
            flagged += 1
        else:
            leak.add(g.class_names[r["cls"][0]])
    assert 0.92 < flagged / len(axes) < 0.97
    assert leak <= {"6+", "6-"}


def test_block_loops_resolve_the_big_classes():
    """k = 3 (6x6 block boundary) resolves every class of 2T unambiguously; k = 2 resolves
    2 and 3+- as to class but leaves 2 marginally ambiguous (margin 0.22)."""
    g = D.FrameGroup(reference("T"))
    for c, members in enumerate(g.classes):
        q = _vortex_q(g, members[0])
        r3 = D.holonomy_full(q, _plaq_loop(3), g)
        assert r3["cls"][0] == c and not r3["ambiguous"][0], g.class_names[c]
        r2 = D.holonomy_full(q, _plaq_loop(2), g)
        assert r2["cls"][0] == c
        if g.class_names[c] != "2":
            assert not r2["ambiguous"][0], g.class_names[c]


def test_block_loops_geometry_and_census():
    lat = Lattice(2, 8)
    assert np.array_equal(lat.block_loops(1), lat.plaquettes)
    b2 = lat.block_loops(2)
    assert b2.shape == (lat.N, 12)
    # consecutive loop sites are nearest neighbours (closed path)
    nb = {a: set(lat.nbr[a]) for a in range(lat.N)}
    assert all(b2[p, (i + 1) % 12] in nb[b2[p, i]] for p in range(0, lat.N, 7) for i in range(12))
    lat3 = Lattice(3, 4)
    assert lat3.block_loops(1).shape == (3 * lat3.N, 4)
    g = D.FrameGroup(reference("T"))
    spins = np.tile(reference("T"), (lat.N, 1, 1))
    out = D.detect(spins, lat.plaquettes, g)
    bc = D.block_census(out["q"], lat, g, k=2)
    assert bc["undefined"] == 0 and bc["ambiguous"] == 0
    assert sum(bc["counts"].values()) == bc["P"] == lat.N


def test_noise_ambiguity_fraction_is_reported():
    rng = np.random.default_rng(8)
    g = D.FrameGroup(reference("T"))
    lat = Lattice(2, 8)
    out = D.detect(random_spins((lat.N, 4), rng), lat.plaquettes, g)
    c = D.census(out, g, lat.N)
    assert c["undefined"] + c["ambiguous"] + int(out["reliable"].sum()) == lat.N
    assert c["link_ambiguous_fraction"] > 0.2
