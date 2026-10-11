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
    """Ideal 1, 4, 6+- vortices: link margins (>= 0.259) stay above the default threshold (0.15)."""
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
        r = D.holonomy_full(_vortex_q(g, gi), _plaq_loop(1), g, margin_thr=0.25)
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
        r = D.holonomy_full(q, _plaq_loop(1), g, margin_thr=0.25)
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
        r3 = D.holonomy_full(q, _plaq_loop(3), g, margin_thr=0.25)
        assert r3["cls"][0] == c and not r3["ambiguous"][0], g.class_names[c]
        r2 = D.holonomy_full(q, _plaq_loop(2), g, margin_thr=0.25)
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


# ------------------------------------------------------- follow-up 4: block statistics
def _random_q(lat, g, rng, noise):
    """Orientation quaternions: uniform reference plus random rotations of angle ~ noise."""
    ax = rng.normal(size=(lat.N, 3))
    ax /= np.linalg.norm(ax, axis=1)[:, None]
    ang = noise * rng.random(lat.N)
    return np.concatenate([np.cos(ang / 2)[:, None], np.sin(ang / 2)[:, None] * ax], 1)


@pytest.mark.parametrize("noise", [0.3, 1.2, 3.0])
def test_block_holonomy_is_product_of_plaquettes(noise):
    """Non-abelian Stokes: the element of a block-boundary loop equals the ordered, transported
    product of its interior plaquette holonomies (any q, any m x n rectangle, any plane)."""
    rng = np.random.default_rng(10)
    g = D.FrameGroup(reference("T"))
    for dim, planes in ((2, [(0, 1)]), (3, [(0, 1), (1, 2), (2, 0)])):
        lat = Lattice(dim, 8)
        q = _random_q(lat, g, rng, noise)
        h = D.link_elements(q, lat, g)
        # reversed links carry the inverse element
        for ax in range(dim):
            assert np.all(g.inverse[h[:, ax]] == h[lat.nbr[:, ax], ax + dim])
        for plane in planes:
            for m, n in ((1, 1), (2, 2), (3, 3), (2, 3), (5, 3)):
                a0 = int(rng.integers(lat.N))
                u, v = plane
                # explicit boundary loop of the m x n rectangle
                path = []
                s = a0
                for ax, cnt, sg in ((u, m, 1), (v, n, 1), (u, m, -1), (v, n, -1)):
                    for _ in range(cnt):
                        path.append(s)
                        s = lat.nbr[s, ax + (0 if sg > 0 else dim)]
                loop = np.array(path)[None, :]
                direct = D.holonomy_full(q, loop, g, match_thr=0.0, margin_thr=0.0)["elem"][0]
                assert D.block_from_plaquettes(lat, h, g, a0, m, n, plane) == direct, (dim, plane, m, n)


def test_misread_2pi_vortex_is_compensated_inside_the_block():
    """A 2pi vortex whose core plaquette is mis-read as 6- carries compensating fake defects in the
    surrounding plaquettes, and the k=2 block net class is the true one (2)."""
    g = D.FrameGroup(reference("T"))
    lat = Lattice(2, L)
    gi = [m[0] for m in g.classes if g.quats[m[0], 0] < -0.99][0]
    cls_2 = g.class_of[gi]
    q = _vortex_q(g, gi)
    h = D.link_elements(q, lat, g)
    core = lat.site_index(X0 - 1, Y0 - 1)               # lower-left site of the 4x4-site block
    inner = [lat.site_index(X0 - 1 + i, Y0 - 1 + j) for j in range(3) for i in range(3)]
    plaq = D.holonomy_full(q, lat.plaquettes[inner], g, match_thr=0.0, margin_thr=0.0)
    triv = g.class_of[g.identity]
    n_nontriv = int((plaq["cls"] != triv).sum())
    assert n_nontriv >= 2                                # core mis-read plus compensating ones
    block = D.holonomy_full(q, D.square_loop(L, X0, Y0, 2)[None, :], g, margin_thr=0.0)
    assert block["cls"][0] == cls_2
    assert D.block_from_plaquettes(lat, h, g, core, 3, 3) == block["elem"][0]


def test_block_census_counts_and_flags():
    g = D.FrameGroup(reference("T"))
    lat = Lattice(2, L)
    # uniform: quantitative, zero density
    q0 = np.tile([1.0, 0, 0, 0], (lat.N, 1))
    bc = D.block_census(q0, lat, g, k=2)
    assert bc["quantitative"] and bc["nontrivial_fraction"] == 0.0
    assert all(v == 0 for v in bc["density"].values())
    # isolated 6+ vortex: (2k-1)^2 = 9 overlapping blocks see it -> density 1/N (away from the seam)
    gi = [m[0] for m in g.classes if g.class_names[g.class_of[m[0]]] == "6+"][0]
    q = _vortex_q(g, gi)
    loops = lat.block_loops(2)
    r = D.holonomy_full(q, loops, g, margin_thr=D.MARGIN_THR_BLOCK)
    x, y = lat.coords[:, 0], lat.coords[:, 1]
    seam = (x == 0) | (x == L - 1) | (y == 0) | (y == L - 1)
    ok = ~seam[loops].any(1)
    c6 = g.class_names.index("6+")
    assert int(((r["cls"] == c6) & ~r["ambiguous"] & ok).sum()) == 9
    # high disorder: flagged not quantitative
    rng = np.random.default_rng(3)
    qn = _random_q(lat, g, rng, 3.0)
    bn = D.block_census(qn, lat, g, k=2)
    assert not bn["quantitative"] and bn["nontrivial_fraction"] > 0.1
    # 3D: line length per volume uses the same normalisation
    lat3 = Lattice(3, 4)
    b3 = D.block_census(np.tile([1.0, 0, 0, 0], (lat3.N, 1)), lat3, g, k=1)
    assert b3["P"] == 3 * lat3.N and b3["quantitative"]


def test_margin_scan_is_monotone():
    rng = np.random.default_rng(4)
    g = D.FrameGroup(reference("T"))
    lat = Lattice(2, 12)
    spins = D.spins_from_quats(_random_q(lat, g, rng, 0.9), g.ref)
    det = D.detect(spins, lat.plaquettes, g)
    sc = D.margin_scan(det, g, thresholds=(0.0, 0.15, 0.2, 0.25, 0.5))
    amb = [sc[t]["ambiguous"] for t in (0.0, 0.15, 0.2, 0.25, 0.5)]
    assert amb == sorted(amb) and amb[0] == 0
    for t, s in sc.items():
        assert s["ambiguous"] + sum(s["counts"].values()) + s["undefined"] == lat.N


def test_scan_detector_reproduces_detect_and_is_monotone():
    rng = np.random.default_rng(12)
    g = D.FrameGroup(reference("T"))
    lat = Lattice(2, 10)
    spins = D.spins_from_quats(_random_q(lat, g, rng, 1.0), g.ref)
    spins = spins + 0.12 * rng.normal(size=spins.shape)              # cluster shape noise
    spins /= np.linalg.norm(spins, axis=-1, keepdims=True)
    combos = ((0.35, 0.9, 0.15), (0.5, 0.9, 0.15), (0.7, 0.7, 0.15), (0.7, 0.5, 0.15))
    for rt, mt, gt in combos:
        det = D.detect(spins, lat.plaquettes, g, res_thr=rt, match_thr=mt, margin_thr=gt)
        sc = D.scan_detector(det, g, [(rt, mt, gt)])[0]
        nc = g.n_classes
        assert int(sc[nc]) == int(det["ambiguous"].sum()) and int(sc[nc + 1]) == int(det["undefined"].sum())
        for c in range(nc):
            assert int(sc[c]) == int(((det["cls"] == c) & det["reliable"]).sum())
    det = D.detect(spins, lat.plaquettes, g, res_thr=1.0, match_thr=0.0, margin_thr=0.0)
    und = [D.scan_detector(det, g, [c])[0][-1] for c in combos]
    assert und == sorted(und, reverse=True)                              # looser cuts -> fewer undefined


# ------------------------------------------------------- follow-up 6: coarse-grained detector
def test_coarse_detector():
    g = D.FrameGroup(reference("T"))
    lat = Lattice(2, 16)
    tab = D.octupole_table(g.ref, 20000)
    uni = D.coarse_detect(np.tile(g.ref, (lat.N, 1, 1)), lat, g, table=tab, refine=300)
    assert uni["reliable"].all() and np.all(uni["cls"] == g.class_of[g.identity])
    assert uni["coarse"].size == 8 and uni["quality"].min() > 0.99
    # a smooth 6+ vortex with core at the corner of four 2x2 blocks, with strong single-cluster shape noise
    gi = [m[0] for m in g.classes if g.class_names[g.class_of[m[0]]] == "6+"][0]
    q = D.vortex_orientations(lat.coords.astype(float), CORE, gi, g)
    sp = D.spins_from_quats(q, g.ref)
    rng = np.random.default_rng(0)
    sp = sp + 0.2 * rng.normal(size=sp.shape)
    sp /= np.linalg.norm(sp, axis=-1, keepdims=True)
    r = D.coarse_detect(sp, lat, g, table=tab, refine=300)
    c = r["coarse"]
    seam = (c.coords[c.plaquettes[:, 0], 0] == 7) | (c.coords[c.plaquettes[:, 0], 1] == 7)
    found = [(int(p), g.class_names[r["cls"][p]]) for p in np.flatnonzero((r["cls"] != g.class_of[g.identity]) & ~seam)]
    assert found == [(3 + 8 * 3, "6+")]
    with pytest.raises(ValueError):
        D.coarse_detect(np.tile(g.ref, (Lattice(2, 6).N, 1, 1)), Lattice(2, 6), g, block=4, table=tab)
