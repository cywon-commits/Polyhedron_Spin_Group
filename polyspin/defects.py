"""Topological-defect detection.

General holonomy detector (T, U, D phases): given a reference cluster
configuration e (n,3) it builds the point group G of proper rotations that map
e to itself as a multiset, lifts it to the unit-quaternion group 2G, fits an
orientation q_a in SU(2) to every cluster, rounds each link q_a^-1 q_b to the
nearest element of 2G and multiplies the four link elements round every
plaquette.  The conjugacy class of the product is the defect type.

Also: Z2 half-vortices of the collinear (C) phase and Berg-Luescher skyrmion
charge of the ferromagnetic (F) phase.

Limitation (see docs/M2_report.md): a plaquette link is rounded to the nearest
group element, which is unambiguous only while a link step is below the
Voronoi radius of 2G.  Classes whose minimal SU(2) rotation angle exceeds
~200 degrees (-1 and the two order-3 classes of 2T) therefore cannot be
resolved on a single plaquette; they show up as 'undefined' there and need a
larger loop (``holonomy`` accepts loops of any length).
"""
from itertools import permutations

import numpy as np
from scipy.spatial.transform import Rotation

UNDEFINED = -1


# --------------------------------------------------------------- quaternions
def qmul(a, b):
    """Hamilton product, (w,x,y,z), broadcasting over leading axes."""
    aw, ax, ay, az = np.moveaxis(a, -1, 0)
    bw, bx, by, bz = np.moveaxis(b, -1, 0)
    return np.stack([aw * bw - ax * bx - ay * by - az * bz,
                     aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw], axis=-1)


def qconj(a):
    return a * np.array([1.0, -1.0, -1.0, -1.0])


def mat_to_quat(R):
    """Rotation matrices (...,3,3) -> unit quaternions (...,4) as (w,x,y,z)."""
    q = Rotation.from_matrix(R.reshape(-1, 3, 3)).as_quat()  # (x,y,z,w)
    q = np.concatenate([q[:, 3:], q[:, :3]], axis=1)
    return q.reshape(R.shape[:-2] + (4,))


def quat_to_mat(q):
    w, x, y, z = np.moveaxis(q, -1, 0)
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
        np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
        np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1)], -2)


# ------------------------------------------------------------- Kabsch fitting
def _perms(n):
    return np.array(list(permutations(range(n))), dtype=np.int64)


def kabsch_all_perms(spins, ref, perms):
    """For every cluster and label permutation the proper rotation R minimising
    sum_i |S[perm_i] - R e_i|^2.  Returns R (N,P,3,3), rms residual (N,P)."""
    Sp = spins[:, perms, :]                       # (N,P,n,3)
    cov = np.einsum("ik,npil->npkl", ref, Sp)     # sum_i e_i (x) s_i
    U, _s, Vt = np.linalg.svd(cov)
    V = np.swapaxes(Vt, -1, -2)
    d = np.sign(np.linalg.det(V @ np.swapaxes(U, -1, -2)))
    d[d == 0] = 1.0
    D = np.zeros(d.shape + (3, 3))
    D[..., 0, 0] = 1.0
    D[..., 1, 1] = 1.0
    D[..., 2, 2] = d
    R = V @ D @ np.swapaxes(U, -1, -2)
    diff = Sp - np.einsum("npkl,il->npik", R, ref)
    res = np.sqrt((diff ** 2).sum(-1).mean(-1))
    return R, res


class FrameGroup:
    """Reference configuration, its symmetry group G and the lift 2G."""

    def __init__(self, ref, tol=1e-6, naming_tol=1e-6):
        ref = np.asarray(ref, float)
        self.ref = ref
        self.n = len(ref)
        if np.linalg.matrix_rank(ref, tol=1e-8) < 2:
            raise ValueError("reference configuration is (anti)collinear: use the C/F detectors")
        self.perms = _perms(self.n)
        R, res = kabsch_all_perms(ref[None], ref, self.perms)
        mats = R[0][res[0] < tol]
        uniq = []
        for M in mats:
            if not any(np.allclose(M, U, atol=1e-6) for U in uniq):
                uniq.append(M)
        self.rotations = np.array(uniq)
        q = mat_to_quat(self.rotations)
        self.quats = np.concatenate([q, -q])      # 2G, shape (2|G|,4)
        self.order_G = len(self.rotations)
        self._build_tables(naming_tol)

    def nearest(self, q, margin=False):
        """Index of nearest 2G element for quaternions (...,4); returns (idx, overlap)
        or, with margin=True, (idx, overlap, margin) where margin = best overlap minus the
        runner-up (a different 2G element)."""
        dots = q @ self.quats.T
        idx = dots.argmax(-1)
        best = np.take_along_axis(dots, idx[..., None], -1)[..., 0]
        if not margin:
            return idx, best
        part = np.partition(dots, -2, axis=-1)
        return idx, best, best - part[..., -2]

    def _build_tables(self, tol):
        m = len(self.quats)
        prod = qmul(self.quats[:, None, :], self.quats[None, :, :])
        self.mult, ov = self.nearest(prod)
        assert np.all(ov > 1 - 1e-8), "2G is not closed under multiplication"
        self.identity = int(self.nearest(np.array([1.0, 0, 0, 0]))[0])
        self.inverse = self.nearest(qconj(self.quats))[0]
        # conjugacy classes
        cls = -np.ones(m, dtype=np.int64)
        classes = []
        for i in range(m):
            if cls[i] >= 0:
                continue
            members = {int(self.mult[self.mult[g, i], self.inverse[g]]) for g in range(m)}
            for k in members:
                cls[k] = len(classes)
            classes.append(sorted(members))
        self.class_of = cls
        self.classes = classes
        self.class_names = [self._name(c) for c in classes]
        assert len(set(self.class_names)) == len(self.class_names), self.class_names

    def element_order(self, i):
        k, x = 1, i
        while x != self.identity:
            x = self.mult[x, i]
            k += 1
        return k

    def _polarity(self, v):
        """Sign that is invariant under conjugation by G: net moment if present,
        otherwise the cubic invariant sum_i (v.e_i)^3 of the axis."""
        nv = np.linalg.norm(v)
        if nv < 1e-9:
            return 0
        v = v / nv
        M = self.ref.sum(0)
        s = float(v @ M)
        if abs(s) < 1e-6:
            s = float(((self.ref @ v) ** 3).sum())
        return 0 if abs(s) < 1e-6 else int(np.sign(s))

    def _name(self, members):
        i = members[0]
        q = self.quats[i]
        name = str(self.element_order(i))
        s = self._polarity(q[1:])
        return name + ("+" if s > 0 else "-" if s < 0 else "")

    @property
    def n_classes(self):
        return len(self.classes)

    def rotation_angle(self, i):
        """SU(2) angle alpha in [0, 2 pi] of element i (q = cos(alpha/2) + ...)."""
        return 2 * np.arccos(np.clip(self.quats[i, 0], -1, 1))


# -------------------------------------------------------------- orientations
def fit_orientations(spins, group: FrameGroup):
    """Per-cluster best proper rotation R_a (N,3,3), rms residual (N,) and chosen perm."""
    R, res = kabsch_all_perms(spins, group.ref, group.perms)
    best = res.argmin(1)
    ar = np.arange(len(spins))
    return R[ar, best], res[ar, best], best


MARGIN_THR = 0.25  # see docs/M3_prep_report.md for the trade-off behind this default


def holonomy_full(q, loops, group: FrameGroup, match_thr=0.9, margin_thr=MARGIN_THR):
    """Holonomy classes with reliability information.

    q      (N,4) orientation quaternions (sign arbitrary)
    loops  (P,m) site indices; the loop closes from the last site to the first
    Returns dict:
      cls        (P,) conjugacy-class id, UNDEFINED where any link overlap < match_thr
      overlap    (P,) minimum link overlap (best element)
      margin     (P,) minimum link margin (best minus runner-up overlap)
      ambiguous  (P,) bool, margin < margin_thr (and not undefined): rounding of at least one
                 link is not decisive, the class may be wrong
      link_margin (P,m) per-link margins
    """
    nxt = np.roll(loops, -1, axis=1)
    rel = qmul(qconj(q[loops]), q[nxt])            # (P,m,4)
    h, ov, mg = group.nearest(rel, margin=True)    # (P,m)
    acc = h[:, 0].copy()
    for k in range(1, loops.shape[1]):
        acc = group.mult[acc, h[:, k]]
    cls = group.class_of[acc]
    minov, minmg = ov.min(1), mg.min(1)
    undefined = minov < match_thr
    cls = np.where(undefined, UNDEFINED, cls)
    return dict(cls=cls, overlap=minov, margin=minmg,
                ambiguous=(minmg < margin_thr) & ~undefined, link_margin=mg)


def holonomy(q, loops, group: FrameGroup, match_thr=0.9):
    """(class id, min link overlap); see ``holonomy_full`` for margins and ambiguity."""
    r = holonomy_full(q, loops, group, match_thr)
    return r["cls"], r["overlap"]


def detect(spins, plaquettes, group: FrameGroup, res_thr=0.35, match_thr=0.9,
           margin_thr=MARGIN_THR, loops=None):
    """Full pipeline on a spin configuration.

    Every loop (default: the plaquettes) ends up in exactly one of three states:
      undefined  fit residual of one of its clusters > res_thr or a link overlap < match_thr
      ambiguous  defined, but a link margin < margin_thr (class unreliable)
      reliable   everything else
    Returns dict(cls (P,), undefined, ambiguous, reliable (P,) bool, residual (N,),
    overlap (P,), margin (P,), link_ambiguous_fraction, q (N,4)).
    """
    loops = plaquettes if loops is None else loops
    R, resid, _ = fit_orientations(spins, group)
    q = mat_to_quat(R)
    r = holonomy_full(q, loops, group, match_thr, margin_thr)
    bad = (resid[loops] > res_thr).any(1)
    cls = np.where(bad, UNDEFINED, r["cls"])
    undefined = cls == UNDEFINED
    ambiguous = r["ambiguous"] & ~undefined
    return dict(cls=cls, undefined=undefined, ambiguous=ambiguous, reliable=~(undefined | ambiguous),
                residual=resid, overlap=r["overlap"], margin=r["margin"], q=q,
                link_ambiguous_fraction=float((r["link_margin"] < margin_thr).mean()))


def reportable_classes(group: FrameGroup, max_angle_deg=180.0):
    """Class ids whose SU(2) rotation angle is <= max_angle_deg.  Only these are resolvable on
    a single plaquette (link step < ~60 deg): 1, 4, 6+- for 2T; 1, 4+- for Z4.  The 2 (2pi) and
    3+- (240 deg) classes of 2T are not."""
    return [c for c, m in enumerate(group.classes)
            if np.degrees(group.rotation_angle(m[0])) <= max_angle_deg + 1e-6]


def census(det, group: FrameGroup, n_sites=None):
    """Defect census of a ``detect`` result.

    Reliable plaquettes are counted per class (``counts``); ``reportable`` lists the classes
    that may be quoted (1, 4, 6+- for 2T), the others are kept for completeness but must not be
    used as densities.  ``undefined`` and ``ambiguous`` are counted separately.  Densities are
    per site (2D: defects per site; 3D: defect-line length per volume, one unit per plaquette).
    """
    cls, rel = det["cls"], det["reliable"]
    counts = {name: int(((cls == c) & rel).sum()) for c, name in enumerate(group.class_names)}
    rep = [group.class_names[c] for c in reportable_classes(group)]
    triv = group.class_names[group.class_of[group.identity]]
    out = dict(counts=counts, reportable=rep, undefined=int(det["undefined"].sum()),
               ambiguous=int(det["ambiguous"].sum()), P=len(cls),
               link_ambiguous_fraction=det.get("link_ambiguous_fraction"))
    if n_sites:
        out["density"] = {k: v / n_sites for k, v in counts.items() if k != triv}
    return out


def block_census(q, lattice, group: FrameGroup, k=2, **kw):
    """Class distribution of the 2k x 2k block-boundary loops (diagnostic of how finely big
    defects are split into small ones when compared with the plaquette census)."""
    loops = lattice.block_loops(k)
    r = holonomy_full(q, loops, group, **kw)
    rel = ~(r["cls"] == UNDEFINED) & ~r["ambiguous"]
    counts = {name: int(((r["cls"] == c) & rel).sum()) for c, name in enumerate(group.class_names)}
    return dict(counts=counts, undefined=int((r["cls"] == UNDEFINED).sum()),
                ambiguous=int(r["ambiguous"].sum()), P=len(loops))


# ------------------------------------------------------- artificial vortices
def vortex_orientations(coords, core, g_index, group: FrameGroup, axis_hint=None, Q0=None):
    """Orientation quaternions of a continuous vortex: q(t) = exp(-t log g), t = theta/2pi,
    theta the polar angle around ``core`` (counter-clockwise).  For g = -1 the rotation
    axis is ``axis_hint`` (generic by default)."""
    g = group.quats[g_index]
    w = np.clip(g[0], -1, 1)
    half = np.arccos(w)
    v = g[1:]
    if np.linalg.norm(v) < 1e-9:
        axis = np.array([0.31, 0.52, 0.79]) if axis_hint is None else np.asarray(axis_hint, float)
        axis = axis / np.linalg.norm(axis)
        half = np.pi if w < 0 else 0.0
    else:
        axis = v / np.linalg.norm(v)
    th = np.arctan2(coords[:, 1] - core[1], coords[:, 0] - core[0]) % (2 * np.pi)
    t = th / (2 * np.pi)
    ang = -t * half
    q = np.concatenate([np.cos(ang)[:, None], np.sin(ang)[:, None] * axis[None, :]], axis=1)
    if Q0 is not None:
        q = qmul(Q0[None, :], q)
    return q


def spins_from_quats(q, ref):
    """Cluster spins R(q_a) e_i."""
    R = quat_to_mat(q)
    return np.einsum("nkl,il->nik", R, ref)


def square_loop(L, x0, y0, k):
    """Site indices of the boundary of the (2k x 2k)-site block whose centre is the
    plaquette with lower-left site (x0,y0), counter-clockwise; periodic L x L lattice."""
    xs = np.arange(x0 - k + 1, x0 + k + 1)
    ys = np.arange(y0 - k + 1, y0 + k + 1)
    path = ([(x, ys[0]) for x in xs[:-1]] + [(xs[-1], y) for y in ys[:-1]]
            + [(x, ys[-1]) for x in xs[::-1][:-1]] + [(xs[0], y) for y in ys[::-1][:-1]])
    return np.array([(x % L) + L * (y % L) for x, y in path], dtype=np.int64)


# ---------------------------------------------------------------- C phase: Z2
def directors(spins):
    """Principal axis n_a of the quadrupole tensor Q_a = sum_i (S S^T - I/3) and the
    collinearity lambda_max / (2n/3) in [0,1]."""
    n = spins.shape[1]
    Q = np.einsum("aik,ail->akl", spins, spins) - n / 3 * np.eye(3)
    w, v = np.linalg.eigh(Q)
    return v[:, :, -1], w[:, -1] / (2 * n / 3)


def z2_vortices(spins, plaquettes, collinearity_thr=0.0):
    """+1 normal, -1 half-vortex, 0 undefined (cluster not collinear enough)."""
    nd, lam = directors(spins)
    nxt = np.roll(plaquettes, -1, axis=1)
    sgn = np.sign((nd[plaquettes] * nd[nxt]).sum(-1))
    sgn[sgn == 0] = 1
    out = sgn.prod(1)
    bad = (lam[plaquettes] < collinearity_thr).any(1)
    return np.where(bad, 0, out).astype(np.int64)


# ------------------------------------------------------------- F phase: skyrmion
def _solid_angle(n1, n2, n3):
    num = (n1 * np.cross(n2, n3)).sum(-1)
    den = 1 + (n1 * n2).sum(-1) + (n2 * n3).sum(-1) + (n3 * n1).sum(-1)
    return 2 * np.arctan2(num, den)


def skyrmion_density(spins, plaquettes, mag_thr=1e-3):
    """Berg-Luescher topological charge per plaquette (two triangles a,b,c / a,c,d) of the
    cluster magnetisation direction (2D).  Returns (charge (P,), undefined (P,))."""
    M = spins.sum(1)
    nm = np.linalg.norm(M, axis=1)
    m = M / np.maximum(nm, 1e-300)[:, None]
    a, b, c, d = (m[plaquettes[:, k]] for k in range(4))
    q = (_solid_angle(a, b, c) + _solid_angle(a, c, d)) / (4 * np.pi)
    bad = (nm[plaquettes] < mag_thr).any(1)
    return np.where(bad, 0.0, q), bad
