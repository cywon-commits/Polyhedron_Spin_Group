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

    def nearest(self, q):
        """Index of nearest 2G element for quaternions (...,4); returns (idx, overlap)."""
        dots = q @ self.quats.T
        idx = dots.argmax(-1)
        return idx, np.take_along_axis(dots, idx[..., None], -1)[..., 0]

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


def holonomy(q, loops, group: FrameGroup, match_thr=0.9):
    """Conjugacy class id of the holonomy around each closed site loop.

    q      (N,4) orientation quaternions (sign arbitrary)
    loops  (P,m) site indices; the loop closes from the last site to the first
    Returns (class_id (P,), min link overlap (P,)); class_id = UNDEFINED where any
    link overlap < match_thr.
    """
    nxt = np.roll(loops, -1, axis=1)
    rel = qmul(qconj(q[loops]), q[nxt])            # (P,m,4)
    h, ov = group.nearest(rel)                      # (P,m)
    acc = h[:, 0].copy()
    for k in range(1, loops.shape[1]):
        acc = group.mult[acc, h[:, k]]
    cls = group.class_of[acc]
    minov = ov.min(1)
    cls = np.where(minov < match_thr, UNDEFINED, cls)
    return cls, minov


def detect(spins, plaquettes, group: FrameGroup, res_thr=0.35, match_thr=0.9):
    """Full pipeline on a spin configuration.

    Returns dict(cls (P,), undefined (P,) bool, residual (N,), overlap (P,), q (N,4)).
    A plaquette is undefined if one of its clusters has fit residual > res_thr or any
    link overlap < match_thr.
    """
    R, resid, _ = fit_orientations(spins, group)
    q = mat_to_quat(R)
    cls, ov = holonomy(q, plaquettes, group, match_thr)
    bad = (resid[plaquettes] > res_thr).any(1)
    cls = np.where(bad, UNDEFINED, cls)
    return dict(cls=cls, undefined=cls == UNDEFINED, residual=resid, overlap=ov, q=q)


def census(cls, group: FrameGroup, n_sites=None):
    """Counts per class, number undefined, and densities per site (3D: line length per
    volume; 2D: defects per site)."""
    out = {"undefined": int((cls == UNDEFINED).sum())}
    for c, name in enumerate(group.class_names):
        out[name] = int((cls == c).sum())
    out["trivial"] = int((cls == group.class_of[group.identity]).sum())
    out["P"] = len(cls)
    if n_sites:
        out["density"] = {k: v / n_sites for k, v in out.items() if k not in ("P", "density")}
    return out


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
