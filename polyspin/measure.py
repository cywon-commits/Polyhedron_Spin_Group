"""Per-temperature measurement: fast order parameters every sweep block and the heavier shape,
correlation and defect diagnostics every ``diag_every`` blocks (accumulated as sums)."""
import numpy as np
from numba import njit, prange

from . import defects as D
from . import observables as O


@njit(parallel=True, cache=True)
def m_l_all(spins, yl, ym, ysc, yc, yp):
    """m_1..m_4 of every replica: (R, 4)."""
    R, N, n, _ = spins.shape
    out = np.empty((R, 4))
    for r in prange(R):
        tot = O.multipole_sum(spins[r], yl, ym, ysc, yc, yp)
        for l in range(1, 5):
            s = 0.0
            for k in range(yl.shape[0]):
                if yl[k] == l:
                    s += tot[k] * tot[k]
            out[r, l - 1] = np.sqrt(s) * np.sqrt(4 * np.pi / (2 * l + 1)) / (N * n)
    return out


def measure_m(spins):
    return m_l_all(spins, O._YL, O._YM, O._YSC, O._YC, O._YP)


# detector-threshold scan recorded for every run: (fit-residual cut, link-overlap cut, margin cut)
# first four: residual-only sequence (overlap 0.9, margin 0.15) used by the pre-registered threshold rule
SCAN_COMBOS = ((0.25, 0.9, 0.15), (0.35, 0.9, 0.15), (0.5, 0.9, 0.15), (0.7, 0.9, 0.15),
               (0.35, 0.9, 0.25), (0.5, 0.7, 0.15), (0.7, 0.7, 0.15), (0.7, 0.5, 0.15))
RES_SEQUENCE = (0.25, 0.35, 0.5, 0.7)
BLOCK_SCAN_THR = SCAN_COMBOS   # kept for the HDF5 attribute name


class Diagnostics:
    """Sums over diagnostic samples, one slot per temperature index."""

    def __init__(self, lattice, params, R, cfg=None):
        cfg = dict(cfg or {})
        self.lat, self.R, self.n = lattice, R, params.n
        self.shape_thr = cfg.get("shape_thr", 0.95)
        self.do_corr = cfg.get("correlators", True)
        self.corr_nb = int(cfg.get("corr_blocks", 0))              # block-wise G_l(r) for error bars (0 = off)
        self.n_expected = max(int(cfg.get("n_expected", 1)), 1)       # diagnostic samples per temperature in the run
        dcfg = dict(cfg.get("defects", {}))
        self.margin_thr = dcfg.get("margin_thr", D.MARGIN_THR)
        self.res_thr = dcfg.get("res_thr", 0.35)
        self.match_thr = dcfg.get("match_thr", 0.9)
        self.block_k = list(dcfg.get("block_k", [2, 3]))
        self.z2_lam_thr = dcfg.get("z2_collinearity_thr", 0.5)
        # defect detection is the expensive part of a diagnostic sample; with stride s it is evaluated
        # only at temperature indices t % s == 0 (counted separately in n_def)
        self.defect_stride = int(dcfg.get("stride", 1))
        self.coarse_blocks = [b for b in dcfg.get("coarse_blocks", [2, 4]) if b]   # scales for R_b = rho(4)/rho(2)
        self.group = None
        ref = dcfg.get("holonomy_ref")
        if ref is not None:
            self.group = D.FrameGroup(np.asarray(ref, float))
        self.do_z2 = bool(dcfg.get("z2", False))
        self.do_sky = bool(dcfg.get("skyrmion", False)) and lattice.dim == 2
        L = lattice.size
        self.n_diag = np.zeros(R)
        self.n_def = np.zeros(R)
        self.split = np.zeros((R, 5))
        self.A_sum = np.zeros((R, 4))
        self.A_hist = np.zeros((R, 4, 32))
        self.M_hist = np.zeros((R, 40))
        self.lam_hist = np.zeros((R, 40))
        self.lam_sum = np.zeros(R)
        self.corr = np.zeros((R, 4, L // 2 + 1))
        if self.corr_nb:
            self.corr_blk = np.zeros((R, self.corr_nb, 4, L // 2 + 1))
            self.corr_blk_n = np.zeros((R, self.corr_nb))
        self.pair_edges = np.linspace(0, L * 0.75, 41)
        if self.group is not None:
            g = self.group
            nc = g.n_classes
            self.class_names = list(g.class_names)
            self.h_counts = np.zeros((R, nc))
            self.h_amb = np.zeros(R)
            self.h_und = np.zeros(R)
            self.h_linkamb = np.zeros(R)
            self.h_resid = np.zeros(R)
            self.h_scan = np.zeros((R, len(SCAN_COMBOS), nc + 2))   # classes, ambiguous, undefined
            self.h_resid_hist = np.zeros((R, 40))                    # per-cluster fit residual, [0, 1.2]
            self.b_counts = {k: np.zeros((R, nc)) for k in self.block_k}
            self.b_amb = {k: np.zeros(R) for k in self.block_k}
            self.b_und = {k: np.zeros(R) for k in self.block_k}
            self.b_frac = {k: np.zeros(R) for k in self.block_k}
            self.b_loops = {k: lattice.block_loops(k) for k in self.block_k}
            self.h_pair = np.zeros((R, 40))
            self.coarse_blocks = [b for b in self.coarse_blocks
                                  if lattice.size % b == 0 and lattice.size // b >= 3]
            self.do_coarse = bool(self.coarse_blocks)
            if self.do_coarse:
                self.coarse_table = D.octupole_table(g.ref, 20000)
                # scale b = 2 keeps the legacy names c_*, other scales use c<b>_*
                self.c = {b: dict(counts=np.zeros((R, nc)), amb=np.zeros(R), und=np.zeros(R), quality=np.zeros(R))
                          for b in self.coarse_blocks}
        if self.do_z2:
            self.z2_count = np.zeros(R)
            self.z2_und = np.zeros(R)
            self.z2_pair = np.zeros((R, 40))
        if self.do_sky:
            self.sky_Q = np.zeros((R, 3))     # sum Q, sum Q^2, sum |Q|
            self.sky_und = np.zeros(R)
            self.sky_abs_density = np.zeros(R)

    # ---------------------------------------------------------------- helpers
    def _pair_hist(self, plaq_idx):
        """Minimum-image distance histogram between plaquettes (2D only)."""
        lat = self.lat
        if lat.dim != 2 or len(plaq_idx) < 2 or len(plaq_idx) > 300:
            return np.zeros(40)
        pos = lat.coords[lat.plaquettes[plaq_idx, 0]].astype(float)
        d = pos[:, None, :] - pos[None, :, :]
        d = (d + lat.size / 2) % lat.size - lat.size / 2
        r = np.sqrt((d ** 2).sum(-1))[np.triu_indices(len(pos), 1)]
        return np.histogram(r, bins=self.pair_edges)[0].astype(float)

    # ---------------------------------------------------------------- sample
    def add(self, t, spins):
        self.n_diag[t] += 1
        ng, big, A, Mabs, lam = O.cluster_shape_arrays(spins, self.shape_thr)
        self.split[t] += O.split_histogram(ng, big)
        self.A_sum[t] += A.mean(0)
        for l in range(4):
            self.A_hist[t, l] += np.histogram(A[:, l], bins=32, range=(0, self.n ** 2))[0] / len(A)
        self.M_hist[t] += np.histogram(Mabs, bins=40, range=(0, 1))[0] / len(A)
        self.lam_hist[t] += np.histogram(lam, bins=40, range=(0, 1))[0] / len(A)
        self.lam_sum[t] += lam.mean()
        if self.do_corr:
            c = O.correlators(spins, self.lat)
            self.corr[t] += c
            if self.corr_nb:
                b = min(int((self.n_diag[t] - 1) * self.corr_nb / self.n_expected), self.corr_nb - 1)
                self.corr_blk[t, b] += c
                self.corr_blk_n[t, b] += 1
        lat = self.lat
        if t % self.defect_stride:
            return
        self.n_def[t] += 1
        if self.group is not None:
            g = self.group
            det = D.detect(spins, lat.plaquettes, g, self.res_thr, self.match_thr, self.margin_thr)
            rel, cls = det["reliable"], det["cls"]
            for c in range(g.n_classes):
                self.h_counts[t, c] += int(((cls == c) & rel).sum())
            self.h_amb[t] += det["ambiguous"].sum()
            self.h_und[t] += det["undefined"].sum()
            self.h_linkamb[t] += det["link_ambiguous_fraction"]
            self.h_resid[t] += det["residual"].mean()
            self.h_scan[t] += D.scan_detector(det, g, SCAN_COMBOS)
            self.h_resid_hist[t] += np.histogram(det["residual"], bins=40, range=(0, 1.2))[0] / lat.N
            triv = g.class_of[g.identity]
            nontriv = np.flatnonzero(rel & (cls != triv) & np.isin(cls, D.reportable_classes(g)))
            self.h_pair[t] += self._pair_hist(nontriv)
            for k in self.block_k:
                bc = D.block_census(det["q"], lat, g, k=k, match_thr=self.match_thr)
                self.b_counts[k][t] += [bc["counts"][nm] for nm in g.class_names]
                self.b_amb[k][t] += bc["ambiguous"]
                self.b_und[k][t] += bc["undefined"]
                self.b_frac[k][t] += bc["nontrivial_fraction"]
        if self.group is not None and self.do_coarse:
            for b in self.coarse_blocks:
                cd = D.coarse_detect(spins, lat, self.group, b, self.coarse_table,
                                     self.match_thr, self.margin_thr, refine=200)
                a = self.c[b]
                for c in range(self.group.n_classes):
                    a["counts"][t, c] += int(((cd["cls"] == c) & cd["reliable"]).sum())
                a["amb"][t] += cd["ambiguous"].sum()
                a["und"][t] += cd["undefined"].sum()
                a["quality"][t] += cd["quality"].mean()
        if self.do_z2:
            z = D.z2_vortices(spins, lat.plaquettes, self.z2_lam_thr)
            self.z2_count[t] += int((z == -1).sum())
            self.z2_und[t] += int((z == 0).sum())
            self.z2_pair[t] += self._pair_hist(np.flatnonzero(z == -1))
        if self.do_sky:
            q, bad = D.skyrmion_density(spins, lat.plaquettes)
            Q = q.sum()
            self.sky_Q[t] += (Q, Q * Q, abs(Q))
            self.sky_und[t] += bad.sum()
            self.sky_abs_density[t] += np.abs(q).sum() / lat.N

    # ---------------------------------------------------------------- io
    def arrays(self):
        out = {k: getattr(self, k) for k in ("n_diag", "n_def", "split", "A_sum", "A_hist", "M_hist",
                                             "lam_hist", "lam_sum", "corr")}
        if self.corr_nb:
            out.update(corr_blk=self.corr_blk, corr_blk_n=self.corr_blk_n)
        if self.group is not None:
            out.update(h_counts=self.h_counts, h_amb=self.h_amb, h_und=self.h_und, h_linkamb=self.h_linkamb,
                       h_resid=self.h_resid, h_scan=self.h_scan, h_pair=self.h_pair,
                       h_resid_hist=self.h_resid_hist)
            for k in self.block_k:
                out[f"b{k}_counts"], out[f"b{k}_amb"] = self.b_counts[k], self.b_amb[k]
                out[f"b{k}_und"], out[f"b{k}_frac"] = self.b_und[k], self.b_frac[k]
        if self.group is not None and getattr(self, "do_coarse", False):
            for b, a in self.c.items():
                p = "c" if b == 2 else f"c{b}"
                out.update({f"{p}_counts": a["counts"], f"{p}_amb": a["amb"], f"{p}_und": a["und"],
                            f"{p}_quality": a["quality"]})
        if self.do_z2:
            out.update(z2_count=self.z2_count, z2_und=self.z2_und, z2_pair=self.z2_pair)
        if self.do_sky:
            out.update(sky_Q=self.sky_Q, sky_und=self.sky_und, sky_abs_density=self.sky_abs_density)
        return out

    def load_arrays(self, d):
        for k, v in self.arrays().items():
            v[...] = d["diag_" + k]
