"""Pre-registered rule for the defect-detector residual threshold and the H2 verdict logic.

Rule (follow-up 6.3): among the residual thresholds 0.25, 0.35, 0.5, 0.7 (link overlap 0.9, margin 0.15)
take the smallest one for which the reliable class densities change by <= 10% when going to the next
larger threshold, at every temperature where the density is statistically meaningful.  Chosen once from
the first production run and then frozen for all points.
H2 verdict: the defect density rho_def(T) and the undefined fraction f_ud(T) are judged together.
"""
import numpy as np

from ..measure import RES_SEQUENCE, SCAN_COMBOS


def residual_sequence_density(run_diag, n_sites, reportable_idx, n_diag):
    """Total reliable density of the reportable (nontrivial) classes per site for the residual-only
    sequence.  run_diag: dict of the HDF5 diag arrays; returns (R, len(RES_SEQUENCE)) and the undefined
    fraction (R, len(RES_SEQUENCE))."""
    scan = run_diag["h_scan"]                         # (R, combos, nc + 2)
    nd = np.maximum(run_diag["n_def"] if "n_def" in run_diag else n_diag, 1)[:, None, None]
    nc = scan.shape[2] - 2
    idx = [SCAN_COMBOS.index((r, 0.9, 0.15)) for r in RES_SEQUENCE]
    dens = scan[:, idx][:, :, reportable_idx].sum(-1) / nd[:, :, 0] / n_sites
    und = scan[:, idx, nc + 1] / nd[:, :, 0]
    return dens, und


def choose_residual_threshold(dens, counts_floor=None, tol=0.10, T_mask=None):
    """dens (R, n_thr): densities.  Returns the smallest threshold index whose density changes <= tol
    relative to the next larger threshold at every temperature in T_mask with dens above the floor.
    Returns None if no threshold satisfies the rule."""
    R, n = dens.shape
    mask = np.ones(R, bool) if T_mask is None else np.asarray(T_mask, bool)
    floor = 0.0 if counts_floor is None else counts_floor
    for i in range(n - 1):
        a, b = dens[mask, i], dens[mask, i + 1]
        use = (np.maximum(a, b) > floor)
        if not use.any():
            continue
        rel = np.abs(b[use] - a[use]) / np.maximum(np.maximum(a[use], b[use]), 1e-30)
        if np.all(rel <= tol):
            return i
    return None


def h2_verdict(T, rho, f_ud, rise_frac=0.5, ud_limit=0.10):
    """T, rho_def(T), f_ud(T) -> ('defect-mediated' | 'amplitude (frame melts first)' | 'undetermined', T_rise).
    T_rise is the lowest temperature at which rho reaches ``rise_frac`` of its maximum."""
    T, rho, f_ud = map(np.asarray, (T, rho, f_ud))
    ok = np.isfinite(rho)
    if not ok.any() or np.nanmax(rho) <= 0:
        return "undetermined", float("nan")
    k = int(np.flatnonzero(ok & (rho >= rise_frac * np.nanmax(rho)))[0])
    if f_ud[k] < ud_limit:
        return "defect-mediated", float(T[k])
    return "amplitude (frame melts first)", float(T[k])


# ---------------------------------------------------------------------------------------------
# H2 (2D) pre-registered scale-ratio rule (M4 tier-3 request, decision 1; thresholds fixed before L = 32).
RB_FREE_LOW, RB_FREE_HIGH, RB_NONE_HIGH, RB_UD_LIMIT = 0.4, 0.7, 0.5, 0.10


def _interp(T, y, t):
    T, y = np.asarray(T, float), np.asarray(y, float)
    ok = np.isfinite(y)
    T, y = T[ok], y[ok]
    if len(T) < 2 or t < T.min() or t > T.max():
        return float("nan")
    return float(np.interp(t, T, y))


def rb_at(T, rb, f_ud4, Tc, fac):
    """R_b and the b = 4 f_ud at fac * Tc by linear interpolation between neighbouring measured temperatures
    (only finite points are used).  Returns (R_b, f_ud, usable) with usable = f_ud < 0.1 and R_b finite."""
    t = fac * Tc
    r, f = _interp(T, rb, t), _interp(T, f_ud4, t)
    return r, f, bool(np.isfinite(r) and np.isfinite(f) and f < RB_UD_LIMIT)


def rb_verdict_one(T, rb, f_ud4, Tc):
    """Per-size verdict of the R_b rule: 'free' | 'none' | 'undetermined' plus the numbers used.  A value taken
    where the b = 4 f_ud is >= 0.1 is not used (recorded in ``usable``); with an unusable value the size's
    verdict is 'undetermined' unless the other value already decides 'none'."""
    r9, f9, u9 = rb_at(T, rb, f_ud4, Tc, 0.9)
    r11, f11, u11 = rb_at(T, rb, f_ud4, Tc, 1.1)
    if u11 and r11 < RB_NONE_HIGH:
        v = "none"
    elif u9 and u11 and r9 <= RB_FREE_LOW and r11 >= RB_FREE_HIGH:
        v = "free"
    else:
        v = "undetermined"
    return dict(verdict=v, Tc=float(Tc), R09=r9, f09=f9, usable09=u9, R11=r11, f11=f11, usable11=u11)


def rb_verdict(per_size):
    """Combine sizes: 'free' / 'none' only if every size gives it; otherwise 'undetermined'.
    ``per_size``: {L: rb_verdict_one(...)}."""
    vs = {d["verdict"] for d in per_size.values()}
    return vs.pop() if len(vs) == 1 else "undetermined"
