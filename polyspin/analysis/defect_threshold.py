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
