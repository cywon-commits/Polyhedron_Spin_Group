"""Decision rules for the 2D stiffness (helicity modulus) measurement (docs/CD_rules.md, fixed before the runs).

Point A (T phase, cut A zG = 1.0, 2D L = 16, 24, 32, axis-averaged Y at >= 3 temperatures in 0.6-1.0 T_peak):
    * 'asymptotic-freedom crossover supported': at EVERY temperature Y(32) < Y(16) by more than Z sigma AND the decrease is roughly linear in ln L
      there (slopes s1 = dY/dlnL over 16 -> 24 and s2 over 24 -> 32 both negative and either within Z combined sigma of each other or with ratio in [1/2, 2]);
    * 'quasi-long-range-order candidate': an L-independent plateau, i.e. >= 2 consecutive temperatures at which Y(16), Y(24), Y(32) agree pairwise within Z sigma;
    * otherwise 'undecidable'.
Point B (cut E' zG = 1.0, director stiffness Y_par, L = 16, 24, 32): T*(L) = temperature at which Y_par(T) crosses the line 2T/pi from above (linear interpolation);
    * 'KT-type crossover consistent': T*(L) exists for all three sizes and the crossing becomes steeper with L: |slope|(16) < |slope|(24) < |slope|(32) and
      |slope|(32) - |slope|(16) > Z combined sigma of the two; slope = d(Y_par - 2T/pi)/dT over the bracketing temperature pair;
    * 'no KT-type behavior': Y_par(32) < Y_par(16) by more than Z sigma at EVERY measured temperature;
    * both satisfied -> 'undecidable (both)'; neither -> 'undecidable'.
Z = 2.  Errors are 16-84% percentile half-widths.
"""
import numpy as np

Z = 2.0
LS = (16, 24, 32)


def _arr(d):
    return {L: np.asarray(d[L], float) for L in d}


def verdict_A(ups, err, z=Z):
    """ups, err: dict L -> array over the same increasing temperatures (>= 3).  Returns (verdict, details)."""
    ups, err = _arr(ups), _arr(err)
    nT = len(ups[16])
    if nT < 3:
        return "undecidable", dict(reason="fewer than 3 temperatures")
    dec = (ups[16] - ups[32]) > z * np.hypot(err[16], err[32])
    s1 = (ups[24] - ups[16]) / np.log(24 / 16); e1 = np.hypot(err[24], err[16]) / np.log(24 / 16)
    s2 = (ups[32] - ups[24]) / np.log(32 / 24); e2 = np.hypot(err[32], err[24]) / np.log(32 / 24)
    both_neg = (s1 < 0) & (s2 < 0)
    consistent = np.abs(s1 - s2) <= z * np.hypot(e1, e2)
    ratio_ok = both_neg & (np.abs(s2) / np.maximum(np.abs(s1), 1e-300) >= 0.5) & (np.abs(s2) / np.maximum(np.abs(s1), 1e-300) <= 2.0)
    linear = both_neg & (consistent | ratio_ok)
    af = bool(dec.all() and linear.all())
    agree = np.ones(nT, bool)
    for a, b in ((16, 24), (24, 32), (16, 32)):
        agree &= np.abs(ups[a] - ups[b]) <= z * np.hypot(err[a], err[b])
    plateau = any(agree[i] and agree[i + 1] for i in range(nT - 1))
    det = dict(decrease_all=bool(dec.all()), linear_all=bool(linear.all()), decrease=dec.tolist(), linear=linear.tolist(), plateau_agree=agree.tolist())
    if af:
        return "asymptotic-freedom crossover supported", det
    if plateau:
        return "quasi-long-range-order candidate", det
    return "undecidable", det


def kt_crossing(T, y, e):
    """First downward crossing of y(T) - 2T/pi.  Returns (T*, |slope|, sigma_slope) or None."""
    T, y, e = (np.asarray(x, float) for x in (T, y, e))
    f = y - 2 * T / np.pi
    for i in range(len(T) - 1):
        if f[i] > 0 >= f[i + 1]:
            t = T[i] + (T[i + 1] - T[i]) * f[i] / (f[i] - f[i + 1])
            slope = (f[i + 1] - f[i]) / (T[i + 1] - T[i])
            return float(t), float(abs(slope)), float(np.hypot(e[i], e[i + 1]) / (T[i + 1] - T[i]))
    return None


def verdict_B(T, ups, err, z=Z):
    ups, err = _arr(ups), _arr(err)
    cross = {L: kt_crossing(T, ups[L], err[L]) for L in LS}
    steep = False
    if all(cross[L] is not None for L in LS):
        s = [cross[L][1] for L in LS]
        steep = bool(s[0] < s[1] < s[2] and (s[2] - s[0]) > z * np.hypot(cross[16][2], cross[32][2]))
    no_kt = bool(((ups[16] - ups[32]) > z * np.hypot(err[16], err[32])).all())
    det = dict(crossings={L: cross[L] for L in LS}, steepening=steep, decrease_all=no_kt)
    if steep and no_kt:
        return "undecidable (both)", det
    if steep:
        return "KT-type crossover consistent", det
    if no_kt:
        return "no KT-type behavior", det
    return "undecidable", det
