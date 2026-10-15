"""Pre-registered rules for stage 2, third priority (cuts C and D, 2D L = 16 first), fixed before the runs (docs/CD_rules.md).

H5 (U phase, cut C): there is a temperature interval where the {211} cluster-shape fraction > 0.5 AND m1 in [0.2, 0.3].
  Implementation: the temperatures of the heating ladder and its cold chains are sorted; an 'interval' is >= 3 consecutive
  temperatures that all satisfy both conditions.  Checked for every U point (zG = 1.0 and 1.2); supported iff both satisfy it.

Re-entrance (cut D, zG = 1.1, F at T = 0): F -> C is seen in BOTH the heating and the cooling ladder.  In a ladder (temperatures sorted)
  it is 'seen' iff some T_low has m1 >= 0.8 (F), some higher T_high has m1 < 0.2 and m2 >= 0.7 (C), and T_re is the first upward
  linear-interpolation crossing of m1 = 0.5 between them.  Quantitative prediction (from the H6 fit zG*(T) = 1 + a T + b T^2 = 1.1, a = 0.392,
  b = 0.15): T_re ~ 0.23, accepted range 0.21-0.26.  Supported iff BOTH T_re(heat) and T_re(cool) are found and lie in the range."""
import numpy as np

SPLIT_211_INDEX = 1                     # diag split classes: 1111, 211, 22, 31, 4
H5_SPLIT_MIN, H5_M1_RANGE, H5_MIN_POINTS = 0.5, (0.2, 0.3), 3
RE_F_M1, RE_C_M1, RE_C_M2, RE_CROSS = 0.8, 0.2, 0.7, 0.5
RE_T_PRED, RE_T_RANGE = 0.23, (0.21, 0.26)


def h5_intervals(T, split211, m1):
    """Maximal runs of >= H5_MIN_POINTS consecutive (sorted) temperatures with split211 > 0.5 and m1 in [0.2, 0.3].
    Returns list of (T_first, T_last, n_points)."""
    T, s, m = (np.asarray(x, float) for x in (T, split211, m1))
    o = np.argsort(T)
    T, s, m = T[o], s[o], m[o]
    ok = (s > H5_SPLIT_MIN) & (m >= H5_M1_RANGE[0]) & (m <= H5_M1_RANGE[1])
    out, i = [], 0
    while i < len(T):
        if ok[i]:
            j = i
            while j + 1 < len(T) and ok[j + 1]:
                j += 1
            if j - i + 1 >= H5_MIN_POINTS:
                out.append((float(T[i]), float(T[j]), j - i + 1))
            i = j + 1
        else:
            i += 1
    return out


def h5_verdict(per_point):
    """per_point: {zG: intervals list}.  'supported' iff every U point has an interval."""
    return "supported" if per_point and all(len(v) > 0 for v in per_point.values()) else "not supported"


def reentrance_T(T, m1, m2):
    """(found, T_re) of F -> C in one ladder (see module docstring)."""
    T, a, b = (np.asarray(x, float) for x in (T, m1, m2))
    o = np.argsort(T)
    T, a, b = T[o], a[o], b[o]
    f = np.where(a >= RE_F_M1)[0]
    if len(f) == 0:
        return False, float("nan")
    i0 = f[0]
    c = np.where((np.arange(len(T)) > i0) & (a < RE_C_M1) & (b >= RE_C_M2))[0]
    if len(c) == 0:
        return False, float("nan")
    for k in range(i0, c[0]):
        if (a[k] - RE_CROSS) * (a[k + 1] - RE_CROSS) <= 0 and a[k] != a[k + 1]:
            return True, float(T[k] + (RE_CROSS - a[k]) * (T[k + 1] - T[k]) / (a[k + 1] - a[k]))
    return False, float("nan")


def reentrance_verdict(heat, cool):
    """heat, cool: (found, T_re).  'supported' iff both found and both inside the accepted range; 'seen, out of range' /
    'seen in one ladder only' / 'not seen' otherwise."""
    (fh, th), (fc, tc) = heat, cool
    if fh and fc:
        inr = all(RE_T_RANGE[0] <= t <= RE_T_RANGE[1] for t in (th, tc))
        return "supported (T_re in the predicted range)" if inr else "F -> C seen in both ladders, T_re outside 0.21-0.26"
    if fh or fc:
        return "F -> C seen in one ladder only"
    return "not seen"
