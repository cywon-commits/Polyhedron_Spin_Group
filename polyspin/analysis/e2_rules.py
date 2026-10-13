"""Pre-registered decision rules for the E' cut (stage 2, first range), fixed before the runs (docs/E2_rules.md).

H7 (residual nematic = two separated transitions): T1 = temperature of the chi_1 maximum (polar order m1),
T2 = temperature of the chi_2 maximum (nematic order m2).  'Separation supported' iff for EVERY size
T2 - T1 > 3 sigma (sigma^2 = s1^2 + s2^2) and the gap at the largest L is >= 70% of the gap at the smallest L
(it must not shrink by more than 30%); otherwise 'undetermined'.

H12 (C -> D transition order; the existing H1 rule applied at T1): 'first-order' iff the corrected three-condition
double-peak rule holds at T1 or the pair exponent a >= 0.9; 'continuous-consistent' iff a <= 0.7 and
|2/3 - V_min| decreases with L; else 'undetermined'.  a = pair exponent of the maximum of c(T) inside
[0.85, 1.15] T1 (an interior maximum is required, otherwise a is undefined -> 'undetermined').
Universality-class identification is exploratory only."""
import numpy as np

from .finite_size import h1_verdict_l14

H7_SIGMA, H7_SHRINK = 3.0, 0.7
T1_WINDOW = (0.85, 1.15)


def chi_curve(m_series, N, T):
    """chi_l(T) = N (<m^2> - <m>^2) / T for series (R, n) (same definition as analysis.summary.thermo)."""
    m = np.asarray(m_series, float)
    return N * (np.mean(m ** 2, axis=1) - np.mean(m, axis=1) ** 2) / np.asarray(T, float)


def peak_location(T, y):
    """Location of the maximum with a parabola through the three points around it (interior maximum only)."""
    T, y = np.asarray(T, float), np.asarray(y, float)
    i = int(np.argmax(y))
    if i == 0 or i == len(T) - 1:
        return float(T[i])
    x, v = T[i - 1:i + 2], y[i - 1:i + 2]
    c = np.polyfit(x, v, 2)
    if c[0] >= 0:
        return float(T[i])
    t = -c[1] / (2 * c[0])
    return float(np.clip(t, x[0], x[-1]))


def peak_sigma(T, m_series, N, nblocks=10):
    """(location, sigma): sigma^2 = block-jackknife variance of the peak location (delete-one of ``nblocks`` time
    blocks) + (local ladder spacing)^2 / 12 (discretisation)."""
    m = np.asarray(m_series, float)
    T = np.asarray(T, float)
    loc = peak_location(T, chi_curve(m, N, T))
    n = m.shape[1]
    b = n // nblocks
    reps = []
    for j in range(nblocks):
        keep = np.concatenate([np.arange(k * b, (k + 1) * b) for k in range(nblocks) if k != j])
        reps.append(peak_location(T, chi_curve(m[:, keep], N, T)))
    reps = np.array(reps)
    jk = (nblocks - 1) / nblocks * np.sum((reps - reps.mean()) ** 2)
    i = int(np.argmin(abs(T - loc)))
    dT = np.mean(np.diff(T)[max(i - 1, 0):i + 1]) if len(T) > 1 else 0.0
    return loc, float(np.sqrt(jk + dT ** 2 / 12.0))


def h7_verdict(per_size):
    """per_size: {L: dict(T1, s1, T2, s2)} -> ('separation supported' | 'undetermined', details)."""
    Ls = sorted(per_size)
    gaps = {L: per_size[L]["T2"] - per_size[L]["T1"] for L in Ls}
    sig = {L: float(np.hypot(per_size[L]["s1"], per_size[L]["s2"])) for L in Ls}
    all_sep = all(gaps[L] > H7_SIGMA * sig[L] for L in Ls)
    keeps = len(Ls) >= 2 and gaps[Ls[-1]] >= H7_SHRINK * gaps[Ls[0]]
    return ("separation supported" if all_sep and keeps else "undetermined"), dict(gap=gaps, sigma=sig, all_3sigma=all_sep, gap_kept=keeps)


def h12_verdict(bimodal_rule, a_pair, dev_small, dev_large):
    """H1 rule on the C->D transition (see module docstring); a_pair NaN (no interior c maximum) -> undetermined."""
    if not np.isfinite(a_pair):
        return "first-order" if bimodal_rule else "undetermined (no interior c maximum at T1)"
    return h1_verdict_l14(bimodal_rule, a_pair, dev_small, dev_large)
