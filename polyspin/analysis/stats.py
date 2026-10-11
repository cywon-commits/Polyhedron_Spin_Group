"""Statistics helpers: integrated autocorrelation time, errors, Binder cumulant, susceptibility."""
import numpy as np


def autocorr(x):
    x = np.asarray(x, float) - np.mean(x)
    n = len(x)
    f = np.fft.rfft(x, 2 * n)
    ac = np.fft.irfft(f * np.conj(f))[:n]
    return ac / ac[0] if ac[0] > 0 else np.zeros(n)


def tau_int(x, c=6.0):
    """Integrated autocorrelation time (in samples) with Sokal's automatic window W >= c * tau(W).
    Returns (tau, window).  tau = 0.5 for uncorrelated data; constant series give tau = 0.5."""
    x = np.asarray(x, float)
    if len(x) < 8 or np.ptp(x) == 0:
        return 0.5, 0
    rho = autocorr(x)
    tau = 0.5
    for W in range(1, len(rho)):
        tau += rho[W]
        if W >= c * tau:
            return float(max(tau, 0.5)), W
    return float(max(tau, 0.5)), len(rho) - 1


def block_error(x, nblocks=20):
    x = np.asarray(x, float)
    m = len(x) // nblocks
    if m < 1:
        return float(np.mean(x)), float("nan")
    b = x[: m * nblocks].reshape(nblocks, m).mean(1)
    return float(b.mean()), float(b.std(ddof=1) / np.sqrt(nblocks))


def binder(m):
    """U = 1 - <m^4> / (3 <m^2>^2)."""
    m = np.asarray(m, float)
    m2 = np.mean(m ** 2)
    return 1.0 - np.mean(m ** 4) / (3.0 * m2 ** 2) if m2 > 0 else np.nan


def susceptibility(m, N, beta):
    m = np.asarray(m, float)
    return N * beta * (np.mean(m ** 2) - np.mean(m) ** 2)


def specific_heat(e, N, beta):
    e = np.asarray(e, float)
    return N * beta ** 2 * np.var(e)


def jackknife(func, x, nblocks=20):
    """Jackknife error of func(x) with block deletion."""
    x = np.asarray(x, float)
    m = len(x) // nblocks
    xb = x[: m * nblocks].reshape(nblocks, m)
    full = func(xb.ravel())
    est = np.array([func(np.delete(xb, i, axis=0).ravel()) for i in range(nblocks)])
    err = np.sqrt((nblocks - 1) / nblocks * np.sum((est - est.mean()) ** 2))
    return float(full), float(err)
