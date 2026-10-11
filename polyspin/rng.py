"""xoshiro256++ generator usable inside numba kernels.

Each replica owns a 4-element uint64 state array, so replicas can be advanced
in parallel threads and still give bit-exact results for a given seed.
"""
import numpy as np
from numba import njit

_MASK = (1 << 64) - 1


def make_state(seed, stream=0):
    """Seed a state array via splitmix64 from (seed, stream)."""
    x = (int(seed) * 0x9E3779B97F4A7C15 + int(stream) * 0xD1B54A32D192ED03 + 1) & _MASK
    out = np.empty(4, dtype=np.uint64)
    for k in range(4):
        x = (x + 0x9E3779B97F4A7C15) & _MASK
        z = x
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _MASK
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _MASK
        out[k] = z ^ (z >> 31)
    return out


@njit(cache=True, inline="always")
def _rotl(x, k):
    return (x << np.uint64(k)) | (x >> np.uint64(64 - k))


@njit(cache=True)
def next_u64(s):
    result = _rotl(s[0] + s[3], 23) + s[0]
    t = s[1] << np.uint64(17)
    s[2] ^= s[0]
    s[3] ^= s[1]
    s[1] ^= s[2]
    s[0] ^= s[3]
    s[2] ^= t
    s[3] = _rotl(s[3], 45)
    return result


@njit(cache=True)
def rand(s):
    """Uniform in [0, 1)."""
    return (next_u64(s) >> np.uint64(11)) * (1.0 / 9007199254740992.0)


@njit(cache=True)
def randn(s):
    """Standard normal (Box-Muller, one variate per call)."""
    u1 = 1.0 - rand(s)
    u2 = rand(s)
    return np.sqrt(-2.0 * np.log(u1)) * np.cos(2.0 * np.pi * u2)


@njit(cache=True)
def randint(s, n):
    return int(rand(s) * n)
