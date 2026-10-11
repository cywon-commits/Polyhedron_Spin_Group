"""Step-size tuning for the Gaussian single-spin and rigid-rotation moves."""
import numpy as np

from . import moves


def tune_steps(spins, nbr, p, beta, s, delta0=0.1, target=(0.3, 0.5), n_rounds=40, block=20,
               use_overlap=False):
    """Adapt (delta, delta_r) until both acceptance rates lie in ``target``.
    Equilibration-phase only: the final values must then be frozen for measurement
    (adapting during measurement would break detailed balance)."""
    delta, delta_r = delta0, delta0
    mid = 0.5 * (target[0] + target[1])
    for _ in range(n_rounds):
        c = np.zeros((moves.N_COUNTERS, 2), dtype=np.int64)
        for _ in range(block):
            moves.mc_sweep(spins, nbr, p, beta, delta, delta_r, 20.0, True, True, use_overlap,
                           False, s, c)
        a_s = c[0, 0] / max(c[0, 1], 1)
        a_r = c[1, 0] / max(c[1, 1], 1)
        if not (target[0] <= a_s <= target[1]):
            delta = min(delta * np.exp(1.5 * (a_s - mid)), 2.0)
        if not (target[0] <= a_r <= target[1]):
            delta_r = min(delta_r * np.exp(1.5 * (a_r - mid)), 1.0)
    return delta, delta_r
