# polyspin — finite-temperature Monte Carlo of polyhedral spin clusters

Full-spin Monte Carlo (single-spin, rigid-cluster rotation and overlap/split moves) with parallel
tempering for a lattice of n = 4 spin clusters, plus defect detectors (2G holonomy, Z2 half-vortices,
skyrmions) and T = 0 / spin-wave tools.  Python + Numba (chosen for the simple local-ΔE kernels and
replica-level `prange` parallelism; per-replica xoshiro states make runs reproducible for any thread count).

```
pip install -e .[test]
pytest -m "not slow"          # ~100 tests, ~2.5 min;  pytest   adds the multi-minute statistical tests
python tools/mutation_check.py  # mutation tests of the move kernels and the replica exchange
```

## Running a simulation
One YAML file = one parallel-tempering run = one HDF5 file (series, diagnostics, final configurations,
seed, parameters, git hash).  Resumable: a `.ckpt.npz` is written periodically and picked up automatically.

```
python tools/make_configs.py --cuts A B --dim 2 --size 16 --outdir configs      # cut-line configs
python -m polyspin.run configs/A_2d_L16_zG1.yaml --out A_2d_L16_zG1.h5
python tools/plot_run.py A_2d_L16_zG1.h5 overview.png
python tools/summarize_runs.py <dir>                                            # table of runs
```
Keys are documented in the docstring of `polyspin/run.py`.  Conventions: input couplings via `zG`, `zH4`
(= z·G, z·H4); `J = 1`; `init: ordered` starts every replica in the T = 0 ground state (needed for
first-order transitions, see docs/M3_report.md), `init: random` is the unbiased start.  Compare both
with `tools/compare_inits.py` — disagreement marks non-equilibrated temperatures.

Modes: `mode: hysteresis` (quasi-static sweep at fixed T, `polyspin/run.py:run_hysteresis`) and the
C|F slab coexistence (`polyspin/slab.py`).

## Layout
```
polyspin/model.py        Hamiltonian, local ΔE, channel form
polyspin/moves.py        single-spin, rigid rotation, overlap/split (detailed balance), label permutation
polyspin/pt.py           replica exchange, step-size tuning, hysteresis scan
polyspin/measure.py      per-temperature order parameters and diagnostics accumulators
polyspin/observables.py  m_l, cluster shapes, correlators
polyspin/defects.py      2G holonomy (plaquette + block loops), Z2 half-vortices, skyrmion charge
polyspin/groundstate.py  single-cluster T = 0 minimisation and analytic boundaries
polyspin/harmonic.py     spin-wave Hessian, harmonic free energy, C-F selection (H6)
polyspin/slab.py         C|F slab coexistence
polyspin/analysis/       stats (tau_int, errors, Binder), summary loader, ladder design
tools/                   config generation, ladder re-planning, plots, cost estimate, mutation tests, scans
docs/                    milestone reports and figures
```
