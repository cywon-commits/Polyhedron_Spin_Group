#!/bin/sh
# H6 auxiliary rescan with the scan upper bound raised to 1.12 (T = 0.175, 0.20 only); the H6 verdict stays fixed (docs/CD_rules.md)
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
for T in 0.175 0.20; do
  [ -f data/stage2/slab_H6_rescan_T$T.npz ] || python3 tools/run_slab_scan.py data/stage2/slab_H6_rescan_T$T.npz --size 24 --T $T --zG 1.040 1.045 1.050 1.055 1.060 1.065 1.070 1.075 1.080 1.085 1.090 1.095 1.100 1.105 1.110 1.115 1.120 1.040 1.045 1.050 1.055 1.060 1.065 1.070 1.075 1.080 1.085 1.090 1.095 1.100 1.105 1.110 1.115 1.120 1.040 1.045 1.050 1.055 1.060 1.065 1.070 1.075 1.080 1.085 1.090 1.095 1.100 1.105 1.110 1.115 1.120 --sweeps 12000 --kinds diag --seed 4000 > runs/slab_H6_rescan_T$T.log 2>&1
  echo "done slab_H6_rescan_T$T" >> runs/stage2/progress.log
done
