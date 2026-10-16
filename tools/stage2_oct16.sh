#!/bin/sh
# (1) re-entrance equilibrium T_re by slab coexistence at zG = 1.1; (5) B 3D slab L = 16.  Each stage is skipped when its output exists.
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
TS=$(python3 -c "print(' '.join(f'{0.20+0.005*i:.3f}' for i in range(15)))")
[ -f data/stage2/slab_reentrance_zG1.1.npz ] || python3 tools/run_slab_scan.py data/stage2/slab_reentrance_zG1.1.npz --size 24 --T $TS --zG 1.1 1.1 1.1 --sweeps 12000 --kinds diag --seed 7000 > runs/slab_reentrance.log 2>&1
echo "done slab_reentrance" >> runs/stage2/progress.log
TB=$(python3 -c "t=[f'{0.350+0.002*i:.3f}' for i in range(9)]; print(' '.join(t*3))")
[ -f data/stage2/slab_B3D_zH41_L16.npz ] || python3 tools/run_slab_B.py data/stage2/slab_B3D_zH41_L16.npz --size 16 --sweeps 12000 --T $TB --seed 8000 > runs/slab_B3D_L16.log 2>&1
echo "done slab_B3D_L16" >> runs/stage2/progress.log
