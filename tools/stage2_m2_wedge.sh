#!/bin/sh
# M2 wedge slab verification at zG = 1.1 (after the Oct-16 slab jobs have finished)
cd "$(dirname "$0")/.."
while ps -eo args | grep -q "[s]tage2_oct16.sh"; do sleep 30; done
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
TA=$(python3 -c "print(' '.join(f'{0.20+0.005*i:.3f}' for i in range(17)))")
TB=$(python3 -c "print(' '.join(f'{0.22+0.005*i:.3f}' for i in range(17)))")
[ -f data/stage2/slab_M2F_zG1.1.npz ] || python3 tools/run_slab_pair.py data/stage2/slab_M2F_zG1.1.npz --left 1 --right 0 --measure 2 --zG 1.1 --T $TA > runs/slab_M2F.log 2>&1
echo "done slab_M2F" >> runs/stage2/progress.log
[ -f data/stage2/slab_CM2_zG1.1.npz ] || python3 tools/run_slab_pair.py data/stage2/slab_CM2_zG1.1.npz --left 2 --right 1 --measure 1 --zG 1.1 --T $TB > runs/slab_CM2.log 2>&1
echo "done slab_CM2" >> runs/stage2/progress.log
