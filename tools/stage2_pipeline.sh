#!/bin/sh
# After the 3D L=12 run: B 3D slab (T_c, latent heat), H6 Clausius-Clapeyron chains, H6 slab scan (diagonal interface).
# Each stage is skipped when its output exists (safe to relaunch after a container restart).
cd "$(dirname "$0")/.."
mkdir -p data/stage2
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
[ -f data/stage2/slab_B3D_zH41_L12.npz ] || python3 tools/run_slab_B.py data/stage2/slab_B3D_zH41_L12.npz --size 12 --sweeps 80000 > runs/slab_B3D.log 2>&1
echo "done slab_B3D" >> runs/stage2/progress.log
[ -f data/stage2/cc_chains.npz ] || python3 tools/run_cc_chains.py data/stage2/cc_chains.npz --size 24 > runs/cc_chains.log 2>&1
echo "done cc_chains" >> runs/stage2/progress.log
[ -f data/stage2/slab_H6_scan.npz ] || python3 tools/run_slab_scan.py data/stage2/slab_H6_scan.npz --size 24 --T 0.10 0.125 0.15 0.175 0.20 \
    --zG $(python3 -c "print(' '.join(f'{1.0+0.005*i:.3f}' for i in range(21)))") --sweeps 100000 --kinds diag > runs/slab_H6.log 2>&1
echo "done slab_H6_scan" >> runs/stage2/progress.log
