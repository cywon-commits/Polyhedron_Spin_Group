#!/bin/sh
cd "$(dirname "$0")/.."
while ps -eo args | grep -qE "[s]tage2_oct16.sh|[s]tage2_m2_wedge.sh"; do sleep 30; done
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 tools/stage2_c3d.py
