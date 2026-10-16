#!/bin/sh
# run the exploratory A zG=0.3 points after the C/D runs (the [s] trick keeps pgrep from matching this script itself)
cd "$(dirname "$0")/.."
while ps -eo cmd | grep -q "[t]ools/stage2_cd.py"; do sleep 30; done
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 tools/stage2_a03.py
