#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
for g in 0 3; do
 /usr/bin/time -v -o logs/phase$g-time.log ./mqft_train phase64 "../mqft_restore60_20261126/checkpoints/g$g.bin" "plans/phase$g.json" "data/phase$g" 16 > logs/phase$g.log 2>&1
 echo PHASE$g COMPLETE
done
echo GENERATION_COMPLETE
