#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
for g in 0 3; do
 ./mqft_train phase65 "../mqft_restore60_20261126/checkpoints/g$g.bin" "plans/phase$g.json" "data/phase$g" 12 > "logs/phase$g.log" 2>&1
 echo PHASE$g COMPLETE
done
