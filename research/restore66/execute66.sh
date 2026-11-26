#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
for g in 0 3; do
 /usr/bin/time -v -o logs/schedule$g-time.log ./mqft_train schedule66 "../mqft_restore60_20261126/checkpoints/g$g.bin" "plans/schedule$g.json" "data/schedule$g" 16 > logs/schedule$g.log 2>&1
 echo SCHEDULE$g COMPLETE
done
