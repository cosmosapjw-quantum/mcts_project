#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
for tag in phase0 phase3 reference3; do
 g=3; if [ "$tag" = phase0 ]; then g=0; fi
 /usr/bin/time -v -o logs/$tag-time.log ./mqft_train phase65 "../mqft_restore60_20261126/checkpoints/g$g.bin" "plans/$tag.json" "data/$tag" 16 > "logs/$tag.log" 2>&1
 echo "$tag COMPLETE"
done
