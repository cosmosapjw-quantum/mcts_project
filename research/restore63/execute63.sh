#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
for tag in fisher0 legacy0 hold0 hold3; do
 g=0; if [ "$tag" = hold3 ]; then g=3; fi
 /usr/bin/time -v -o "logs/$tag-time.log" ./mqft_train control62 "../mqft_restore60_20261126/checkpoints/g$g.bin" "plans/$tag.json" "data/$tag" 16 > "logs/$tag.log" 2>&1
 echo "$tag COMPLETE"
done
echo GENERATION_COMPLETE
