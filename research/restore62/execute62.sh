#!/bin/bash
set -eu
cd "$(dirname "$0")"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
O=../mqft_restore60_20261126
for tag in response0w1 response0w4 response3w1 response3w4 legacy0 symmetry0 symmetry3; do
 case "$tag" in response3*|symmetry3) g=3;; *) g=0;; esac
 p=plans/$tag.json; if [ -e plans/${tag}_rest.json ]; then p=plans/${tag}_rest.json; fi
 /usr/bin/time -v -o logs/$tag-time.log ./mqft_train control62 "$O/checkpoints/g$g.bin" "$p" "data/$tag" 16 > "logs/$tag.log" 2>&1
 echo "$tag COMPLETE"
done
echo GENERATION_COMPLETE
