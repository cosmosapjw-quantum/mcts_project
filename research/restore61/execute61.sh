#!/bin/bash
set -eu
cd "$(dirname "$0")"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
O=../mqft_restore60_20261126
for g in 0 3; do
 /usr/bin/time -v -o logs/response$g-time.log ./mqft_train control "$O/checkpoints/g$g.bin" plans/rest$g.json data/response$g 16 > logs/response$g.log 2>&1
 echo RESPONSE$g COMPLETE
 /usr/bin/time -v -o logs/symmetry$g-time.log ./mqft_train control "$O/checkpoints/g$g.bin" plans/symmetry$g.json data/symmetry$g 16 > logs/symmetry$g.log 2>&1
 echo SYMMETRY$g COMPLETE
 ./mqft_train infer "$O/checkpoints/g$g.bin" plans/d4_fixture.json > data/infer_single$g.json
 ./mqft_train infer_d4 "$O/checkpoints/g$g.bin" plans/d4_fixture.json > data/infer_d4_$g.json
done
echo GENERATION_COMPLETE
