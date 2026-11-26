#!/bin/bash
set -eu
cd "$(dirname "$0")"
export PYTHONPATH="$HOME/.cache/uv/archive-v0/bQo8pywhxBw50J5E" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
/usr/bin/time -v -o logs/self3-time.log ./mqft_train play checkpoints/g2.bin checkpoints/g2.bin plans/self3.json data/self3 14 > logs/self3.log 2>&1
echo SELFPLAY3_COMPLETE
python -u scale60.py fit3 | tee logs/fit3.log
for pair in 10 21 32 30; do
 new=${pair:0:1}; old=${pair:1:1}
 for budget in 32 128; do
  tag=a${pair}b${budget}
  /usr/bin/time -v -o logs/$tag-time.log ./mqft_train play checkpoints/g$new.bin checkpoints/g$old.bin plans/$tag.json data/$tag 14 > logs/$tag.log 2>&1
  python -u scale60.py arena "$tag" | tee logs/$tag-audit.log
 done
done
echo SCALE_COMPLETE
