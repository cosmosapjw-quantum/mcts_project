#!/bin/bash
set -eu
cd "$(dirname "$0")"
export PYTHONPATH="$HOME/.cache/uv/archive-v0/bQo8pywhxBw50J5E" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
for i in 1 2; do
 p=$((i-1))
 /usr/bin/time -v -o "logs/self$i-time.log" ./mqft_train play "checkpoints/g$p.bin" "checkpoints/g$p.bin" "plans/self$i.json" "data/self$i" 10 > "logs/self$i.log" 2>&1
 echo "SELFPLAY $i complete"
 python -u train60.py fit "$i" > "logs/fit$i.log" 2>&1
 tail -1 "logs/fit$i.log"
 python -u train60.py infer "$i" > "logs/infer$i.log" 2>&1
 cat "logs/infer$i.log"
 /usr/bin/time -v -o "logs/arena$i-time.log" ./mqft_train play "checkpoints/g$i.bin" "checkpoints/g$p.bin" "plans/arena$i.json" "data/arena$i" 10 > "logs/arena$i.log" 2>&1
 python -u train60.py arena "$i" > "logs/arena$i-audit.log" 2>&1
 cat "logs/arena$i-audit.log"
done
echo CYCLE_COMPLETE
