#!/bin/bash
set -eu
B="$HOME/quantum/research/reboot_20261120"; S="$B/mqft_restore59_20261126"; O="$B/mqft_restore49_20261126"; P="$B/mqft_restore39_20261125/repo"
run() { local L=$1; export PYTHONPATH="$O/family/L$L/python" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1; "$P/.venv/bin/python" -m mqft.cli generate "$S/plans/heldout-L$L.json" "$S/data/heldout-L$L" --backend "$O/runtime_verified/mqft-L$L" --rapfi "$P/.external/rapfi" --execute > "$S/logs/L$L-generate.log" 2>&1; echo "GEN $L done"; "$P/.venv/bin/python" -m mqft.cli analyze "$S/plans/heldout-L$L.json" "$S/data/heldout-L$L" "$S/analysis/heldout-L$L" > "$S/logs/L$L-analyze.log" 2>&1; echo "ANALYZE $L done"; }
run 13 & a=$!; run 21 & b=$!; x=0; y=0; wait "$a" || x=$?; wait "$b" || y=$?; echo "EXIT $x $y"; test "$x$y" = 00
