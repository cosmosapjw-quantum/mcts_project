#!/bin/bash
set -u
B="$HOME/quantum/research/reboot_20261120"; S="$B/mqft_restore53_20261126"; P="$B/mqft_restore39_20261125/repo"
run() {
 local L="$1" n="$2" f="$2-L$1"
 PYTHONPATH="$S/family/L$L/python" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /usr/bin/time -v -o "$S/logs/$f-time.log" timeout --signal=INT 480s "$P/.venv/bin/python" -m mqft.cli generate "$S/plans/$f.json" "$S/data/$f" --backend "$S/runtime/mqft-L$L" --rapfi "$P/.external/rapfi" --execute > "$S/logs/$f-generate.log" 2>&1
 local rc=$?;echo "$rc" > "$S/logs/$f.exit";echo "$f generation $rc";test "$rc" = 0 || return "$rc"
 PYTHONPATH="$S/family/L$L/python" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 "$P/.venv/bin/python" -m mqft.cli analyze "$S/plans/$f.json" "$S/data/$f" "$S/analysis/$f" > "$S/logs/$f-analyze.log" 2>&1
 rc=$?;echo "$rc" > "$S/logs/$f-analyze.exit";echo "$f analysis $rc";test "$rc" = 0
}
(run 13 pilot)& a=$!;(run 21 pilot)& b=$!
wait "$a" || exit $?;wait "$b" || exit $?
(run 13 g0 && run 13 variants && run 13 game)& a=$!
(run 21 g0 && run 21 variants && run 21 game)& b=$!
(run 13 g1 && run 21 g1)& c=$!
wait "$a";echo "stream a $?";wait "$b";echo "stream b $?";wait "$c";echo "stream c $?"
