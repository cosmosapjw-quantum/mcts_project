#!/bin/bash
set -u
B="$HOME/quantum/research/reboot_20261120"; S="$B/mqft_restore55_20261126"; O="$B/mqft_restore54_20261126"; F="$B/mqft_restore53_20261126"; P="$B/mqft_restore39_20261125/repo"
run() {
 local L="$1" n="$2" f="$2-L$1"
 PYTHONPATH="$O/python/L$L" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /usr/bin/time -v -o "$S/logs/$f-time.log" timeout --signal=INT 600s "$P/.venv/bin/python" -m mqft.cli generate "$S/plans/$f.json" "$S/data/$f" --backend "$F/runtime/mqft-L$L" --rapfi "$P/.external/rapfi" --execute > "$S/logs/$f-generate.log" 2>&1
 local rc=$?;echo "$rc" > "$S/logs/$f.exit";echo "$f generation $rc";test "$rc" = 0 || return "$rc"
 PYTHONPATH="$O/python/L$L" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 "$P/.venv/bin/python" -m mqft.cli analyze "$S/plans/$f.json" "$S/data/$f" "$S/analysis/$f" > "$S/logs/$f-analyze.log" 2>&1
 rc=$?;echo "$rc" > "$S/logs/$f-analyze.exit";echo "$f analysis $rc";test "$rc" = 0
}
(run 13 root0 && run 13 root1)& a=$!
(run 21 root0 && run 21 root1)& b=$!
(run 13 game16 && run 21 game16 && run 13 game32 && run 21 game32)& c=$!
wait "$a";echo "stream a $?";wait "$b";echo "stream b $?";wait "$c";echo "stream c $?"
