#!/bin/bash
set -u
B="$HOME/quantum/research/reboot_20261120"; S="$B/mqft_restore58_20261126"; O="$B/mqft_restore49_20261126"; F="$O"; P="$B/mqft_restore39_20261125/repo"
run() {
 local L="$1" n="$2" f="$2-L$1"
 PYTHONPATH="$O/family/L$L/python" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /usr/bin/time -v -o "$S/logs/$f-time.log" timeout --signal=INT 600s "$P/.venv/bin/python" -m mqft.cli generate "$S/plans/$f.json" "$S/data/$f" --backend "$F/runtime_verified/mqft-L$L" --rapfi "$P/.external/rapfi" --execute > "$S/logs/$f-generate.log" 2>&1
 local rc=$?;echo "$rc" > "$S/logs/$f.exit";echo "$f generation $rc";test "$rc" = 0 || return "$rc"
 PYTHONPATH="$O/family/L$L/python" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 "$P/.venv/bin/python" -m mqft.cli analyze "$S/plans/$f.json" "$S/data/$f" "$S/analysis/$f" > "$S/logs/$f-analyze.log" 2>&1
 rc=$?;echo "$rc" > "$S/logs/$f-analyze.exit";echo "$f analysis $rc";test "$rc" = 0
}
run_size() { for n in B16b0 B16b1 B16b2 B32b0 B32b1 B32b2; do run "$1" "$n" || return $?; done; }
(run_size 13 && run_size 19)& a=$!
(run_size 15 && run_size 21)& b=$!
(run_size 17)& c=$!
wait "$a"; x=$?; wait "$b"; y=$?; wait "$c"; z=$?; echo "streams $x $y $z"; test "$x$y$z" = 000
