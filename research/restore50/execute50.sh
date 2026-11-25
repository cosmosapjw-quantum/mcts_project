#!/bin/bash
set -u
S="$HOME/quantum/research/reboot_20261120/mqft_restore50_20261126"
O="$HOME/quantum/research/reboot_20261120/mqft_restore49_20261126"
P="$HOME/quantum/research/reboot_20261120/mqft_restore39_20261125/repo"
run_size() {
 L="$1"
 for n in $(python3 -c "import json;print(' '.join(json.load(open('$S/REGISTRATION_L$L.json'))['plans']))"); do
  f="$n-L$L"
  PYTHONPATH="$O/family/L$L/python" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /usr/bin/time -v -o "$S/logs/$f-time.log" timeout --signal=INT 360s "$P/.venv/bin/python" -m mqft.cli generate "$S/plans/$f.json" "$S/data/$f" --backend "$O/runtime_verified/mqft-L$L" --rapfi "$P/.external/rapfi" --execute > "$S/logs/$f-generate.log" 2>&1
  rc=$?; echo "$rc" > "$S/logs/$f.exit"; echo "$f generation rc=$rc"; test "$rc" = 0 || return "$rc"
  PYTHONPATH="$O/family/L$L/python" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 "$P/.venv/bin/python" -m mqft.cli analyze "$S/plans/$f.json" "$S/data/$f" "$S/analysis/$f" > "$S/logs/$f-analyze.log" 2>&1
  rc=$?; echo "$rc" > "$S/logs/$f-analyze.exit"; echo "$f analysis rc=$rc"; test "$rc" = 0 || return "$rc"
 done
}
( run_size 15; run_size 19 ) & a=$!
( run_size 21 ) & b=$!
( run_size 13; run_size 17 ) & c=$!
wait "$a"; echo "stream a $?"
wait "$b"; echo "stream b $?"
wait "$c"; echo "stream c $?"
