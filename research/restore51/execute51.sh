#!/bin/bash
set -u
B="$HOME/quantum/research/reboot_20261120"; S="$B/mqft_restore51_20261126"; O="$B/mqft_restore49_20261126"; P="$B/mqft_restore39_20261125/repo"
run_one() {
 local L="$1" n="$2" f="$2-L$1"
 PYTHONPATH="$O/family/L$L/python" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /usr/bin/time -v -o "$S/logs/$f-time.log" timeout --signal=INT 480s "$P/.venv/bin/python" -m mqft.cli generate "$S/plans/$f.json" "$S/data/$f" --backend "$O/runtime_verified/mqft-L$L" --rapfi "$P/.external/rapfi" --execute > "$S/logs/$f-generate.log" 2>&1
 local rc=$?; echo "$rc" > "$S/logs/$f.exit"; echo "$f generation rc=$rc"; test "$rc" = 0 || return "$rc"
 PYTHONPATH="$O/family/L$L/python" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 "$P/.venv/bin/python" -m mqft.cli analyze "$S/plans/$f.json" "$S/data/$f" "$S/analysis/$f" > "$S/logs/$f-analyze.log" 2>&1
 rc=$?; echo "$rc" > "$S/logs/$f-analyze.exit"; echo "$f analysis rc=$rc"; test "$rc" = 0
}
run_size() {
 local L="$1"
 run_one "$L" pilot || return $?
 PYTHONPATH="$O/family/L$L/python" OPENBLAS_NUM_THREADS=1 "$P/.venv/bin/python" "$S/plan51.py" "$L" confirm || return $?
 # Sorted registration is immutable before perturbed runs.
 for n in $(python3 -c "import json;print(' '.join(k for k in json.load(open('$S/REGISTRATION_L$L.json'))['plans'] if k!='pilot'))"); do run_one "$L" "$n" || return $?; done
}
(run_size 13) & a=$!
(run_size 17) & b=$!
(run_size 21) & c=$!
wait "$a"; echo "L13 stream $?"
wait "$b"; echo "L17 stream $?"
wait "$c"; echo "L21 stream $?"
