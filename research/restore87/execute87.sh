#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
O="$PWD/../mqft_restore49_20261126"; P="$PWD/../mqft_restore39_20261125/repo"
for L in 13 15 17 19 21; do
  script=extract87.py
  if [ "$L" -ge 19 ]; then script=extract87_fast.py; fi
  PYTHONPATH="$O/family/L$L/python" "$P/.venv/bin/python" "$script" "$L"
done
python3 erratum86.py
python3 critical87.py
python3 verify87.py
python3 summary87.py
python3 plot87.py
