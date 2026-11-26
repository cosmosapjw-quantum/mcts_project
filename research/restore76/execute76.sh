#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
O="$PWD/../mqft_restore49_20261126"; P="$PWD/../mqft_restore39_20261125/repo"
for L in 13 15 17 19 21; do PYTHONPATH="$O/family/L$L/python" "$P/.venv/bin/python" audit76.py "$L"; done
python3 critical76.py
python3 verify76.py
python3 pulse76.py
python3 plot76.py
python3 summary76.py
