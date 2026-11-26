#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
O="$PWD/../mqft_restore49_20261126"; P="$PWD/../mqft_restore39_20261125/repo"
for L in 13 15 17 19 21; do PYTHONPATH="$O/family/L$L/python" "$P/.venv/bin/python" extract77.py "$L"; done
python3 original77.py
python3 closure77.py
python3 rg77.py
python3 verify77.py
python3 summary77.py
python3 plot77.py
