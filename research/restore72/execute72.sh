#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
B="$(dirname "$PWD")"; O="$B/mqft_restore49_20261126"; P="$B/mqft_restore39_20261125/repo"
for L in 13 15 17 19 21; do PYTHONPATH="$O/family/L$L/python" "$P/.venv/bin/python" audit72.py "$L"; done
python3 memory72.py
python3 scale72.py
python3 verify72.py
python3 plot72.py
