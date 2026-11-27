#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4
python3 extract82.py
python3 population82.py fit
python3 population82.py evaluate
python3 verify82.py
python3 summary82.py
python3 plot82.py
