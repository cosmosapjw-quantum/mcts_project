#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python3 source100.py
python3 native100.py
python3 therm100.py
python3 verify100.py
python3 plot100.py
