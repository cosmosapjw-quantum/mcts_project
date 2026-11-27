#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2
# Reproduce evaluation on immutable frozen model, landscape and raw paths.
python3 extract84.py
python3 population84.py evaluate
python3 verify84.py
python3 summary84.py
python3 plot84.py
