#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python3 calibrate79.py
python3 inventory79.py
python3 verify79.py
python3 assemble79.py
python3 plot79.py
