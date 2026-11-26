#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
# Existing registered raw data only; do not overwrite native data.
python3 first75.py
python3 transport75.py
python3 games75.py
python3 verify75.py
python3 plot75.py
