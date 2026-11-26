#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
# Existing registered raw data are read, not overwritten.
python3 extract71.py
python3 closure71.py
python3 verify71.py
python3 plot71.py
