#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
# Existing registered raw data are read, not overwritten.
python3 audit69.py
python3 marked70.py
python3 verify70.py
