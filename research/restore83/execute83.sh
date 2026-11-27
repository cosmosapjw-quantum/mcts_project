#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
# Recompute analysis from existing registered diagnostic data.
python3 audit83.py
python3 adaptive83.py
python3 verify83.py
python3 fragments83.py
python3 summary83.py
python3 plot83.py
