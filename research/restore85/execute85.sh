#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python3 native85.py
python3 branches85.py
python3 thermal85.py
python3 verify85.py
python3 summary85.py
python3 plot85.py
