#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python3 qec89.py
python3 native89.py
python3 summary89.py
python3 verify89.py
python3 plot89.py
python3 report89.py
