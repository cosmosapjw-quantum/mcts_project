#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
# Reanalyze registered immutable raw histories; fresh generation separate.
python3 response74.py
python3 petz74.py
python3 verify74.py
python3 plot74.py
