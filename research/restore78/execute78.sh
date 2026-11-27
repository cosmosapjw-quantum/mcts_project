#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python3 native78.py
python3 gauge78.py
python3 verify78.py
python3 original78.py
python3 plot78.py
