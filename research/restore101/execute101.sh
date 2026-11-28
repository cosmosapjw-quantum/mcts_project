#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python3 source101.py
python3 field101.py
python3 quantum101.py
python3 verify101.py
python3 plot101.py
