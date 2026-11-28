#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python3 source102.py
python3 markov102.py
python3 lag102.py
python3 verify102.py
python3 plot102.py
