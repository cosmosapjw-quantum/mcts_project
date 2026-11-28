#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python3 source98.py
python3 rg98.py
python3 verify98.py
python3 report98.py
python3 plot98.py
