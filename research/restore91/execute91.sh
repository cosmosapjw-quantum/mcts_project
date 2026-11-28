#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export OPENBLAS_NUM_THREADS=1
python3 thermal91.py
python3 verify91.py
python3 plot91.py
