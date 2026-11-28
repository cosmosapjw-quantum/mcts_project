#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
if [[ ! -f SCALE92.json.gz ]]; then
  if [[ ! -f SCALE92.json ]]; then python3 scale92.py; fi
  python3 pack92.py
fi
python3 verify92.py
python3 plot92.py
