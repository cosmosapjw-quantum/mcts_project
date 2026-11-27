#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
# Reproduce analysis from existing registered raw and forced replays.
python3 extract80.py
python3 rng80.py
python3 action80.py
python3 verify80.py
python3 summary80.py
python3 plot80.py
