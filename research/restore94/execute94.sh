#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python3 decoherence94.py
python3 records94.py
python3 verify94.py
python3 plot94.py
