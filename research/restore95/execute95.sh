#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python3 entangle95.py
python3 verify95.py
python3 base_control95.py
python3 plot95.py
python3 plot_base95.py
