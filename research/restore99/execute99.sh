#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python3 source99.py
python3 scaling99.py
python3 fit99.py
python3 action99.py
python3 plot99.py
