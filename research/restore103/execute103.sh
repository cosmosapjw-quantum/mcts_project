#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python3 source103.py
python3 measure103.py
python3 prior103.py
python3 fit103.py
python3 verify103.py
python3 cross103.py
python3 plot103.py
python3 verify_fit103.py
