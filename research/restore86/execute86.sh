#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python3 extract86.py
python3 memory86.py
python3 scale86.py
python3 verify86.py
python3 summary86.py
python3 plot86.py
