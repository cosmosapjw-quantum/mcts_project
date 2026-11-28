#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python3 barrier88.py
python3 gauge88.py
python3 topology88.py
python3 transport88.py
python3 verify88.py
python3 landscape88.py
python3 summary88.py
python3 persistence_rank88.py
python3 plot88.py
python3 report88.py
