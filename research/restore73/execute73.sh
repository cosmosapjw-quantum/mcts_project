#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
# Registered raw data already exist. No overwrite of raw data.
python3 decision73.py
python3 instrument73.py
python3 verify73.py
python3 plot73.py
