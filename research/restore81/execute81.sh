#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
# Analysis rerun on existing registered raw traces.
python3 audit81.py
python3 effects81.py
python3 record81.py
python3 verify81.py
python3 summary81.py
python3 plot81.py
