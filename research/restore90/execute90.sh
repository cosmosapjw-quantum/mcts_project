#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export OPENBLAS_NUM_THREADS=1
for s in path90 dual90 response90 work90 phase90 verify90 plot90 report90; do
  python "$s.py" > "logs/$s.log" 2>&1
done
