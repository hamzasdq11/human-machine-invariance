#!/bin/bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
set -x
cd "$(dirname "$0")"
DREPS=40 python3 scripts/run_diagnostics.py all
echo DIAG_DONE
