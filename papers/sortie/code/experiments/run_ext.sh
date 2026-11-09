#!/usr/bin/env bash
# Runs the SORTIE extension experiments E9-E17 in order. Usage: run_ext.sh [--quick]
# Logs go to <results>/<name>.log (results = papers/sortie/results unless SORTIE_RESULTS is set).
# Each script resumes from its CSV, so a rerun only computes missing jobs. E11 and E12 read
# the E3/E8 plans from SORTIE_SOURCE_RESULTS (default papers/sortie/results).
set -u

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE="$(dirname "$HERE")"
RESULTS="${SORTIE_RESULTS:-$(cd "$CODE/../results" && pwd)}"
PY="${PYTHON:-python}"
QUICK="${1:-}"

mkdir -p "$RESULTS"
cd "$CODE" || exit 1

status=0
for name in e9_budget e10_fleet e11_dayfactor e12_misspec e13_kappa e14_screen e15_saa e16_baselines e17_family; do
    log="$RESULTS/$name.log"
    echo "[run_ext] $name -> $log"
    if "$PY" "experiments/$name.py" $QUICK > "$log" 2>&1; then
        echo "[run_ext] $name ok"
    else
        echo "[run_ext] $name FAILED (see $log)"
        status=1
    fi
done
echo "[run_ext] finished with status $status"
exit $status
