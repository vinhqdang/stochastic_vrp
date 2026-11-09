#!/usr/bin/env bash
# Runs the SORTIE experiments in order. Usage: run_all.sh [--quick]
# Logs go to <results>/<name>.log (results = papers/sortie/results unless SORTIE_RESULTS is set).
# Each script resumes from its CSV, so a rerun only computes missing jobs.
set -u

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE="$(dirname "$HERE")"
RESULTS="${SORTIE_RESULTS:-$(cd "$CODE/../results" && pwd)}"
PY="${PYTHON:-python}"
QUICK="${1:-}"

mkdir -p "$RESULTS"
cd "$CODE" || exit 1

status=0
for name in e0_seed e1_accuracy e2_exact e3_main e4_inconsistency e5_ablation e6_sensitivity e7_scale e8_congestion; do
    log="$RESULTS/$name.log"
    echo "[run_all] $name -> $log"
    if "$PY" "experiments/$name.py" $QUICK > "$log" 2>&1; then
        echo "[run_all] $name ok"
    else
        echo "[run_all] $name FAILED (see $log)"
        status=1
    fi
done
echo "[run_all] finished with status $status"
exit $status
