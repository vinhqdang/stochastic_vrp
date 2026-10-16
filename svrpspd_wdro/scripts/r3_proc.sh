#!/bin/bash
# One shard of the round-2 experiments: runs every part for task slice i of n, one worker.
# usage: r3_proc.sh OUT_ROOT i n [parts...]
cd "$(dirname "$0")/.." || exit 1
OUT=$1; I=$2; N=$3; shift 3
PARTS=${@:-passrate mhat unconf rivals2 coverage2 hubcrude long}
for p in $PARTS; do
  OMP_NUM_THREADS=1 python scripts/run_rarecert_r3.py --part "$p" --workers 1 --shard "$I/$N" \
    --out-dir "$OUT/s$I/$p" >> "$OUT/s$I.log" 2>&1
done
echo "all parts done" >> "$OUT/s$I.log"
