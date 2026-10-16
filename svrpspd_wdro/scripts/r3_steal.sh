#!/bin/bash
# Work the later parts of another machine's shard from the other end (reverse task order).
# usage: r3_steal.sh OUT_ROOT slot n [parts...]
cd "$(dirname "$0")/.." || exit 1
OUT=$1; I=$2; N=$3; shift 3
PARTS=${@:-coverage2 hubcrude long}
for p in $PARTS; do
  OMP_NUM_THREADS=1 python scripts/run_rarecert_r3.py --part "$p" --workers 1 --shard "$I/$N" --reverse \
    --out-dir "$OUT/v$I/$p" >> "$OUT/v$I.log" 2>&1
done
echo "all parts done" >> "$OUT/v$I.log"
