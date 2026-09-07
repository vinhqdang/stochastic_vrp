#!/bin/bash
# Drive a benchmark run to completion across daily-quota exhaustion.
#
# A run stops when every configured project's daily allowance for the
# chosen model is spent. Resume is idempotent (musique_benchmark.py skips
# calls already in the output file), so the fix is simply to launch again.
#
# Waits by WAITING ON ITS OWN CHILD, never by pgrep on a command line:
# an earlier version matched its own ancestor shell -- whose cmdline
# contained the run's command string -- and blocked forever.
#
# Usage: run_until_done.sh <expected_records> <model> <grid> <scorer> [passes]
set -u
cd "$(dirname "$0")"

TARGET=${1:?expected record count}
MODEL=${2:-gemini-3.5-flash-lite}
GRID=${3:-min}
SCORER=${4:-lexical}
PASSES=${5:-8}

OUT="results/musique_${MODEL}_${SCORER}_${GRID}.jsonl"

count() { [ -f "$OUT" ] && wc -l < "$OUT" || echo 0; }

for pass in $(seq 1 "$PASSES"); do
    n=$(count)
    if [ "$n" -ge "$TARGET" ]; then
        echo "target reached: $n / $TARGET"
        break
    fi
    echo "=== pass $pass: $n / $TARGET ==="
    python3 musique_benchmark.py --instances 60 --grid "$GRID" \
        --scorer "$SCORER" --model "$MODEL" --workers 3 \
        > "results/pass_${pass}.log" 2>&1 &
    child=$!
    wait "$child"
    after=$(count)
    echo "pass $pass added $((after - n)) records (now $after)"
    # No progress means every project's allowance for this model is spent;
    # further passes today would just re-probe an exhausted quota.
    if [ "$after" -le "$n" ]; then
        echo "no progress -- quota exhausted for $MODEL across all keys"
        break
    fi
done

echo "=== final: $(count) / $TARGET records ==="
python3 analyze_musique.py "$OUT" 2>/dev/null | tail -30
echo
python3 paired_test.py "$OUT" 2>/dev/null | tail -16
