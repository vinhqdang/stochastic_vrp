#!/bin/bash
# One-factor-at-a-time sensitivity of the fleet cost parameters around the
# defaults (F_standby=20, p_late=1.5, s_emg=2.5, F_emg=40), Dethloff subset,
# Det+SAA gates, plans cached -> evaluation only.
cd "$(dirname "$0")/.."
for cfg in "F_standby:10" "F_standby:35" "p_late:0.5" "p_late:3.0" \
           "s_emg:1.5" "s_emg:4.0" "F_emg:25" "F_emg:60"; do
  tag=$(echo "$cfg" | tr ':.' '__')
  python scripts/run_realistic_eval.py policies=Det,SAA max=12 workers=3 \
    "costs=$cfg" out="results_costsens_$tag" > "results/costsens_$tag.log" 2>&1
  echo "done $cfg"
done
