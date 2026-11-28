#!/bin/sh
# Sensitivity runs for the TEMPO paper (run from svrpspd_wdro/). Outputs: results/tempo_pool/*_sens_*.csv, trace_*.csv, timing_*.csv
set -e
for f in Dethloff SalhiNagy City; do
  python3 scripts/tempo_pool_trace.py fam=$f days=100 scen=mixed tag=_mixed
  python3 scripts/tempo_pool_trace.py fam=$f days=100 scen=zone tag=_zone
  python3 scripts/tempo_pool_trace.py fam=$f days=100 scen=city tag=_city
done
for H in 5 10 20 30; do for f in Dethloff SalhiNagy City; do
  python3 scripts/tempo_pool_eval.py fam=$f days=100 scen=mixed H=$H pols=hier,hier_eb tag=_sens_H$H
done; done
for sz in 0.00 0.05 0.10 0.30 0.40; do for f in Dethloff SalhiNagy City; do
  python3 scripts/tempo_pool_eval.py fam=$f days=100 sc=0.2 sz=$sz tag=_sens_sz$sz
done; done
for Z in 2 4 8; do for f in Dethloff SalhiNagy City; do
  python3 scripts/tempo_pool_eval.py fam=$f days=100 scen=mixed Z=$Z tag=_sens_Z$Z
done; done
for f in Dethloff SalhiNagy City; do python3 scripts/tempo_pool_timing.py fam=$f n_inst=5; done
echo CHAIN_DONE
