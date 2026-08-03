@echo off
setlocal
cd /d "%~dp0"

python probe_freight_recoverability_v2.py ^
  instance="HANOI-100-1(1).vrpspd" ^
  seeds=0:20 ^
  ks=7,8 ^
  base_t=3 ^
  t=8 ^
  noimp=3 ^
  init_tries=30 ^
  min_route_len=5 ^
  min_route_delivery_frac=0.20 ^
  sub_min_len=10 ^
  sub_min_work_frac=0.50 ^
  deltas=0.05,0.10,0.20,0.40 ^
  pair_tol=0.02 ^
  out=hanoi_recoverability_v2

pause
