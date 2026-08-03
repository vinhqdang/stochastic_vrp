@echo off
setlocal
cd /d "%~dp0"

rem No hard-coded instance argument here: the Python script selects
rem HANOI-100-1.vrpspd from this directory by default.
python probe_freight_recoverability_v2_fixed.py ^
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
