@echo off
setlocal

REM Run the two operating points separately so a long run cannot lose both outputs.
python probe_freight_recoverability_v3_two_outages.py ^
  instance=HANOI-100-1.vrpspd ^
  routes=hanoi_recoverability_v2_routes.json ^
  deltas=0.10 ^
  pair_tol=0.02 ^
  sub_min_len=10 ^
  sub_min_work_frac=0.50 ^
  out=hanoi_recoverability_v3_d10

if errorlevel 1 goto :fail

python probe_freight_recoverability_v3_two_outages.py ^
  instance=HANOI-100-1.vrpspd ^
  routes=hanoi_recoverability_v2_routes.json ^
  deltas=0.20 ^
  pair_tol=0.02 ^
  sub_min_len=10 ^
  sub_min_work_frac=0.50 ^
  out=hanoi_recoverability_v3_d20

if errorlevel 1 goto :fail

echo.
echo Finished both delta runs.
echo Upload these six files:
echo   hanoi_recoverability_v3_d10_plans.csv
echo   hanoi_recoverability_v3_d10_two_outages.csv
echo   hanoi_recoverability_v3_d10_pairs.csv
echo   hanoi_recoverability_v3_d20_plans.csv
echo   hanoi_recoverability_v3_d20_two_outages.csv
echo   hanoi_recoverability_v3_d20_pairs.csv
goto :end

:fail
echo.
echo Probe failed. Keep the traceback and send it back.

:end
pause
