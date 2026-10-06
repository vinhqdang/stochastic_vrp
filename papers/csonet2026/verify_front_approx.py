"""verify_front_approx.py -- brute-force checks of Theorem 22 and Proposition 23
(constant-factor approximation of the chained route for a hazard front, the
failure of the round-trip surrogate on a ray, and strong NP-hardness at Gamma=1).

  python3 verify_front_approx.py        (about 1 minute)

* checks/front/indep.py : independent implementation. Exact chained optimum
  (Held-Karp over ordered subsets) against the algorithm of Theorem 22 run with
  an exact point-to-point orienteering oracle; asserts that the output is
  protected and that its weight is at least OPT/(P*m).
* checks/front/tests.py : the lemmas of the proof (excess monotone, shortcut,
  restart, splitting, concatenation, comparison), with an exact and with a
  deliberately weak oracle.
* checks/front/g3.py    : Proposition 23 (ray) and the star calibration.
* checks/front/ham.py   : Theorem 24 (Hamiltonian-path reduction, equidistant sites).
"""
import re, subprocess, sys
from pathlib import Path
here = Path(__file__).resolve().parent / "checks" / "front"
bad = 0
for cmd in ([sys.executable, "indep.py", "1", "150"], [sys.executable, "indep.py", "2", "150"],
            [sys.executable, "indep.py", "3", "150"], [sys.executable, "tests.py", "0"],
            [sys.executable, "tests.py", "1"], [sys.executable, "g3.py"], [sys.executable, "ham.py", "7"], [sys.executable, "ham.py", "8"]):
    r = subprocess.run(cmd, cwd=here, capture_output=True, text=True)
    print("==", " ".join(cmd[1:]), "(exit %d)" % r.returncode)
    print(r.stdout.strip()); 
    # failures are reported as "violations N", "FAIL <name>" lines or "TOTAL .. FAIL N"
    bad_counts = [m for m in re.findall(r"violations\s*=?\s*(\d+)", r.stdout) if int(m)]
    bad_counts += [m for m in re.findall(r"TOTAL \d+ FAIL (\d+)", r.stdout) if int(m)]
    if r.returncode or bad_counts or re.search(r"^FAIL ", r.stdout, re.M):
        bad += 1
print("ALL FRONT-APPROXIMATION CHECKS PASSED" if not bad else "FAILURES: %d" % bad)
sys.exit(1 if bad else 0)
