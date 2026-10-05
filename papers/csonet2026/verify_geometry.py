"""verify_geometry.py -- independent brute-force checks of the results of
Section 4.7 (hazard geometry): the spoke-chain gap (Theorems 13-14,
Proposition 15), the chained route on a line (Theorems 17-18), affine
deadlines and a front from the depot (Proposition 19, Theorems 20-21).
The scripts in checks/geometry/ were written to test the statements,
exhaustively on small instances, and exit non-zero on any mismatch.
Run: python3 verify_geometry.py (about 45 s; writes
verify_geometry_results.txt)."""
import subprocess
import sys
from pathlib import Path

here = Path(__file__).resolve().parent
d = here / "checks" / "geometry"
out, rc = [], 0
for script in ("h1_verify_basic.py", "h1_thmA.py", "h2_run_all.py"):
    r = subprocess.run([sys.executable, script], capture_output=True,
                       text=True, cwd=d)
    out.append(f"== checks/geometry/{script} (exit {r.returncode})")
    out.append(r.stdout.rstrip())
    if r.stderr.strip():
        out.append(r.stderr.rstrip())
    rc |= r.returncode
out.append("ALL GEOMETRY CHECKS PASSED" if rc == 0 else "FAILURES")
text = "\n".join(out) + "\n"
(here / "verify_geometry_results.txt").write_text(text)
print(text[-1500:])
sys.exit(rc)
