"""verify_extensions.py -- independent checks of the results added in the
second revision (equal-size sites on vehicles with different dispatch
times, hardness for a constant hazard-arrival time, the refined FPTAS
guarantee and its tight families, the value-indexed recursion, tie
handling in Lemma 1, the union-find variant, and the Partition
preprocessing of Theorem 2). Each script in checks/ exits non-zero on any
mismatch. Run: python3 verify_extensions.py (writes
verify_extensions_results.txt)."""
import subprocess
import sys
from pathlib import Path

here = Path(__file__).resolve().parent
out, rc = [], 0
for i in (1, 2, 3, 4):
    r = subprocess.run([sys.executable, str(here / "checks" / f"check_item{i}.py")],
                       capture_output=True, text=True, cwd=here)
    out.append(f"== checks/check_item{i}.py (exit {r.returncode})")
    out.append(r.stdout.rstrip())
    if r.stderr.strip():
        out.append(r.stderr.rstrip())
    rc |= r.returncode
out.append("ALL EXTENSION CHECKS PASSED" if rc == 0 else "FAILURES")
text = "\n".join(out) + "\n"
(here / "verify_extensions_results.txt").write_text(text)
print(text)
sys.exit(rc)
