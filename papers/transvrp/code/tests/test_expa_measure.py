"""Portable process RSS measurement, and UTF-8 reading of the research log by the gate."""
import subprocess
import sys
from pathlib import Path

from transvrp.expa import paths
from transvrp.expa.measure import process_peak_rss_mb

GOOD_HASH = "a" * 40
CODE_DIR = Path(__file__).resolve().parents[1]

# Runs in a fresh interpreter: the process peak is monotone for the whole pytest session,
# so an in-process check depends on what earlier tests already reached.
_RSS_PROBE = (
    "import numpy as np\n"
    "from transvrp.expa.measure import process_peak_rss_mb as f\n"
    "before = f()\n"
    "block = np.ones(25_000_000)\n"          # about 200 MB of float64, touched by the sum
    "assert block.sum() > 0\n"
    "print(before, f())\n"
)


def test_process_peak_rss_is_positive_and_monotone():
    out = subprocess.run([sys.executable, "-c", _RSS_PROBE], cwd=CODE_DIR, capture_output=True,
                         text=True, check=True).stdout.split()
    before, after = float(out[0]), float(out[1])
    assert before > 0
    assert after >= before + 100


def test_gate_reads_research_log_as_utf8(tmp_path, monkeypatch):
    # 'Á' is c3 81 in UTF-8 and 0x81 is undefined in cp1252, so a non-UTF-8 read fails here.
    text = f"≤ é Á frozen {GOOD_HASH}\n"
    log = tmp_path / "RESEARCH_LOG.md"
    log.write_bytes(text.encode("utf-8"))
    monkeypatch.setattr(paths, "RESEARCH_LOG", log)
    monkeypatch.setattr(paths, "current_commit", lambda *a, **k: GOOD_HASH)
    monkeypatch.setattr(paths, "tree_is_clean", lambda *a, **k: True)
    assert paths.verify_test_gate(GOOD_HASH).commit == GOOD_HASH
