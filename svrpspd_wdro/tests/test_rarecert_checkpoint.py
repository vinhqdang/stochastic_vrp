import multiprocessing as mp

import numpy as np

from rarecert.checkpoint import load, resumable_map


def _sq(t):
    return {"x": t["x"], "y": np.float64(t["x"]) ** 2, "key": (t["x"], "a")}


def _key(t):
    return ("t", t["x"])


def test_resume_skips_finished(tmp_path):
    path = tmp_path / "ck.jsonl"
    tasks = [{"x": i} for i in range(5)]
    first = list(resumable_map(_sq, tasks[:3], _key, path))
    assert len(first) == 3 and len(load(path)) == 3
    with open(path, "a") as fh:
        fh.write('{"key": ["t", 9], "res": {"x"')  # truncated line from a killed run
    calls = []

    def counting(t):
        calls.append(t["x"])
        return _sq(t)

    out = list(resumable_map(counting, tasks, _key, path))
    assert calls == [3, 4]
    assert sorted(r["x"] for r in out) == list(range(5))
    assert out[0]["y"] == 0.0 and out[0]["key"] == [0, "a"]
    assert len(load(path)) == 5  # records after the truncated line survive


def test_pool(tmp_path):
    path = tmp_path / "ck.jsonl"
    tasks = [{"x": i} for i in range(6)]
    with mp.get_context("fork").Pool(2) as pool:
        out = list(resumable_map(_sq, tasks, _key, path, pool))
    assert sorted(r["x"] for r in out) == list(range(6))
    assert len(load(path)) == 6
