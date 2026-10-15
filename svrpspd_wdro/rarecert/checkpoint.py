"""Resumable task maps for the long RARECERT runs.

Each finished task result is appended as one JSON line {"key": ..., "res": ...} to a
checkpoint file in the output directory. A rerun with the same output directory loads
the finished results, yields them first and runs only the remaining tasks, so a killed
run loses at most the tasks that were in progress.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _plain(x):
    """JSON-safe copy: numpy scalars and arrays to Python, tuples to lists."""
    if isinstance(x, dict):
        return {str(k): _plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_plain(v) for v in x]
    if isinstance(x, np.ndarray):
        return _plain(x.tolist())
    if isinstance(x, np.generic):
        return x.item()
    return x


def _key(k) -> str:
    return json.dumps(_plain(k))


def load(path: Path) -> dict:
    """Finished results by key string. A truncated last line (killed mid-write) is skipped."""
    done = {}
    path = Path(path)
    if not path.exists():
        return done
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                doc = json.loads(line)
            except json.JSONDecodeError:
                continue
            done[_key(doc["key"])] = doc["res"]
    return done


def append(path: Path, key, res) -> None:
    with open(path, "a") as fh:
        fh.write(json.dumps({"key": _plain(key), "res": _plain(res)}, allow_nan=True) + "\n")
        fh.flush()


def resumable_map(fn, tasks, keyfn, path, pool=None):
    """Yield fn(t) for every task: checkpointed results first, then new ones as they finish.

    pool: a multiprocessing pool (imap_unordered) or None for a serial map.
    """
    path = Path(path)
    done = load(path)
    todo = []
    for t in tasks:
        k = _key(keyfn(t))
        if k in done:
            yield done[k]
        else:
            todo.append(t)
    if not todo:
        return
    if path.exists() and path.stat().st_size:
        with open(path, "rb") as fh:
            fh.seek(-1, 2)
            if fh.read(1) != b"\n":  # killed mid-write: start the next record on a new line
                with open(path, "a") as fa:
                    fa.write("\n")
    keys = {id(t): keyfn(t) for t in todo}
    if pool is None:
        for t in todo:
            res = fn(t)
            append(path, keys[id(t)], res)
            yield res
        return
    for t_key, res in pool.imap_unordered(_KeyedCall(fn, keyfn), todo, chunksize=1):
        append(path, t_key, res)
        yield res


class _KeyedCall:
    """Picklable wrapper returning (key, fn(task)), so results map back to their tasks."""

    def __init__(self, fn, keyfn):
        self.fn, self.keyfn = fn, keyfn

    def __call__(self, task):
        return self.keyfn(task), self.fn(task)

