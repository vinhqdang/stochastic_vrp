"""Chunk publish under a Windows race: when another worker already holds or has published
the same deterministic chunk, os.replace fails with PermissionError and the local copy
must be discarded without changing the returned arrays."""
import numpy as np
import pytest

from transvrp.expa import paths

INST = "CON3-0"


def _no_pause(monkeypatch):
    monkeypatch.setattr(paths, "CHUNK_REPLACE_PAUSE_S", 0.0)


def test_other_worker_wins_race_returns_fresh_arrays_and_drops_tmp(tmp_path, monkeypatch):
    _no_pause(monkeypatch)
    expected = paths._generate_chunk(INST, "train", 0, 0)
    real_replace = paths.os.replace

    def other_worker_wins(src, dst):
        dst.write_bytes(b"published by another worker")   # its file is now at the destination
        raise PermissionError("destination open by another process")

    monkeypatch.setattr(paths.os, "replace", other_worker_wins)
    got = paths._chunk_arrays(INST, "train", 0, 0, tmp_path)
    monkeypatch.setattr(paths.os, "replace", real_replace)

    assert np.array_equal(got[0], expected[0]) and np.array_equal(got[1], expected[1])
    assert got[0].dtype == np.float64 and got[1].dtype == np.float64
    assert not list(tmp_path.glob("*.tmp.npy"))


def test_replace_retried_after_two_permission_errors(tmp_path, monkeypatch):
    _no_pause(monkeypatch)
    expected = paths._generate_chunk(INST, "train", 0, 0)
    real_replace = paths.os.replace
    calls = {"n": 0}

    def flaky_then_ok(src, dst):
        calls["n"] += 1
        if calls["n"] <= 2:
            raise PermissionError("transient")
        return real_replace(src, dst)

    monkeypatch.setattr(paths.os, "replace", flaky_then_ok)
    got = paths._chunk_arrays(INST, "train", 0, 0, tmp_path)

    assert calls["n"] == 3
    assert np.array_equal(got[0], expected[0]) and np.array_equal(got[1], expected[1])
    assert len(list(tmp_path.glob("*.npy"))) == 1
    assert not list(tmp_path.glob("*.tmp.npy"))


def test_replace_gives_up_after_the_retry_budget(tmp_path, monkeypatch):
    _no_pause(monkeypatch)
    calls = {"n": 0}

    def always_denied(src, dst):
        calls["n"] += 1
        raise PermissionError("denied")

    monkeypatch.setattr(paths.os, "replace", always_denied)
    with pytest.raises(PermissionError):
        paths._chunk_arrays(INST, "train", 0, 0, tmp_path)
    assert calls["n"] == paths.CHUNK_REPLACE_TRIES
