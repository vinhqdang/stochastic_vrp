"""Model pickles must not record classes under a main-module name.

Running ``python -m transvrp.expa.sweep`` used to pickle ``Job`` as ``__main__.Job``
(``__mp_main__.Job`` in spawn workers), so the pilot report could not load the models.
The CLI now runs through the importable module, and ``sweep.load_payload`` still
resolves the names that the existing pickles carry.
"""
import contextlib
import pickle
import subprocess
import sys
import types
from pathlib import Path

import pytest

from transvrp.expa import sweep

CODE_DIR = Path(__file__).resolve().parents[1]
_MISSING = object()


def _run_module(module, args):
    return subprocess.run([sys.executable, "-m", module, *args], cwd=CODE_DIR,
                          capture_output=True, text=True, timeout=900)


def _run_cli(args):
    proc = _run_module("transvrp.expa.sweep", args)
    assert proc.returncode == 0, proc.stderr
    return proc


@contextlib.contextmanager
def _pickled_as(cls, modname):
    """Make ``cls`` pickle under ``modname``, as the pre-fix CLI did."""
    created = modname not in sys.modules
    if created:
        sys.modules[modname] = types.ModuleType(modname)
    mod = sys.modules[modname]
    old_attr = getattr(mod, cls.__name__, _MISSING)
    old_module = cls.__module__
    setattr(mod, cls.__name__, cls)
    cls.__module__ = modname
    try:
        yield
    finally:
        cls.__module__ = old_module
        if old_attr is _MISSING:
            delattr(mod, cls.__name__)
        else:
            setattr(mod, cls.__name__, old_attr)
        if created:
            del sys.modules[modname]


def test_cli_fit_pickles_job_under_its_importable_name(tmp_path):
    _run_cli(["fit", "--stage", "smoke", "--jobs", "2", "--results-dir", str(tmp_path)])
    files = sorted((tmp_path / "models" / "smoke").glob("*.pkl"))
    assert len(files) == 5
    for path in files:
        raw = path.read_bytes()
        assert b"__main__" not in raw and b"__mp_main__" not in raw
        with open(path, "rb") as fh:          # plain pickle: no remapping involved
            job = pickle.load(fh)["job"]
        assert type(job) is sweep.Job
        assert type(job).__module__ == "transvrp.expa.sweep"


@pytest.mark.parametrize("modname", ["__main__", "__mp_main__"])
def test_load_payload_restores_legacy_main_job(tmp_path, modname):
    job = sweep.Job("neural", "CON3-0", 0, 100, 0, arm="full")
    with _pickled_as(sweep.Job, modname):
        blob = pickle.dumps({"job": job, "kind": "neural"})
    assert modname.encode() in blob
    path = tmp_path / "legacy.pkl"
    path.write_bytes(blob)
    payload = sweep.load_payload(path)
    assert type(payload["job"]) is sweep.Job
    assert type(payload["job"]).__module__ == "transvrp.expa.sweep"
    assert payload["job"] == job


class NotHere:                          # not defined in sweep: must not be remapped
    pass


def test_load_payload_leaves_other_main_names_alone(tmp_path):
    with _pickled_as(NotHere, "__main__"):
        blob = pickle.dumps(NotHere())
    path = tmp_path / "other.pkl"
    path.write_bytes(blob)
    with pytest.raises(AttributeError):
        sweep.load_payload(path)


def test_pilot_cli_loads_smoke_models_including_legacy_names(tmp_path):
    _run_cli(["fit", "--stage", "smoke", "--results-dir", str(tmp_path)])
    _run_cli(["eval", "--stage", "smoke", "--eval", "dev", "--results-dir", str(tmp_path)])
    _run_cli(["merge", "--stage", "smoke", "--results-dir", str(tmp_path)])
    models = sorted((tmp_path / "models" / "smoke").glob("*.pkl"))
    legacy = models[0]
    with open(legacy, "rb") as fh:
        payload = pickle.load(fh)
    with _pickled_as(sweep.Job, "__main__"):
        legacy.write_bytes(pickle.dumps(payload))
    proc = _run_module("transvrp.expa.pilot", ["--results-dir", str(tmp_path),
                                               "--stage", "smoke"])
    assert proc.returncode == 0, proc.stderr
    assert "AttributeError" not in proc.stderr
    assert (tmp_path / "smoke_report.md").exists()
