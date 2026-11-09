"""H = None (and H identically one) must reproduce the homogeneous code bit for bit.

The reference values come from the original tree (papers/sortie/code), run in a
subprocess that writes no bytecode and no numba cache there. Set SORTIE_ORIG_CODE to
point at another copy; the test is skipped when that tree is absent.
"""

import json
import os
import subprocess
import sys
from dataclasses import replace

import numpy as np
import pytest

from identity_reference import compute
from sortie.instances import DATA_PATH, load_x641, make_instance
from sortie.search import solve_fair
from sortie.instances import noise_matrix

TESTS = os.path.dirname(os.path.abspath(__file__))
ORIG_CODE = os.environ.get(
    "SORTIE_ORIG_CODE", "/home/user/stochastic_vrp/papers/sortie/code")

pytestmark = pytest.mark.skipif(
    not os.path.isdir(os.path.join(ORIG_CODE, "sortie")),
    reason="original homogeneous code tree not available",
)


@pytest.fixture(scope="module")
def reference_orig(tmp_path_factory):
    cache = str(tmp_path_factory.mktemp("numba_cache"))
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", NUMBA_CACHE_DIR=cache)
    script = ("import sys, json; sys.path.insert(0, sys.argv[1]); sys.path.insert(1, sys.argv[2]); "
              "import identity_reference as r; print(json.dumps(r.compute(sys.argv[3])))")
    proc = subprocess.run([sys.executable, "-c", script, ORIG_CODE, TESTS, DATA_PATH],
                          capture_output=True, text=True, env=env, cwd=cache, timeout=900)
    assert proc.returncode == 0, proc.stderr[-2000:]
    return json.loads(proc.stdout.strip().splitlines()[-1])


@pytest.fixture(scope="module")
def reference_dev():
    return compute(DATA_PATH)


def test_homogeneous_outputs_identical_to_original(reference_orig, reference_dev):
    assert reference_dev.keys() == reference_orig.keys()
    for key in reference_orig:
        assert reference_dev[key] == reference_orig[key], key


def test_identically_one_multiplier_equals_none():
    coords, demand = load_x641(DATA_PATH)
    base = make_instance(coords, demand, 15, 0, m=5)
    ones = replace(base, H=np.ones_like(base.C))
    assert np.array_equal(noise_matrix(ones), base.C)
    start = [[1, 2, 3, 4], [5, 6, 7], [8, 9, 10], [11, 12, 13], [14, 15]]
    budget = 1.10 * sum(sum(base.C[a, b] for a, b in zip([0] + r, r + [0])) for r in start)
    kw = dict(mode="pboi", measure="range", v=0.3, kappa=1.0, budget=budget, time_limit=1e6,
              seed=11, max_iters=15)
    r0 = solve_fair(base, start, **kw)
    r1 = solve_fair(ones, start, **kw)
    assert r0["value"] == r1["value"]
    assert r0["routes"] == r1["routes"]
    assert r0["W"] == r1["W"]
