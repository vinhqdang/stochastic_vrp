import math
import os

import numpy as np
import pytest

from sortie.instances import DATA_PATH, load_x641, make_instance

DATA = os.path.join(os.path.dirname(__file__), "..", "..", "data", "X-n641-k35.vrp")


@pytest.fixture(scope="module")
def x641():
    return load_x641(DATA)


def test_data_path_resolves():
    assert os.path.normpath(DATA) == os.path.normpath(DATA_PATH)
    assert os.path.exists(DATA_PATH)


def test_load_shapes_and_first_node(x641):
    coords, demand = x641
    assert coords.shape == (641, 2)
    assert demand.shape == (641,)
    assert coords.dtype == np.float64
    assert np.array_equal(coords[0], [0.0, 0.0])
    assert demand[0] == 0
    assert demand[1] == 59 and demand[2] == 87


def test_block_counts():
    assert 641 // 16 == 40
    assert 641 // 36 == 17


def test_make_instance_n15_block0(x641):
    coords, demand = x641
    inst = make_instance(coords, demand, n=15, b=0)
    assert inst.name == "X641-n15-b0"
    assert inst.m == 5
    assert inst.coords.shape == (16, 2)
    assert inst.demand.shape == (16,)
    assert inst.demand[0] == 0
    assert np.array_equal(inst.demand[1:], demand[1:16])
    assert inst.Q == int(math.floor(demand[1:16].sum() / 4.0))
    assert inst.C.shape == (16, 16)
    assert np.array_equal(inst.C, inst.C.T)
    assert np.all(np.diag(inst.C) == 0)


def test_distances_are_ceiled_euclidean(x641):
    coords, demand = x641
    inst = make_instance(coords, demand, n=15, b=0)
    for i in range(16):
        for j in range(16):
            if i == j:
                continue
            d = math.hypot(*(coords[i] - coords[j]))
            assert inst.C[i, j] == math.ceil(d)
            assert inst.C[i, j] - 1 < d <= inst.C[i, j]


def test_block_index_bounds(x641):
    coords, demand = x641
    inst = make_instance(coords, demand, n=15, b=39)
    assert np.array_equal(inst.coords, coords[39 * 16:40 * 16])
    with pytest.raises(ValueError):
        make_instance(coords, demand, n=15, b=40)
