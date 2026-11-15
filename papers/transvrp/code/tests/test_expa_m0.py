"""M0: copied BATON modules and requirements."""
import filecmp
from pathlib import Path

from transvrp.benchmark import instance_table  # noqa: F401  (import check)
from transvrp.baton.dp_exec import fit_dp
from transvrp.data import route_problem

CODE = Path(__file__).resolve().parent.parent
SRC = CODE.parent.parent.parent / "svrpspd_wdro" / "core"


def test_copied_modules_are_byte_identical():
    for name in ("dp_exec.py", "otr21.py"):
        assert filecmp.cmp(SRC / name, CODE / "transvrp" / "baton" / name,
                           shallow=False)


def test_requirements_file_exists():
    assert (CODE / "requirements-expA.txt").read_text().strip()


def test_fit_dp_keys_on_con3_route0():
    p = next(route_problem("CON3-0", n_train=500, n_test=10))
    m = len(p["route"])
    model = fit_dp(p["g_train"], p["B"], p["H"], p["E"])
    assert set(model.keys()) == set(range(1, m))
