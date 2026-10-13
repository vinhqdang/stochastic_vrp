"""Tests for scripts/run_quasar_m4b.py (QUASAR M4b task B: repair-step statistics).

Covers the candidate enumeration and sorting of a repair step, the agreement of
the insertion-peak risks with the explicitly inserted routes, the CRN identity
a_plus - a_minus = p_j - p_1, the k* / s consistency with quantum/search.py, and
two extreme-Q cases. Synthetic plan, hand-made symmetric distances, N = 4000.
Fast."""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest

from quantum.experiments import route_peak
from quantum.search import kstar, stop_index

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_quasar_m4b.py"


def _load_driver():
    spec = importlib.util.spec_from_file_location("run_quasar_m4b", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


m4b = _load_driver()

N = 4000
N_NODES = 7                       # depot 0 and customers 1..6
ROUTES = [[1, 3, 5], [2, 4, 6]]   # the synthetic plan
Q_MID = 11.0


def _distances() -> np.ndarray:
    """Hand-made symmetric distances with zero diagonal (depot = 0)."""
    D = np.zeros((N_NODES, N_NODES), dtype=np.int64)
    for i in range(N_NODES):
        for j in range(N_NODES):
            if i != j:
                D[i, j] = abs(i - j) + (i + j) % 3 + 1
    return D


def _scenarios(seed: int = 3):
    """Integer-valued demands, so every load and peak is exact in float64."""
    rng = np.random.default_rng(seed)
    dsc = rng.integers(0, 5, size=(N_NODES, N)).astype(np.float64)
    psc = rng.integers(0, 5, size=(N_NODES, N)).astype(np.float64)
    dsc[0] = 0.0
    psc[0] = 0.0
    return dsc, psc


D = _distances()
DSC, PSC = _scenarios()


def _reduced(routes, c):
    out = []
    for r in routes:
        rr = [x for x in r if x != c]
        if rr:
            out.append(rr)
    return out


def _inserted(reduced, ri, pos, c):
    r = reduced[ri]
    return [list(reduced[k]) if k != ri else r[:pos - 1] + [c] + r[pos - 1:] for k in range(len(reduced))]


def test_candidate_count_and_delta_sorting():
    c = 4
    T = m4b._candidate_table(ROUTES, c, D, DSC, PSC, Q_MID)
    reduced = _reduced(ROUTES, c)
    assert T["M"] == sum(len(r) + 1 for r in reduced) == 7
    assert T["reduced"] == reduced
    # Deltas recomputed from the distance matrix, and the list is sorted by (delta, ri, pos).
    recomputed = []
    for ri, r in enumerate(reduced):
        for pos in range(1, len(r) + 2):
            u = r[pos - 2] if pos >= 2 else 0
            v = r[pos - 1] if pos <= len(r) else 0
            recomputed.append((float(D[u, c] + D[c, v] - D[u, v]), ri, pos))
    assert T["cands"] == sorted(recomputed)
    assert T["cands"] == sorted(T["cands"])
    # The original position is a candidate with rank orig_rank and risk p_orig.
    assert 1 <= T["orig_rank"] <= T["M"]
    assert T["p"][T["orig_rank"] - 1] == T["p_orig"]


def test_dropped_route_has_no_original_candidate():
    routes = [[1, 3, 5], [2]]
    T = m4b._candidate_table(routes, 2, D, DSC, PSC, Q_MID)
    assert T["dropped"] and T["orig_rank"] == 0 and math.isnan(T["p_orig"])
    assert T["M"] == 4 and len(T["reduced"]) == 1
    assert m4b.repair_step([[2]], 2, D, DSC, PSC, Q_MID) is None


def test_insertion_risks_match_explicit_routes():
    c = 4
    T = m4b._candidate_table(ROUTES, c, D, DSC, PSC, Q_MID)
    reduced = T["reduced"]
    for j, (_, ri, pos) in enumerate(T["cands"]):
        new = _inserted(reduced, ri, pos, c)[ri]
        explicit = float((route_peak(new, DSC, PSC) > Q_MID).mean())
        assert T["p"][j] == explicit, f"candidate {j} (route {ri}, pos {pos})"


def test_original_position_reproduces_the_plan():
    c = 4
    T = m4b._candidate_table(ROUTES, c, D, DSC, PSC, Q_MID)
    orig_route = ROUTES[1]
    assert T["p_orig"] == float((route_peak(orig_route, DSC, PSC) > Q_MID).mean())
    # Base risks of the reduced routes come from their own peaks.
    for ri, r in enumerate(T["reduced"]):
        assert T["base_risk"][ri] == float((route_peak(r, DSC, PSC) > Q_MID).mean())


def test_crn_identity_a_plus_minus_a_minus_equals_p_minus_p1():
    c = 4
    T = m4b._candidate_table(ROUTES, c, D, DSC, PSC, Q_MID)
    diff = (T["ap"] - T["am"]) - (T["p"] - T["p"][0])
    assert np.max(np.abs(diff)) < 1e-12
    varD = T["ap"] + T["am"] - (T["ap"] - T["am"]) ** 2
    assert np.allclose(varD, T["varD"], rtol=0, atol=1e-15)


@pytest.mark.parametrize("c", [1, 4, 6])
def test_k_and_s_consistent_with_search(c):
    rec = m4b.repair_step(ROUTES, c, D, DSC, PSC, Q_MID)
    T = m4b._candidate_table(ROUTES, c, D, DSC, PSC, Q_MID)
    p = T["p"]
    assert rec["M"] == T["M"] == len(p)
    assert rec["p_1"] == p[0]
    assert rec["s"] == stop_index(p, m4b.ALPHA)
    for eps in m4b.EPS:
        assert rec[f"k_{m4b.EPS_TAG[eps]}"] == kstar(p, m4b.ALPHA, eps)
    head = [float(x) for x in rec["p_sorted_head"].split()]
    assert len(head) == min(len(p), m4b.HEAD)


def test_every_feasibility_band_and_cost_column_is_finite():
    rec = m4b.repair_step(ROUTES, 4, D, DSC, PSC, Q_MID)
    for eps in m4b.EPS:
        t = m4b.EPS_TAG[eps]
        for stem in ("Nq_full_wc", "Nq_full_opt", "Nc_full_wc", "Nc_full_opt", "Nq_core",
                     "Nc_core", "Nq_crn_core", "Nc_crn_core", "v_opt", "vD_opt", "varD_mean"):
            assert math.isfinite(rec[f"{stem}_{t}"]), f"{stem}_{t}"
        assert 0 <= rec[f"band_{t}"] <= rec["M"]


def test_extreme_q_all_feasible_and_all_infeasible():
    rec = m4b.repair_step(ROUTES, 4, D, DSC, PSC, 1e9)
    assert rec["p_1"] == 0.0 and rec["s"] == 1
    for eps in m4b.EPS:
        t = m4b.EPS_TAG[eps]
        assert rec[f"k_{t}"] == 1
        assert rec[f"band_{t}"] == 0
        assert rec[f"v_opt_{t}"] == pytest.approx((m4b.ALPHA - eps) * (1 - m4b.ALPHA + eps))
    rec0 = m4b.repair_step(ROUTES, 4, D, DSC, PSC, -1.0)
    assert rec0["s"] == 0
    for eps in m4b.EPS:
        assert rec0[f"k_{m4b.EPS_TAG[eps]}"] == 0


def test_summary_smoke():
    import pandas as pd

    rows = []
    for c in (1, 4, 6):
        rec = m4b.repair_step(ROUTES, c, D, DSC, PSC, Q_MID)
        rows.append({"instance": "SYN", "family": "city", "gate": "Det", "customer": c, **rec})
    text = m4b.summary_markdown(pd.DataFrame(rows))
    assert "## eps = 0.01" in text and "| ALL | ALL |" in text
