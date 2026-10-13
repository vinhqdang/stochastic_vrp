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


# ============================================================
# step_record refactor and mode rr
# ============================================================

BIG_ROUTES = [[1, 4, 7, 10, 12], [2, 3, 5, 8], [6, 9, 11]]   # 12 customers, depot 0
BIG_N_NODES = 13


def _big_synthetic(seed: int = 5):
    """Symmetric Manhattan-type distances on random grid points, integer demands."""
    rng = np.random.default_rng(seed)
    xy = rng.integers(0, 30, size=(BIG_N_NODES, 2))
    Db = np.zeros((BIG_N_NODES, BIG_N_NODES), dtype=np.int64)
    for i in range(BIG_N_NODES):
        for j in range(BIG_N_NODES):
            if i != j:
                Db[i, j] = int(np.abs(xy[i] - xy[j]).sum()) + 1
    dsc = rng.integers(0, 5, size=(BIG_N_NODES, N)).astype(np.float64)
    psc = rng.integers(0, 5, size=(BIG_N_NODES, N)).astype(np.float64)
    dsc[0] = 0.0
    psc[0] = 0.0
    return Db, dsc, psc


BIG_D, BIG_DSC, BIG_PSC = _big_synthetic()


def _same(a, b) -> bool:
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return True
    return a == b


def test_step_record_on_full_arrays_matches_repair_step():
    for c in (1, 4, 6):
        T = m4b._candidate_table(ROUTES, c, D, DSC, PSC, Q_MID)
        stats = m4b.step_record(T["p"], T["ap"], T["am"], T["varD"], T["M"])
        rec = m4b.repair_step(ROUTES, c, D, DSC, PSC, Q_MID)
        for key, val in stats.items():
            assert _same(rec[key], val), key
        assert list(rec)[:9] == ["M", "n_routes", "dropped", "p_1", "s", "orig_rank",
                                 "p_orig", "feasible_routes", "p_sorted_head"]
        assert set(stats) == set(rec) - {"M", "n_routes", "dropped", "orig_rank", "p_orig",
                                         "feasible_routes"}


@pytest.mark.parametrize("Q", [8.0, 10.0, 12.0, 14.0, 1e9])
@pytest.mark.parametrize("seed", [1, 2])
def test_rr_lazy_evaluation_matches_full(Q, seed):
    lazy_routes, lazy = m4b.rr_round(BIG_ROUTES, BIG_D, BIG_DSC, BIG_PSC, Q,
                                     np.random.default_rng(seed), full=False)
    full_routes, full = m4b.rr_round(BIG_ROUTES, BIG_D, BIG_DSC, BIG_PSC, Q,
                                     np.random.default_rng(seed), full=True)
    assert lazy_routes == full_routes
    assert [(k, c) for k, c, _ in lazy] == [(k, c) for k, c, _ in full]
    for (_, _, r1), (_, _, r2) in zip(lazy, full):
        assert r2["n_evaluated"] == r2["M"]
        assert 1 <= r1["n_evaluated"] <= r1["M"]
        for key in r2:
            if key != "n_evaluated":
                assert _same(r1[key], r2[key]), key


def test_rr_lazy_evaluation_truncates_when_candidates_are_easy():
    # Q huge: every candidate has p = 0, so k_e2 = 1 and j_stop = min(M, HEAD).
    _, steps = m4b.rr_round(BIG_ROUTES, BIG_D, BIG_DSC, BIG_PSC, 1e9, np.random.default_rng(1))
    recs = [r for _, _, r in steps]
    assert any(r["M"] > m4b.HEAD and r["n_evaluated"] == m4b.HEAD for r in recs)
    assert all(r["s"] == 1 and r["new_route"] == 0 for r in recs)


@pytest.mark.parametrize("Q", [8.0, 10.0, 12.0, 14.0, 1e9])
@pytest.mark.parametrize("seed", [1, 2, 3])
def test_rr_ruin_reinserts_each_customer_once(Q, seed):
    all_c = sorted(c for r in BIG_ROUTES for c in r)
    routes, steps = m4b.rr_round(BIG_ROUTES, BIG_D, BIG_DSC, BIG_PSC, Q, np.random.default_rng(seed))
    removed = [c for _, c, _ in steps]
    assert len(removed) == len(set(removed)) == max(1, int(m4b.Q_FRAC * len(all_c)))
    assert set(removed) <= set(all_c)
    assert sorted(c for r in routes for c in r) == all_c
    assert all(len(r) > 0 for r in routes)
    for k, _, rec in steps:
        assert rec is not None
        assert rec["chosen_rank"] == rec["s"]
        assert rec["new_route"] == int(rec["s"] == 0)


def test_rr_insertion_is_cheapest_feasible_candidate():
    Q = 12.0
    for c in (2, 6, 9):
        routes = [[x for x in r if x != c] for r in BIG_ROUTES]
        routes = [r for r in routes if r]
        reduced = [list(r) for r in routes]
        cands = []
        for ri, r in enumerate(reduced):
            for pos in range(1, len(r) + 2):
                u = r[pos - 2] if pos >= 2 else 0
                v = r[pos - 1] if pos <= len(r) else 0
                cands.append((float(BIG_D[u, c] + BIG_D[c, v] - BIG_D[u, v]), ri, pos))
        cands.sort()
        risks = [float((route_peak(_inserted(reduced, ri, pos, c)[ri], BIG_DSC, BIG_PSC) > Q).mean())
                 for _, ri, pos in cands]
        expected_s = next((j + 1 for j, pj in enumerate(risks) if pj <= m4b.ALPHA), 0)
        rec = m4b.rr_insert_step(routes, c, BIG_D, BIG_DSC, BIG_PSC, Q, {})
        assert rec["M"] == len(cands) and rec["s"] == expected_s
        if expected_s >= 1:
            _, ri, pos = cands[expected_s - 1]
            assert routes == _inserted(reduced, ri, pos, c)
        else:
            assert routes[-1] == [c] and rec["new_route"] == 1


def test_rr_summary_smoke():
    import pandas as pd

    _, steps = m4b.rr_round(BIG_ROUTES, BIG_D, BIG_DSC, BIG_PSC, 12.0, np.random.default_rng(4))
    rows = [{"instance": "SYN", "family": "VNN", "gate": "Det", "round": 1, "step": k,
             "customer": c, **rec} for k, c, rec in steps]
    text = m4b.summary_markdown(pd.DataFrame(rows), rr=True)
    assert text.startswith("# QUASAR M4b repair-step statistics (task B, mode rr")
    assert "## eps = 0.01" in text and "| VNN | Det |" in text
    assert "new route share" in text and "chosen rank median" in text


def test_family_of_region_and_legacy_names():
    assert m4b.family_of("VNN-HANOI") == "VNN"
    assert m4b.family_of("GD-SHENZHENSHI") == "GD"
    assert m4b.family_of("CON3-0") == "CON"
    assert m4b.family_of("SCA8-1") == "SCA"
    assert m4b.family_of("HANOI") == "city"
    assert "Region" in m4b.DATA_FAMILIES
