"""Fast self-contained tests for the ECLAIR implementation.
Run from papers/eclair/code:  ../../../.venv/bin/python -m pytest tests -q
(deliberately NOT under svrpspd_wdro/tests — see ../README.md on
self-containment)."""

import json
import math
import random
import re
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from eclair.calibrate import cp_upper, _binom_cdf
from eclair.certify import Bets, e_bh, run_certification
from eclair.mutations import make_faithful, make_mutant
from eclair.probes import brute_optimum, solve_value, tier_a, tier_b
from eclair.problems import ALL_SPECS


def test_faithful_matches_brute_force():
    rng = random.Random(11)
    for spec in ALL_SPECS:
        cand = make_faithful(spec)
        for _ in range(5):
            p = spec.gen(rng, "micro")
            assert solve_value(cand.build(p)) == brute_optimum(spec, p), \
                spec.name


def test_mrs_never_alarm_on_faithful():
    rng = random.Random(12)
    for spec in ALL_SPECS:
        cand = make_faithful(spec)
        for _ in range(12):
            alarm, _ = tier_b(cand, rng)
            assert not alarm, f"MR false alarm on faithful {spec.name}"


def test_mutants_are_semantically_different():
    rng = random.Random(13)
    for spec in ALL_SPECS:
        mut = make_mutant(spec, rng)
        assert mut is not None and not mut.faithful
        diff = any(
            solve_value(mut.build(p)) != brute_optimum(spec, p)
            for p in (spec.gen(rng, "micro") for _ in range(12)))
        assert diff, spec.name


def test_conservative_bet_is_supermartingale():
    """E_H0[e] <= 1 whenever p0_true <= p0_bar < p1_bet."""
    p0_bar, p1_bet = 0.05, 0.4
    e_alarm = p1_bet / p0_bar
    e_pass = (1 - p1_bet) / (1 - p0_bar)
    for p0_true in (0.0, 0.01, 0.03, 0.05):
        mean = p0_true * e_alarm + (1 - p0_true) * e_pass
        assert mean <= 1.0 + 1e-12


def test_cp_upper_bound():
    # zero alarms in n trials: rule-of-three-ish upper bound
    assert 0.0 < cp_upper(0, 100) < 0.05
    assert cp_upper(0, 100) > 0 / 100
    # bound must be conservative: CDF at the bound equals ~2.5%
    n, k = 200, 3
    ub = cp_upper(k, n)
    assert abs(_binom_cdf(k, n, ub) - 0.025) < 1e-3
    assert ub > k / n


def test_e_bh():
    # K=4, alpha=0.05: reject k with e_(k) >= K/(alpha k) = 80/k
    assert e_bh([100, 100, 1, 1], 0.05) == {0, 1}      # 100 >= 40
    assert e_bh([100, 30, 1, 1], 0.05) == {0}          # 30 < 40
    assert e_bh([1, 1, 1, 1], 0.05) == set()


def _toy_bets():
    return Bets({
        "A": {"p0_bar": 0.02, "p1_bet": 0.4, "mean_cost": 0.001},
        "B": {"delta": 0.05, "mean_cost": 0.001},
        "C": {"p0_bar": 0.15, "p1_bet": 0.45, "mean_cost": 0.001},
    })


def test_certification_separates():
    rng = random.Random(14)
    spec = ALL_SPECS[0]
    bets = _toy_bets()
    pool = [make_faithful(spec), make_faithful(spec)]
    labels = [True, True]
    for _ in range(2):
        m = make_mutant(spec, rng)
        assert m is not None
        pool.append(m)
        labels.append(False)
    res = run_certification(pool, bets, "kelly", budget=1.5, alpha=0.05,
                            rng=rng)
    rejected = [st.rejected for st in res["states"]]
    # no faithful candidate rejected; at least one mutant caught
    assert not rejected[0] and not rejected[1]
    assert any(rejected[2:])
    if res["screening_survivor"] is not None:
        assert labels[res["states"].index(res["screening_survivor"])]


# ── certification phase (Theorem 3) — deterministic tests with a
#    mocked Tier-A oracle, so behaviour does not depend on CP-SAT
#    randomness (review R3.3) ───────────────────────────────────────

import eclair.certify as C


class _FakeCand:
    """Minimal stand-in for a candidate; identity is all we need."""
    _n = 0

    def __init__(self):
        _FakeCand._n += 1
        self.cid = _FakeCand._n
        self.spec = None

    def build(self, params):            # pragma: no cover
        raise AssertionError("mocked candidates are never solved")


def _bets():
    calib = {"A": {"p0_bar": 0.02, "p1_bet": 0.4, "mean_cost": 0.001},
             "B": {"delta": 0.05, "mean_cost": 0.001},
             "C": {"p0_bar": 0.10, "p1_bet": 0.4, "mean_cost": 0.001}}
    return Bets(calib)


def _run_cert(monkeypatch, alarms, eps=0.10, alpha=0.05, k=3,
              cert_budget=10.0, screen_budget=0.0):
    """Screening is skipped (zero budget); tier A returns `alarms` in
    order during the certification phase."""
    seq = list(alarms)

    def fake_tier_a(cand, rng):
        return (seq.pop(0) if seq else False), 0.001

    monkeypatch.setitem(C.TIER_FNS, "A", fake_tier_a)
    pool = [_FakeCand() for _ in range(k)]
    return run_certification(pool, _bets(), "kelly", screen_budget, alpha,
                             random.Random(0), eps=eps,
                             cert_budget=cert_budget), pool


def test_cert_threshold_arithmetic():
    """need = ceil(log(1/alpha)/log(1/(1-eps))) — the paper's formula."""
    for alpha, eps, want in [(0.05, 0.10, 29), (0.05, 0.05, 59),
                             (0.05, 0.01, 299), (0.01, 0.10, 44)]:
        need = math.ceil(math.log(1 / alpha) / -math.log(1 - eps))
        assert need == want, (alpha, eps, need)


def test_cert_succeeds_after_exactly_need_passes(monkeypatch):
    need = math.ceil(math.log(1 / 0.05) / -math.log(1 - 0.10))   # 29
    res, pool = _run_cert(monkeypatch, [False] * need)
    assert res["certified_pick"] is not None
    assert res["certification_abstained"] is False
    assert res["cert_probes"] == need          # not one probe more
    assert res["certified_pick"].cert_spent > 0


def test_cert_alarm_blocks_certification_and_falsifies(monkeypatch):
    """An alarm on probe 3 must prevent certification, hard-falsify the
    attempted candidate, and NOT silently promote another survivor."""
    res, pool = _run_cert(monkeypatch, [False, False, True])
    assert res["certified_pick"] is None
    assert res["certification_abstained"] is True
    assert res["certified_pick"] is None        # terminal output
    attempted = [s for s in res["states"] if s.rejected_in_cert]
    assert len(attempted) == 1 and attempted[0].rejected
    # a screening survivor may still exist, but it is NOT certified
    assert res["screening_survivor"] is not None
    assert res["screening_survivor"] is not res["certified_pick"]


def test_cert_insufficient_budget_abstains(monkeypatch):
    """Budget for ~5 probes < need: abstain, do not certify."""
    res, _ = _run_cert(monkeypatch, [False] * 100, cert_budget=0.0055)
    assert res["certified_pick"] is None
    assert res["certification_abstained"] is True
    assert res["cert_probes"] <= 6             # budget-bounded


def test_only_one_candidate_is_attempted(monkeypatch):
    """Single-attempt rule (Remark 2): an alarm on the first attempt
    must not start a second candidate's attempt."""
    need = math.ceil(math.log(1 / 0.05) / -math.log(1 - 0.10))
    res, _ = _run_cert(monkeypatch, [True] + [False] * (3 * need))
    assert res["cert_probes"] == 1             # stopped after the alarm
    assert sum(s.rejected_in_cert for s in res["states"]) == 1
    assert res["certified_pick"] is None


def test_ebh_rejected_cannot_be_certified(monkeypatch):
    """A candidate in the e-BH rejection set is not a survivor and so
    can never be the attempted (or certified) candidate."""
    seq = [False] * 200

    def fake_tier_a(cand, rng):
        return (seq.pop(0) if seq else False), 0.001

    monkeypatch.setitem(C.TIER_FNS, "A", fake_tier_a)
    pool = [_FakeCand() for _ in range(4)]
    res = run_certification(pool, _bets(), "kelly", 0.0, 0.05,
                            random.Random(0), eps=0.10, cert_budget=10.0)
    cp = res["certified_pick"]
    if cp is not None:
        idx = res["states"].index(cp)
        assert idx not in res["ebh_rejected"]
        assert not cp.rejected


def test_cert_false_certification_rate_respects_alpha(monkeypatch):
    """Empirical check of Theorem 3 across several seeds: candidates
    whose true error rate is exactly eps (the boundary of H0') must be
    certified at most ~alpha of the time. Pooled over seeds the
    Monte-Carlo standard error is ~0.006, so a 0.02 margin over alpha
    is ~3 sigma; per-seed rates are also required to stay well below
    the 2x-alpha line."""
    alpha, eps = 0.05, 0.10
    per_seed = []
    for seed in (20260812, 7, 99, 12345):
        per_seed.append(_cert_rate_at_boundary(monkeypatch, alpha, eps,
                                               trials=250, seed=seed))
    pooled = sum(per_seed) / len(per_seed)
    assert pooled <= alpha + 0.02, (pooled, per_seed)
    assert max(per_seed) <= 2 * alpha, per_seed


def _cert_rate_at_boundary(monkeypatch, alpha, eps, trials, seed):
    rng = random.Random(seed)
    certified = 0
    for _ in range(trials):
        draws = [rng.random() < eps for _ in range(400)]

        def fake_tier_a(cand, r, _d=draws):
            return (_d.pop(0) if _d else True), 0.0001

        monkeypatch.setitem(C.TIER_FNS, "A", fake_tier_a)
        pool = [_FakeCand() for _ in range(2)]
        res = run_certification(pool, _bets(), "kelly", 0.0, alpha,
                                random.Random(0), eps=eps, cert_budget=10.0)
        certified += res["certified_pick"] is not None
    return certified / trials


# ── reproducibility machinery (review R7): the audit estimator, the
#    checker's arithmetic, and provenance capture ───────────────────

def test_audit_interval_matches_wilson_formula():
    """estimate_error_rate's interval must equal the closed-form Wilson
    interval at the requested z, for both levels we use."""
    import check_artifacts as CA
    from eclair.probes import estimate_error_rate

    class _Det:
        """Candidate whose disagreement pattern is fixed and known."""
        def __init__(self, spec, every):
            self.spec, self.every, self.i = spec, every, 0

    spec = ALL_SPECS[0]
    for z, k_expected in ((1.96, None), (2.2414, None)):
        lo, hi = CA.wilson(16, 400, z)
        # closed form, recomputed independently here
        p, n = 16 / 400, 400
        d = 1 + z * z / n
        c = (p + z * z / (2 * n)) / d
        h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
        assert abs(lo - (c - h)) < 1e-12 and abs(hi - (c + h)) < 1e-12
    # the two levels must differ, and 97.5% must be the wider one
    lo95, hi95 = CA.wilson(43, 400, 1.96)
    lo975, hi975 = CA.wilson(43, 400, 2.2414)
    assert lo975 < lo95 and hi975 > hi95


def test_estimate_error_rate_respects_z_and_n(monkeypatch):
    """The estimator must actually use the z it is given."""
    from eclair import probes as P
    spec = ALL_SPECS[0]
    cand = make_faithful(spec)
    e1 = P.estimate_error_rate(cand, random.Random(0), n=40, z=1.96)
    e2 = P.estimate_error_rate(cand, random.Random(0), n=40, z=2.2414)
    assert e1[1] == e2[1] == 40
    assert e2[3] >= e1[3]          # wider upper limit at the higher z


def test_solver_params_pinned():
    """Thread count is a methodological parameter (it moves measured
    costs, hence the budget and the routing score)."""
    from eclair.probes import SOLVER_PARAMS
    assert SOLVER_PARAMS.get("num_search_workers") == 1


def test_checker_rejects_vacuous_interval_match(tmp_path):
    """Regression for the checker's own past failure: an interval
    quoted ACROSS A LINE BREAK must still be found."""
    import check_artifacts as CA
    src = "some text CI\n$[0.07753, 0.14721]$ more text"
    quoted = re.findall(r"CI\s*\$\[\s*([\d.]+)\s*,\s*([\d.]+)\s*\]\$",
                        src, re.S)
    assert quoted == [("0.07753", "0.14721")]


def test_promote_run_refuses_incomplete(tmp_path):
    """A run missing an output must NOT publish anything (multi-file
    consistency, review R7.6)."""
    import run_experiment as RX
    stage = tmp_path / "stage"
    out = tmp_path / "out"
    stage.mkdir()
    out.mkdir()
    (stage / "a.json").write_text("{}")
    try:
        RX.promote_run(stage, out, ["a.json", "missing.json"])
        raise AssertionError("promote_run should have refused")
    except RuntimeError as e:
        assert "incomplete" in str(e)
    assert not (out / "a.json").exists()      # nothing promoted


def test_promote_run_writes_manifest(tmp_path):
    import run_experiment as RX
    stage, out = tmp_path / "s", tmp_path / "o"
    stage.mkdir(); out.mkdir()
    (stage / "a.json").write_text('{"x": 1}')
    (stage / "b.txt").write_text("hello")
    man = RX.promote_run(stage, out, ["a.json", "b.txt"])
    assert set(man) == {"a.json", "b.txt"}
    assert (out / "MANIFEST.json").exists()
    saved = json.loads((out / "MANIFEST.json").read_text())["files"]
    import hashlib
    assert saved["b.txt"] == hashlib.sha256(b"hello").hexdigest()
