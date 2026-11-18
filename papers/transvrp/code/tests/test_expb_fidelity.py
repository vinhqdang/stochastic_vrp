"""Dev fidelity gates (EXPERIMENT_B.md section 7, M22). Every test uses constructed routes,
pools and payloads (a real THR fit on a constructed route for the end-to-end checks); no
test or test_b data is read. Each gate has a pass and a fail case, and the boundary values
of the section-0 thresholds are checked."""
import json
import math
import pickle
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from transvrp.expa import sweep
from transvrp.expa.backward import split_80_20
from transvrp.expa.features import RouteCache, RouteStats
from transvrp.expa.paths import Paths, RouteSpec
from transvrp.expa.simulate import simulate_policy
from transvrp.expb import codecs, fidelity as fd, gatecheck as gc
from transvrp.expb.dqn import stage_counts
from transvrp.expb.thr import fit_thr
from transvrp.expb.registry import B_ARMS

B, H, E = 4.0, np.array([1.0, 0.5]), np.array([3.0, 3.0])
SPEC = RouteSpec(inst="TOY", route_id=0, route=(1, 2), m=2, B=B, H=H, E=E, E0=3.0)
N_TRAIN, N_DEV = 2000, 1000


def draw(seed: int, n: int, scale: float = 1.0) -> Paths:
    rng = np.random.default_rng(seed)
    p = np.floor(rng.integers(0, 4, size=(n, 2)) * scale).astype(float)
    return Paths(p=p, d=np.zeros_like(p))


class FakeSource(fd.DevSource):
    """Constructed route data: one spec, one dev pool, one train pool (any rep)."""

    def __init__(self, dev: Paths, train: Paths, spec: RouteSpec = SPEC):
        self.dev, self.train, self.spec_ = dev, train, spec
        self.purposes = []

    def spec(self, inst, route_id):
        return self.spec_

    def pool(self, inst, route_id, purpose, rep=0, N=None):
        self.purposes.append(purpose)
        return self.dev if purpose == "dev" else self.train


def job_of(rep: int, arm: str, N: int = N_TRAIN) -> sweep.Job:
    return sweep.Job(kind="neural", inst="TOY", route_id=0, N=N, rep=rep, arm=arm)


def thr_payload(train: Paths, rep: int) -> dict:
    models, stats, diag, budget = fit_thr(SPEC, train, N_TRAIN, rep)
    return {"kind": "neural", "models": codecs.pack_models(models, "thr", SPEC.E0),
            "stats": stats, "diag": diag, "budget": budget, "job": job_of(rep, "thr"),
            "fit_commit": "0" * 40, "fit_tree_clean": True, "libraries": {}}


def write_world(tmp: Path, dev: Paths, train: Paths, reps=(0,), tamper=None):
    """Writes THR payloads, dev rows and per-day dev costs; returns (df, models, rows)."""
    models_dir, rows_dir = tmp / "models", tmp / "rows"
    models_dir.mkdir(parents=True), rows_dir.mkdir(parents=True)
    rows = []
    for rep in reps:
        payload = thr_payload(train, rep)
        if tamper is not None:
            tamper(payload)
        pickle.dump(payload, open(models_dir / f"{payload['job'].key}.pkl", "wb"))
        models = codecs.unpack_models(payload["models"])
        cache = RouteCache(SPEC, dev, payload["stats"])
        arm_c = simulate_policy(cache, B, H, E, models, H_bill=H)[0]
        none_c = simulate_policy(cache, B, H, E, {}, H_bill=H)[0]
        np.savez(rows_dir / f"{payload['job'].key}.dev.npz", thr=arm_c, none=none_c)
        base = dict(inst="TOY", route_id=0, m=2, stratum="s", N=N_TRAIN, rep=rep,
                    eval_set="dev", fit_seconds=0.0, fit_commit="x", eval_commit="y",
                    gap_closed=np.nan, n_eval=dev.p.shape[0], handoff_rate=0.0, fail_rate=0.0)
        rows.append(dict(base, policy="thr", mean_cost=float(arm_c.mean())))
        rows.append(dict(base, policy="none", mean_cost=float(none_c.mean())))
        rows.append(dict(base, policy="oracle", mean_cost=0.5 * float(arm_c.mean())))
        rows.append(dict(base, policy="baton_eq", mean_cost=float(arm_c.mean())))
    return pd.DataFrame(rows), models_dir, rows_dir


@pytest.fixture(scope="module")
def train_pool():
    return draw(11, N_TRAIN)


@pytest.fixture(scope="module")
def dev_pool():
    return draw(23, N_DEV)


# ---------------------------------------------------------------- pure helpers

def test_within_band_boundary_is_inclusive_at_three_se():
    assert fd._within_band(10.0 + 3.0 * 0.5, 10.0, 0.5)          # exactly 3 SE: passes
    assert not fd._within_band(10.0 + 3.0 * 0.5 + 1e-9, 10.0, 0.5)


def test_thr_dp_value_matches_closed_form_on_constant_continuation():
    g1 = np.array([0.0, 1.0, 5.0, 2.0])          # 5 > B = 4 breaches at stop 1
    w_grid = np.linspace(-1.0, B, 11)
    c1 = np.full(11, 0.8)                         # C_1 = 0.8 everywhere
    H1, E_first = 0.5, 3.0
    # g1 = 0, 1, 2 hand off at min(H, C) = 0.5; g1 = 5 breaches at cost E[0]
    expected = (0.5 + 0.5 + 0.5 + 3.0) / 4
    assert fd.thr_dp_value(g1, w_grid, c1, H1, E_first, B) == pytest.approx(expected)


def test_monotone_check_refuses_a_decrease_beyond_tolerance():
    c = np.array([[0.0, 0.5, 1.0], [0.0, 0.5, 0.5 - 1e-6]])
    assert not fd._monotone(c, 3.0)
    assert fd._monotone(np.array([[0.0, 0.5, 0.5 - 1e-12]]), 3.0)   # within 1e-9 * E0


def test_budget_label_thresholds_are_strict_at_half_a_point():
    assert not fd.budget_label({2: 1.0, 3: 0.5})["budget_limited"]          # gain = 0.5
    assert fd.budget_label({2: 1.0 + 1e-9, 3: 0.5})["budget_limited"]
    cut = fd.budget_label({1: 3.0, 2: 1.0, 3: 0.9}, p_main=1, p_star=3)
    assert cut["cut"] and cut["budget_limited"]                              # 3.0 - 0.9 > 0.5


def test_tuning_scores_are_percent_over_baton_eq():
    rows = pd.DataFrame([dict(inst="A", route_id=0, rep=0, policy="baton_eq", mean_cost=10.0),
                         dict(inst="A", route_id=0, rep=0, policy="dqn@p1", mean_cost=11.0),
                         dict(inst="A", route_id=0, rep=0, policy="dqn@p2", mean_cost=10.5),
                         dict(inst="A", route_id=0, rep=0, policy="dqn@p3", mean_cost=10.0)])
    s = fd.tuning_scores(rows, "dqn")
    assert s == pytest.approx({1: 10.0, 2: 5.0, 3: 0.0})


def test_independence_item_is_full_against_thr_in_percent():
    rows = pd.DataFrame([dict(inst="A", route_id=0, rep=0, policy="thr", mean_cost=10.0),
                         dict(inst="A", route_id=0, rep=0, policy="nn_full", mean_cost=9.0)])
    assert fd.independence_items(rows)["full_vs_thr_pct"] == pytest.approx(-10.0)


# ---------------------------------------------------------------- refusal of non-dev rows

def test_require_dev_refuses_non_dev_rows_and_test_files(tmp_path):
    rows_dir = tmp_path / "rows"
    rows_dir.mkdir()
    good = pd.DataFrame({"eval_set": ["dev", "dev"]})
    fd.require_dev(good, rows_dir)
    with pytest.raises(ValueError, match="dev rows only"):
        fd.require_dev(pd.DataFrame({"eval_set": ["dev", "test_b"]}), rows_dir)
    with pytest.raises(ValueError, match="no eval_set"):
        fd.require_dev(pd.DataFrame({"mean_cost": [1.0]}), rows_dir)
    (rows_dir / "thr__TOY__r0__N2000__rep0.test_b.npz").write_bytes(b"")
    with pytest.raises(ValueError, match="non-dev"):
        fd.require_dev(good, rows_dir)


def test_source_refuses_test_purposes():
    with pytest.raises(ValueError, match="fidelity reads"):
        fd.DevSource().pool("TOY", 0, "test_b")
    with pytest.raises(ValueError, match="fidelity reads"):
        fd.DevSource().pool("TOY", 0, "test")


def test_gate_report_refuses_arm_stage_mismatch(tmp_path):
    with pytest.raises(ValueError, match="does not hold arm"):
        fd.gate_report("b_thr", "dos", pd.DataFrame(), tmp_path, tmp_path)


# ---------------------------------------------------------------- end-to-end THR gate

def test_thr_gate_passes_on_constructed_independent_world(tmp_path, train_pool, dev_pool):
    df, models, rows = write_world(tmp_path, dev_pool, train_pool, reps=(0, 1, 2))
    rep = fd.gate_report("b_thr", "thr", df, models, rows, source=FakeSource(dev_pool, train_pool))
    gates = {g["id"]: g for g in rep["gates"]}
    assert gates["b_thr"]["passed"], gates["b_thr"]
    assert gates["b0"]["passed"] and gates["p1"]["passed"]
    assert rep["passed"] and rep["dropped"] == []


def test_thr_gate_fails_when_the_table_is_not_monotone(tmp_path, train_pool, dev_pool):
    def break_monotone(payload):
        payload["models"]["states"][1][0]["c"][0, 5] -= 1.0
    df, models, rows = write_world(tmp_path, dev_pool, train_pool, tamper=break_monotone)
    rep = fd.gate_report("b_thr", "thr", df, models, rows, source=FakeSource(dev_pool, train_pool))
    assert not {g["id"]: g for g in rep["gates"]}["b_thr"]["passed"]
    assert "thr:b_thr" in rep["dropped"] and not rep["passed"]


def test_thr_gate_fails_when_dev_days_do_not_match_the_model(tmp_path, train_pool):
    far_dev = draw(31, N_DEV, scale=3.0)          # dev stop-1 marginal far from training
    df, models, rows = write_world(tmp_path, far_dev, train_pool)
    rep = fd.gate_report("b_thr", "thr", df, models, rows,
                         source=FakeSource(far_dev, train_pool))
    assert not {g["id"]: g for g in rep["gates"]}["b_thr"]["passed"]


def test_b0_fails_when_an_arm_row_beats_the_oracle(tmp_path, train_pool, dev_pool):
    df, models, rows = write_world(tmp_path, dev_pool, train_pool)
    df.loc[df.policy == "oracle", "mean_cost"] = 1e6      # oracle worse than the arm
    rep = fd.gate_report("b_thr", "thr", df, models, rows, source=FakeSource(dev_pool, train_pool))
    assert not {g["id"]: g for g in rep["gates"]}["b0"]["passed"]


def test_p1_fails_when_arm_costs_far_above_trivial_policies(tmp_path, train_pool, dev_pool):
    df, models, rows = write_world(tmp_path, dev_pool, train_pool)
    for f in rows.glob("*.npz"):
        z = dict(np.load(f))
        z["thr"] = z["thr"] + 10.0
        np.savez(f, **z)
    rep = fd.gate_report("b_thr", "thr", df, models, rows, source=FakeSource(dev_pool, train_pool))
    assert not {g["id"]: g for g in rep["gates"]}["p1"]["passed"]


def test_arm_stage_with_no_payloads_is_refused(tmp_path, dev_pool, train_pool):
    empty = tmp_path / "models"
    empty.mkdir()
    with pytest.raises(FileNotFoundError):
        fd.load_jobs(empty)


# ---------------------------------------------------------------- RLSM and DOS

def jobs_with(diag):
    return [{"job": job_of(0, "rlsm"), "diag": diag, "models": {"states": {}}}]


def rlsm_stage(n_val, action="fit", mse=0.5, const=1.0, resid=1e-12):
    d = dict(kind="stage", arm="rlsm", k=1, fold="A", n_fit=50, n_val=n_val, action=action)
    if action == "fit":
        d.update(ne_residual_rel=resid, val_mse=mse, const_val_mse=const)
    return d


def ctx_for(stage, arm, jobs):
    return fd._Ctx(stage=stage, arm=arm, jobs=jobs, source=None, rows_dir=Path("."),
                   independent=False, g2_keys=frozenset(), thr_dir=None)


def test_rlsm_gate_pass_and_share_boundary_at_ninety_percent():
    ten = [rlsm_stage(100) for _ in range(9)] + [rlsm_stage(100, mse=2.0)]
    assert fd._gate_rlsm(ctx_for("b_rlsm", "rlsm", jobs_with(ten))).passed   # 9/10 = 0.90
    eleven = [rlsm_stage(100) for _ in range(8)] + [rlsm_stage(100, mse=2.0)] * 2
    assert not fd._gate_rlsm(ctx_for("b_rlsm", "rlsm", jobs_with(eleven))).passed  # 0.80


def test_rlsm_stages_below_100_val_rows_are_not_counted_and_residual_is_gated():
    stages = [rlsm_stage(99, mse=9.0)] + [rlsm_stage(100)]
    assert fd._gate_rlsm(ctx_for("b_rlsm", "rlsm", jobs_with(stages))).passed
    bad_resid = [rlsm_stage(100, resid=2e-8)]
    assert not fd._gate_rlsm(ctx_for("b_rlsm", "rlsm", jobs_with(bad_resid))).passed


def dos_stage(obj, never, n_val=100, k=1, chosen=1):
    return dict(kind="stage", arm="dos", k=k, fold="A", n_val=n_val, action="cold",
                chosen_ckpt=chosen, val_loss_per_ckpt=[(0, 9.0), (chosen, obj)],
                never_val=never)


def dos_ctx(stages):
    jobs = [{"job": job_of(0, "dos"), "diag": stages, "models": {"states": {}}}]
    return fd._Ctx(stage="b_dos", arm="dos", jobs=jobs, source=FakeSource(None, None),
                   rows_dir=Path("."), independent=False, g2_keys=frozenset(), thr_dir=None)


def test_dos_gate_uses_min_of_always_and_never_with_tolerance():
    # always stop = H[0]/E0 = 1/3 for k = 1 (the minimum here); never = 0.5
    always = 1.0 / 3.0
    assert fd._gate_dos(dos_ctx([dos_stage(always + 0.5e-6, 0.5)])).passed
    assert not fd._gate_dos(dos_ctx([dos_stage(always + 2e-6, 0.5)])).passed


def test_dos_gate_refuses_when_never_objective_is_not_recorded():
    stage = dos_stage(0.3, 0.5)
    del stage["never_val"]
    assert not fd._gate_dos(dos_ctx([stage])).passed


def test_dos_share_boundary_at_ninety_five_percent():
    ok = [dos_stage(0.3, 0.5) for _ in range(19)] + [dos_stage(0.9, 0.5)]   # 19/20 = 0.95
    assert fd._gate_dos(dos_ctx(ok)).passed
    bad = [dos_stage(0.3, 0.5) for _ in range(18)] + [dos_stage(0.9, 0.5)] * 2   # 0.90
    assert not fd._gate_dos(dos_ctx(bad)).passed


# ---------------------------------------------------------------- DQN transitions

def test_dqn_gate_accepts_alive_pairs_times_passes_and_refuses_other_counts():
    train = draw(41, 500)
    folds, _ = split_80_20(500)
    fit = np.sort(np.concatenate(folds))
    W = np.cumsum((train.p - train.d)[fit], axis=1)
    counts = stage_counts(W, SPEC.B, SPEC.m)[1:]
    good = dict(transitions_per_stage=list(counts * 2))
    job = job_of(0, "dqn", N=500)
    source = FakeSource(None, train)
    ok = fd._Ctx("b_dqn", "dqn", [{"job": job, "budget": good, "models": {"nets": []}}],
                 source, Path("."), False, frozenset(), None)
    assert fd._gate_dqn(ok).passed
    bad = dict(transitions_per_stage=list(counts * 2 + 1))
    ctx = fd._Ctx("b_dqn", "dqn", [{"job": job, "budget": bad, "models": {"nets": []}}],
                  source, Path("."), False, frozenset(), None)
    assert not fd._gate_dqn(ctx).passed


def test_dqn_gate_refuses_a_non_finite_checkpoint():
    train = draw(41, 200)
    folds, _ = split_80_20(200)
    fit = np.sort(np.concatenate(folds))
    counts = stage_counts(np.cumsum((train.p - train.d)[fit], axis=1), SPEC.B, SPEC.m)[1:]
    net = {"state": {"w": np.array([1.0, np.nan])}}
    job = job_of(0, "dqn", N=200)
    ctx = fd._Ctx("b_dqn", "dqn", [{"job": job, "budget": dict(transitions_per_stage=list(counts)),
                                    "models": {"nets": [net]}}],
                  FakeSource(None, train), Path("."), False, frozenset(), None)
    assert not fd._gate_dqn(ctx).passed


# ---------------------------------------------------------------- report, family, round trip

def fake_reports(passing):
    out = {}
    for arm in B_ARMS:
        gates = [dict(id="b0", passed=True, value=1.0, threshold="t", detail=""),
                 dict(id=fd.GATE_OF_ARM[arm], passed=arm in passing, value=0.5,
                      threshold="t", detail="d|x"),
                 dict(id="p1", passed=True, value=-1.0, threshold="t", detail="")]
        out[arm] = dict(passed=arm in passing and True, gates=gates)
    return out


def test_family_lists_follow_gate_verdicts():
    lists = fd.derive_family(fake_reports({"thr", "alpz", "dqn"}))
    assert lists["family"] == ["thr", "alpz", "dqn"]
    assert lists["descriptive"] == ["rlsm", "dos", "ppo"]
    assert lists["dropped"] == ["rlsm:b_rlsm", "dos:b_dos", "ppo:b_ppo"]
    assert lists["references"] == ["full", "ladder"]


def test_family_derivation_requires_every_arm():
    reports = fake_reports(set(B_ARMS))
    del reports["ppo"]
    with pytest.raises(ValueError, match="ppo"):
        fd.derive_family(reports)


def test_round_trip_through_gatecheck_parser(tmp_path):
    lists = fd.derive_family(fake_reports({"thr", "alpz", "dqn"}))
    text = fd.render_report(fake_reports({"thr", "alpz", "dqn"}), lists)
    match = gc.DERIVATION_RE.search(text)
    assert match is not None
    block = json.loads(match.group(1))
    for key in gc.FAMILY_KEYS:
        assert block[key] == lists[key]
    hashes = {k: "a" * 64 for k in gc.HASH_KEYS}
    line = " ".join(["TEST_B FAMILY CLOSED:",
                     *(f"{k}=[{','.join(lists[k])}]" for k in gc.FAMILY_KEYS),
                     *(f"{k}={hashes[k]}" for k in gc.HASH_KEYS)])
    found = gc.FAMILY_LINE.fullmatch(line)
    assert found is not None
    assert found.group(1) == "thr,alpz,dqn"
    assert found.group(3) == "rlsm:b_rlsm,dos:b_dos,ppo:b_ppo"


def test_report_rendering_is_deterministic():
    reports = fake_reports({"thr"})
    lists = fd.derive_family(reports)
    assert fd.render_report(reports, lists) == fd.render_report(reports, lists)


def test_gate_report_is_deterministic_across_runs(tmp_path, train_pool, dev_pool):
    df, models, rows = write_world(tmp_path, dev_pool, train_pool)
    src = FakeSource(dev_pool, train_pool)
    a = fd.gate_report("b_thr", "thr", df, models, rows, source=src)
    b = fd.gate_report("b_thr", "thr", df, models, rows, source=src)
    assert a["gates"] == b["gates"] and a["dropped"] == b["dropped"]


@pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")
def test_round_trip_read_derivation_from_a_committed_file(tmp_path):
    repo = tmp_path / "repo"
    (repo / gc.FIDELITY_REL).parent.mkdir(parents=True)
    lists = fd.derive_family(fake_reports({"thr"}))
    text = fd.render_report(fake_reports({"thr"}), lists)
    (repo / gc.FIDELITY_REL).write_text(text, encoding="utf-8", newline="\n")
    subprocess.run(["git", "-C", str(repo), "init", "-q"], check=True)
    subprocess.run(["git", "-C", str(repo), "add", "-A"], check=True)
    subprocess.run(["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t",
                    "commit", "-q", "-m", "fidelity"], check=True)
    head = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True,
                          text=True, check=True).stdout.strip()
    block = gc.read_derivation(head, repo)
    assert block["family"] == ["thr"]
    assert block["dropped"] == [f"{a}:{g}" for a in B_ARMS[1:] for g in [fd.GATE_OF_ARM[a]]]


# ---------------------------------------------------------------- ALP-Z (real LP, small N)

def alpz_payload(train: Paths, n: int, rep: int) -> dict:
    from transvrp.expb.alp import fit_alpz
    sub = Paths(p=train.p[:n], d=train.d[:n])
    models, stats, diag, budget = fit_alpz(SPEC, sub, n, rep)
    job = sweep.Job(kind="neural", inst="TOY", route_id=0, N=n, rep=rep, arm="alpz")
    return {"kind": "neural", "models": codecs.pack_models(models, "alpz", SPEC.E0),
            "stats": stats, "diag": diag, "budget": budget, "job": job,
            "fit_commit": "0" * 40, "fit_tree_clean": True, "libraries": {}}


def test_alpz_gate_passes_on_a_real_lp_and_fails_on_a_bad_status(train_pool, dev_pool):
    payload = alpz_payload(train_pool, 600, 0)
    src = FakeSource(dev_pool, Paths(p=train_pool.p[:600], d=train_pool.d[:600]))
    jobs = [payload]
    ctx = fd._Ctx("b_alpz", "alpz", jobs, src, Path("."), False, frozenset({("TOY", 0, 0)}))
    assert fd._gate_alpz(ctx).passed
    bad = alpz_payload(train_pool, 600, 0)
    for d in bad["diag"]:
        d["lp_status"] = 1
    ctx = fd._Ctx("b_alpz", "alpz", [bad], src, Path("."), False, frozenset({("TOY", 0, 0)}))
    assert not fd._gate_alpz(ctx).passed


# ---------------------------------------------------------------- boundaries found by mutation

def test_share_helper_is_inclusive_at_the_threshold():
    assert fd._meets(0.99, 0.99)
    assert not fd._meets(0.99 - 1e-12, 0.99)


def test_thr_dp_breach_is_strict_at_the_load_limit():
    w_grid = np.linspace(-1.0, B, 11)
    c1 = np.full(11, 0.8)
    g1 = np.array([B])                      # exactly at the limit: not a breach
    assert fd.thr_dp_value(g1, w_grid, c1, 0.5, 3.0, B) == pytest.approx(0.5)


def test_per_day_standard_error_uses_the_sample_variance():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    assert fd._se(x) == pytest.approx(np.std(x, ddof=1) / 2.0)


def test_p1_boundary_is_two_standard_errors_and_the_competitor_is_the_minimum(tmp_path):
    dev = draw(23, N_DEV)
    stats = RouteStats.fit(dev)
    cache = RouteCache(SPEC, dev, stats)
    none_c = simulate_policy(cache, B, H, E, {}, H_bill=H)[0]
    h1_c = np.where(cache.W[:, 0] > B, SPEC.E[0], SPEC.H[0])
    assert none_c.mean() < h1_c.mean(), "the test needs none as the minimum competitor"
    rng = np.random.default_rng(5)
    z = rng.standard_normal(N_DEV)
    z = (z - z.mean()) / z.std(ddof=1)      # zero mean, unit sample SD
    s = 1.0
    se = s / math.sqrt(N_DEV)

    def gate_at(delta):
        arm_c = none_c + (2.0 * se + delta) + s * z
        rows = tmp_path / f"rows{delta}"
        rows.mkdir(exist_ok=True)
        job = job_of(0, "thr")
        np.savez(rows / f"{job.key}.dev.npz", thr=arm_c)
        ctx = fd._Ctx("b_thr", "thr", [{"job": job, "stats": stats}],
                      FakeSource(dev, dev), rows, False, frozenset())
        return fd._gate_p1(ctx)

    assert gate_at(-1e-9).passed
    assert not gate_at(1e-6).passed


def test_family_includes_every_arm_that_passes():
    lists = fd.derive_family(fake_reports(set(B_ARMS)))
    assert lists["family"] == list(B_ARMS)
    assert lists["descriptive"] == [] and lists["dropped"] == []


def test_dqn_gate_accepts_three_passes():
    train = draw(41, 500)
    folds, _ = split_80_20(500)
    fit = np.sort(np.concatenate(folds))
    counts = stage_counts(np.cumsum((train.p - train.d)[fit], axis=1), SPEC.B, SPEC.m)[1:]
    job = job_of(0, "dqn", N=500)
    ctx = fd._Ctx("b_dqn", "dqn", [{"job": job, "budget": dict(transitions_per_stage=list(counts * 3)),
                                    "models": {"nets": []}}],
                  FakeSource(None, train), Path("."), False, frozenset(), None)
    assert fd._gate_dqn(ctx).passed


def test_dos_unrecorded_objective_fails_even_when_the_share_passes():
    stages = [dos_stage(0.3, 0.5) for _ in range(19)]
    missing = dos_stage(0.3, 0.5)
    del missing["never_val"]
    assert fd._gate_dos(dos_ctx(stages + [missing])).value == pytest.approx(0.95)
    assert not fd._gate_dos(dos_ctx(stages + [missing])).passed
