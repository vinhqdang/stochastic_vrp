"""ECLAIR statistical-core prototype (PROJECT.md §7). Stdlib-only.

Monte-Carlo simulation of the certification layer with NO LLM and NO CP
solver: candidates are faithful/unfaithful coin flips and the three
probe tiers are simulated binary probes with known true error rates,
bet against with CONSERVATIVELY calibrated rates (PROJECT.md §5.2 —
the router never sees the true rates). Questions answered before any
real engineering:

  Q1 (validity)  Is the empirical probability that a FAITHFUL candidate
                 is ever rejected <= alpha, under adaptive routing,
                 optional stopping, and a shared budget?
  Q2 (power)     What fraction of UNFAITHFUL candidates does each
                 routing policy kill within the budget, at what
                 solver-cost per kill?
  Q3 (routing)   Does Kelly (expected log-evidence per solver-second)
                 beat round-robin and cost-blind routing?

Probe model (per single probe execution on one candidate):

  tier  cost   true P(alarm|H0)  true P(alarm|H1)  bet params (assumed)
  B     1.0    0                 0.06              alarm => e = inf (hard
                                                   certificate); pass => e = 1
  A     8.0    0.01              0.35              p0_bar=0.02, p1_bet=0.30
  C     0.4    0.08              0.30              p0_bar=0.10, p1_bet=0.25

Tier A/C e-variable: Bernoulli LR e = (p1_bet/p0_bar)^X *
((1-p1_bet)/(1-p0_bar))^(1-X). Since p0_true <= p0_bar <= p1_bet,
E_H0[e] <= 1 (conservative calibration preserves validity), so the
running product is a nonnegative supermartingale under H0 for ANY
predictable routing rule; Ville gives the alpha guarantee.

Routing policies:
  kelly       argmax over (alive candidate, tier) of
              [q*G1 + (1-q)*G0] / cost, where q is a Bayes posterior of
              unfaithfulness maintained with the ASSUMED rates, G1/G0
              are the assumed expected log-e under H1/H0, and tier B's
              G1 is delta_B * (remaining log-gap to the threshold)
              with G0 = 0 (zero null bleed).
  round_robin cycle candidates; within a candidate cycle B -> C -> A.
  cost_blind  same score as kelly WITHOUT dividing by cost.

Usage: python3 eclair_prototype.py   (writes prototype_results.txt)
"""

from __future__ import annotations

import math
import random
import statistics

ALPHA = 0.05
LOG_THRESH = math.log(1.0 / ALPHA)
K_CANDIDATES = 8
P_UNFAITHFUL = 0.5
BUDGET = 200.0          # shared solver-seconds per replication
N_REPS = 500
SEED = 20260807

TIERS = ("B", "C", "A")
COST = {"B": 1.0, "C": 0.4, "A": 8.0}
TRUE_P0 = {"B": 0.0, "C": 0.08, "A": 0.01}
TRUE_P1 = {"B": 0.06, "C": 0.30, "A": 0.35}
BET_P0 = {"C": 0.10, "A": 0.02}          # conservative upper bounds
BET_P1 = {"C": 0.25, "A": 0.30}
DELTA_B_ASSUMED = 0.06                    # router's belief about tier B power


def _log_lr(tier: str, alarm: bool) -> float:
    p0, p1 = BET_P0[tier], BET_P1[tier]
    return math.log(p1 / p0) if alarm else math.log((1 - p1) / (1 - p0))


def _assumed_growth(tier: str, under_h1: bool) -> float:
    """Assumed expected log-e of one probe, computed with bet params only."""
    p0, p1 = BET_P0[tier], BET_P1[tier]
    p = p1 if under_h1 else p0
    return p * math.log(p1 / p0) + (1 - p) * math.log((1 - p1) / (1 - p0))


G1 = {t: _assumed_growth(t, True) for t in ("A", "C")}
G0 = {t: _assumed_growth(t, False) for t in ("A", "C")}


class Candidate:
    __slots__ = ("unfaithful", "logE", "q", "rejected", "spent")

    def __init__(self, unfaithful: bool):
        self.unfaithful = unfaithful
        self.logE = 0.0
        self.q = P_UNFAITHFUL     # router's posterior of unfaithfulness
        self.rejected = False
        self.spent = 0.0          # solver-seconds spent on this candidate


def run_probe(c: Candidate, tier: str, rng) -> None:
    """Execute one probe, update wealth and posterior, check Ville."""
    p_alarm = TRUE_P1[tier] if c.unfaithful else TRUE_P0[tier]
    alarm = rng.random() < p_alarm
    c.spent += COST[tier]

    if tier == "B":
        if alarm:                          # hard certificate, e = inf
            c.rejected = True
            c.q = 1.0
            return
        # Bayes on a pass with assumed delta_B (routing only)
        num = c.q * (1 - DELTA_B_ASSUMED)
        c.q = num / (num + (1 - c.q))
        return

    c.logE += _log_lr(tier, alarm)
    p0, p1 = BET_P0[tier], BET_P1[tier]
    like1 = p1 if alarm else 1 - p1
    like0 = p0 if alarm else 1 - p0
    num = c.q * like1
    c.q = num / (num + (1 - c.q) * like0)
    if c.logE >= LOG_THRESH:
        c.rejected = True


def kelly_score(c: Candidate, tier: str, use_cost: bool) -> float:
    if tier == "B":
        gap = max(LOG_THRESH - c.logE, 0.0)
        val = c.q * DELTA_B_ASSUMED * gap
    else:
        val = c.q * G1[tier] + (1 - c.q) * G0[tier]
    return val / COST[tier] if use_cost else val


def run_replication(policy: str, rng) -> list[Candidate]:
    cands = [Candidate(rng.random() < P_UNFAITHFUL) for _ in range(K_CANDIDATES)]
    budget = BUDGET
    rr_state = 0
    while budget >= min(COST.values()):
        alive = [c for c in cands if not c.rejected]
        if not alive:
            break
        affordable = [t for t in TIERS if COST[t] <= budget]
        if not affordable:
            break
        if policy == "round_robin":
            c = alive[rr_state % len(alive)]
            t = affordable[(rr_state // len(alive)) % len(affordable)]
            rr_state += 1
        else:
            use_cost = policy == "kelly"
            best, c, t = -math.inf, None, None
            for cand in alive:
                for tier in affordable:
                    s = kelly_score(cand, tier, use_cost)
                    if s > best:
                        best, c, t = s, cand, tier
        budget -= COST[t]
        run_probe(c, t, rng)
    return cands


def evaluate(policy: str, seed: int) -> dict:
    rng = random.Random(seed)
    n_faith = n_faith_rej = n_unf = n_unf_rej = 0
    kill_costs = []
    surv_logE = []
    for _ in range(N_REPS):
        for c in run_replication(policy, rng):
            if c.unfaithful:
                n_unf += 1
                if c.rejected:
                    n_unf_rej += 1
                    kill_costs.append(c.spent)
            else:
                n_faith += 1
                if c.rejected:
                    n_faith_rej += 1
                else:
                    surv_logE.append(c.logE)
    return dict(
        policy=policy,
        false_rej=n_faith_rej / n_faith,
        n_faith=n_faith,
        detect=n_unf_rej / n_unf,
        n_unf=n_unf,
        mean_kill_cost=statistics.fmean(kill_costs) if kill_costs else float("nan"),
        mean_surv_logE=statistics.fmean(surv_logE) if surv_logE else float("nan"),
    )


def main() -> str:
    lines = [
        "ECLAIR statistical-core prototype",
        f"alpha={ALPHA}  K={K_CANDIDATES} candidates/rep  "
        f"P(unfaithful)={P_UNFAITHFUL}  budget={BUDGET:.0f} solver-sec/rep  "
        f"reps={N_REPS}  seed={SEED}",
        "",
        f"{'policy':<12}{'false-rej (faithful)':>22}{'detect (unfaithful)':>21}"
        f"{'mean cost/kill':>16}{'surv logE':>11}",
    ]
    for policy in ("kelly", "round_robin", "cost_blind"):
        r = evaluate(policy, SEED)
        lines.append(
            f"{r['policy']:<12}"
            f"{r['false_rej']:>14.4f} (n={r['n_faith']})"
            f"{r['detect']:>13.4f} (n={r['n_unf']})"
            f"{r['mean_kill_cost']:>16.1f}"
            f"{r['mean_surv_logE']:>11.2f}"
        )
    lines += [
        "",
        f"Validity target: false-rej <= alpha = {ALPHA} for EVERY policy",
        "(conservative calibration should land it well below).",
        "Routing target: kelly >= round_robin and cost_blind on detect,",
        "at lower mean cost per kill.",
    ]
    out = "\n".join(lines)
    print(out)
    return out


if __name__ == "__main__":
    import pathlib

    text = main()
    path = pathlib.Path(__file__).with_name("prototype_results.txt")
    path.write_text(text + "\n")
    print(f"\nwritten: {path}")
