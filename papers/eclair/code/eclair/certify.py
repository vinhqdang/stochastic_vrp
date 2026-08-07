"""The certification layer: per-candidate e-processes with calibrated
conservative Bernoulli bets (Tiers A/C), hard certificates (Tier B),
Kelly/GROW routing of solver-seconds, Ville-threshold rejection, and
e-BH selection over the final pool (PROJECT.md §3 Steps 3–5).

Validity note (PROJECT.md §5.2): Tier A/C e-factors use a calibrated
UPPER bound p0_bar on the null alarm rate; E_H0[e] <= 1 whenever
p0_true <= p0_bar < p1_bet, so each candidate's running product is a
nonnegative supermartingale under "this candidate is faithful" for ANY
predictable routing policy. Tier B alarms have probability exactly 0
under faithfulness (verified in calibration), so an alarm is a hard
certificate (e = infinity).
"""

from __future__ import annotations

import math

from .probes import TIER_FNS, tier_c

TIERS = ("A", "B", "C")
HARD_LOG_E = math.log(1e18)          # stand-in for e = infinity


class Bets:
    """Calibrated betting/routing parameters (see calibrate.py)."""

    def __init__(self, calib: dict):
        self.p0 = {t: calib[t]["p0_bar"] for t in ("A", "C")}
        self.p1 = {t: calib[t]["p1_bet"] for t in ("A", "C")}
        self.delta_b = calib["B"]["delta"]
        self.cost = {t: calib[t]["mean_cost"] for t in TIERS}
        for t in ("A", "C"):
            assert self.p0[t] < self.p1[t], f"tier {t} bet is powerless"

    def log_e(self, tier, alarm):
        p0, p1 = self.p0[tier], self.p1[tier]
        return math.log(p1 / p0) if alarm else math.log((1 - p1) / (1 - p0))

    def growth(self, tier, under_h1):
        p0, p1 = self.p0[tier], self.p1[tier]
        p = p1 if under_h1 else p0
        return p * math.log(p1 / p0) + (1 - p) * math.log((1 - p1) / (1 - p0))

    def bayes(self, tier, alarm, q):
        """Posterior of unfaithfulness after a tier outcome (routing only)."""
        if tier == "B":
            like1, like0 = (self.delta_b, 0.0) if alarm \
                else (1 - self.delta_b, 1.0)
        else:
            p0, p1 = self.p0[tier], self.p1[tier]
            like1, like0 = (p1, p0) if alarm else (1 - p1, 1 - p0)
        num = q * like1
        den = num + (1 - q) * like0
        return num / den if den > 0 else 1.0


class State:
    __slots__ = ("cand", "logE", "q", "spent", "rejected", "n_probes",
                 "error")

    def __init__(self, cand, prior):
        self.cand = cand
        self.logE = 0.0
        self.q = prior
        self.spent = 0.0
        self.rejected = False
        self.n_probes = 0
        self.error = None


def _kelly_score(st, tier, bets, log_thresh, use_cost=True):
    if tier == "B":
        gap = max(log_thresh - st.logE, 0.0)
        val = st.q * bets.delta_b * gap
    else:
        val = st.q * bets.growth(tier, True) + (1 - st.q) * bets.growth(tier, False)
    return val / bets.cost[tier] if use_cost else val


def run_certification(pool, bets, policy, budget, alpha, rng,
                      prior=0.5, max_probes=4000, tiers=TIERS,
                      trace=None):
    """Certify a candidate pool. Returns dict with per-candidate final
    states, the e-BH rejection set, and the certified pick.
    `tiers` restricts the probe pool (ablations). If `trace` is a
    list, (candidate_index, tier, cumulative_cost, logE) is appended
    after every probe (figures/diagnostics only)."""
    log_thresh = math.log(1.0 / alpha)
    states = [State(c, prior) for c in pool]
    rr = 0
    n_total = 0
    spent_total = 0.0

    while budget > 0 and n_total < max_probes:
        alive = [s for s in states if not s.rejected]
        if not alive:
            break
        if policy == "round_robin":
            st = alive[rr % len(alive)]
            tier = tiers[(rr // len(alive)) % len(tiers)]
            rr += 1
        else:
            use_cost = policy == "kelly"
            st, tier, best = None, None, -math.inf
            for s in alive:
                for t in tiers:
                    sc = _kelly_score(s, t, bets, log_thresh, use_cost)
                    if sc > best:
                        best, st, tier = sc, s, t
        try:
            if tier == "C":
                alarm, cost = tier_c(st.cand, [s.cand for s in alive], rng)
            else:
                alarm, cost = TIER_FNS[tier](st.cand, rng)
        except Exception as e:          # LLM candidate crashed at runtime:
            st.logE = HARD_LOG_E        # a faithful model never raises, so
            st.rejected = True          # a crash is a hard certificate
            st.error = f"{type(e).__name__}: {e}"
            st.n_probes += 1
            n_total += 1
            continue
        budget -= cost
        st.spent += cost
        st.n_probes += 1
        n_total += 1
        spent_total += cost
        if tier == "B":
            if alarm:
                st.logE = HARD_LOG_E
                st.rejected = True
            st.q = bets.bayes("B", alarm, st.q)
        else:
            st.logE += bets.log_e(tier, alarm)
            st.q = bets.bayes(tier, alarm, st.q)
            if st.logE >= log_thresh:
                st.rejected = True
        if trace is not None:
            trace.append((states.index(st), tier, spent_total,
                          min(st.logE, HARD_LOG_E)))

    ebh = e_bh([math.exp(min(s.logE, HARD_LOG_E)) for s in states], alpha)
    survivors = [s for s in states if not s.rejected]
    pick = min(survivors, key=lambda s: s.logE) if survivors else None
    return {"states": states, "ebh_rejected": ebh, "pick": pick,
            "abstained": pick is None}


def e_bh(e_values, alpha):
    """e-BH (Wang & Ramdas): reject the k* largest e-values where
    k* = max{k: e_(k) >= K/(alpha*k)}. Returns set of indices."""
    K = len(e_values)
    order = sorted(range(K), key=lambda i: -e_values[i])
    k_star = 0
    for k in range(1, K + 1):
        if e_values[order[k - 1]] >= K / (alpha * k):
            k_star = k
    return set(order[:k_star])
