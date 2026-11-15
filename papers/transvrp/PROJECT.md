# TRANSVRP — project notes (2026-11-15)

## 1. Seed paper: OR-Transformer (arXiv 2609.01933v2), as read

Problem: stochastic **joint replenishment** (n items, shared fixed order cost K,
lead times, AR(1) market factor driving Poisson demand), infinite-horizon
discounted cost. Not a routing problem.

Method: three separate Transformer encoders (order-opening, quantity, critic),
no positional or item-index embeddings, so the policy is permutation-equivariant
(Eq. 2, 23). Hybrid action: Bernoulli "open an order" head trained by a
score-function (PPO-surrogate) gradient; continuous quantities trained by a
**pathwise gradient through the differentiable inventory simulator**
(Eq. 30, truncated BPTT with a frozen-critic terminal value). d=128, 4 blocks,
8 heads, 16,000 updates, 1,024 parallel rollouts of length 10.

Claims: at n=1,024, 0.35M cost vs 1.39M for the best learner (about -75%) and
9.35M for rolling-horizon MILP (about -96%); versus Gurobi given 6 h per decision
(8 decisions), -19.1% cost and a ">4 million times" decision-time ratio.

What I would flag when reading it critically:
1. **The MILP baseline at n=1,024 is degenerate.** Appendix D.3 says the 100-scenario,
   50-period MILP has about 20M columns and *all* 6,400 decisions of both Gurobi
   and HiGHS hit the 10-minute limit and returned zero orders; the identical
   9.35M figures are the no-order trajectory. So the "-96%" and "4M x" headlines
   compare against "do nothing", not against a converged optimiser. The 6 h
   comparison (Fig. 4.1) is the more meaningful one.
2. At n=1 and n=4 HPO (MLP with pathwise gradients) is best, not OR-Transformer
   (Table D.5). The advantage is a scale effect.
3. n=1,024 uses **one training run** (the shaded band is a 10th-90th percentile
   inside that run); n<=64 uses 30 seeds.
4. Evaluation set is 128 held-out episodes, and the 1,024-item comparison covers
   a 50- or 8-decision horizon only.
5. PPO, Transformer-PPO and HPO diverge at scale, so part of the "scaling"
   story is the baselines failing to train, which the authors tie to plasticity
   (their ref. 34). The paper does not isolate that cause.
6. It is a short preprint (5 pages of body); no ablation separating "Transformer"
   from "pathwise gradient". Transformer-PPO vs OR-Transformer does isolate the
   gradient estimator, HPO vs OR-Transformer isolates the architecture, so the
   2x2 is partly there, but no table presents it as one.

What transfers to routing, and what does not:
- **Permutation-equivariant set encoder over entities**: transfers directly
  (customers are an unordered set; this is the Kool et al. / POMO line).
- **Pathwise gradient through a differentiable simulator**: transfers only to
  *continuous* decisions. The SVRPSPD load recursion L_k = L_{k-1} - d_k + p_k
  is differentiable in demands, but "next customer", "hand off now" are discrete,
  so the pathwise estimator has nothing to differentiate unless the decision is
  relaxed or continuous (see direction B).
- **Shared fixed cost coupling a discrete open/close with continuous quantities**:
  analogue is dispatch-or-not (fixed cost of a standby/emergency vehicle) plus a
  continuous quantity such as a safety load or pre-positioned stock.

## 2. Literature map (all arXiv-verified for existence, see VERIFY_CITATIONS.md)

| strand | papers | note |
|---|---|---|
| Neural constructive routing | Kool+ 2018 (1803.08475), Nazari+ 2018 (1802.04240), POMO (2010.16011) | deterministic CVRP baselines; attention encoders |
| Neural / RL for stochastic VRP | Iklassov+ 2024 SVRPTW (2402.09765); Dastpak-Errico-Jabali 2026 VRPSD with outsourcing (2607.16875) | stochastic travel/demand; the 2026 one learns a value function with a GAT-style encoder |
| Transformer for dynamic PDP | Zou+ MAPT (2511.17435) | stochastic *requests*, not stochastic demand |
| LLM as heuristic designer | ReEvo (2402.01145), EoH (2401.02051) | LLM writes code, not routes |
| LLM agents for VRP | Zhang+ AFL (2510.16701); survey Xiu+ 2026 (2607.00604) | deterministic complex VRPs; survey is the entry point |
| LLM formulation of stochastic models | Talebi 2025 (2508.17200) | modelling, not solving |
| LLM adaptive routing | Tan-Fan (2512.10610) | traffic, not VRP optimisation |

A search found **no paper** combining transformer RL, stochastic demand *and*
simultaneous pickup/delivery with online recourse (a search result, not proof; repeat
before claiming novelty). Several entries were seen only as snippets and must be read.

## 3. Our assets

- Engine with instances (Dethloff, Salhi-Nagy, city), a scenario generator, a
  three-class fleet cost model, and a cache of ALNS plans.
- BATON as the strong *online* baseline (LS optimal stopping), plus DP and
  clairvoyant bounds, giving a ladder to place a neural policy on.
- BATON's own null result (`core/otr21.py`): richer learned features lost to a
  scalar isotonic model at *historical-data* scale in 23/25 routes. That is the
  prior against a transformer winning here, and the reason direction A must use
  simulator-scale (unlimited) data.

## 4. Candidate directions

**A. Amortised continuation value / handoff policy.** A permutation-invariant
encoder over the remaining-route set (customers left, their price schedule,
current load) outputs a value C_k(state) or the three-way action. Trained on
unlimited simulated paths. Tests whether BATON's scalar-state model is the right
complexity point once data is not the limit. Cheap, in-engine, a clean baseline
ladder. Risk: a neural model may only tie BATON.

**B. OR-Transformer-style hybrid training on a routing relaxation.** Discrete
dispatch decision by score gradient, continuous decisions (safety-load target per
route, pre-positioned standby location/time) by pathwise gradient through the load
recursion. Closest to the seed paper. Risk: the continuous part may be too small
to matter; must check that a Lagrangian/threshold sweep does not already match it.

**C. Neural planner for the SAA-CVaR gate.** Learn a route-feasibility surrogate
(peak-overflow probability) to speed up ALNS moves. Overlaps QUASAR/RARECERT
territory (they study the same quantity); separation must be argued.

**D. LLM role.** Two honest options: (i) LLM as heuristic designer for the ALNS
operators (ReEvo/EoH style) evaluated on the stochastic objective; (ii) **weight-space
programming** (below). Directly asking an LLM to solve routes is a weak baseline
and should only appear as one.

**Recommendation:** start with A as the minimum testable claim (it can finish as a
clean negative like QUASAR if it ties), add B only if A shows a gap that a
continuous decision could close. Treat D(ii) as a late, optional add-on.

### D(ii) Weight-space programming (the author's idea)

Primary source: Takehi, Teoh, Zhang, Khattab (MIT CSAIL), blog post 2026-11-02
(`takehi2026wsp`, see VERIFY_CITATIONS.md). It is a blog, not a paper, and the
authors call it a capability demonstration, not a method.

What it reports that matters here: an LLM edited 3.5% of the weights of a failing
**PPO policy** (car on a ramp) from 0/256 to 1,024/1,024 episode success, and removed
a looping fault in Qwen3.5-0.8B by editing 0.016% of weights. It did not help a
Gemma 3 270M control, and the authors say edits remove obstacles but cannot add
missing ability. There is **no comparison with fine-tuning or gradient-based editing**.

Possible analogue: a trained routing policy has a diagnosable pathology (e.g. it
ignores the running peak and overflows on pickup-heavy routes); an LLM with tools
locates the responsible weights and edits them, with no retraining.

My assessment:
- Policy repair is a reported success case (Exp 4), so this is less speculative than
  I first wrote. But it matches "remove an obstacle"; a policy that never learned the
  peak is "missing ability", the case the post says fails.
- Our policies are small (about 0.8M parameters per network, versus 171k to 7M in the
  post's experiments), so size is not the obstacle. The real baseline is cheap:
  (1) a few hundred fine-tuning or constrained-RL steps, (2) a scripted rank-1
  ablation or activation steering. The post never runs these, so it cannot tell us
  the LLM edit beats them. That comparison is the contribution if there is one.
- Tool access varies in the post (Python allowed in some experiments), so a fair
  protocol must state what the LLM may run.
- Go/no-go: plant a known defect in a routing policy and ask whether an LLM-driven
  edit repairs it as well as the cheap baselines at equal budget. If not, drop it.
- Single-vendor, early, few seeds; keep it an optional late add-on to A or B.

## 5. First concrete steps (no code written yet)

1. Read in full: Dastpak-Errico-Jabali (2607.16875), Iklassov (2402.09765),
   Xiu survey (2607.00604); read the weight-space programming post in full and its logs (github.com/RikiyaT/lm_understand-logs).
2. Define the minimum experiment for A: state features, training-data size sweep
   (10^2 ... 10^6 paths) versus BATON on the same plans and the same 2,000 test days.
3. Decide the target venue after step 2; do not choose it first.

## 6. Open decisions for the author

- ~~Share the engine or stay self-contained?~~ **Resolved 2026-11-15: self-contained**
  (`code/`). Caveat kept: a neural policy that *builds on* BATON's fitted values is a
  related paper; one benchmarked *against* BATON is not. Keep it the second.
- Whether the LLM angle (D) is in scope or only the transformer angle (A, B).
