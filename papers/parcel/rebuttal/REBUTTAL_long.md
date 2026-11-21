# AAMAS 2027 — Submission 106 — Author rebuttal

AAMAS allows a rebuttal only per review, with no general comment, so
each reply below stands on its own. Paste each one under its review.
Every number comes from `NEW_RESULTS.md` in this directory, which lists
the script and committed summary file behind each one. The submitted PDF
is final for review, so all new evidence is stated in the text.

---

## Reply to Reviewer DXj8

Thank you for the careful and fair review. We ran new experiments
for each point. Throughout, [a,b] is a 95% cluster-bootstrap CI and Δ is
in accuracy points.

**1. The 2.6× headline.** Agreed. We re-ran the grid on 120 new held-out
instances (240 receivers, twice the primary grid), with k=5 and q=0.3
frozen from the submission. The density rule at q=0.3 (2.6×) loses −4.6
[−8.7,−0.4] (permutation p=.062): probably a small real loss, not a free
region. Uniform k=5 gives −7.9 (submitted −8.3), and per-receiver vs.
seed set reproduces at +19.2 [+12.5,+25.8]. We are replacing the
practical headline with a stronger-scorer result:
- With per-receiver top-3 fixed in advance, swapping the lexical scorer
  for an off-the-shelf cross-encoder (ms-marco-MiniLM) moves accuracy
  vs. broadcast from −13.8 to −1.3 [−5.0,+2.5], at 6.7× fewer tokens.
- Top-1, chosen after the fact, gives −0.4 [−4.6,+3.7] at 23.6×.
- On Llama-3.3-70B, top-2 is −4.2 [−9.2,+0.8] and top-1 is −5.8
  [−11.7,0.0].

We will report this as "no detectable loss on either model; losses up
to ~5 (Gemini) and ~12 (Llama) points not excluded", not as free. The
oracle is +5.0 [+0.4,+9.6], p=.049 on the held-out set: borderline, same
sign in all three runs.

**2. Quadratic growth.** Testing it changed our claim. We built a
pipeline of 120 MuSiQue chains with 2–4 agents. Each agent's question is
filled in with the upstream agent's answer, and the baseline forwards
the whole accumulating conversation (all documents plus every earlier
reply). Cost grows roughly *linearly* in agents: 4964/8325/11188 tokens
at 2/3/4 (different chains, not one extended). History adds only 11% to
the 4th agent's prompt; re-sent documents dominate. We will replace
"quadratically" with this. The allocation findings carry over (both
p<.001):
- the oracle is +20.8 [+11.7,+29.2] over full history at 13.6× fewer
  tokens;
- per-receiver beats one shared set by +20.8.

**3. Stronger matched-cost baseline.** In Table 2 both arms use the same
lexical relevance scores. Per-receiver takes each receiver's top-k by
score per token; the seed set takes the top-k by score summed over
receivers. Tokens are not exactly matched, and the mismatch favours the
seed set (283 vs. 254). Giving both sides the strongest scorer we had,
per-receiver top-2 still beats the cross-encoder seed set by +7.5
[+3.3,+11.7], p=.001, while spending 32% fewer tokens (Llama: +9.2,
p=.013). In fused instances the shared set degrades as receivers are
added. At n=16, delivery recall is 51% for a seed set with one item per
receiver, vs. 85% for per-receiver top-1 at 1/18 of the tokens (recall
only, word-count tokens).

**4. One model.** On Llama-3.3-70B-Instruct (a different family, 60
held-out instances), per-receiver vs. seed set is +24.2 [+15.8,+32.5],
p<.001, and the oracle is +4.2 (n.s., same sign). Separately, the
held-out `gemini-3.5-flash-lite` run reproduces +19.2.

**5. "No cost in approximability."** Agreed. The precise statement:
under integer costs (so the algorithm is pseudo-polynomial) and a
(1−ε)-oracle for each receiver at every budget level, the multi-receiver
coupling adds no approximation loss beyond the single-receiver
subproblem (Corollary 5). For general submodular relevance, the
transferred ratio is (1−1/e). Whether saturation adds hardness (Remark 3)
remains open. The abstract and conclusion will say exactly this.

We also apologise that the OpenReview TL;DR/abstract fields came from an
earlier draft ("no accuracy cost"). The PDF is authoritative and the
metadata will be fixed.

---

## Reply to Reviewer jkMC

Thank you. We ran new experiments for each concern ([a,b] = 95%
cluster-bootstrap CI; Δ in accuracy points).

**Q1 / real multi-agent collaboration.** We ran 120 MuSiQue chains of
2–4 agents (40 each), one agent per hop. Each agent's question is filled
in with the upstream agent's answer, so errors propagate. The baseline
forwards the whole accumulating conversation (all documents plus every
earlier reply). Model `gemini-3.5-flash-lite`; tokens are API-reported.

| policy | acc | tok/instance | vs. full history |
|---|---|---|---|
| full history | 0.408 | 8159 | — |
| oracle | 0.617 | 598 | +20.8 [+11.7,+29.2], 13.6× fewer |
| per-receiver, cross-encoder top-2 | 0.400 | 980 | −0.8 [−10.8,+8.3], 8.3× fewer |
| per-receiver, lexical top-3 | 0.283 | 1110 | −12.5 [−22.5,−2.5] |
| one shared set, top-3 | 0.075 | 1036 | −33.3 [−42.5,−24.2] |

The findings transfer to dependent agents:
- Over-delivery hurts (oracle +20.8, p<.001).
- Per-receiver beats one shared set (+20.8, p<.001).
- A better estimator helps (+11.7 for the cross-encoder over lexical,
  p=.013), but a 21.7-point gap to the oracle remains. The logs show
  UNKNOWN answers cascading downstream.

One correction: cost grows roughly linearly in agents (4964/8325/11188
tokens at 2/3/4), not quadratically.

On additivity: the theory assumes utility additive across receivers;
chains are not. For the separation this makes things worse for the
shared set. Under a conjunctive utility (success only if every agent
gets its item), Proposition 1(a)'s instance gives the shared set 0 and
per-receiver 1: an unbounded ratio without saturation. The exact
decomposition (Proposition 4) does need additivity. We will state that
as its scope and list the non-additive case as open.

**Q2 / estimation vs. allocation.** We now test this directly, in two
directions.
- *Fixed allocators, varying scores.* Algorithm 1 and uniform top-k get
  the same scores, interpolated from the practical scorer to perfect
  relevance labels (250 instances with heterogeneous receivers;
  delivery recall at matched tokens). Algorithm 1's gain over uniform is
  +0.9 with lexical scores, +1.8 with a cross-encoder, and **+20.0
  [+19.1,+21.0] with perfect scores**. With homogeneous receivers it is
  ≤0 at every quality level.
- *Fixed allocation, varying estimator.* With per-receiver top-3 on 120
  new held-out instances, replacing lexical with a cross-encoder moves
  accuracy vs. broadcast from −13.8 to −1.3 [−5.0,+2.5], at 6.7× fewer
  tokens.

So the allocation rule has headroom that only near-perfect, calibrated
estimates unlock. The first test is at the recall level; Algorithm 1
with perfect scores was not run end-to-end with an LLM.

**Novelty vs. RCR-Router.** RCR-Router selects a per-agent memory subset
under a token budget, without optimality guarantees. We add three
things:
1. A proof that the one-shared-set object loses a factor n (unbounded
   with saturation) even on monotone submodular instances. Its measured
   counterpart is +19 to +24 points across two model families (Gemini,
   Llama-3.3-70B), held-out data and the pipeline.
2. An exact decomposition: per-receiver allocation inherits
   single-receiver approximability (under Corollary 5's conditions).
3. Evidence on where further gains lie. A per-agent relevance-model
   router (our cross-encoder arm) recovers most of the gap, and the
   factorial test shows a smarter allocator adds little until estimates
   are near-perfect.

We agree the separation is intuitive once stated. Its use is settling
which object to optimise.

**Conclusions vs. evidence.**
- *2.6×:* with q=0.3 frozen, the held-out loss is −4.6 [−8.7,−0.4]
  (p=.062). We will call it a small loss.
- *22×:* the oracle is +5.0 [+0.4,+9.6], p=.049, but needs gold labels.
  The practical result does not: −1.3 at 6.7× with a pre-specified k,
  or −0.4 [−4.6,+3.7] at 23.6× with top-1 (chosen post hoc).
- *Bottleneck claim:* it rests on the two-direction test above, not on
  Table 4 alone.

---

## Reply to Reviewer r7t8

Thank you for the detailed and direct review. We ran new experiments
for most points ([a,b] = 95% cluster-bootstrap CI; Δ in accuracy points).

**Theory.** We accept that the propositions are elementary and will
present them so. Proposition 3 becomes a remark. The influence-
maximization framing is dropped: the object is "one shared context set
vs. per-receiver sets", as in shared-memory and global top-k designs.
We keep the controlled comparison, now replicated (below): +19.2
held-out, +24.2 on Llama-3.3-70B, +20.8 in a dependent pipeline, and
+7.5 even with a cross-encoder on both sides.

**Not a multi-agent system.** We added one: 120 MuSiQue chains of 2–4
agents. Each agent's question is filled in with the upstream answer
(errors propagate), against a baseline that forwards the whole
accumulating conversation. The oracle beats full history by +20.8
[+11.7,+29.2] at 13.6× fewer tokens. Per-receiver beats one shared set
by +20.8 (both p<.001). A cross-encoder per-receiver policy matches full
history (−0.8 [−10.8,+8.3]) at 8.3× fewer tokens. Cost grows roughly
linearly in agents (4964/8325/11188 tokens at 2/3/4), not quadratically;
we will correct that sentence.

**Baselines.**
- With a cross-encoder (ms-marco-MiniLM), per-receiver selection is not
  a poor policy: −1.3 [−5.0,+2.5] vs. broadcast at 6.7× fewer tokens
  (top-3, k fixed in advance), or −0.4 at 23.6× (top-1, chosen post hoc).
  It beats the cross-encoder seed set by +7.5 [+3.3,+11.7] (p=.001) at
  32% fewer tokens.
- We did not run RCR-Router, AgentPrune or KVComm themselves. Our
  cross-encoder arm is a per-agent relevance-model router in the spirit
  of RCR-Router, not a reimplementation.
- A message-pruning ablation (keep only referenced upstream answers,
  drop the transcript; not AgentPrune) saves just 2.3% of tokens in the
  pipeline (+2.5, n.s.), because documents dominate. This is why we
  optimise content.
- KVComm changes the medium (shared KV caches, open weights) and is
  complementary.

**Evidence.**
- *Held-out set:* 120 new instances (240 receivers), with k=5 and q=0.3
  frozen from the submission. Per-receiver vs. seed set reproduces at
  +19.2 [+12.5,+25.8], and k=5 and q=0.3 give −7.9 and −4.6 (submitted
  −8.3 and −4.2).
- *Oracle significance:* +5.0 [+0.4,+9.6], p=.049 on the held-out set
  (borderline, same sign in every run). In the pipeline it is +20.8,
  p<.001.
- *Models:* Llama-3.3-70B-Instruct, 60 held-out instances: per-receiver
  vs. seed set +24.2 [+15.8,+32.5], p<.001.

**Bottleneck test.** Two cross-encoders rank the gold paragraph first
for 91% (MiniLM) and 80% (bge-reranker) of receivers, vs. 58% for
lexical. The same allocation then improves from −13.8 to −1.3. A
factorial test feeds Algorithm 1 and uniform top-k the same scores,
interpolated to perfect labels. Algorithm 1 gains +20.0 recall points
over uniform with perfect scores, but only +1–2 with real ones. An LLM
relevance judge was blocked by API quota in the window; we will add it.

**Inconsistencies.** The TL;DR and abstract metadata came from an
earlier draft; the PDF is authoritative, and we apologise and will fix
it.

On Table 4: before submission we found that the table in an earlier
draft could not be reproduced from the code we still had. We rebuilt it
from scratch with `value_curves_check.py`, which is in the supplement,
and every submitted number regenerates from that script. One cell
differs from the lost version: captured-mass on the plain set changed
from a tie to a loss. The caption sentence was our disclosure of this.
We will replace it with a clearer provenance note and record the
changed cell.

**Writing.** Agreed. Section 4 will be cut substantially and the hedging
removed, to make room for these results.
