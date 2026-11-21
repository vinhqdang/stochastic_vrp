# AAMAS 2027 — Submission 106 — Author rebuttal

Four OpenReview comments, ready to paste: one general response (post it
as a comment to all reviewers / the AC) and one reply per review. Every
number below comes from `NEW_RESULTS.md` in this directory, which lists
the script and committed summary file each one comes from. The submitted
PDF is final for review, so all new evidence is stated in the text.

---

## 1. General response (to all reviewers and the AC)

We thank the reviewers. Their shared concerns: (a) the proxy is not a
real multi-agent system; (b) the 2.6× headline is weak and the grid
small; (c) weak baselines, no held-out tuning; (d) one model family;
(e) the bottleneck claim needs a stronger scorer and a test separating
estimation from allocation. We ran new experiments for each.

**(a) A dependent multi-agent pipeline.** We ran 120 MuSiQue chains of
2, 3 or 4 agents (40 each), one agent per hop. Each agent's question is
filled in with the upstream agent's answer, so errors propagate. The
baseline forwards the whole accumulating conversation (all documents
plus every earlier agent's full reply). Tokens are API-reported;
`gemini-3.5-flash-lite`.

| policy | acc | tok/instance | vs. full history |
|---|---|---|---|
| full history | 0.408 | 8159 | — |
| oracle | 0.617 | 598 | +20.8 [+11.7,+29.2], p<.001, 13.6× fewer |
| per-receiver, cross-encoder top-2 | 0.400 | 980 | −0.8 [−10.8,+8.3], 8.3× fewer |
| per-receiver, lexical top-3 | 0.283 | 1110 | −12.5 [−22.5,−2.5] |
| one shared set, top-3 | 0.075 | 1036 | −33.3 [−42.5,−24.2] |

All three findings transfer to interacting agents:
- Over-delivery hurts: the oracle is +20.8.
- Per-receiver allocation beats one shared set: +20.8, p<.001.
- A better estimator helps (+11.7 for the cross-encoder over lexical,
  p=.013), but 21.7 points to the oracle remain.

One correction: across these chains, cost grows roughly *linearly* in
the number of agents (4964/8325/11188 tokens at 2/3/4 agents). The
accumulated replies make the 4th agent's prompt only 11% longer than
the 1st's; re-sending the documents dominates. "Quadratic" will be
replaced by this.

**(b, c) Held-out replication, frozen settings.** 120 new instances (240
receivers) never used before; k=5 and q=0.3 frozen from the submission.
- Per-receiver vs. seed set: +19.2 [+12.5,+25.8], p<.001 (submitted
  +19.2).
- Oracle: +5.0 at 23.9× fewer tokens, [+0.4,+9.6], p=.049. This is
  borderline but consistent across three runs.
- Uniform k=5: −7.9 (submitted −8.3).
- Density q=0.3: −4.6 [−8.7,−0.4], permutation p=.062. This suggests a
  small real loss at 2.6×, and we will say so.

**(c, e) A stronger scorer.** An off-the-shelf cross-encoder
(ms-marco-MiniLM) ranks the gold paragraph first for 91% of receivers,
against 58% for lexical (raw ranking). With the *same* allocation,
per-receiver top-3, accuracy vs. broadcast moves from −13.8 (lexical) to
−1.3 (cross-encoder, [−5.0,+2.5]). Top-1 and top-2, chosen after seeing
this grid, give −0.4 [−4.6,+3.7] at 23.6× fewer tokens. On Llama-3.3-70B,
top-2 is −4.2 [−9.2,+0.8]. So the estimator sets the practical frontier:
~10–24× without a detectable loss, though a loss of about 5 (Gemini) to
9–12 (Llama) points is not excluded.

**(e) Estimation vs. allocation.** We gave Algorithm 1 and uniform top-k
the same scores, interpolated from practical to perfect relevance (250
instances with a composite receiver, delivery recall at matched tokens).
- Algorithm 1's gain is +0.9 (lexical) and +1.8 (cross-encoder).
- With perfect scores it is **+20.0 [+19.1,+21.0]**.
- With homogeneous receivers it is ≤0 at every quality level.

The allocation rule has headroom, but only accurate, calibrated value
estimates unlock it. This is a recall-level result.

**(d, f) Another model family; the strongest seed set.**
- Llama-3.3-70B-Instruct (60 held-out instances): per-receiver vs. seed
  set +24.2 [+15.8,+32.5], p<.001; oracle +4.2 (n.s.).
- With the cross-encoder on *both* sides, per-receiver top-2 beats the
  seed set by +7.5 [+3.3,+11.7] (Llama +9.2) at 32% fewer tokens.
- In fused instances (delivery recall, word-count tokens) the shared set
  degrades with n: at n=16 a seed set with one item per receiver
  reaches 51%, against 85% for per-receiver top-1 at 1/18 of the tokens.

**Corrections.**
- The OpenReview TL;DR and abstract ("no accuracy cost") came from an
  earlier draft. We apologise and will fix them.
- The influence-maximization framing goes.
- Section 4 will be cut and de-hedged.
- "Quadratic" becomes the measured linear growth.

---

## 2. Reply to Reviewer DXj8

Thank you for the careful and fair review.

**1. The 2.6× headline.** Agreed. On 120 held-out instances with q=0.3
frozen, the loss is −4.6 [−8.7,−0.4] (permutation p=.062): probably a
small real loss, not a free region. We are replacing the practical
headline with the stronger-scorer result.
- With per-receiver top-3 fixed in advance, a cross-encoder scorer gives
  −1.3 [−5.0,+2.5] at 6.7× fewer tokens. Lexical gives −13.8.
- Top-1, chosen after the fact, gives −0.4 [−4.6,+3.7] at 23.6×.
- On Llama-3.3-70B, top-2 is −4.2 [−9.2,+0.8] and top-1 is −5.8
  [−11.7,0.0].

We will report this as "no detectable loss on either model; losses up
to ~5 (Gemini) and ~12 (Llama) points not excluded", not as free. The
oracle is +5.0 [+0.4,+9.6], p=.049: borderline, same sign in all three
runs.

**2. Quadratic growth.** Testing it changed our claim. In a new
pipeline, agents consume upstream answers and receive the whole
accumulating conversation (120 chains, 2–4 agents). Cost grows roughly
linearly in agents (4964/8325/11188 tokens at 2/3/4; these are
different chains, not one chain extended). History adds only 11% to
the 4th agent's prompt; re-sent documents dominate. We will replace
"quadratically" with this. The allocation findings carry over: oracle
+20.8 at 13.6× fewer tokens; per-receiver vs. one shared set +20.8 (both
p<.001).

**3. Stronger matched-cost baseline.** In Table 2 both arms use the same
lexical relevance scores. The per-receiver arm takes each receiver's
top-k by score per token; the seed set takes the top-k by score summed
over receivers. Tokens are not exactly matched, and the mismatch favours
the seed set (283 vs. 254). Giving both sides the strongest scorer we
had, per-receiver top-2 still beats the cross-encoder seed set by
+7.5 [+3.3,+11.7], p=.001, while spending 32% fewer tokens (Llama +9.2,
p=.013). In fused instances the shared set degrades as receivers are
added (delivery recall at n=16: 51% for a seed set with one item per
receiver, vs. 85% for per-receiver top-1 at 1/18 of the tokens).

**4. One model.** Llama-3.3-70B-Instruct, 60 held-out instances:
per-receiver vs. seed set +24.2 [+15.8,+32.5], p<.001; oracle +4.2
(n.s., same sign). Separately, the held-out `gemini-3.5-flash-lite` run
reproduces +19.2.

**5. "No cost in approximability."** Agreed. The precise statement:
under integer costs (so the algorithm is pseudo-polynomial) and a
(1−ε)-oracle for each receiver at every budget level, the multi-receiver
coupling adds no approximation loss beyond the single-receiver
subproblem (Corollary 5). For general submodular relevance, the
transferred ratio is (1−1/e). Whether saturation adds hardness (Remark 3)
remains open. The abstract and conclusion will say exactly this.

---

## 3. Reply to Reviewer jkMC

Thank you.

**Q1 / real multi-agent collaboration.** We ran this experiment: 120
MuSiQue chains of 2–4 agents, where each agent's question is filled in
with the upstream agent's answer (errors propagate) and the baseline
forwards the whole accumulating conversation. The findings transfer:
- The oracle beats full history by +20.8 at 13.6× fewer tokens.
- Per-receiver beats one shared set by +20.8. Both p<.001.
- A cross-encoder per-receiver policy matches full history (−0.8
  [−10.8,+8.3]) at 8.3× fewer tokens but stays 21.7 below the oracle.
  The estimation gap persists, and the logs show UNKNOWN answers
  cascading downstream.

On additivity: the theory assumes utility additive across receivers;
chains are not. For the separation this makes things worse for the
shared set. Under a conjunctive utility (success only if every agent
gets its item), Proposition 1(a)'s instance gives the shared set 0 and
per-receiver 1, an unbounded ratio without saturation. The exact
decomposition (Proposition 4) does need additivity. We will state that
as its scope and list the non-additive case as open.

**Q2 / estimation vs. allocation.** We now test this directly, in two
directions.
- *Fixed allocators, varying scores.* Scores are interpolated from the
  practical scorer to perfect relevance labels (250 instances with
  heterogeneous receivers; delivery recall at matched tokens).
  Algorithm 1's gain over uniform is +0.9 with lexical scores, +1.8 with
  a cross-encoder, and **+20.0 [+19.1,+21.0] with perfect scores**. With
  homogeneous receivers it is ≤0 at every quality level.
- *Fixed allocation, varying estimator.* With per-receiver top-3,
  replacing lexical with a cross-encoder moves accuracy vs. broadcast
  from −13.8 to −1.3 on held-out data.

So the allocation rule has headroom that only near-perfect, calibrated
estimates unlock. The first test is at the recall level; Algorithm 1
with perfect scores was not run end-to-end with an LLM.

**Novelty vs. RCR-Router.** RCR-Router selects a per-agent memory subset
under a token budget, without optimality guarantees. We add three
things:
1. A proof that the one-shared-set object loses a factor n (unbounded
   with saturation) even on monotone submodular instances. Its measured
   counterpart is +19 to +24 points across two model families, held-out
   data and the pipeline.
2. An exact decomposition: per-receiver allocation inherits the
   single-receiver approximability (under the conditions of
   Corollary 5).
3. Evidence on where further gains lie. A per-agent relevance-model
   router (our cross-encoder arm) recovers most of the gap, and the
   factorial test shows a smarter allocator adds little until estimates
   are near-perfect.

We agree the separation is intuitive once stated. Its use is settling
which object to optimise.

**Conclusions vs. evidence.** The practical result no longer relies on
gold labels: −1.3 [−5.0,+2.5] at 6.7× with a pre-specified k, held-out.
The bottleneck claim now rests on the two-direction test above, not on
Table 4 alone.

---

## 4. Reply to Reviewer r7t8

Thank you for the detailed and direct review.

**Theory.** We accept that the propositions are elementary and will
present them so. Proposition 3 becomes a remark. The influence-
maximization framing is dropped: the object is "one shared context set
vs. per-receiver sets", as in shared-memory and global top-k designs.
We keep the controlled comparison and its replications: +19.2
held-out, +24.2 on Llama-3.3-70B, +20.8 in a dependent pipeline, and
+7.5 even with a cross-encoder on both sides.

**Not a multi-agent system.** We added one: 120 chains of 2–4 agents,
each consuming upstream answers, against a baseline that forwards the
accumulating conversation. The oracle is +20.8 over full history at
13.6× fewer tokens; per-receiver beats the shared set by +20.8 (both
p<.001). Cost grows roughly linearly in agents, not quadratically, and
we will correct that sentence.

**Baselines.**
- With a cross-encoder, per-receiver selection is no longer a poor
  policy: −1.3 [−5.0,+2.5] vs. broadcast at 6.7× fewer tokens (top-3), or
  −0.4 at 23.6× (top-1, chosen post hoc). It beats the cross-encoder
  seed set by +7.5 (p=.001) at 32% fewer tokens.
- We did not run RCR-Router, AgentPrune or KVComm themselves. Our
  cross-encoder arm is a per-agent relevance-model router in the spirit
  of RCR-Router, not a reimplementation.
- A message-pruning ablation (keep only referenced upstream answers,
  drop the transcript; not AgentPrune) saves just 2.3% of tokens in the
  pipeline (+2.5 points, n.s.), because documents dominate. This is why
  we optimise content.
- KVComm changes the medium (shared KV caches, open weights) and is
  complementary.

**Evidence.**
- *Size and significance:* a held-out set of 120 new instances (240
  receivers). There the oracle is +5.0 [+0.4,+9.6], p=.049 (borderline,
  same sign in every run). In the pipeline it is +20.8, p<.001.
- *Held-out tuning:* k=5 and q=0.3 were frozen and reproduce (−7.9 and
  −4.6, vs. −8.3 and −4.2 submitted).
- *Models:* Llama-3.3-70B-Instruct was added.

**Bottleneck test.** Two cross-encoders rank the gold paragraph first
for 91% (MiniLM) and 80% (bge-reranker) of receivers, vs. 58% for
lexical. The same allocation then improves from −13.8 to −1.3. The
factorial test (general response) shows the allocator gains +20 recall
points only with near-perfect estimates. An LLM relevance judge was
blocked by API quota in the window; we will add it.

**Inconsistencies.** The TL;DR and abstract metadata came from an
earlier draft; the PDF is authoritative, and we apologise and will fix
it.

On Table 4: before submission we found that the table in an earlier
draft could not be reproduced from the code we still had. We therefore
rebuilt it from scratch with `value_curves_check.py`, which is in the
supplement. Every submitted number regenerates from that script. The
rebuilt table is not identical to the lost one: captured-mass on the
plain set changed from a tie to a loss. The caption sentence was our
disclosure of this. We will replace it with a clearer provenance note
in the supplement and record the changed cell.

**Writing.** Agreed. Section 4 will be cut substantially and the hedging
removed.
