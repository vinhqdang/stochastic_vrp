# AAMAS 2027 — Submission 106 — Author rebuttal

Four OpenReview comments, ready to paste: one general response (post it
as a comment to all reviewers / the AC) and one reply per review. Every
number below comes from `NEW_RESULTS.md` in this directory, which lists the
script and committed summary file each one comes from. The submitted PDF is
final for review, so all new evidence is stated in the text.

---

## 1. General response (to all reviewers and the AC)

We thank the reviewers. Their shared concerns: (a) the proxy is not a
real multi-agent system; (b) the 2.6× headline is weak and the grid
small; (c) weak baselines, no held-out tuning; (d) one model family;
(e) the bottleneck claim needs a stronger scorer and a test separating
estimation from allocation. We ran new experiments for each.

**(a) A real dependent multi-agent pipeline.** We ran 120 MuSiQue
chains with 2, 3 or 4 agents (40 each), one agent per hop. Each agent's
question is filled in with the upstream agent's answer, so errors
propagate, and the baseline forwards the *whole* accumulating
conversation (all documents plus every earlier agent's full reply).
Token counts are API-reported. Results (`gemini-3.5-flash-lite`):

| policy | acc | tokens/instance | vs. full history |
|---|---|---|---|
| full history (common practice) | 0.408 | 8159 | — |
| oracle allocation | 0.617 | 598 | **+20.8**, CI [+11.7,+29.2], p<0.001, 13.6× fewer |
| per-receiver, cross-encoder top-2 | 0.400 | 980 | −0.8, CI [−10.8,+8.3], 8.3× fewer |
| per-receiver, lexical top-3 | 0.283 | 1110 | −12.5, CI [−22.5,−2.5] |
| one shared set (seed set), top-3 | 0.075 | 1036 | −33.3, CI [−42.5,−24.2] |

All three main findings hold with interacting agents. Over-delivery
hurts (+20.8 for the oracle). Per-receiver allocation beats one shared
set (+20.8, p<0.001). Better relevance estimation buys most of what is
cheaply available (+11.7 cross-encoder over lexical, p=0.013), while a
21.7-point gap to the oracle remains. One correction this experiment
forces: in this architecture, cost grows **linearly** with the number of
agents (4964/8325/11188 tokens at 2/3/4 agents): the accumulated replies
make the 4th agent's prompt only 11% longer than the 1st's; re-sending
the documents to every agent is what costs. We will replace "grows quadratically" by this measured statement.

**(b, c) Held-out replication with frozen settings.** We re-ran the
grid on 120 *new* instances that no earlier run had touched (240
receivers, twice the primary grid), with k=5 and q=0.3 frozen from the
submitted grid. Results reproduce: per-receiver vs. seed set +19.2
(CI [+12.5,+25.8], p<0.001; submitted +19.2). The oracle is +5.0 at 23.9×
fewer tokens, **now significant** (CI [+0.4,+9.6], p=0.049). Frozen
uniform k=5 is −7.9 (submitted −8.3); frozen density q=0.3 is −4.6, CI
[−8.7,−0.4]. So the 2.6× point does carry a small real loss, and we will
say so.

**(c, e) A strong relevance scorer changes the practical answer.** With
an off-the-shelf cross-encoder (ms-marco-MiniLM-L-6) as the per-receiver
scorer, the gold paragraph is ranked first for 91% of receivers (lexical:
under 60%). Under the *same* uniform per-receiver allocation, accuracy
moves from −13.8 (lexical top-3) to **−0.4 points at 23.6× fewer tokens**
(cross-encoder top-1, CI [−4.6,+3.7]). The practical frontier is
therefore set by the estimator: ~24× is reachable with no detectable
loss (bounded at 4.6 points) once relevance is estimated well. We will
make this the practical headline, replacing the 2.6× point.

**(e) Estimation vs. allocation, separated.** We fed Algorithm 1 and
uniform top-k the same scores while interpolating them from the
practical scorer to perfect relevance labels (250 instances with a
composite receiver, matched tokens). Algorithm 1's recall gain over
uniform is +1–2 points with lexical or cross-encoder scores, and
**+20.0 points (CI [+19.1,+21.0])** with perfect scores. With homogeneous
receivers it is ≈0 even with perfect scores. The allocation rule has
real headroom, but only accurate, *calibrated* per-receiver value
estimates unlock it. This is the experiment behind our claim that the
bottleneck is estimation.

**(d) A second model family.** Llama-3.3-70B-Instruct on 60 held-out
instances: per-receiver vs. seed set +24.2 (CI [+15.8,+32.5], p<0.001),
oracle +4.2 (n.s.), cross-encoder top-2 −4.2 (CI [−9.2,+0.8]) at 10.8×
fewer tokens.

**(f) The seed-set baseline given the strongest scorer.** With the
cross-encoder on both sides, per-receiver top-2 still beats the seed set
(+7.5, CI [+3.3,+11.7], p=0.001) while sending 32% fewer tokens. As
receivers are added, the shared set collapses even with the
cross-encoder. At n=16 receivers, a seed set with one item per receiver
(1830 tokens/receiver) delivers the needed paragraph to 51%, versus 85%
for per-receiver top-1 at 100 tokens/receiver.

**Corrections we will make.** (i) The OpenReview TL;DR/abstract fields
("no accuracy cost") are from an earlier draft, not the submitted PDF;
we apologise and will fix them. (ii) The influence-maximization framing
goes: the object is "one shared context set vs. per-receiver sets".
(iii) Section 4 is cut and de-hedged to make room for the new results.
(iv) "Quadratic" becomes the measured linear growth.

---

## 2. Reply to Reviewer DXj8

Thank you for the careful and fair review. Point by point:

**1. The 2.6× headline is weak.** Agreed, and the held-out run makes it
precise rather than better: frozen q=0.3 costs −4.6 points, CI [−8.7,
−0.4] (120 new instances, 240 receivers), so it is a small real loss.
We are therefore replacing the practical headline. A cross-encoder
per-receiver policy reaches **23.6× fewer tokens at −0.4 points, CI
[−4.6,+3.7]** on the same held-out set. On Llama-3.3-70B it is −5.8, CI
[−11.7,0.0]. We will state this as "no detectable loss, bounded at about
5 points", not as "free". The oracle's gain is now significant on the
held-out grid (+5.0, CI [+0.4,+9.6], p=0.049).

**2. Quadratic growth is not tested.** Correct, and testing it changed
our claim. In a new pipeline where agents depend on upstream answers and
receive the whole accumulating conversation (120 chains, 2–4 agents), cost
grows linearly with the number of agents (4964/8325/11188 tokens per
instance at 2/3/4 agents). History makes the 4th agent's prompt only 11%
longer than the 1st's; re-sent documents dominate. We will replace "quadratically" with this measured statement.
The allocation results carry over: oracle +20.8 points at 13.6× fewer
tokens (p<0.001); per-receiver beats one shared set by +20.8 (p<0.001).
Details are in the general response.

**3. A stronger matched-cost baseline.** The per-receiver arm in Table 2
*is* per-receiver greedy with the same scorer: both arms use the same
lexical scores and the same ranking, and differ only in one shared set
vs. one set per receiver. We will say so explicitly. To remove the
"weak scorer" explanation, we gave both sides the strongest scorer we
had (a cross-encoder). Per-receiver top-2 beats the cross-encoder seed
set by +7.5 (CI [+3.3,+11.7], p=0.001) while spending 32% fewer tokens;
on Llama it is +9.2 (p=0.013). The gap also widens with the number of
receivers: at n=16, a seed set with one item per receiver reaches 51%
delivery recall vs. 85% for per-receiver top-1 at 1/18 of the tokens.

**4. One model.** Llama-3.3-70B-Instruct (a different family, 70B) on 60
held-out instances: per-receiver vs. seed set +24.2 (CI [+15.8,+32.5],
p<0.001), oracle +4.2 (n.s., same sign). The held-out
`gemini-3.5-flash-lite` run independently reproduces the +19.2 gap.

**5. "At no cost in approximability."** Agreed, the wording conflates
two things. What Corollary 5 shows is that the multi-receiver coupling
adds no approximation loss beyond the single-receiver subproblem.
Whether saturation itself adds hardness (Remark 3) is a separate
question that we leave open. We will use exactly this wording in the
abstract and the conclusion.

---

## 3. Reply to Reviewer jkMC

Thank you. On the three weaknesses and two questions:

**Q1 / "too far from real multi-agent collaboration."** We ran the
experiment you describe: 120 MuSiQue chains of 2, 3 or 4 agents. Each
agent's question is filled in with the upstream agent's answer (errors
propagate), and the baseline forwards the whole accumulating
conversation. The findings transfer, more strongly. The oracle beats
full history by +20.8 points at 13.6× fewer tokens (p<0.001), so
over-delivery hurts more when agents interact. Per-receiver allocation
beats one shared set by +20.8 (p<0.001). A cross-encoder per-receiver
policy matches full history (−0.8, CI [−10.8,+8.3]) at 8.3× fewer tokens
but stays 21.7 below the oracle, which is the estimation gap again.

On additivity: the theory assumes utility additive across receivers, and
a chain is not additive. For the separation, this makes things worse for
the shared set. Under a conjunctive utility, where the task succeeds only
if every agent gets its item, Proposition 1(a)'s instance gives the
shared set value 0 and per-receiver value 1, so the ratio is unbounded
already without saturation. The exact decomposition (Proposition 4) does
need additivity. We will state this as its scope and list the
non-additive case as open.

**Q2 / estimation vs. allocation.** This was the right question, and we
now answer it directly. We hold the allocators fixed and vary only the
quality of their input scores, interpolating from the practical scorer
to perfect relevance labels. On 250 instances with heterogeneous
receivers, at matched tokens, Algorithm 1's recall gain over tuned
uniform is +0.9 points with lexical scores and +1.8 with a cross-encoder,
but **+20.0 (CI [+19.1,+21.0]) with perfect scores**. With homogeneous
receivers it is ≈0 even with perfect scores. Conversely, keeping the
allocation fixed (uniform per-receiver top-k) and improving only the
estimator moves accuracy from −13.8 to −0.4 points vs. broadcast. So the
allocation rule is not the limitation; its input is. Moreover, ranking
quality alone is not enough: Algorithm 1 needs calibrated value
estimates.

**Novelty vs. RCR-Router.** RCR-Router is empirical: it selects a
per-agent memory subset under a token budget, with no formal analysis.
We add three things it does not have. (i) A proof that the one-shared-set
object is off by a factor n (unbounded with saturation) on monotone
submodular instances, together with the measured counterpart: +19 to +24
points across two model families, held-out data and the pipeline,
growing with n. (ii) An exact decomposition showing that per-receiver
allocation costs nothing extra in approximability. (iii) Evidence of
*where* further gains are: an RCR-style router with a strong scorer is
our cross-encoder arm (24× at −0.4 points), and the factorial experiment
shows that a smarter allocator will not add to it unless estimates
become near-perfect. We agree that the separation is intuitive once
stated. Its value is that it settles which object to optimise, and the
data confirm the size of the gap.

**"Experiments do not support the conclusions."** On 2.6× vs. 22×: see
the practical policy above (23.6× at −0.4, CI [−4.6,+3.7], held-out),
which no longer relies on gold labels. On Table 4: see Q2. The factorial
design, not Table 4 alone, is now our evidence for the bottleneck claim.

---

## 4. Reply to Reviewer r7t8

Thank you for the detailed and direct review. We have acted on most of
it.

**Theory.** We accept that the propositions are elementary, and we will
present them that way: Proposition 3 becomes a remark, and the influence-
maximization framing is dropped. The object is "one shared context set
vs. per-receiver sets", which is what shared-memory and global top-k
context-sharing designs implement, not a claim about the IM literature.
What we keep is the controlled comparison and its measured counterpart,
now replicated: +19.2 held-out, +24.2 on Llama-3.3-70B, +20.8 in a
dependent pipeline, +7.5 even when both sides use a cross-encoder, and
growing with n.

**Not a multi-agent system.** We added one. In 120 chains of 2–4
agents, each agent consumes upstream answers and the baseline forwards
the accumulating conversation. The oracle is +20.8 points over full
history at 13.6× fewer tokens (p<0.001); per-receiver beats the shared
set by +20.8 (p<0.001). Cost grows *linearly* in agents (4964/8325/11188
tokens at 2/3/4), not quadratically. We will correct that sentence.

**Baselines.** Per-receiver selection with a cross-encoder is −0.4
points vs. broadcast at 23.6× fewer tokens (CI [−4.6,+3.7]), so it is no
longer a "poor policy". It still beats the cross-encoder seed set by +7.5
(p=0.001) at 32% fewer tokens. This arm is a router in the RCR-Router
style: per-agent selection by a relevance model under a token budget. We
did not reproduce RCR-Router's role-aware scorer and will say so.
AgentPrune decides *which agent-to-agent edges* carry messages; its
analogue in our pipeline is dropping the transcript and keeping only
referenced answers. That pruning saves only ~2% of tokens there (+2.5
points, n.s.), because documents dominate, which is why we optimise the
content. KVComm changes the medium (shared KV caches; open weights, one
model) and is complementary: what to share still has to be chosen.

**Evidence.**
- *Sample size:* there is now a held-out set of 120 new instances (240
  receivers).
- *Oracle significance:* the oracle is significant there (+5.0, CI
  [+0.4,+9.6], p=0.049) and in the pipeline (p<0.001).
- *Held-out tuning:* k=5 and q=0.3 were frozen from the submitted grid
  and reproduce on the new instances (−7.9 and −4.6, vs. −8.3 and −4.2
  submitted).
- *Models:* Llama-3.3-70B-Instruct was added.

**Bottleneck test.** We added two cross-encoders. The gold paragraph is
ranked first for 91% of receivers (MiniLM) and 80% (bge-reranker-base),
against under 60% for lexical. The same allocation then moves from −13.8
to −0.4 points. The factorial test (general response, (e)) shows the
allocator gains +20 points only with near-perfect estimates. An LLM
relevance judge was blocked by API quota within the window; we will add
it in the revision.

**Inconsistencies.** The TL;DR/abstract metadata came from an earlier
draft; the PDF is authoritative and the metadata will be fixed. We
apologise. Table 4: the caption's remark refers to an exploratory script
behind a pre-submission draft of that table. Every number in the
*submitted* Table 4 is regenerated by `value_curves_check.py` in the
supplement. The remark was an over-literal disclosure and will be
removed.

**Writing.** Agreed. Section 4 will be cut substantially and the hedging
removed, to make room for the new results.
