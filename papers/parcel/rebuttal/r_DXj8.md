Thank you. New experiments ([a,b] = 95% cluster CI, Δ in accuracy points):

**1. 2.6× headline.** Agreed. On 120 new held-out instances (240 receivers) with q=0.3 and k=5 frozen, q=0.3 loses −4.6 [−8.7,−0.4] (p=.062), a small real loss; k=5 gives −7.9 (submitted −8.3). We replace the headline with a stronger scorer. With per-receiver top-3 fixed in advance, an off-the-shelf cross-encoder moves Δ vs. broadcast from −13.8 (lexical) to −1.3 [−5.0,+2.5] at 6.7× fewer tokens. Top-1 (post hoc) gives −0.4 [−4.6,+3.7] at 23.6×; on Llama-3.3-70B it gives −5.8 [−11.7,0.0]. We will write "no detectable loss; up to ~5 (Gemini) / ~12 (Llama) points not excluded". Held-out oracle: +5.0 [+0.4,+9.6], p=.049 (borderline, same sign in all runs).

**2. Quadratic growth.** Tested; it changed our claim. In 120 MuSiQue chains of 2–4 dependent agents, each fed upstream answers, the baseline forwards the whole accumulating conversation. Cost grows roughly *linearly* (4964/8325/11188 tokens at 2/3/4 agents); history adds only 11% to the 4th prompt. We will correct the sentence. Allocation findings carry over: oracle +20.8 [+11.7,+29.2] at 13.6× fewer tokens; per-receiver beats one shared set by +20.8 (both p<.001).

**3. Matched baseline.** Both Table 2 arms use the same lexical scores; per-receiver takes each receiver's top-k, the seed set takes the top-k of summed scores, and the token mismatch favours the seed set (283 vs. 254). With the cross-encoder on both sides, per-receiver top-2 still wins by +7.5 [+3.3,+11.7] (p=.001) at 32% fewer tokens (Llama +9.2, p=.013). At n=16 fused receivers, a seed set with one item per receiver delivers 51% vs. 85% for per-receiver top-1 at 1/18 of the tokens (recall).

**4. One model.** Llama-3.3-70B, 60 held-out instances: per-receiver vs. seed set +24.2 [+15.8,+32.5], p<.001; oracle +4.2 (n.s.). The held-out Gemini run reproduces +19.2.

**5. Wording.** Agreed. Corollary 5 shows the multi-receiver coupling adds no loss beyond the single-receiver subproblem, given integer costs and a (1−ε)-oracle at every level, or (1−1/e) for submodular relevance. Saturation hardness (Remark 3) stays open; the abstract will say this.

The OpenReview TL;DR/abstract came from an earlier draft; sorry, we will fix it.
