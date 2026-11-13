# INFORMS Journal on Computing — decision (received 2026-11-13)

Manuscript JOC-2026-10-OA-2667, "Certifying rare overflow risk of regional
pickup-and-delivery plans". Area editor: Shane Henderson. **Decision: reject**
(editor's decision, no referee reports).

Editor's reasons, verbatim:

> Your model is very narrowly defined, so the results are restricted to a narrow
> class of models. It is good that you provide an analytical analysis of the highly
> stylized importance sampling problem. As such, I think you should pursue
> publication in another outlet that confines its attention to probability-related
> contributions, e.g. Probability in the Engineering and Informational Sciences.

## Response plan (adopted 2026-11-13)

Retarget to *Probability in the Engineering and Informational Sciences* (PEIS,
Cambridge University Press), as the editor suggested. The IJOC version was a
computing/OR paper with proof sketches; PEIS needs the probability to be the
contribution. Changes, in `submission_peis/`:

1. Complete proofs of everything (the IJOC appendix had sketches): gamma-quantile
   lemma, spectral facts, one-block lemma, logarithmic asymptotics with explicit
   error terms (−c/λ − O(√c) ≤ log P ≤ −c/λ + O(log c)).
2. Single shift: sharp dichotomy at ϱ = 1/4 (proved both ways), plus a new
   defensive-mixture statement (inefficient for ϱ > 1/2).
3. New result: silent underestimation (Proposition 1) — exact conditional mean of
   the estimator on the event that no draw reaches the weaker end mode; checked
   against the measured bias on 30 routes with quadrature end-mode probabilities.
4. Mixture efficiency proved (was a sketch) under a covering condition with an
   explicit covering constant Ψ (a 2-D min–max, `svrpspd_wdro/rarecert/covering.py`),
   closed-form cases, and weights that may decay exponentially (covers the weights
   SMIS uses). Condition checked on all 5 920 routes: holds for every route at
   ρ = 0.6 and 0.8.
5. Numerics cut to an illustration; quantum section, route-day simulator and most
   logistics detail removed (they remain in the IJOC version).

## Venue change (same day)

PEIS has been gold open access since volume 38 (2024), so publishing there carries an
article processing charge that the author cannot pay. New target: *Methodology and
Computing in Applied Probability* (Springer, subscription/hybrid, free without the
open-access option). The package moved from `submission_peis/` to `submission_mcap/`.
