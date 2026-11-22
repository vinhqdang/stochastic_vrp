# Methodology and Computing in Applied Probability — decision on the RARECERT submission

Submission ID 660189cb-22ec-4177-ae02-1c60e1fc15a1. Editor: Markos Koutras.
Submitted 2026-11-13 (commit `68f1c6b`). **Decision: reject after review** (the email
landed in the author's spam folder; Springer's transfer page showed it as a desk
reject, which it was not). Single set of editor/reviewer comments, quoted below.
Author reported it on 2026-11-22 — the paper had already been transferred to
Queueing Systems by then (see `../STATUS.md`).

## Comments (verbatim)

> The manuscript investigates rare-event probabilities for the maximum load of
> pickup-and-delivery routes under gamma-distributed demands coupled through Gaussian
> copulas. It develops logarithmic tail asymptotics, spectral characterizations of
> dominant overflow modes and importance sampling estimators, supported by theoretical
> efficiency results and extensive numerical investigations.
>
> Although the mathematical analysis has merit, the principal contribution is closely
> tied to a specific stochastic vehicle-routing model and the associated rare-event
> simulation problem. The methodological developments largely build upon established
> large-deviation arguments, spectral analysis and importance sampling techniques,
> rather than introducing a sufficiently general probabilistic framework.
>
> The results concerning the two dominant overflow modes, the efficiency thresholds for
> individual shifts and the proposed mixture estimator provide useful insights within
> the model considered. Nevertheless, their broader methodological significance for
> Applied Probability appears relatively limited, particularly given the model-specific
> assumptions underlying the theoretical conclusions.
>
> The numerical experiments are extensive and demonstrate the practical effectiveness of
> the proposed approach. However, these investigations primarily establish computational
> advantages for the particular class of routing problems studied, rather than a more
> generally applicable advance in probabilistic methodology.
>
> Furthermore, the reference list shows limited engagement with the broader Applied
> Probability journal literature. This may indicate that the subject is not particularly
> aligned with the interests of MCAP's readership. Overall, the manuscript appears
> better suited to a specialized journal focusing on stochastic optimization,
> transportation systems or computational operations research, where its contributions
> may receive greater attention.

## Reading of the comments

- Same objection as INFORMS J. Computing (model too narrow), now from the probability
  side: a *fit* verdict, not a flaw in the proofs ("the mathematical analysis has
  merit"; numerics "extensive").
- The editor names the right audience: stochastic optimization, transportation
  systems, computational OR — i.e. not a pure applied-probability journal.
- Actionable for any resubmission: (1) state the results for the general structure
  (maximum over stages of positive parts of Gaussian quadratic forms with disjoint
  blocks) instead of only the routing model; (2) cite the applied-probability and
  rare-event literature more broadly (the reference list was built around routing and
  IS for credit portfolios).
