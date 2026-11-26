# Informative Pickup Routing

- **Working title** — Informative Pickup Routing
- **State** — Exploratory research, open for edits
- **Venue** — Undecided
- **Identification result** — A synthetic construction shows that identical marginal
  distributions through order three can conceal which early visits are useful
  before a standby dispatch. The pilot computes exact minimax values and
  simulates identification from synchronized pickup records.
- **Policy optimizer** — The two-visit problem is solved from overload-labeled
  pickup records with a dispatch deadline and travel costs. The implementation
  matches exhaustive enumeration on small instances and learns the optimal
  adaptive policy in a synthetic regional example. An independent holdout can
  certify improvement over an immediate dispatch decision.
- **Open questions** — Compare the identification result and policy guarantees
  with prior work. Test whether the policy can improve a feasible service plan
  on synchronized operational records.

See [the research note](README.md) for the model, proof, and reproduction
command.
