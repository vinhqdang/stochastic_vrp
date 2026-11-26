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
- **Fleet formulation** — The [mathematical model](FLEET_MODEL.md) covers
  arbitrary operating and standby fleets. The dispatch stage is a matching
  problem. With independent route signals, singleton information can have
  zero value while joint information has value linear in the number of standby
  vehicles.
- **Open questions** — Compare the results with prior work, develop a scalable
  pre-dispatch optimizer, and evaluate on synchronized operational records.

See [the research note](README.md) for the model, proof, and reproduction
command.
