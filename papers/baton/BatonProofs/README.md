# BatonProofs — machine-checked proofs for the BATON manuscript

Lean 4 (toolchain in `lean-toolchain`) with Mathlib. The library checks
Propositions 1–4 of Section 3 of `../main.tex` (revision 1).

## Build and check

```bash
curl -sSfL https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh | sh -s -- -y
cd papers/baton/BatonProofs
lake exe cache get        # prebuilt Mathlib
lake build                # compiles every proof: no errors, no warnings
lake env lean Axioms.lean # axiom audit of every headline theorem
```

Every headline theorem depends only on Lean's three standard axioms
(`propext`, `Classical.choice`, `Quot.sound`); there is no `sorry` or
`admit` anywhere in the library.

## Correspondence with the paper

| Paper | Lean theorem | File |
|---|---|---|
| Prop. 1, inclusion `{W_m > B} ⊆ {max W_k > B}` | `endpoint_subset_peak` | `EndpointBias.lean` |
| Prop. 1, `P(Y_end) ≤ P(Y_peak)` | `prob_endpoint_le_peak` | `EndpointBias.lean` |
| Prop. 1, endpoint labels vanish on collect-then-deliver routes, a trigger `p̂ > τ > 0` never fires | `endpoint_label_zero`, `endpoint_frequency_zero` | `EndpointBias.lean` |
| Prop. 1, two-sided bounds (eq. `biasbounds`) on `Δ = V_react − V⋆` | `endpoint_loss_bounds` | `EndpointBias.lean` |
| Prop. 1, upper bound attained when the breach is announced one stop ahead | `endpoint_loss_attained` | `EndpointBias.lean` |
| Prop. 2, inclusion of stopping regions (kernel form, any menu) | `C_le_C0`, `stop_region_subset` | `Monotone.lean` |
| Prop. 2, inclusion (history form, no Markov assumption) | `Cn_le_C0`, `opt_stop_imp_myopic_stop` | `Regret.lean` |
| Prop. 2, strictness when the myopic rule fires later with positive probability | `Cn_lt_C0` (via `V_lt_C0_of_fires`) | `Regret.lean` |
| Prop. 2, strict inclusion on a right-neighbourhood of the myopic boundary | `myopic_boundary` | `Boundary.lean` |
| Prop. 3, `σ⁰ ≤ σ*` (pathwise), regret identity, clairvoyant bound (eq. `regret`) | `opt_stop_imp_myopic_stop`, `price_of_overtriggering` | `Regret.lean` |
| Prop. 3, flat prices: regret ≤ `ω·P(σ⁰<σ*, T=∞)` ≤ `ω·P(T=∞)` | `price_flat` | `Regret.lean` |
| Clairvoyant lower bound used in Props. 1 and 3 | `Lfrom_le_V`, `Lafter_le_Cn`, `J_ge_lb` | `Regret.lean`, `EndpointBias.lean` |
| Prop. 4, monotone continuation value, any menu priced independently of the load | `C_monotone'` | `Monotone.lean` |
| Prop. 4 for BATON's menu (handoff, depot return to reset level `x₀`) | `baton_C_monotone` | `Monotone.lean` |
| `C⁰` monotone (used in the proof of Prop. 2) | `C0_monotone` | `Monotone.lean` |
| `C_k ≤ E_{k+1}` (the step that removes the need for `H ≤ E`) | `C_le_E` | `Monotone.lean` |

## Modelling choices in the formalisation

* **Proposition 4** (`Monotone.lean`). The transition of the load chain
  from stop `k` to `k+1` enters only through its expectation operator,
  abstracted as a map `P k : (ℝ → ℝ) → ℝ → ℝ` that is positive (`f ≤ g`
  ⇒ `P f ≤ P g`), preserves constants, and maps nondecreasing functions
  to nondecreasing functions (Assumption 1). These are exactly the
  properties the paper's proof uses, and the expectation operator of any
  stochastically monotone Markov kernel has them on the bounded functions
  the recursion produces. Assumption 2 is used only as "E is non-negative
  and non-increasing"; no ordering between handoff/return prices and the
  emergency price is assumed.
* **Propositions 2 (inclusion, strictness) and 3** (`Regret.lean`). A day
  is a finite probability tree whose nodes are full histories. This is
  more general than the paper's Markov setting (where values depend on
  the history only through `(k, W_k)`), and it is exactly the setting of
  the fitted policy, which works on a finite set of training days. The
  strict-inclusion claim near the boundary (`Boundary.lean`) is a
  statement about continuous functions of the load.
* **Proposition 1** (`EndpointBias.lean`). A finite set of days with
  probabilities. The optimum `V⋆` is an infimum over an admissible class
  of handoff rules; the proof needs only that admissible rules act after
  stop 1 or later and that the two constant rules ("never", "after stop
  1") are admissible — true for the stopping times of the paper. The
  formalisation surfaced one side condition that the paper now states:
  the lower bound needs `m ≥ 2` stops, so that a decision epoch exists.
