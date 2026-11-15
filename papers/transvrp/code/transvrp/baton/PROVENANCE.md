# Provenance of the copied BATON code

`otr2.py` and `costs.py` are **verbatim copies** (unmodified) of
`svrpspd_wdro/core/otr2.py` and `svrpspd_wdro/core/costs.py` at repo commit
`37888fa`. They are copied, not imported, so this paper stays independent of
BATON (under review at C&OR, frozen). `costs.py` imports `_overflow_step` and
`_ConstantModel` from the sibling `otr2.py`, so the pair stays together.

Do not edit these two files here. New methods go in `transvrp/` next to this
folder. If an upstream fix is needed, copy a fresh version and update the
commit hash above. `tests/test_parity.py` records the upstream numbers.
