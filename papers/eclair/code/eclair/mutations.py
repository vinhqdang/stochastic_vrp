"""Mutation operators — unfaithful-candidate injection.

A candidate model is a FUNCTION params -> cp.Model. The faithful
candidate is the reference builder; an unfaithful candidate is the
reference builder composed with a systematic, instance-generic
corruption of the params — the error classes LLM modellers actually
exhibit (misread constants, misaligned data columns, dropped
constraint rows/families, flipped comparison operators):

  scalar_shift   bound misread by +/- delta            (cap = 56 -> 54)
  vec_entry      one vector entry misread (positional) (w[3] += 2)
  vec_rotate     data-wiring misalignment: item i gets vector[(i+1)%n]
  row_drop       one structural row lost (conflict/edge/coverage entry)
  family_drop    half of a structural family lost
  matrix_entry   one matrix entry misread
  op_flip        one constraint family's comparison operator inverted

Every operator is deterministic given its descriptor, so the SAME
corruption applies to every instance the candidate ever sees — a
candidate is a wrong *model*, not a wrong *instance*.

Mutants are screened for semantic difference (mutation-testing
practice: discard equivalent mutants): a mutant must disagree with the
spec's brute-forced optimum on at least one of SCREEN_TRIES micro
instances, else it is resampled.
"""

from __future__ import annotations

import copy

from .probes import brute_optimum, solve_value


class Candidate:
    """spec + corruption descriptor; label is ground truth for scoring."""

    _ids = [0]

    def __init__(self, spec, descriptor):
        self.spec = spec
        self.descriptor = descriptor            # None => faithful
        self._ids[0] += 1
        self.cid = self._ids[0]

    @property
    def faithful(self):
        return self.descriptor is None

    def corrupt(self, params):
        if self.descriptor is None:
            return params
        p = copy.deepcopy(params)
        kind = self.descriptor[0]
        if kind == "scalar_shift":
            _, key, delta = self.descriptor
            p[key] = max(1, p[key] + delta)
        elif kind == "vec_entry":
            _, key, pos, delta = self.descriptor
            vec = p[key]
            j = pos % len(vec)
            vec[j] = max(1, vec[j] + delta)
        elif kind == "vec_rotate":
            _, key = self.descriptor
            vec = p[key]
            p[key] = vec[1:] + vec[:1]
        elif kind == "row_drop":
            _, key, pos = self.descriptor
            rows = p[key]
            if rows:
                del rows[pos % len(rows)]
        elif kind == "family_drop":
            _, key = self.descriptor
            p[key] = p[key][: len(p[key]) // 2]
        elif kind == "matrix_entry":
            _, key, pos_r, pos_c, delta = self.descriptor
            mat = p[key]
            r = pos_r % len(mat)
            c = pos_c % len(mat[r])
            mat[r][c] = max(1, mat[r][c] + delta)
        elif kind == "op_flip":
            _, family = self.descriptor
            p["_flip"] = family
        else:
            raise ValueError(kind)
        return p

    def build(self, params):
        return self.spec.build(self.corrupt(params))


def _draw_descriptor(spec, rng):
    mut = spec.mut
    kinds = []
    for key in mut.get("scalars", []):
        kinds.append(("scalar_shift", key))
    for key in mut.get("vectors", []):
        kinds.append(("vec_entry", key))
        kinds.append(("vec_rotate", key))
    for key in mut.get("rows", []):
        kinds.append(("row_drop", key))
        kinds.append(("family_drop", key))
    for key in mut.get("matrices", []):
        kinds.append(("matrix_entry", key))
    for fam in mut.get("flips", []):
        kinds.append(("op_flip", fam))
    kind, key = kinds[rng.randrange(len(kinds))]
    delta = rng.choice([-3, -2, -1, 1, 2, 3])
    if kind == "scalar_shift":
        return (kind, key, delta)
    if kind == "vec_entry":
        return (kind, key, rng.randrange(64), delta)
    if kind == "matrix_entry":
        return (kind, key, rng.randrange(64), rng.randrange(64), delta)
    if kind in ("vec_rotate", "family_drop"):
        return (kind, key)
    if kind == "row_drop":
        return (kind, key, rng.randrange(64))
    return (kind, key)                          # op_flip


SCREEN_TRIES = 6


def make_mutant(spec, rng, max_attempts=40):
    """Draw a semantically non-equivalent mutant (or None if unlucky)."""
    for _ in range(max_attempts):
        cand = Candidate(spec, _draw_descriptor(spec, rng))
        for _ in range(SCREEN_TRIES):
            params = spec.gen(rng, "micro")
            truth = brute_optimum(spec, params)
            got = solve_value(cand.build(params))
            if got != truth:
                return cand
    return None


def make_faithful(spec):
    return Candidate(spec, None)
