"""Route eligibility and the 10^6 subset (module M8).

EXPERIMENT_A.md sections 7.1 and M8.

Eligibility is computed on the dev stream only (section 5). A route is
eligible when m >= 4 and its dev peak-breach rate is >= 2%, where a path
breaches if the running sum W_k exceeds B for some stop k in 1..m (the same
event as ``_overflow_step(cum, B) <= m``).

The 10^6 subset: 6 medium routes then 6 long routes, each drawn from the
stratum's eligible list L sorted by (inst, route_id) with
``numpy.random.default_rng(2026)`` (medium) or ``default_rng(2027)`` (long),
``.choice(len(L), 6, replace=False)``.

The output file is written only where the caller points. Nothing is written
into the repository by default.
"""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd

from .paths import route_pool, route_specs

MIN_M = 4
MIN_BREACH = 0.02
SUBSET_PER_STRATUM = 6
SUBSET_SEEDS = {"medium": 2026, "long": 2027}
COLUMNS = ["inst", "route_id", "m", "stratum", "dev_breach_rate",
           "eligible", "in_1e6_subset"]


def stratum_of(m: int) -> str:
    """Descriptive strata of section 7.1 (tiny is m <= 3, outside the sweep)."""
    if m < 4:
        return "tiny"
    if m <= 9:
        return "short"
    if m <= 14:
        return "medium"
    return "long"


def dev_breach_rates(inst: str) -> pd.DataFrame:
    """Dev peak-breach rate of every non-empty Det route of one instance."""
    pool = route_pool(inst, "dev")
    rows = []
    for spec in route_specs(inst):
        paths = pool[spec.route_id]
        peak = np.cumsum(paths.g, axis=1).max(axis=1)
        rows.append(dict(inst=inst, route_id=spec.route_id, m=spec.m,
                         stratum=stratum_of(spec.m),
                         dev_breach_rate=float((peak > spec.B).mean())))
    return pd.DataFrame(rows, columns=["inst", "route_id", "m", "stratum",
                                       "dev_breach_rate"])


def build_eligibility(insts: list[str]) -> pd.DataFrame:
    """Eligibility table for the given instances, dev stream only (no I/O)."""
    parts = [dev_breach_rates(inst) for inst in insts]
    df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(
        columns=COLUMNS[:5])
    df["eligible"] = (df["m"] >= MIN_M) & (df["dev_breach_rate"] >= MIN_BREACH)
    df["in_1e6_subset"] = False
    for stratum, seed in SUBSET_SEEDS.items():
        mask = df["eligible"] & (df["stratum"] == stratum)
        L = df.index[mask].tolist()
        L = sorted(L, key=lambda i: (df.at[i, "inst"], int(df.at[i, "route_id"])))
        # The subset is defined on the full 20-instance set (6 + 6). Smaller
        # inputs draw min(6, len(L)) routes so that the function still runs.
        take = min(SUBSET_PER_STRATUM, len(L))
        rng = np.random.default_rng(seed)
        pick = rng.choice(len(L), take, replace=False) if take else []
        for j in pick:
            df.at[L[j], "in_1e6_subset"] = True
    df = df[COLUMNS].sort_values(["inst", "route_id"], kind="stable")
    return df.reset_index(drop=True)


def _normalised(df: pd.DataFrame) -> pd.DataFrame:
    """Round-trip through CSV text so that file and fresh frames compare fairly."""
    return pd.read_csv(io.StringIO(df.to_csv(index=False)))


def eligible_routes(insts: list[str], out_path: Path,
                    overwrite: bool = False) -> pd.DataFrame:
    """Eligibility table; writes ``out_path`` unless it exists and overwrite is False.

    In the latter case the table is recomputed and must equal the file, else
    ValueError is raised.
    """
    out_path = Path(out_path)
    df = build_eligibility(list(insts))
    if out_path.exists() and not overwrite:
        stored = pd.read_csv(out_path)
        if not _normalised(df).equals(_normalised(stored)):
            raise ValueError(f"recomputed eligibility differs from {out_path}")
        return df
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return df
