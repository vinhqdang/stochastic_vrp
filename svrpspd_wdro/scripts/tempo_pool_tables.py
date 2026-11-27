"""Tables for the TEMPO pooling experiment.

Usage (from svrpspd_wdro/):
    python scripts/tempo_pool_tables.py [fam=Dethloff,City,SalhiNagy] [tag=]

Reads results/tempo_pool/<fam><tag>.csv (rows sorted by instance and day
before any resampling, so output does not depend on completion order) and
writes results/tempo_pool/tables.md with, per family and scenario:
  * mean cost per policy;
  * paired cost differences (hier minus each other policy; hier_eb minus
    route, zone, full) with 95% instance-clustered bootstrap intervals
    (resample instances, then days within each drawn instance; 2000
    resamples; fixed seed);
  * share of the fixed-to-known gap closed by each policy;
  * slope of log(mean excess cost over known) on log(mean squared factor
    error) across instances and policies (points with non-positive excess
    or error are dropped and counted).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_W = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_W))
from ev.pooling import POL, SCENARIOS  # noqa: E402

OUT = _W / "results" / "tempo_pool"
BOOT = 2000
SEED = 20261127
PAIRS = [("hier", p) for p in ["fixed", "route", "full", "zone", "hier_eb", "mle", "known"]]
PAIRS += [("hier_eb", p) for p in ["route", "zone", "full"]]
SCEN_ORDER = list(SCENARIOS)


def _md_table(header, rows):
    out = ["| " + " | ".join(header) + " |",
           "|" + "|".join("---" for _ in header) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(out)


def _matrix(sub, col):
    """(n_inst, n_days) matrix of one column, instances and days sorted."""
    days = sorted(set.intersection(*[set(g["day"]) for _, g in sub.groupby("instance")]))
    mats = []
    for inst in sorted(sub["instance"].unique()):
        g = sub[sub["instance"] == inst].set_index("day").loc[days]
        mats.append(g[col].to_numpy(dtype=float))
    return np.vstack(mats)


def boot_ci(M, rng, B=BOOT):
    """Instance-clustered bootstrap of the mean over instances of the
    per-instance day means. M: (n_inst, n_days) paired differences."""
    n_i, n_d = M.shape
    point = float(M.mean(axis=1).mean())
    stats = np.empty(B)
    for b in range(B):
        sel = rng.integers(0, n_i, n_i)
        days = rng.integers(0, n_d, (n_i, n_d))
        stats[b] = M[sel[:, None], days].mean(axis=1).mean()
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return point, float(lo), float(hi)


def slope_points(sub):
    xs, ys = [], []
    n_tot = 0
    for inst, g in sub.groupby("instance"):
        for p in POL:
            if p == "known":
                continue
            n_tot += 1
            ex = float((g[f"cost_{p}"] - g["cost_known"]).mean())
            mse = float(g[f"se_{p}"].sum() / max(1, g[f"nd_{p}"].sum()))
            if ex > 0 and mse > 0:
                xs.append(np.log(mse))
                ys.append(np.log(ex))
    slope = float(np.polyfit(xs, ys, 1)[0]) if len(xs) >= 2 else float("nan")
    return slope, len(xs), n_tot


def family_section(fam, df):
    lines = [f"## {fam}", ""]
    n_inst = df["instance"].nunique()
    lines.append(f"{n_inst} instances; days per instance: "
                 f"{df.groupby('instance')['day'].nunique().min()}-"
                 f"{df.groupby('instance')['day'].nunique().max()}; "
                 f"zones per instance: {int(df['Z'].min())}-{int(df['Z'].max())}.")
    lines.append("")
    scens = [s for s in SCEN_ORDER if s in set(df["scenario"])]

    # 1. mean cost per policy
    lines += ["### Mean cost per policy", ""]
    header = ["scenario"] + POL
    rows = []
    for s in scens:
        sub = df[df["scenario"] == s]
        rows.append([s] + [f"{sub[f'cost_{p}'].mean():.2f}" for p in POL])
    lines += [_md_table(header, rows), ""]

    # 2. paired differences with bootstrap intervals
    lines += ["### Paired cost differences (A minus B; negative favours A)", "",
              "95% instance-clustered bootstrap intervals (2000 resamples).", ""]
    header = ["scenario", "A", "B", "mean diff", "95% CI", "instances A<B"]
    rows = []
    for s in scens:
        sub = df[df["scenario"] == s]
        rng = np.random.default_rng(SEED)
        for a, b in PAIRS:
            M = _matrix(sub, f"cost_{a}") - _matrix(sub, f"cost_{b}")
            pt, lo, hi = boot_ci(M, rng)
            frac = float((M.mean(axis=1) < 0).mean())
            rows.append([s, a, b, f"{pt:+.2f}", f"[{lo:+.2f}, {hi:+.2f}]",
                         f"{frac * 100:.0f}%"])
    lines += [_md_table(header, rows), ""]

    # 3. share of the fixed-to-known gap closed
    lines += ["### Share of the fixed-to-known gap closed", "",
              "(mean fixed cost minus mean policy cost) / (mean fixed cost minus "
              "mean known cost).", ""]
    header = ["scenario"] + [p for p in POL if p not in ("fixed", "known")]
    rows = []
    for s in scens:
        sub = df[df["scenario"] == s]
        gap = sub["cost_fixed"].mean() - sub["cost_known"].mean()
        row = [s]
        for p in header[1:]:
            share = (sub["cost_fixed"].mean() - sub[f"cost_{p}"].mean()) / gap
            row.append(f"{share:.2f}")
        rows.append(row)
    lines += [_md_table(header, rows), ""]

    # 4. slope of log excess on log MSE
    lines += ["### Slope of log(excess cost over known) on log(mean squared factor error)", "",
              "Points: instance x policy (known excluded); points with non-positive excess "
              "or error are dropped.", ""]
    header = ["scenario", "slope", "points used", "points total"]
    rows = []
    for s in scens:
        sub = df[df["scenario"] == s]
        slope, n_used, n_tot = slope_points(sub)
        rows.append([s, f"{slope:.3f}", n_used, n_tot])
    lines += [_md_table(header, rows), ""]
    return lines


def main(kw):
    fams = kw.get("fam", "Dethloff,City,SalhiNagy").split(",")
    tag = kw.get("tag", "")
    out = ["# TEMPO pooling experiment: tables", "",
           "Generated by `scripts/tempo_pool_tables.py` from "
           "`results/tempo_pool/<family>.csv`. Costs are daily recourse costs; "
           "`known` is a reference (true factors), not a bound.", ""]
    done = []
    for fam in fams:
        path = OUT / f"{fam}{tag}.csv"
        if not path.exists():
            print(f"skip {fam}: {path} missing")
            continue
        df = pd.read_csv(path)
        out += family_section(fam, df)
        done.append(fam)
    (OUT / "tables.md").write_text("\n".join(out) + "\n")
    print(f"wrote {OUT / 'tables.md'} for {done}")


if __name__ == "__main__":
    kw = dict(x.split("=", 1) for x in sys.argv[1:])
    main(kw)
