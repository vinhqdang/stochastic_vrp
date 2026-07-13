#!/usr/bin/env python3
"""fetch_amazon_pilot.py — stream a pilot slice of the 2021 Amazon
Last-Mile Routing Research Challenge evaluation data from the public
AWS bucket (see data/AmazonLMC/README.md for citation/licence).

Streams the monolithic travel-time (843 MB) and package (166 MB) JSONs
over HTTP with ijson, keeping only n_per_metro routes per metro, so
nothing large ever touches disk. Requires: pip install ijson.

Usage (from svrpspd_wdro/): python scripts/fetch_amazon_pilot.py [n_per_metro=5]
"""
import json
import sys
import urllib.request
from collections import defaultdict
from pathlib import Path

import ijson

_WDRO = Path(__file__).resolve().parent.parent
DATA = _WDRO / "data" / "AmazonLMC"
BASE = ("https://amazon-last-mile-challenges.s3.amazonaws.com/almrrc2021/"
        "almrrc2021-data-evaluation/model_apply_inputs/")


class NanFilter:
    """File-like wrapper replacing literal NaN (non-standard JSON in the
    package file) with null across chunk boundaries."""

    def __init__(self, raw):
        self.raw = raw
        self.tail = b""

    def read(self, n=65536):
        chunk = self.raw.read(n)
        if not chunk:
            out, self.tail = self.tail, b""
            return out
        buf = (self.tail + chunk).replace(b"NaN", b"null")
        out, self.tail = buf[:-3], buf[-3:]
        return out


def stream_pick(fname, wanted, out_name, nan_filter=False):
    got = {}
    with urllib.request.urlopen(BASE + fname, timeout=900) as resp:
        src = NanFilter(resp) if nan_filter else resp
        for rid, obj in ijson.kvitems(src, "", use_float=True):
            if rid in wanted:
                got[rid] = obj
                if len(got) == len(wanted):
                    break
    json.dump(got, open(DATA / out_name, "w"))
    print(f"{out_name}: {len(got)} routes", flush=True)


def main():
    n_per = 5
    for a in sys.argv[1:]:
        if a.startswith("n_per_metro="):
            n_per = int(a[12:])
    DATA.mkdir(parents=True, exist_ok=True)
    raw = DATA / "eval_route_data.json"
    if not raw.exists():
        urllib.request.urlretrieve(BASE + "eval_route_data.json", raw)
    d = json.load(open(raw))
    by_metro = defaultdict(list)
    for rid, v in d.items():
        by_metro[v["station_code"][:3]].append(rid)
    pilot = set()
    for metro in ("DLA", "DBO", "DSE", "DCH", "DAU"):
        pilot.update(sorted(by_metro[metro])[:n_per])
    print(f"pilot: {len(pilot)} routes")
    stream_pick("eval_travel_times.json", pilot, "pilot_travel_times.json")
    stream_pick("eval_package_data.json", pilot, "pilot_package_data.json",
                nan_filter=True)
    json.dump({r: d[r] for r in pilot},
              open(DATA / "pilot_route_data.json", "w"))
    print("done")


if __name__ == "__main__":
    main()
