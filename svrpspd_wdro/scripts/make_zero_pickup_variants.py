#!/usr/bin/env python3
"""
make_zero_pickup_variants.py — deliver-only sensitivity twins of the city
instances. A seeded share z of the customers of every data/City instance
gets pickup 0 (deliveries, coordinates and road distances unchanged);
written to data/CityZP<100z>/.

Usage: python scripts/make_zero_pickup_variants.py [z=0.25,0.5]
"""
import hashlib
import sys
from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parent.parent / "data"


def variant(src: Path, z: float, out_dir: Path) -> None:
    lines = src.read_text().splitlines()
    i0 = lines.index("PICKUP_AND_DELIVERY_SECTION") + 1
    i1 = next(i for i in range(i0, len(lines)) if lines[i].strip() == "EOF" or
              lines[i].strip().endswith("SECTION"))
    cust = [i for i in range(i0, i1) if lines[i].split()[0] != "1"]
    seed = int(hashlib.md5(f"{src.stem}-{z}".encode()).hexdigest(), 16) % 2**32
    rng = np.random.default_rng(seed)
    pick = rng.choice(cust, size=int(round(z * len(cust))), replace=False)
    for i in pick:
        t = lines[i].split()
        lines[i] = f"{t[0]} {t[1]} 0"
    for j, ln in enumerate(lines):
        if ln.startswith("COMMENT"):
            lines[j] = ln + f"; {int(100 * z)}% deliver-only customers"
    out_dir.mkdir(exist_ok=True)
    (out_dir / src.name).write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    zs = [0.25, 0.5]
    for a in sys.argv[1:]:
        if a.startswith("z="):
            zs = [float(x) for x in a[2:].split(",")]
    for z in zs:
        out = DATA / f"CityZP{int(100 * z)}"
        for f in sorted((DATA / "City").glob("*.vrpspd")):
            variant(f, z, out)
        print(f"wrote {out}")
