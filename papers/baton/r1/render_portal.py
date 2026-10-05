#!/usr/bin/env python3
"""
render_portal.py — fill r1/portal_replies_template.md with the numbers of
../tables/macros.tex and the section/table/figure numbers of ../main.aux,
writing r1/portal_replies.md (plain text for the Elsevier portal).

Placeholders: {m:macroName} and {r:label}. Run after compiling main.tex.
"""
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE.parent


def load_macros():
    out = {}
    for ln in (BASE / "tables" / "macros.tex").read_text().splitlines():
        m = re.match(r"\\newcommand\{\\(\w+)\}\{(.*)\}\s*$", ln)
        if m:
            v = m.group(2)
            v = v.replace(r"\%", "%").replace("{,}", ",").replace(r"\,", " ")
            v = v.replace(r"$\mu$s", "µs").replace("$", "")
            v = v.replace(r"p \le 10^{", "p ≤ 1e").replace("}", "")
            out[m.group(1)] = v
    return out


def load_refs():
    out = {}
    for ln in (BASE / "main.aux").read_text().splitlines():
        m = re.match(r"\\newlabel\{([^}]+)\}\{\{([^}]*)\}", ln)
        if m:
            out[m.group(1)] = m.group(2)
    return out


def main():
    mac, ref = load_macros(), load_refs()
    t = (HERE / "portal_replies_template.md").read_text()
    missing = []

    def sub(m):
        kind, key = m.group(1), m.group(2)
        table = mac if kind == "m" else ref
        if key not in table:
            missing.append(m.group(0))
            return m.group(0)
        return table[key]
    t = re.sub(r"\{([mr]):([A-Za-z0-9:_-]+)\}", sub, t)
    (HERE / "portal_replies.md").write_text(t)
    print("wrote r1/portal_replies.md", "missing: " + ", ".join(missing) if missing else "")


if __name__ == "__main__":
    main()
