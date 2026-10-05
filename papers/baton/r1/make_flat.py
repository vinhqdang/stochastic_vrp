#!/usr/bin/env python3
"""
make_flat.py — build the flat LaTeX bundle for the Elsevier submission
system: one .tex file plus the PNG figures it uses, no subfolders.

The macros file and every table are inlined, the bibliography is inlined
from the compiled main.bbl (so the journal system does not need BibTeX),
and figure paths lose their directory. The bundle is compiled in an
empty directory as a check before it is zipped.

Usage (from papers/baton/, after compiling main.tex so main.bbl is
current): python r1/make_flat.py
Output: r1/submission/BATON_R1_latex.zip and the response letter PDF.
"""
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent          # papers/baton
OUT = HERE / "r1" / "submission"
NAME = "BATON_R1"


def main():
    src = (HERE / "main.tex").read_text()
    bbl = HERE / "main.bbl"
    if not bbl.exists():
        sys.exit("compile main.tex first (main.bbl missing)")

    def inline(m):
        return (HERE / (m.group(1) + ".tex")).read_text().rstrip("\n") + "\n"
    src = re.sub(r"\\input\{(tables/[^}]+)\}", inline, src)

    figs = []

    def fig(m):
        name = Path(m.group(2)).name
        figs.append(name)
        return m.group(1) + "{" + name + "}"
    src = re.sub(r"(\\includegraphics(?:\[[^\]]*\])?)\{([^}]+)\}", fig, src)

    bib = re.compile(r"\\bibliographystyle\{[^}]*\}\s*\\bibliography\{[^}]*\}")
    assert bib.search(src), "bibliography commands not found"
    src = bib.sub(lambda _m: "% bibliography inlined from main.bbl "
                  "(style elsarticle-harv)\n" + bbl.read_text(), src)
    assert "\\input{" not in src, "an \\input remains"

    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    tex = OUT / f"{NAME}.tex"
    tex.write_text(src)
    for f in figs:
        shutil.copy(HERE / "figures" / f"{f}.png", OUT / f"{f}.png")

    # check: compile in a directory holding only the bundle
    chk = OUT / "_check"
    chk.mkdir()
    for p in OUT.glob("*.*"):
        shutil.copy(p, chk / p.name)
    for _ in range(3):
        subprocess.run(["pdflatex", "-interaction=nonstopmode", tex.name],
                       cwd=chk, capture_output=True, timeout=600)
    log = (chk / f"{NAME}.log").read_text(errors="ignore")
    bad = [ln for ln in log.splitlines()
           if ln.startswith("!") or "undefined" in ln or "Missing" in ln]
    if bad or not (chk / f"{NAME}.pdf").exists():
        sys.exit("bundle does not compile cleanly:\n" + "\n".join(bad[:20]))
    pages = re.search(r"Output written on .*?\((\d+) pages", log).group(1)
    shutil.rmtree(chk)

    zpath = OUT / f"{NAME}_latex.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(tex, tex.name)
        for f in figs:
            z.write(OUT / f"{f}.png", f"{f}.png")
    # keep only the zip and the response letter next to it
    tex.unlink()
    for f in figs:
        (OUT / f"{f}.png").unlink()
    shutil.copy(HERE / "r1" / "response_letter.pdf",
                OUT / f"{NAME}_response_letter.pdf")
    print(f"wrote {zpath.relative_to(HERE)}: 1 tex + {len(figs)} png, "
          f"compiles to {pages} pages")


if __name__ == "__main__":
    main()
