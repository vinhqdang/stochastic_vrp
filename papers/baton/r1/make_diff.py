#!/usr/bin/env python3
"""
make_diff.py — build r1/main_diff.pdf: revision 1 with the changes marked
against the submitted manuscript (git revision SUBMITTED below).

Both versions are flattened (\\input files inlined) and their generated
number macros expanded to literal values, so latexdiff compares text and
numbers rather than macro names. Tables and figures are shown in their
revised form without markup inside the floats.

Usage (from papers/baton/): python r1/make_diff.py
"""
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

SUBMITTED = "d88142b"          # repository state holding the submitted text
HERE = Path(__file__).resolve().parent.parent          # papers/baton
REPO = HERE.parent.parent


def macros(tex_dir: Path) -> dict:
    out = {}
    for ln in (tex_dir / "tables" / "macros.tex").read_text().splitlines():
        m = re.match(r"\\newcommand\{\\(\w+)\}\{(.*)\}\s*$", ln)
        if m:
            out[m.group(1)] = m.group(2)
    return out


def flatten(tex_dir: Path) -> str:
    src = (tex_dir / "main.tex").read_text()

    def inline(m):
        name = m.group(1)
        f = tex_dir / (name if name.endswith(".tex") else name + ".tex")
        return f.read_text() if f.exists() else m.group(0)
    src = src.replace(r"\input{tables/macros}", "")
    src = re.sub(r"\\input\{(tables/[^}]+)\}", inline, src)
    mac = macros(tex_dir)
    src = src.replace(r"\input{tables/macros}", "")
    for k in sorted(mac, key=len, reverse=True):
        src = re.sub(r"\\" + k + r"(\{\})?(?![A-Za-z])",
                     lambda _m, v=mac[k]: v,
                     src)
    return src


def main():
    tmp = Path(tempfile.mkdtemp())
    subprocess.run(["git", "-C", str(REPO), "archive", SUBMITTED, "papers/baton"],
                   check=True, stdout=open(tmp / "old.tar", "wb"))
    subprocess.run(["tar", "-xf", str(tmp / "old.tar"), "-C", str(tmp)], check=True)
    old_dir = tmp / "papers" / "baton"
    (tmp / "old.tex").write_text(flatten(old_dir))
    (tmp / "new.tex").write_text(flatten(HERE))
    work = tmp / "build"
    shutil.copytree(HERE, work, ignore=shutil.ignore_patterns("r1", "*.pdf", "BatonProofs"))
    diff = subprocess.run(
        ["latexdiff", "--type=UNDERLINE", "--math-markup=coarse",
         "--config", "PICTUREENV=(?:picture|DIFnomarkup|table|sidewaystable|figure|algorithm|tikzpicture)[\\w\\d*@]*",
         str(tmp / "old.tex"), str(tmp / "new.tex")],
        check=True, capture_output=True, text=True).stdout
    (work / "main_diff.tex").write_text(diff)
    for cmd in (["pdflatex", "-interaction=nonstopmode", "main_diff"],
                ["bibtex", "main_diff"],
                ["pdflatex", "-interaction=nonstopmode", "main_diff"],
                ["pdflatex", "-interaction=nonstopmode", "main_diff"]):
        subprocess.run(cmd, cwd=work, capture_output=True, timeout=600)
    pdf = work / "main_diff.pdf"
    if not pdf.exists():
        sys.exit("latexdiff build failed; see " + str(work / "main_diff.log"))
    shutil.copy(pdf, HERE / "r1" / "main_diff.pdf")
    shutil.rmtree(tmp, ignore_errors=True)
    print("wrote r1/main_diff.pdf")


if __name__ == "__main__":
    main()
