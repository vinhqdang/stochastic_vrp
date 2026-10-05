"""make_submission.py -- builds the flat upload package submission/:
ONE main.tex (bibliography inlined as thebibliography, no .bib/.bst, no
subfolders) plus the figure PNGs, nothing else.

  python3 make_submission.py

Needs pdflatex, bibtex and pdftoppm. Re-run after any change to main.tex,
references.bib or the figures. The package compiles with Springer's
sn-jnl.cls (supplied by the journal system, deliberately not included).
"""
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

here = Path(__file__).resolve().parent
out = here / "submission"
tex = (here / "main.tex").read_text()

with tempfile.TemporaryDirectory() as tmp:
    tmp = Path(tmp)
    for f in ("main.tex", "references.bib", "sn-jnl.cls", "sn-mathphys-ay.bst"):
        shutil.copy(here / f, tmp / f)
    shutil.copytree(here / "figures", tmp / "figures")
    for cmd in (["pdflatex", "-interaction=nonstopmode", "main.tex"],
                ["bibtex", "main"],
                ["pdflatex", "-interaction=nonstopmode", "main.tex"]):
        subprocess.run(cmd, cwd=tmp, capture_output=True, text=True)
    bbl = (tmp / "main.bbl").read_text().strip()

assert "\\bibliography{references}" in tex
tex = tex.replace("\\bibliography{references}", bbl)
tex = re.sub(r"\\bibliographystyle\{[^}]*\}\n?", "", tex)
names = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{figures/([^}]*)\.pdf\}", tex)
tex = re.sub(r"(\\includegraphics(?:\[[^\]]*\])?)\{figures/([^}]*)\.pdf\}",
             r"\1{\2.png}", tex)

if out.exists():
    shutil.rmtree(out)
out.mkdir()
(out / "main.tex").write_text(tex)
for n in dict.fromkeys(names):
    subprocess.run(["pdftoppm", "-r", "300", "-png", "-singlefile",
                    str(here / "figures" / f"{n}.pdf"), str(out / n)], check=True)
print("wrote", out, sorted(p.name for p in out.iterdir()))
