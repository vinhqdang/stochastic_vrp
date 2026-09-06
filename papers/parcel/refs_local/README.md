# refs_local — downloaded papers, NEVER committed

Drop PDFs of prior work here for reading. **Everything in this
directory except this README is gitignored**, deliberately.

`stochastic_vrp` is a public repository. Publisher PDFs (Elsevier,
Springer, ACM, IEEE) are copyrighted and must not be committed to it.
Keep them here, read them, cite them in `references.bib` — but never
push the files themselves.

arXiv preprints are generally redistributable under their posted
licence, but there is no reason to commit those either; the arXiv ID in
`references.bib` is enough.

## Wanted

- **Shi & Lai (2024)**, *Approximation algorithm of maximizing
  non-monotone non-submodular functions under knapsack constraint*,
  Theoretical Computer Science 990:114409.
  <https://doi.org/10.1016/j.tcs.2024.114409>
  Save as `shi_lai_2024_tcs.pdf`.
  **Why:** does their weak-supermodular case already absorb a penalty
  that rises with receiver load? Decides whether PARCEL's
  state-dependent penalty is a real increment — `PROJECT.md` §10.6.
