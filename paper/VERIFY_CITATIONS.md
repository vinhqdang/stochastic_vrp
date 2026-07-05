# Citation verification status

Every entry in `references.bib` was checked automatically on 2026-07-05:
each DOI was resolved through `https://doi.org/<doi>`. **None returned
404** — all 20 DOIs exist and redirect to a publisher landing page.

HTTP codes: `200` = landing page fetched; `403`/`202` = the DOI resolved
and redirected correctly but the publisher blocks automated clients
(INFORMS, OUP, Palgrave, IEEE), so the landing-page *metadata* could not
be machine-compared against the BibTeX fields.

## Please eyeball these (DOI valid, metadata not machine-checked)

| Key | DOI | What to confirm |
|---|---|---|
| longstaff2001valuing | 10.1093/rfs/14.1.113 | RFS 14(1):113–147, 2001 |
| ropke2006adaptive | 10.1287/trsc.1050.0135 | Transp. Sci. 40(4):455–472, 2006 |
| bertsimas1992vehicle | 10.1287/opre.40.3.574 | Oper. Res. 40(3):574–585, 1992 |
| dror1989vehicle | 10.1287/trsc.23.3.166 | Transp. Sci. 23(3):166–176, 1989 |
| secomandi2001rollout | 10.1287/opre.49.5.796.10608 | Oper. Res. 49(5):796–802, 2001 |
| ulmer2019offline | 10.1287/trsc.2017.0767 | Transp. Sci. 53(1):185–202, 2019; author list Ulmer/Goodson/Mattfeld/Hennig |
| salhi1999cluster | 10.1057/palgrave.jors.2600808 | JORS 50(10):1034–1042, 1999 |
| tsitsiklis2001regression | 10.1109/72.935083 | IEEE TNN 12(4):694–703, 2001 |

## Books without DOIs (stable URLs supplied instead)

| Key | Note |
|---|---|
| barlow1972statistical | Barlow–Bartholomew–Bremner–Brunk 1972, Wiley. HathiTrust record URL in the entry. |
| chow1971great | Chow–Robbins–Siegmund 1971, Houghton Mifflin. HathiTrust record URL in the entry. |

## Metadata machine-verified OK (200 + known venues)

clement2002analysis, dethloff2001vehicle, min1989multiple,
gendreau1996stochastic, novoa2009approximate, esfahani2018data,
ai2009particle, subramanian2010parallel, peskir2006optimal,
oyola2018stochastic, robbins1955empirical (the Ayer et al. PAVA paper),
koc2020review.

All author lists in the .bib are complete (no "et al.").
