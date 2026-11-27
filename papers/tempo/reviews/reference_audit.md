# TEMPO reference audit (2026-11-27)

Method: every DOI in `references.bib` resolved against Crossref (and OpenAlex where Crossref was thin); Part B items found by Crossref bibliographic search, so every DOI below was returned by Crossref with matching title/authors. Chen et al. (2025) full text read from the CC-BY TU/e repository PDF. No repository file was edited.

## Headline findings

1. **gmira2021rollingopt is a fabricated attribution.** No "Gmira, Glize, Hoenig, Rousseau 2021" paper was found. Chen et al. (2025), Section "Benchmarks", write: "Rolling-greedy and Rolling-opt proposed by Gmira et al. 2021a", and their reference list gives 2021a = **Gmira, M., Gendreau, M., Lodi, A., Potvin, J.-Y., "Managing in real-time a vehicle routing plan with time-dependent travel times on a road network", Transp. Res. Part C 132, 103379** (DOI 10.1016/j.trc.2021.103379). The referee is right. (2021b = Gmira et al., "Tabu search for the time-dependent VRP with time windows on a road network", EJOR 288(1):129-140, which Chen et al. use as the separate "GmiraTS" baseline.)
2. **chen2025dgta has wrong author first names**: it is Dawei Chen, Christina Imdahl, David Lai, Tom Van Woensel (not Xin, Christian, Yun). Confirmed by Crossref, OpenAlex and the TU/e record.
3. **he2026smpc has wrong authors** (all six given names/surnames differ from Crossref).
4. VERIFY_CITATIONS.md's claim that the other entries were "independently cross-checked" was not reliable: three further entries needed fixes (He, Chen, Merchan).

## Part A: all 24 entries

Summary: **OK 18, FIX 5, NOT FOUND 1.**

| Key | Status | Notes / corrected fields |
|---|---|---|
| huber1964robust | OK | Ann. Math. Stat. 35(1):73-101, matches |
| wald1945sequential | OK | 16(2):117-186 matches |
| lorden1970excess | OK | 41(2):520-527 matches |
| woodroofe1982nonlinear | OK | SIAM CBMS-NSF 39, 1982, DOI resolves |
| ouertani2024dvrp | OK | Chapter pp. 1-36 (add `pages`); online Nov 2023, print 2024; Crossref container series "Unsupervised and Semi-Supervised Learning", book "Advances in Computational Logistics and Supply Chain Analytics", Springer Nature Switzerland |
| kadyrov2025drl | OK | Oper. Res. Perspectives 15:100351 |
| ramdaswang2025hypothesis | OK | FnT Stat. 1(1-2):1-390 (this also clears the earlier flag on page count) |
| koningvanmeer2026 | FIX | add `volume={88}, number={4}, pages={1366--1384}` (JRSSB 88(4), 2026) |
| dethloff2001vehicle | OK | OR Spektrum 23(1):79-96 |
| salhi1999cluster | OK | JORS 50(10):1034-1042 |
| pillac2013review | OK | EJOR 225(1):1-11 |
| he2026smpc | FIX | Authors must be: He, Guosong and Li, Qiuchi and Li, Xingchen and Huang, Yu and Huang, Yi and Duan, Qianqian. Mathematics 14(6):1032, 2026 (18 Mar 2026) otherwise correct |
| chen2025dgta | FIX | Authors: Chen, Dawei and Imdahl, Christina and Lai, David and Van Woensel, Tom. Rest correct (TR-C 172:105022, March 2025). Imdahl is "Christina" per Crossref/TU/e |
| gmira2021rollingopt | NOT FOUND | Replace by gmira2021managing (Part B item 4); drop the `\citet{chen2025dgta}` note |
| page1954 | OK | Biometrika 41(1/2):100-115; DOI 10.1093/biomet/41.1-2.100 also exists (the JSTOR DOI 10.2307/2333009 resolves too) |
| gama2014survey | OK | CSUR 46(4), Article 44 (pp. 1-37) |
| adams2007bayesian | OK | arXiv:0710.3742, Adams and MacKay, submitted 2007-10-19 |
| shafer2021testing | OK | JRSS-A 184(2):407-431 |
| ramdas2023safi | OK | Stat. Sci. 38(4):576-601; DOI confirmed (earlier flag cleared) |
| ville1939etude | FIX | Not a `phdthesis` with school Gauthier-Villars. Use `@book`: Gauthier-Villars, Paris, 1939, series Monographies des probabilites, fascicule III (Cambridge Math. Gazette review; Numdam lists it as Paris thesis no. 218). Page count sources disagree (144 vs 124): omit pages |
| waudbysmithramdas2024 | OK | JRSSB 86(1):1-27; print Feb 2024, online Feb 2023 (earlier flag cleared) |
| vovkwang2021combining | OK | Ann. Stat. 49(3):1736-1754 |
| shin2024edetectors | OK | NEJSDS pp. 229-260 confirmed; Crossref/OpenAlex give online date 2023-12 and no volume field, so vol 2(2) / year 2024 not independently confirmed (recalled); check on nejsds.org before submission |
| merchan2024amazon | FIX | Title is "2021 Amazon Last Mile Routing Research Challenge: Data Set" (no hyphen); pages **8--11** (Crossref and OpenAlex agree), not 8--21. Transp. Sci. 58(1), Jan 2024 |

## Part B: verified metadata (all DOIs returned by Crossref with matching authors/title)

Notes on the target list:
- Item 5: Bertsimas 1992 is Oper. Res. 40(3):574-585.
- Item 8: Part I is 2018 (7(3)), Part II is 2017 (6(4)).
- Item 15: Salavati-Khoshghalb et al. have three 2019 papers; all three are listed.
- Item 16: Louveaux and Salazar-Gonzalez's exact paper is "Exact Approach for the VRP with Stochastic Demands and **Preventive Returns**", Transp. Sci. 2018. Jabali, Rei, Gendreau, Laporte (2014) is "Partial-route inequalities for the multi-vehicle routing problem with stochastic demands", DAM 177:121-136; there is no 2014 paper by them titled "partial-route inventory/preventive restocking", so cite the DAM title only.
- Item 17: I did not find a verified Ulmer paper on "re-optimization timing" per se; the closest is Ulmer (2019, Networks), reactive reoptimization vs anticipation. The rest are real but are about waiting/diversion/event-driven frameworks, so state the claim accordingly.

```bibtex
@article{yang2000restocking,
  author  = {Yang, Wen-Huei and Mathur, Kamlesh and Ballou, Ronald H.},
  title   = {Stochastic Vehicle Routing Problem with Restocking},
  journal = {Transportation Science},
  volume  = {34}, number = {1}, pages = {99--112}, year = {2000},
  doi     = {10.1287/trsc.34.1.99.12278}
}
@article{secomandi2009reopt,
  author  = {Secomandi, Nicola and Margot, Fran{\c c}ois},
  title   = {Reoptimization Approaches for the Vehicle-Routing Problem with Stochastic Demands},
  journal = {Operations Research},
  volume  = {57}, number = {1}, pages = {214--230}, year = {2009},
  doi     = {10.1287/opre.1080.0520}
}
@article{florio2022partial,
  author  = {Florio, Alexandre M. and Feillet, Dominique and Poggi, Marcus and Vidal, Thibaut},
  title   = {Vehicle Routing with Stochastic Demands and Partial Reoptimization},
  journal = {Transportation Science},
  volume  = {56}, number = {5}, pages = {1393--1408}, year = {2022},
  doi     = {10.1287/trsc.2022.1129}
}
@article{gmira2021managing,
  author  = {Gmira, Maha and Gendreau, Michel and Lodi, Andrea and Potvin, Jean-Yves},
  title   = {Managing in real-time a vehicle routing plan with time-dependent travel times on a road network},
  journal = {Transportation Research Part C: Emerging Technologies},
  volume  = {132}, pages = {103379}, year = {2021},
  doi     = {10.1016/j.trc.2021.103379}
}
@article{gmira2021tabu,
  author  = {Gmira, Maha and Gendreau, Michel and Lodi, Andrea and Potvin, Jean-Yves},
  title   = {Tabu search for the time-dependent vehicle routing problem with time windows on a road network},
  journal = {European Journal of Operational Research},
  volume  = {288}, number = {1}, pages = {129--140}, year = {2021},
  doi     = {10.1016/j.ejor.2020.05.041}
}
@article{bertsimas1992vrpsd,
  author  = {Bertsimas, Dimitris J.},
  title   = {A Vehicle Routing Problem with Stochastic Demand},
  journal = {Operations Research},
  volume  = {40}, number = {3}, pages = {574--585}, year = {1992},
  doi     = {10.1287/opre.40.3.574}
}
@article{gendreau1996stochastic,
  author  = {Gendreau, Michel and Laporte, Gilbert and S{\'e}guin, Ren{\'e}},
  title   = {Stochastic vehicle routing},
  journal = {European Journal of Operational Research},
  volume  = {88}, number = {1}, pages = {3--12}, year = {1996},
  doi     = {10.1016/0377-2217(95)00050-X}
}
@article{gendreau2016future,
  author  = {Gendreau, Michel and Jabali, Ola and Rei, Walter},
  title   = {50th Anniversary Invited Article---Future Research Directions in Stochastic Vehicle Routing},
  journal = {Transportation Science},
  volume  = {50}, number = {4}, pages = {1163--1173}, year = {2016},
  doi     = {10.1287/trsc.2016.0709}
}
@article{oyola2018svrp1,
  author  = {Oyola, Jorge and Arntzen, Halvard and Woodruff, David L.},
  title   = {The stochastic vehicle routing problem, a literature review, part I: models},
  journal = {EURO Journal on Transportation and Logistics},
  volume  = {7}, number = {3}, pages = {193--221}, year = {2018},
  doi     = {10.1007/s13676-016-0100-5}
}
@article{oyola2017svrp2,
  author  = {Oyola, Jorge and Arntzen, Halvard and Woodruff, David L.},
  title   = {The stochastic vehicle routing problem, a literature review, part II: solution methods},
  journal = {EURO Journal on Transportation and Logistics},
  volume  = {6}, number = {4}, pages = {349--388}, year = {2017},
  doi     = {10.1007/s13676-016-0099-7}
}
@article{ulmer2020modeling,
  author  = {Ulmer, Marlin W. and Goodson, Justin C. and Mattfeld, Dirk C. and Thomas, Barrett W.},
  title   = {On modeling stochastic dynamic vehicle routing problems},
  journal = {EURO Journal on Transportation and Logistics},
  volume  = {9}, number = {2}, pages = {100008}, year = {2020},
  doi     = {10.1016/j.ejtl.2020.100008}
}
@article{psaraftis2016dvrp,
  author  = {Psaraftis, Harilaos N. and Wen, Min and Kontovas, Christos A.},
  title   = {Dynamic vehicle routing problems: Three decades and counting},
  journal = {Networks},
  volume  = {67}, number = {1}, pages = {3--31}, year = {2016},
  doi     = {10.1002/net.21628}
}
@article{ritzinger2016survey,
  author  = {Ritzinger, Ulrike and Puchinger, Jakob and Hartl, Richard F.},
  title   = {A survey on dynamic and stochastic vehicle routing problems},
  journal = {International Journal of Production Research},
  volume  = {54}, number = {1}, pages = {215--231}, year = {2016},
  doi     = {10.1080/00207543.2015.1043403}
}
@article{novoa2009adp,
  author  = {Novoa, Clara and Storer, Robert},
  title   = {An approximate dynamic programming approach for the vehicle routing problem with stochastic demands},
  journal = {European Journal of Operational Research},
  volume  = {196}, number = {2}, pages = {509--515}, year = {2009},
  doi     = {10.1016/j.ejor.2008.03.023}
}
@article{goodson2016restocking,
  author  = {Goodson, Justin C. and Thomas, Barrett W. and Ohlmann, Jeffrey W.},
  title   = {Restocking-Based Rollout Policies for the Vehicle Routing Problem with Stochastic Demand and Duration Limits},
  journal = {Transportation Science},
  volume  = {50}, number = {2}, pages = {591--607}, year = {2016},
  doi     = {10.1287/trsc.2015.0591}
}
@article{salavati2019rulebased,
  author  = {Salavati-Khoshghalb, Majid and Gendreau, Michel and Jabali, Ola and Rei, Walter},
  title   = {A Rule-Based Recourse for the Vehicle Routing Problem with Stochastic Demands},
  journal = {Transportation Science},
  volume  = {53}, number = {5}, pages = {1334--1353}, year = {2019},
  doi     = {10.1287/trsc.2018.0876}
}
@article{salavati2019hybrid,
  author  = {Salavati-Khoshghalb, Majid and Gendreau, Michel and Jabali, Ola and Rei, Walter},
  title   = {A hybrid recourse policy for the vehicle routing problem with stochastic demands},
  journal = {EURO Journal on Transportation and Logistics},
  volume  = {8}, number = {3}, pages = {269--298}, year = {2019},
  doi     = {10.1007/s13676-018-0126-y}
}
@article{salavati2019exact,
  author  = {Salavati-Khoshghalb, Majid and Gendreau, Michel and Jabali, Ola and Rei, Walter},
  title   = {An exact algorithm to solve the vehicle routing problem with stochastic demands under an optimal restocking policy},
  journal = {European Journal of Operational Research},
  volume  = {273}, number = {1}, pages = {175--189}, year = {2019},
  doi     = {10.1016/j.ejor.2018.07.039}
}
@article{louveaux2018exact,
  author  = {Louveaux, Fran{\c c}ois V. and Salazar-Gonz{\'a}lez, Juan-Jos{\'e}},
  title   = {Exact Approach for the Vehicle Routing Problem with Stochastic Demands and Preventive Returns},
  journal = {Transportation Science},
  volume  = {52}, number = {6}, pages = {1463--1478}, year = {2018},
  doi     = {10.1287/trsc.2017.0780}
}
@article{jabali2014partial,
  author  = {Jabali, Ola and Rei, Walter and Gendreau, Michel and Laporte, Gilbert},
  title   = {Partial-route inequalities for the multi-vehicle routing problem with stochastic demands},
  journal = {Discrete Applied Mathematics},
  volume  = {177}, pages = {121--136}, year = {2014},
  doi     = {10.1016/j.dam.2014.05.040}
}
@article{ichoua2003dispatching,
  author  = {Ichoua, Soumia and Gendreau, Michel and Potvin, Jean-Yves},
  title   = {Vehicle dispatching with time-dependent travel times},
  journal = {European Journal of Operational Research},
  volume  = {144}, number = {2}, pages = {379--396}, year = {2003},
  doi     = {10.1016/S0377-2217(02)00147-9}
}
% Item 17: when to re-optimize / reoptimization policies
@article{ulmer2019anticipation,
  author  = {Ulmer, Marlin W.},
  title   = {Anticipation versus reactive reoptimization for dynamic vehicle routing with stochastic requests},
  journal = {Networks},
  volume  = {73}, number = {3}, pages = {277--291}, year = {2019},
  doi     = {10.1002/net.21861}
}
@article{pillac2012event,
  author  = {Pillac, Victor and Gu{\'e}ret, Christelle and Medaglia, Andr{\'e}s L.},
  title   = {An event-driven optimization framework for dynamic vehicle routing},
  journal = {Decision Support Systems},
  volume  = {54}, number = {1}, pages = {414--423}, year = {2012},
  doi     = {10.1016/j.dss.2012.06.007}
}
@article{abdallah2017periodic,
  author  = {AbdAllah, Abdel Monaem F. M. and Essam, Daryl L. and Sarker, Ruhul A.},
  title   = {On solving periodic re-optimization dynamic vehicle routing problems},
  journal = {Applied Soft Computing},
  volume  = {55}, pages = {1--12}, year = {2017},
  doi     = {10.1016/j.asoc.2017.01.047}
}
@article{ninikas2014reopt,
  author  = {Ninikas, George and Minis, Ioannis},
  title   = {Reoptimization strategies for a dynamic vehicle routing problem with mixed backhauls},
  journal = {Networks},
  volume  = {64}, number = {3}, pages = {214--231}, year = {2014},
  doi     = {10.1002/net.21567}
}
@article{vieira2003rescheduling,
  author  = {Vieira, Guilherme E. and Herrmann, Jeffrey W. and Lin, Edward},
  title   = {Rescheduling Manufacturing Systems: A Framework of Strategies, Policies, and Methods},
  journal = {Journal of Scheduling},
  volume  = {6}, number = {1}, pages = {39--62}, year = {2003},
  doi     = {10.1023/A:1022235519958}
}
@article{ichoua2000diversion,
  author  = {Ichoua, Soumia and Gendreau, Michel and Potvin, Jean-Yves},
  title   = {Diversion Issues in Real-Time Vehicle Dispatching},
  journal = {Transportation Science},
  volume  = {34}, number = {4}, pages = {426--438}, year = {2000},
  doi     = {10.1287/trsc.34.4.426.12325}
}
@article{mitrovic2004waiting,
  author  = {Mitrovi{\'c}-Mini{\'c}, Sne{\v z}ana and Laporte, Gilbert},
  title   = {Waiting strategies for the dynamic pickup and delivery problem with time windows},
  journal = {Transportation Research Part B: Methodological},
  volume  = {38}, number = {7}, pages = {635--655}, year = {2004},
  doi     = {10.1016/j.trb.2003.09.002}
}
@article{yang2004realtime,
  author  = {Yang, Jian and Jaillet, Patrick and Mahmassani, Hani},
  title   = {Real-Time Multivehicle Truckload Pickup and Delivery Problems},
  journal = {Transportation Science},
  volume  = {38}, number = {2}, pages = {135--148}, year = {2004},
  doi     = {10.1287/trsc.1030.0068}
}
% Item 18: sequential change detection in transportation
@article{qi2005cusum,
  author  = {Qi, Yi Grace and Martinelli, David R. and Teng, Hualiang Harry and Jiang, Pengguo},
  title   = {An application of the {CUSUM} algorithm to freeway incident detection based on two contiguous detectors},
  journal = {Journal of Advanced Transportation},
  volume  = {39}, number = {2}, pages = {221--241},
  year    = {2005},  % YEAR UNVERIFIED: Crossref lists only an online date of 2010-04-16; vol. 39 is the 2005 volume. Confirm on Wiley.
  doi     = {10.1002/atr.5670390206}
}
@article{teng2003delay,
  author  = {Teng, Hualiang and Qi, Yi},
  title   = {Detection-delay-based freeway incident detection algorithms},
  journal = {Transportation Research Part C: Emerging Technologies},
  volume  = {11}, number = {3--4}, pages = {265--287}, year = {2003},
  doi     = {10.1016/S0968-090X(03)00022-6}
}
@article{ahsani2020changepoint,
  author  = {Ahsani, Vesal and Sharma, Anuj and Hegde, Chinmay and Knickerbocker, Skylar and Hawkins, Neal},
  title   = {Improving Probe-Based Congestion Performance Metrics Accuracy by Using Change Point Detection},
  journal = {Journal of Big Data Analytics in Transportation},
  volume  = {2}, number = {1}, pages = {61--74}, year = {2020},
  doi     = {10.1007/s42421-020-00017-w}
}
@article{anbaroglu2014nonrecurrent,
  author  = {Anbaroglu, Berk and Heydecker, Benjamin and Cheng, Tao},
  title   = {Spatio-temporal clustering for non-recurrent traffic congestion detection on urban road networks},
  journal = {Transportation Research Part C: Emerging Technologies},
  volume  = {48}, pages = {47--65}, year = {2014},
  doi     = {10.1016/j.trc.2014.08.002}
}
@article{shiryaev2010quickest,
  author  = {Shiryaev, Albert N.},
  title   = {Quickest Detection Problems: Fifty Years Later},
  journal = {Sequential Analysis},
  volume  = {29}, number = {4}, pages = {345--385}, year = {2010},
  doi     = {10.1080/07474946.2010.520580}
}
```

Replacement entries for Part A fixes:

```bibtex
@article{chen2025dgta,
  author  = {Chen, Dawei and Imdahl, Christina and Lai, David and Van Woensel, Tom},
  title   = {The Dynamic Traveling Salesman Problem with Time-Dependent and Stochastic travel times: A deep reinforcement learning approach},
  journal = {Transportation Research Part C: Emerging Technologies},
  volume  = {172}, pages = {105022}, year = {2025},
  doi     = {10.1016/j.trc.2025.105022}
}
@article{he2026smpc,
  author  = {He, Guosong and Li, Qiuchi and Li, Xingchen and Huang, Yu and Huang, Yi and Duan, Qianqian},
  title   = {A Stochastic Model Predictive Control Strategy for Vehicle Routing with Correlated Stochastic Service Times},
  journal = {Mathematics}, volume = {14}, number = {6}, pages = {1032}, year = {2026},
  doi     = {10.3390/math14061032}
}
@article{koningvanmeer2026,
  author  = {Koning, Nick W. and van Meer, Sam},
  title   = {Anytime validity is free: inducing sequential tests},
  journal = {Journal of the Royal Statistical Society Series B: Statistical Methodology},
  volume  = {88}, number = {4}, pages = {1366--1384}, year = {2026},
  doi     = {10.1093/jrsssb/qkag050}
}
@article{merchan2024amazon,
  author  = {Merch{\'a}n, Daniel and Arora, Jatin and Pachon, Julian and Konduri, Karthik and Winkenbach, Matthias and Parks, Steven and Noszek, Joseph},
  title   = {2021 Amazon Last Mile Routing Research Challenge: Data Set},
  journal = {Transportation Science}, volume = {58}, number = {1}, pages = {8--11}, year = {2024},
  doi     = {10.1287/trsc.2022.1173}
}
@book{ville1939etude,
  author    = {Ville, Jean},
  title     = {{\'E}tude critique de la notion de collectif},
  publisher = {Gauthier-Villars},
  address   = {Paris},
  series    = {Monographies des probabilit{\'e}s},
  number    = {3},
  year      = {1939}
}
```

## Unverified / caveats

- **Ulmer "re-optimization timing"** (item 17): no paper by that description was found; Ulmer (2019) is the nearest verified one.
- **Jabali et al. 2014 "partial-route inventory / preventive restocking"**: no paper with that title exists in Crossref; the DAM partial-route inequalities paper is the real one.
- **Item 18** is thin: only the Qi et al. CUSUM freeway-incident paper is a direct CUSUM application, and its print year (2005) is inferred from the volume number. Teng and Qi (2003) is detection-delay framed rather than explicitly CUSUM; Ahsani et al. and Anbaroglu et al. are change-point/clustering based, not CUSUM/SPRT. I found no verified SPRT application to travel-time or logistics monitoring; do not claim one.
- Shin et al. volume/year (2(2), 2024) not independently confirmed; Ville page count omitted (sources conflict).
- Merchan pages 8-11 rests on Crossref and OpenAlex agreeing; the INFORMS page returned 403.
- The Chen et al. rolling-opt attribution was read from the full text (benchmarks section and reference list); the exact description of Gmira 2021a as the source of "Rolling-opt" is Chen et al.'s own statement. The text of Gmira 2021a itself was not read.
