# Amazon Last-Mile Routing Research Challenge — pilot slice

Source: 2021 Amazon Last Mile Routing Research Challenge data set,
AWS Open Data bucket `amazon-last-mile-challenges` (CC-BY-4.0).
Citation: Merchan, Arora, Pachon, Konduri, Winkenbach, Parks, Noszek,
"2021 Amazon Last Mile Routing Research Challenge: Data Set,"
Transportation Science 58(1), 2024. DOI 10.1287/trsc.2022.1173.

Files here:
- pilot_route_data.json     25 evaluation routes (5 per metro: DLA los
                            angeles, DBO boston, DSE seattle, DCH
                            chicago, DAU austin): station, capacity,
                            stops with lat/lng/zone.
- pilot_travel_times.json   full inter-stop travel-time matrices
                            (seconds) for those routes — the planner's
                            forecast, used as TEMPO's null P0.
- pilot_package_data.json   per-package dimensions and planned service
                            times for those routes.
- eval_route_data.json      full evaluation route file (38 MB, kept for
                            enlarging the pilot; the 166 MB package and
                            843 MB travel-time files are NOT stored —
                            they are streamed on demand).

Regenerate / enlarge: scripts/fetch_amazon_pilot.py [n_per_metro=5]
Adapter: ev/amazon.py. Experiment: scripts/ev_amazon_pilot.py.
