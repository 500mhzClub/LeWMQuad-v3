# Startup-correction screen: interim status

Five episodes evaluated; 05/0 is running. The fixed screen continues unchanged.

| Episode | Round trip | Outcome | Containment |
|---|---|---|---|
| 00/0 | Pass | 130.42 simulated seconds | Exact native trajectory, requests and sensor hashes |
| 01/0 | Fail | Startup recovered; visual pose lost at 341 s | Original startup failure; equality not required |
| 02/0 | Fail | Map-bound rejection removed; visual pose lost at 58.5 s | Original startup failure; equality not required |
| 03/0 | Pass | 463.12 simulated seconds | Exact native trajectory, requests and sensor hashes |
| 04/0 | Pass | 110.92 simulated seconds | Failed; command divergence begins at 84.3 s |

All five evaluated episodes have zero disallowed contacts, hard-clearance
violations, operating-margin violations and unresolved sampled clearance.

## Containment diagnosis and accounting

This combined change now counts as a harness version under the user's
containment rule. It cannot be reported as a contained correctness-only change.

04/0 has identical consumed sensor hashes and measured poses until frame 844.
At planning frame 840, both logs report 7,004 floor cells and 5,224 fine obstacle
cells, but the fine-goal route cost and waypoint differ: old waypoint
[3.475, 0.925], new [3.475, 1.025]. First requested-command divergence is at
84.3 mission seconds: forward versus right arc. Both missions succeed, but
that does not waive exact containment.

Source inspection identified a missed map-origin offset in the correction:
`clearance_costs` now places the origin at index 160 in its 320×320 grid, while
`cached_fine_connectivity_development.search_graph` and the older
`fine_goal_route_development` consumer still index that grid with `+100`.
Consequently the fine-goal route reads clearance costs from the wrong cells.
This is an implementation defect, not evidence of a navigation improvement.
No source or assignments have been changed during this screen. Preserve its
full results; correct this indexing consistently and test it before any later
version. No passing-gate claim may be based on the mismatched implementation.

## Preliminary cost of the larger grid

On the two exactly reproduced episodes, wall time increased 8.8% (00/0) and
7.3% (03/0). Median planning latency rose from 137 to 213 ms and from 132 to
210 ms respectively; p95 is about 227 ms. Peak process-tree RSS was 7.46 GiB
versus 7.42 GiB for 00/0, and 8.63 versus 8.77 GiB for 03/0. These are two
paired observations, not a completed-cohort cost estimate. Re-project the
programme budget after the full cohort.
