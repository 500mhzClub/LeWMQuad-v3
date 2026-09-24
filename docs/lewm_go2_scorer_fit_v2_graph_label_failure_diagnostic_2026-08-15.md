# Scorer-fit V2 graph-label failure diagnostic

Status: diagnostic only; no relabelling, execution, encoding, training, or qualification.

Audited source commit: `5c67135ad83b9206e6520e507f1ecaf980fd3d8d`

Frozen corpus digest: `5216e2182a4e165a673714fcccbd6b769d01fa565a69a466b3cab066ab01ccc3`

Machine-readable inventory and evidence: `docs/lewm_go2_scorer_fit_v2_graph_label_failure_diagnostic_2026-08-15.json`

Audit digest (compact canonical JSON excluding `audit_digest`): `90dda36b7e85a650a75d1efb5d21faf3f3ed40f0860f3bdb3f6a4e69b8bd3741`

## Result

The eighteen records expose two mechanisms, not a graph-localisation or BFS/Dijkstra implementation bug:

- 4 are `OFF_NAVIGABLE_GRAPH_OUTCOME`. Their final continuous poses remain inside the open-field scene bounds and in positive-clearance free space, but their nearest graph nodes are 2.005–2.070 m away, beyond the frozen 2.0 m location radius.
- 14 are `LOCATABLE_GOAL_UNREACHABLE_OUTCOME`. Their final nearest cells are connected to the goal in the raw graph, but every route crosses a cell blocked by the frozen navigation mask. Independent BFS exactly agrees with `SceneGraph.bfs_distance` and `GeodesicField`.

There is a narrower diagnostic-observability defect: the producer held all 20 tick rows in memory, but persisted only snapshot/H1–H4 camera poses and collapsed both refusal causes into `unlocatable_or_unreachable_geodesic`. The exact first failing tick, last finite tick, path contacts, per-tick clearance/stuck/fall/termination evidence, and at-or-before-horizon completion trace therefore cannot be recovered.

No record receives `INSUFFICIENT_TRACE_FOR_LABEL` as its primary failure mechanism because the final refusal operation is recoverable in all eighteen cases. The missing tick trace nevertheless makes every record insufficient for a complete progress/safety/completion/utility relabel.

No utility or replacement label is assigned here.

## Complete eighteen-record inventory

`Observed` is the first preserved block-end sample with unavailable locate/geodesic evidence. `Possible range` is deliberately conservative: a valid block endpoint cannot exclude an earlier transient failure followed by recovery, so the range always starts at tick 0. The exact first tick is not persisted for any record. Positions are reconstructed from the persisted camera pose and the frozen zero-RPY mount; RPY is world-frame roll/pitch/yaw. `raw→masked` gives raw-graph hops followed by frozen masked reachability.

| # | Role | State | Candidate | Goal cell | Start cell / finite d (m) | Observed / possible range | Last observed valid | Final xyz (m) | Final RPY (rad) | Final cell / locate d (m) | raw→masked | Operation | Category |
|---:|---|---|---|---:|---|---|---|---|---|---|---|---|---|
| 0 | fit | local-composite completion-03 | 10 reverse_then_turn | 8 | 7 / 1.141798 | 9 / 0–9 | 4 | 1.945393, 0.824196, 0.350656 | -0.041883, 0.005503, 0.421245 | 4 / 0.346239 | 2→unreachable | geodesic | `LOCATABLE_GOAL_UNREACHABLE_OUTCOME` |
| 1 | fit | medium-maze general-04 | 5 turn_left_sustained | 24 | 5 / 2.766771 | 9 / 0–9 | 4 | 0.379211, -3.389338, 0.348074 | 0.007126, 0.013189, 0.170162 | 4 / 0.423851 | 4→unreachable | geodesic | `LOCATABLE_GOAL_UNREACHABLE_OUTCOME` |
| 2 | fit | medium-maze general-04 | 6 turn_right_sustained | 24 | 5 / 2.766771 | 19 / 0–19 | 14 | 0.394381, -3.285421, 0.353983 | 0.027043, 0.015465, -1.126951 | 4 / 0.403526 | 4→unreachable | geodesic | `LOCATABLE_GOAL_UNREACHABLE_OUTCOME` |
| 3 | fit | medium-maze general-04 | 10 reverse_then_turn | 24 | 5 / 2.766771 | 4 / 0–4 | snapshot | 0.350955, -3.293335, 0.344253 | -0.040028, 0.003858, 0.406543 | 4 / 0.363154 | 4→unreachable | geodesic | `LOCATABLE_GOAL_UNREACHABLE_OUTCOME` |
| 4 | calibration | open-field general-00 | 0 straight_fast | 6 | 2 / 10.618443 | 19 / 0–19 | 14 | 3.728742, -3.986709, 0.353521 | -0.008035, -0.023243, -0.792047 | 2 / 2.069590 | 4→4 | locate | `OFF_NAVIGABLE_GRAPH_OUTCOME` |
| 5 | calibration | open-field general-00 | 1 straight_medium | 6 | 2 / 10.618443 | 19 / 0–19 | 14 | 3.623115, -4.013990, 0.346791 | -0.022386, -0.005995, -0.672699 | 2 / 2.025085 | 4→4 | locate | `OFF_NAVIGABLE_GRAPH_OUTCOME` |
| 6 | fit | open-field general-02 | 0 straight_fast | 6 | 2 / 10.580656 | 19 / 0–19 | 14 | 3.586856, -4.095414, 0.350166 | -0.069295, -0.020016, -1.009049 | 2 / 2.069555 | 4→4 | locate | `OFF_NAVIGABLE_GRAPH_OUTCOME` |
| 7 | fit | open-field general-02 | 1 straight_medium | 6 | 2 / 10.580656 | 19 / 0–19 | 14 | 3.482873, -4.087942, 0.356946 | -0.063919, -0.038201, -0.985920 | 2 / 2.005433 | 4→4 | locate | `OFF_NAVIGABLE_GRAPH_OUTCOME` |
| 8 | calibration | visual-stress safety-00 | 0 straight_fast | 38 | 30 / 1.749912 | 4 / 0–4 | snapshot | -0.065097, 0.789597, 0.343895 | -0.078783, -0.013740, 0.270024 | 31 / 0.065923 | 3→unreachable | geodesic | `LOCATABLE_GOAL_UNREACHABLE_OUTCOME` |
| 9 | calibration | visual-stress safety-00 | 1 straight_medium | 38 | 30 / 1.749912 | 4 / 0–4 | snapshot | -0.071756, 0.777211, 0.346812 | -0.036550, 0.005003, -0.048620 | 31 / 0.075288 | 3→unreachable | geodesic | `LOCATABLE_GOAL_UNREACHABLE_OUTCOME` |
| 10 | calibration | visual-stress safety-00 | 2 straight_slow | 38 | 30 / 1.749912 | 4 / 0–4 | snapshot | -0.078233, 0.754806, 0.339795 | -0.052529, 0.015525, 0.159848 | 31 / 0.090349 | 3→unreachable | geodesic | `LOCATABLE_GOAL_UNREACHABLE_OUTCOME` |
| 11 | calibration | visual-stress safety-00 | 3 arc_left_sustained | 38 | 30 / 1.749912 | 9 / 0–9 | 4 | -0.077045, 0.765284, 0.342454 | -0.050420, 0.012025, 1.034435 | 31 / 0.084506 | 3→unreachable | geodesic | `LOCATABLE_GOAL_UNREACHABLE_OUTCOME` |
| 12 | calibration | visual-stress safety-00 | 4 arc_right_sustained | 38 | 30 / 1.749912 | 4 / 0–4 | snapshot | -0.051514, 0.654050, 0.342948 | -0.017095, -0.008717, -0.463735 | 31 / 0.154775 | 3→unreachable | geodesic | `LOCATABLE_GOAL_UNREACHABLE_OUTCOME` |
| 13 | calibration | visual-stress safety-00 | 5 turn_left_sustained | 38 | 30 / 1.749912 | 4 / 0–4 | snapshot | -0.291250, 0.751643, 0.346535 | -0.027694, -0.005740, 1.119806 | 31 / 0.295237 | 3→unreachable | geodesic | `LOCATABLE_GOAL_UNREACHABLE_OUTCOME` |
| 14 | calibration | visual-stress safety-00 | 7 turn_left_then_go | 38 | 30 / 1.749912 | 4 / 0–4 | snapshot | -0.141708, 0.787150, 0.348771 | -0.055390, -0.015360, 0.899972 | 31 / 0.142289 | 3→unreachable | geodesic | `LOCATABLE_GOAL_UNREACHABLE_OUTCOME` |
| 15 | calibration | visual-stress safety-00 | 8 turn_right_then_go | 38 | 30 / 1.749912 | 4 / 0–4 | snapshot | -0.116701, 0.737367, 0.347046 | -0.071670, -0.027789, -0.110152 | 31 / 0.132446 | 3→unreachable | geodesic | `LOCATABLE_GOAL_UNREACHABLE_OUTCOME` |
| 16 | calibration | visual-stress safety-00 | 9 go_then_turn_left | 38 | 30 / 1.749912 | 4 / 0–4 | snapshot | -0.186170, 0.654856, 0.346754 | -0.060949, -0.004178, 0.216137 | 31 / 0.236064 | 3→unreachable | geodesic | `LOCATABLE_GOAL_UNREACHABLE_OUTCOME` |
| 17 | calibration | visual-stress safety-00 | 11 hold_all | 38 | 30 / 1.749912 | 4 / 0–4 | snapshot | -0.287451, 0.853563, 0.349817 | -0.030714, 0.007675, 0.292876 | 31 / 0.292399 | 3→unreachable | geodesic | `LOCATABLE_GOAL_UNREACHABLE_OUTCOME` |

Each machine-readable record additionally contains the exact scene/state/branch/assignment/snapshot identities, designated goal, all four offline block-end replays, five neighbouring graph-node checks, requested and realised/post-slew 20-tick actions, raw row binding, and all relevant source/lineage digests.

For every record:

- all four blocks and all 20 action ticks completed (`blocks_completed=4`, no truncation);
- 3 context and 4 horizon RGB receipts exist and their bytes validate, but there is no 20-tick RGB trace;
- only 15 pre-branch proprioception/control samples are stored; future proprioception is explicitly unavailable;
- snapshot and four block-end poses are recoverable from camera metadata, but the full pose trace is absent;
- contact, per-tick clearance, stuck, fall/termination, and completion-event traces are absent;
- every aggregate oracle field is null because the frozen oracle refused the row.

## Counts

- Category: 14 locatable/goal-unreachable; 4 off represented graph; 0 locator defects; 0 other.
- Source operation: 14 geodesic/reachability; 4 locate threshold.
- Role: 6 fit; 12 calibration.
- Family: local composite motifs 1; medium enclosed maze 3; open obstacle field 4; visual sensor stress 10.
- Stratum: completion-enriched 1; general 7; safety-enriched 10.
- First observed unavailable endpoint: tick 4 = 10; tick 9 = 3; tick 19 = 5.
- Exact first failure tick: not persisted = 18. Conservative possible ranges: ticks 0–4 = 10; 0–9 = 3; 0–19 = 5.
- Candidate: c0 3, c1 3, c2 1, c3 1, c4 1, c5 2, c6 1, c7 1, c8 1, c9 1, c10 2, c11 1.

## Offline graph-path checks

The final context receipt reproduces each frozen snapshot cell and finite start distance. H4 is captured immediately after the twentieth action tick. With the fixed mount `[0.326, 0, 0.043]`, zero mount RPY, and the persisted camera forward/up vectors, the base transform is invertible at the precision of the stored float32 camera pose.

For all preserved block endpoints:

1. `SceneGraph.locate` equals a brute-force nearest-node calculation.
2. An independent queue-based BFS equals `SceneGraph.bfs_distance` with and without `nav_blocked_cells`.
3. `GeodesicField` finiteness equals masked BFS reachability.

The four open-field endpoints remain within ±5 m scene bounds and have 2.97–3.54 m wall/obstacle clearance. They are outside the sparse 3×3 graph's 2.0 m coverage, not outside the physical world.

For the other fourteen, raw graph paths remain present (2, 4, or 3 hops), while masked paths are absent. Their nearest-cell assignments are correct. This is meaningful under the frozen transit policy, though it does not prove physical-space unreachability.

## Can the preserved evidence support a future deterministic relabel?

| Category | Progress | Safety | Completion | Composite utility |
|---|---|---|---|---|
| `OFF_NAVIGABLE_GRAPH_OUTCOME` | Only after an explicit future off-graph rule | No: 20-tick safety trace absent | No: at-or-before-horizon trace absent | No |
| `LOCATABLE_GOAL_UNREACHABLE_OUTCOME` | Only after an explicit future reachability rule | No: 20-tick safety trace absent | No: at-or-before-horizon trace absent | No |

The exact final endpoint and failure operation are reproducible. The exact first failing tick, last finite geodesic tick, complete path-level safety evidence, and completion state through the horizon are not. Consequently, the preserved artifacts are insufficient for deterministic full relabelling, and this pass freezes no new oracle semantics.

## Qualification-set consequence

All 24 historical calibration identities remain intact and retain their frozen `calibration` role. Prospectively, all 24 are marked `DEVELOPMENT_ONLY_PENDING_NEXT_DECISION`: they participated in diagnosing an outcome-dependent label boundary and must not later be represented as untouched qualification data. They are not discarded or replaced.

The integrity audit verifies 96 fit and 24 calibration state identities, 120 complete candidate banks, 1,440 row self-digests, equality of every individual row record with the compiled ledger, and 6,120 unique RGB file hashes (10,080 row references). Corpus artifacts were not changed.

## Recommendations — not implemented

1. Explicitly version any later semantic choice: treat graph-coverage/reachability loss as a terminal outcome, or revise graph coverage/transit masks.
2. Persist the 20-tick pose, locate distance/cell, geodesic finiteness, contact, clearance, stuck, termination, and completion evidence in any future corpus.
3. Do not relabel the eighteen V2 rows. If complete utilities are required, generate a separately versioned corpus under the chosen oracle.
4. Use fresh qualification identities after the next semantic decision; keep these 24 calibration identities development-only.

No branch was rerun; no latent was encoded; no scorer was trained or qualified; no predictor checkpoint was opened; no state/candidate was replaced; no invalid record was deleted; and no utility was invented.
