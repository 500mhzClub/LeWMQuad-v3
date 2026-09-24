# Paired planning-memory readout complete

Session49332 exited0, result
`3d11666a31d229355bafbd50f622c73ef0dd7acf2882654eb07fe1b8e7747214`,
launch `9320c50cdcc2e3f936019ad429b8caa475f6fbbfbac5101cce5df70e62aa3171`,
1,688bound sources. Root
`go2_current_observation_planning_maze_readout_v1_attempt_001`. The readout
preserves the completed original outcomes and verifies the paired definitions
against the original learned readout and its native artifacts. No new execution.

Current-observation planning travels0.4981579341564623m over10.2simulated
seconds, comes no closer than3.783563617810178m to the outbound goal, and
terminates3.7835777021721033m from it. Maximum observed XY pose error is
0.005478224998792014m. It never leaves the native start cell or reaches an
arrival; it stops on visual missingness at observation92. Strict visibility
fails at18and84, as recorded without modification in the native result report.

Median observation/control wall time898.425796ms; receipt-inclusive median
945.811852ms, maximum1221.033506ms. All103receipt-inclusive iterations exceed
100ms. The original persistent-planning baseline has receipt-inclusive median
1237.033925ms, one outbound arrival and no home return. Its strict visibility
also fails. Different trajectories, map sizes and durations prevent treating
these whole-cycle times as an isolated cost of memory or inference.

Both methods retain contact history, tracking/floor anchors, learned temporal
history and online residuals, mission/settling state and selector scan state.
The changed factor is whether accumulated observed planning cells are queried;
this is not a memoryless comparator. The native first intervention was physically
reproduced with complete prospective decisions, but one short failed trajectory
does not establish memory advantage, statistical reliability or independent-maze
generalization. Both have zero verified round trips and no real-time/hardware
qualification. The goal remains active.

Full native result and artifact identities:
`docs/go2_current_observation_planning_maze_pilot_result_2026-09-09.md`.
