# Sample-bound native probe: longer execution, zero verified goals

All 35 focused tests and four fresh native sensor-to-command replays passed.
The refined geometry removes specific overapproximation vetoes and lets direct
039 continue beyond its previous stop. Verified goals remain 0/4; the other
three cases reproduce their predecessor's complete command tapes and outcomes.

| Model / layout | Terminal | Terminal / minimum goal distance |
|---|---|---:|
| RGB direct / 039 | All candidates vetoed, tick 218 | 1.503888 / 1.115987 m |
| RGB JEPA / 039 | Mission budget exhausted | 1.129313 / 1.129313 m |
| RGB JEPA / 052 | Nominal start infeasible, tick 28 | 1.068954 / 1.068859 m |
| RGB direct / 052 | No nominally feasible forecast, tick 28 | 1.083074 / 1.083011 m |

There were no physical stops, disallowed contacts, visual failures or hard
measurement failures. All 561 RGB-D frames passed strict physical visibility.
Every selected forecast passed the unchanged nominal and refined surface tests.
Fixed model snapshots remained unchanged. These bounded clean audits do not
turn the zero-goal outcomes into navigation success.

Direct 039 remains in WAYPOINT mode from tick 23 through termination. Its earlier
nominal-action predecessor terminated at tick 83; this version continues until
218 and ends at initial-frame XY [-0.038095, -0.853698] m. It selects 18 holds,
nine right turns, five right arcs, two forwards, four left arcs and five left
turns before the final no-candidate choice. The unfinished detour moves farther
laterally but ends farther from the mission goal. Extra execution time and
distance alone are not evidence of useful goal-reaching.

At the final state nominal clearance is 0.953617 m. The remaining vetoes are
exclusively front-foot sphere intersections with retained measured-point
enclosures: hold, forward, left arc and left turn hit both front feet; right arc
and right turn hit the left front foot. The model would otherwise rank left
turn highest (utility 0.094517), followed by left arc (0.070741) and hold
(0.024912). Thus the late stop is still a foot/terrain-geometry limitation,
not lack of a route or violation of the nominal obstacle radius.

The [recorded geometry diagnostic](go2_measured_sample_bounds_result_2026-09-08.md)
shows why this was a justified prospective refinement: it encloses every return,
retains mixed observations and first witnesses, and removes empty voxel-volume
and primitive-AABB overlaps without deleting terrain points. It does not solve
the distinction between admissible foot support and obstacle collision. Before
introducing that distinction, classify all contributing observations in affected
bounds, preserving non-floor and unknown evidence; the earlier first-witness
floor classification alone is insufficient. Any future foot-support rule must
be explicit and remain separate from non-foot collision checks and calibration.

A separate source inspection confirms another limitation worth measuring, not
silently changing: two-component model translations are lifted with zero body-Z,
and yaw is applied around the current body-Z axis. In the earlier tick-83 state,
measured map up in body coordinates was [-0.007396, -0.085555, 0.996306]; the
candidate translation lift implied map-height changes of roughly -0.36 to
-2.28 mm. That may affect foot queries, but it does not establish a replacement
motion model. Current nominal foot bottoms also differ from the fixed measured
floor hypothesis by several millimetres. Future posture, terrain interaction and
prediction uncertainty remain unresolved; no plane correction was applied here.

The improved comparison ends at the first command OR terminal intervention.
For direct 039 the first terminal-state difference is tick 83, while the first
different requested command is tick 93. All RGB, observer/map evidence, forecasts
and native/public arrays match through the 84 observations preceding that
terminal intervention. For each other case, the full command tape, complete RGB
sequence, observer/map evidence, forecasts and native/body/gyro arrays match the
nominal-action predecessor exactly. No later unexecuted outcomes are inferred.

Preflight recorded 82.26 GB available RAM, 87.73 GB artifact free space, 0.3% CPU
activity and idle GPUs. Four fresh single-thread CPU processes used the existing
verified native scaling configuration. Post-launch work took 203.401 seconds.
The native result binds 980 sources and 2,374 artifacts totaling 948,877,886 bytes,
with 30,850 physics samples and 561 RGB-D frames; the readout binds 982 sources.
All 557 command iterations exceed 100 ms. Median full iteration times range
from 314.649 to 367.266 ms under changing concurrent load, with simulation paused
during compute. These are not real-time or uncontended performance comparisons.

| Artifact under the development base | SHA-256 |
|---|---|
| `go2_sample_bounds_goal_probe_v1_attempt_001/launch.json` | `1d592efdc2ce32a866e0482ee44ceb6ed004d3bc9f8fa93124e9091de90a63b2` |
| Native `result.json` | `76bcd4079e7271a2e8ecd6fde9c551cfc1a83918b84239f4449e138345a36dc4` |
| `go2_sample_bounds_goal_readout_v1_attempt_001/launch.json` | `6ac54c77273e94a39faf173bbcaed49a7a1029ba0173b80f129d3902f7fa97b6` |
| Readout `result.json` | `2e2c89b3593d9d14662ffbd5fbd0ac39e5a9fd0716c1107c3a8d873407090b2f` |

Every predecessor and failure is preserved. Independent-maze goal-reaching,
physical backtracking, model/planning/memory contributions, calibrated sensing,
realistic timing and bounded hardware evidence remain unproven. The full goal
is active and unfulfilled.
