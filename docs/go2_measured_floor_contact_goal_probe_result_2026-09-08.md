# Measured-floor foot contacts: scoped rule works, navigation remains incomplete

All 43 focused tests and all four fresh native replays passed. The scoped
foot-floor contact rule lets direct 039 advance beyond the predecessor's
blockage, but verified goals remain 0/4. The other three cases reproduce their
predecessor's complete command tapes and outcomes exactly.

| Model / layout | Terminal | Terminal / minimum goal distance |
|---|---|---:|
| RGB direct / 039 | Mission budget exhausted | 1.547352 / 1.115987 m |
| RGB JEPA / 039 | Mission budget exhausted | 1.129313 / 1.129313 m |
| RGB JEPA / 052 | Nominal start infeasible, tick 28 | 1.068954 / 1.068859 m |
| RGB direct / 052 | No nominally feasible forecast, tick 28 | 1.083074 / 1.083011 m |

All 586 RGB-D frames passed strict physical visibility and hard measurement
checks. There were no disallowed contacts, physical stops or visual failures.
Each chosen forecast passed the nominal obstacle constraint and the scoped
surface/contact rule. Fixed direct and JEPA snapshots remained unchanged.

The [all-return partition diagnostic](go2_measured_floor_partition_result_2026-09-08.md)
classified every contributing observation before this prospective change. The
runtime rule permits floor overlap only for the four exact foot spheres whose
entire nominal projected disk lies on previously measured floor. Non-foot
contacts, non-floor/unknown returns and incomplete footprint coverage remain
vetoes. Every frame retains classification accounting and both original and
revised candidate checks. This is explicit experimental contact handling, not
support, friction, penetration, model/pose uncertainty or hardware certification.

Direct 039 stays in WAYPOINT mode after tick 23. Its first changed command is
tick 83: forward becomes available instead of holding. It advances through ticks
88, 93 and 98, then holds at every selection from 103 through 238. Terminal
initial-frame XY is [-0.016210, -0.956625] m. The 48 selections comprise 29 holds,
six right turns, four right arcs, three forwards, five left arcs and one left
turn. This is an unfinished detour ending farther from the mission goal, not
successful exploration or arrival.

The late holding state has a route and 0.993472-m nominal clearance. At tick 238,
left turn has utility 0.083766 and left arc 0.070043, but both remain vetoed by
incomplete projected floor coverage. Left turn's left-front foot lacks floor
cell [4, -19]; left arc's right-rear foot lacks [-7, -18] and [-6, -18]. Their
intersecting bounds contain floor returns and no non-floor/unknown hits, but the
coverage condition is intentionally retained. Hold is the highest-valued
remaining candidate (0.020426); forward is feasible but scores -0.001738.
The same nominal and prediction limitations persist in the unchanged cases.

The next coverage check should query the actual nominal foot footprint against
retained measured depth patches, keeping all existing pixel-validity, normal,
planarity and height tests. Requiring every touched 5-cm grid cell may demand
coverage outside a 44-mm foot disk; determine whether those specific missing
cells reflect grid overapproximation or genuinely unavailable observations.
Do not mark the cells observed, waive unknown ground or change their original
receipts to resolve this. Moving-state prediction/braking uncertainty remains
a separate unresolved requirement, especially for the two 052 cases.

RGB, complete observer/map evidence, forecasts and native/body/gyro arrays
match the predecessor through the 84 observations before direct 039's first
changed command. Its first terminal-state difference occurs later, at tick
218, so it is excluded from the common prefix. For the three unaffected cases,
the entire tapes, RGB sequences, observer/map evidence, forecasts and native/
public arrays match. Newly added classification receipts are separately rebuilt
and verified by each complete raw replay. Maximum accepted visual XY errors
remain 4.780 mm, 2.532 mm, 0.877 mm and 0.877 mm in table order.

Runtime classification retains 9,288,049 inserted returns across the four cases:
2,759,132 measured-floor returns and 6,528,917 non-floor/unknown returns. Every
observation is classified independently; a floor first witness never exempts
a later unknown return. All original observations and failed attempts remain.

Preflight recorded 82.00 GB available RAM, 86.74 GB artifact free space, 0.3%
CPU activity and idle GPUs. Four fresh single-thread CPU processes used the
bound native scaling evidence. Post-launch work took 246.446 seconds. The
native result binds 994 sources and 2,474 artifacts totaling 1,057,621,796 bytes,
with 32,100 physics samples and 586 RGB-D frames; the readout binds 996 sources.
All 582 command iterations exceed 100 ms; case median full-iteration times
range from 398.775 to 479.051 ms under concurrent load. Simulation pauses during
compute, so these are not real-time or uncontended performance results.

| Artifact under the development base | SHA-256 |
|---|---|
| `go2_measured_floor_contact_goal_probe_v1_attempt_001/launch.json` | `69a517bb848be2eb7ffec409df6a75731f2d1ed42e564a713e3bb2f623846271` |
| Native `result.json` | `607524c7b6cc86356bc0bdceac1dd310723c6012e1dc377fe80c890f5057f17e` |
| `go2_measured_floor_contact_goal_readout_v1_attempt_001/launch.json` | `65c8e3bbeb51a8d1db4e4dccc52fd0439ee0bd17d1c71be51eea2784e24478d0` |
| Readout `result.json` | `473c1eec74d205672017efeaec0c742dedcf401f601dc37f0fba969d7b538c08` |

The full goal remains active and unfulfilled. Independent-maze goal-reaching,
physical backtracking, model/planning/memory contributions, calibrated sensing,
realistic timing and bounded hardware evidence remain unproven.
