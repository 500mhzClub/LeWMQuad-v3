# Completed collection inspected; native raw audit still pending

This is a read-only inspection of the closed collection log, not admission of
the native result or a claim of physically verified navigation. The original
worker PID 2994743, creation time 1789194027.81, remains active in its original
audit. Its compressed decision-stream read position advanced from 503,058,432
to 514,068,480 bytes during this inspection. No worker-terminal, final result,
failure or raw-audit JSON had been recorded at the last observation.

## Exact inputs

Artifact root:
`go2_measured_plane_chained_maze02_v1_attempt_001` under the existing
`navigation_development_artifacts_v1` artifact base.

Case: `no_rgb_direct_measured_plane_chained_maze_02`.

| Input | SHA-256 |
| --- | --- |
| Root `launch.json` | `0f2963636abc4f1fc738e9323d2a74db02888ca7062dc1afa21023f8c443b8ff` |
| Case `result.json` | `6ec45708018a96dc039ddf7db7a578a40a3f8cde5e3d2e8c6706c71352a8527d` |
| Case `context_decisions.jsonl.gz` | `c7d131924208f8afdbf8e5d4ab2ba9f211ed2ca4a62faf70e498bb19a379fb7a` |

The compressed decision log is 609,591,253 bytes. A complete stream scan
checked exactly 4,014 ordered tick/observation indices. Its SHA-256 was checked
before and after that scan and before and after the subsequent targeted
selection inspections. No model was loaded or controller replayed by these
inspections; the existing native worker continues its full raw audit.

## Logged collection outcome

The collection receipt reports 4,014 observations, 4,013 completed commands,
201,400 physics samples and ten terminal zero commands. It reports no physical
or acquisition stop. The controller reached `MISSION_TICK_BUDGET_EXHAUSTED` at
mission frame 4003 after 4,000 navigation ticks, still in RETURN with observed
distance 2.621759960041774 m to home and `verified_round_trip: false`.

Its recorded outbound arrival is frame 3062, observed distance
0.02753891729016619 m with ten quiet intervals. That arrival receipt explicitly
has `native_verified: false`; physical arrival verification remains with the
pending raw audit.

All 940 decisions in frames 3063–4002 have null controller terminal status.
Requested commands and selected action counts agree:

| Action | Count |
| --- | ---: |
| right_turn | 451 |
| left_arc | 347 |
| hold | 118 |
| left_turn | 13 |
| right_arc | 11 |

At the former failure frame 3113 the recorded action is right_turn, with null
terminal and failure. The independently reconstructed old/new prefix and
actual changed-command physics proof are still pending in the native worker;
the closed log alone is not that proof.

## Distance plateau does not establish stalled route progress

The observed home distance falls from 4.700948681402086 m at frame 3113 to
2.6127165366826244 m at frame 3800, then remains near 2.62 m. However, the
sampled waypoint and route records continue advancing:

| Frame | Observed home distance (m) | Waypoint map XY (m) | Remaining route cells |
| --- | ---: | --- | ---: |
| 3700 | 2.669370 | [2.575, 0.075] | 103 |
| 3800 | 2.612717 | [2.525, -0.375] | 94 |
| 3900 | 2.629745 | [2.525, -0.825] | 84 |
| 4002 | 2.623161 | [2.275, -1.225] | 74 |

All four route proposals are `OBSERVED_FLOOR_ROUTE_TO_FRONTIER` with the same
route target [0.475, -0.025] m. Their route-grid lengths are respectively 102,
93, 83 and 73 cell edges. These are observed planner records, not native
physical route lengths or proof that a robot can traverse the proposal. They
show why straight-line home distance alone would give a misleading diagnosis.

## One late hold is selected by utility, not an empty candidate set

At frame 4002 the log has six phase-admissible candidates. All six recorded
nominal path checks are clear and all six surface `possible_intersection`
flags are false. These experimental filters do not certify safe motion.

The final selected hold has utility -0.004221389787986082. Forward has greater
predicted executed-waypoint distance progress (0.023295203123220865 m versus
-0.0015628152845802434 m for hold), but utility -0.04533539388698428. Its
full-plan contact score is 0.05726688584151375 versus 0.002284528431206187 for
hold. These are uncalibrated model scores, not measured collision probabilities.
Every candidate's recorded utility is negative, and hold has the highest one.

This motivates examining progress/risk scoring and the available mission
budget after the audit. It does not establish that the entire return was
stuck, justify changing safety filters, prove that another action would have
worked, or demonstrate success from a longer budget. No controller, live-bound
source, command, budget or native attempt was changed.

## Source follow-up for possible next experiments

`lewm/observed_round_trip_mission_development.py` limits navigation to 4,000
ticks. `lewm/joint_visual_surface_memory_development.py` limits retained route
history to 4,096 frames and fails rather than evicting evidence at capacity.
The current extended collector and replay bind the 4,014-observation population
through their writer, reader, sensor, renderer-witness and command-audit
dependencies. A longer mission therefore needs coordinated source changes and
resource validation; changing a launch budget alone would be insufficient.

`lewm/executed_waypoint_score_development.py` explicitly scores the causal
100 ms waypoint displacement against the 800 ms full-plan contact score while
retaining all eight nominal path segments and the existing surface filter.
Its selection is the maximum utility among the remaining phase-allowed
candidates. This source behavior is consistent with the frame-4002 record.
It does not prove that reducing the contact horizon would improve navigation;
such a change would need its own prospective comparison and physical outcome.
No source limit or scoring rule was modified in this follow-up.

## Complete hold-form census and supported utility reconstruction

The first attempt to apply the existing whole-log intermediate-waypoint hold
classifier was rejected when it encountered another scoring form. That
diagnostic rejection is preserved byte-for-byte in
`docs/go2_measured_plane_chained_hold_classifier_rejection_2026-09-12.json`,
SHA-256 `7aee2eab01c4f74869ff515bf4515fc5c258cb31d7a9141b5861c81a78735a2d`.
It is a read-only diagnostic limitation, not a native execution failure.

A subsequent census accounted for every hold form across all 4,014 ordered
observations and matching completed command-tape entries. It applied the
existing exact utility reconstruction only to its supported intermediate
waypoint form, retaining an explicit count of every other form. The complete
result is
`docs/go2_measured_plane_chained_hold_schema_census_2026-09-12.json`, SHA-256
`edffa273bd66717adea7ae628210b9cb35fef0eec356f380083473d08f9c1f69`.

There are 604 holds: 593 intermediate-waypoint holds and eleven final-goal
holds during outbound travel (first at frame 2962, last at 3047). All 118 return
holds are in the supported intermediate-waypoint group. Their utilities and
components reconstruct exactly from the saved forecasts, causal residuals and
existing potential/contact rule. Every return hold has reason
`no_strictly_better_allowed_nonhold`: no allowed nonhold action has strictly
greater recorded utility. This is not a claim that every possible action was
geometrically admissible or that changing the contact penalty alone would
eliminate every hold.

Across all 593 reconstructed holds, 410 have that utility reason and 183 have
`every_better_nonhold_has_a_first_point_invariant_veto`. The eleven final-goal
holds are counted but their utilities were not reconstructed by this helper;
the report explicitly sets `all_hold_utilities_reconstructed: false`.

Both inspections bind the decision-stream hash above and command-tape SHA-256
`f325bfc1c5eacc87642bd584b42d6a90612dc4f8264541422ab2ce0f4fc1d14a`.
The successful census rehashed both inputs before and after processing.
The unchanged reconstruction helper is
`lewm/residual_hold_veto_readout_development.py`, SHA-256
`a59fc3811dd9a6470a0fb2e49c20ea2dc897e57c7405ab0c46f1dd007c26a066`.
No model or controller was executed by these inspections, no alternative
physical outcomes are inferred, and no live-bound source was changed. The
native worker remains active in final verification at this snapshot.
