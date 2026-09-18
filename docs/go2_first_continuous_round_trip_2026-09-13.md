# First verified continuous development round trip

The learned world-model controller completed an outbound-and-return mission
on development layout 0 with independently checked native arrivals. The
owner exited 0 and all 2,540 paired camera observations and registered poses
were saved. This is the first continuous round trip to pass the physical
arrival checks; the earlier successful mission paused simulation for planning.

Artifact: `go2_indexed_geometry_precise_goal_round_trip_native_layout00_v1_attempt_001`
under the navigation development artifact root in RecoveryStorage.
Result SHA-256:
`bd42df893c1ad8df2ba042d9aeec35f01aaae174b4b04f396fe035a45b6626a4`.
Independent measurements: `continuous_native_arrival_evaluation.json`.

| Measurement | Result |
|---|---:|
| Outbound arrival frame | 1666 |
| Return arrival frame | 2538 |
| Maximum outbound distance during one-second dwell | 6.73 mm |
| Maximum home distance during one-second dwell | 16.09 mm |
| Physical arrival requirement | 40 mm |
| Zero requests throughout both dwells | Yes |
| Maximum native 100 ms speed, outbound / return dwell | 0.0206 / 0.0267 m/s |
| Disallowed contacts / pipeline faults | 0 / 0 |
| On-time plans | 594 / 625 (95.0%) |
| Timed execution wall / simulated duration | 254.70 / 254.22 s |
| Registered position median / maximum error | 10.64 / 21.05 mm |

The controller uses the frozen `seed_2026091001_full_jepa` model assignment,
the frozen learned motion residual, paired RGB-D visual tracking and body
sensors, persistent observed floor/obstacle memory, learned candidate-action
forecasts, measured frontier inspections and actual goal/home coordinates.
The residual fit SHA-256 is
`ef7511b29afd0117291f600d7afad6adccdcd90199d0ea1f45519d7b81b01638`.
One initial panorama and one frontier panorama completed. Tracker and map
history were retained across the outbound/return transition. Native poses
were used only by the evaluator.

Recent changes that enabled this attempt include joint-camera registration
for missing retained-reference and consecutive-frame fits; an explicit 2 cm
floor spread treatment; indexed, cached exact obstacle geometry; an exact
goal endpoint beyond the containing cell centre; and a stricter 2 cm observed
arrival target. The physical requirement remains 4 cm. The complete experiment
history, including failures, is in
`go2_feature150_memory_clearance_experiment_2026-09-13.md`.

Scope: nominal ideal-sensor simulation, with measured worker/acquisition costs
charged to the simulation clock. Physics does not wait for planning, but it
pauses during rendering. This is not host-real-time or hardware qualification,
and no full separate raw-sensor audit is claimed. Layout 0 has been heavily
used for development; this result alone does not establish unseen-maze
reliability, JEPA advantage, prediction necessity or memory benefit.

The first transfer to layout 1 failed after 95.152 simulated navigation
seconds. Native translational speed reached 0.300170836 m/s, exceeding the
unchanged 0.30 m/s limit. There were no disallowed contacts; the final native
contacts were foot support, and the robot remained inside the allowed domain.
All 952 camera pairs were saved. This is a failed transfer, with no arrivals.
The artifact is
`go2_indexed_geometry_precise_goal_round_trip_native_layout01_v1_attempt_001`;
`physical_stop_diagnosis.json` separates this physical cause from the secondary
cleanup clock-reversal error. The latter is fixed for subsequent runs by
retaining partial physics-step time and propagating the original stop without
attempting worker-drain physics. Two focused policy-service tests pass.

The layout-2 transfer also failed to reach either destination: it exhausted
3,600 navigation ticks after turning drift left it unable to select an action
that met its nominal clearance requirement. The process exited 0, all 3,605
camera pairs and poses were saved, and there were no disallowed contacts or
pipeline faults. Independent evaluation found a maximum position error of
14.50 mm and no arrivals. Clean execution is not navigation success.
Detailed diagnosis and the matched reactive comparison are recorded in
`go2_continuous_controller_comparison_2026-09-13.md`.

Next: complete the reactive comparison using the current continuous execution
loop and investigate turn drift, stopping and clearer viewing positions.
Realistic sensing, timing and bounded physical-platform evidence remain
outstanding.
