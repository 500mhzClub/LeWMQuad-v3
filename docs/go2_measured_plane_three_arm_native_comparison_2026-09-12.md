# Three completed measured-plane native pilots on development maze 02

All three actual simulator executions and their raw audits are complete. Only
the learned no-RGB direct world-model controller reached the outbound goal.
None completed a round trip. These are single runs on a reused development
maze, not independent-maze qualification or a general world-model advantage.

| Arm | Observations | Completed commands | Physics samples | Outbound crossings / distinct edges | Goal arrivals | Terminal decision |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Learned no-RGB direct forecast | 3124 | 3123 | 156900 | 15 / 10 | 1 outbound | Frame 3113: visual bridge budget exhausted during return turnaround |
| Nominal requested-twist forecast | 66 | 65 | 4000 | 0 / 0 | 0 | Frame 55: no candidate passes nominal clearance constraints |
| Fully reactive | 461 | 460 | 23750 | 5 / 5 | 0 | Frame 450: neither camera provides an acceptable current pose |

All three have zero recorded native contact samples, no invalid crossings,
no return crossings, no route retracing, and no verified round trip. Their
raw sensor/controller/command audits and strict physical visibility checks
pass, with no hard measurement-failed frames. Zero recorded contacts is not
a general collision-clearance certificate.

## What the comparison establishes

The nominal arm retains predictive action evaluation, the shared scoring
machinery, and observed residual correction while substituting requested-twist
forecasts; its audit confirms zero learned-model forward calls. The reactive
arm loads no high-level world model and replaces the whole action-selection
method. It does not preserve predictive constraint gates or isolate prediction
ranking. Neither is an isolated JEPA-training or memory ablation.

Both comparison arms have an actual four-observation shared physical/public
prefix with the learned run. The first intervention is frame 3: learned left
arc `[0.16, 0, 0.45]` versus nominal/reactive forward `[0.2, 0, 0]`. The shared
900-sample physics prefix hash is
`8419be1a3143128fcef2a1cf843d6476063177cc52f83316c42fd9b2bc7789fa`.
Complete baseline and candidate decisions match their prospective prefix
reports, and the changed commands were physically completed. Later paths
diverge and are not interchangeable observation histories.

All launch records retain 4000 navigation ticks, 4013 maximum commands,
4014 maximum observations, the same robot URDF and native geometry/scene
source identities, and no native-pose controller input. Physics pauses during
computation. Observed control timings therefore do not establish continuous
real-time execution:

| Arm | Observation/control median | p95 | Samples over 100 ms |
| --- | ---: | ---: | ---: |
| Learned | 1440.4902195 ms | 2996.3371336999994 ms | 3124 / 3124 |
| Nominal | 931.7922085 ms | 1151.48065175 ms | 66 / 66 |
| Reactive | 780.368284 ms | 854.450711 ms | 461 / 461 |

Different trajectory lengths and histories make these descriptive timings,
not a matched-history speed comparison. The separate full-history timing
replay is still required.

## Reactive terminal diagnosis

At frame 449, the last accepted pose uses auxiliary visual motion and measured
floor transport. The primary camera has insufficient rigid-pose matches; the
auxiliary camera supplies a fifth measured increment bridge without a retained
anchor. The last measured joint floor plane is unavailable because of
`insufficient_combined_two_axis_extent`.

At frame 450, primary registration still has insufficient matches, and the
auxiliary increment is rejected for rigid consensus fraction, grid support,
or displacement. No retained anchor passes the unchanged registration and
increment gates. The controller latches `SENSOR_OR_MODEL_FAILURE` with
`same-episode current visual evidence required`, then records ten terminal
zero commands. This is not the learned run's ten-bridge exhaustion mechanism.

The measured-plane estimator intentionally retains the original qualified
image fit when a reference or current floor plane is genuinely missing;
`measured_plane_constrained=False` on that witness is an explicit existing
fallback, not proof of the wrong controller being instantiated. The source
branch in `lewm/measured_plane_dual_camera_pose_development.py` was inspected.
No live or frozen controller, threshold, bridge budget, or map state was
changed in response to these failures.

## Frozen result identities

Artifact base:
`/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/navigation_development_artifacts_v1`.

| Root | `result.json` SHA-256 |
| --- | --- |
| `go2_measured_plane_dispatch_recovery_v1_attempt_001` | `4ecd63cd96aff9277103609b3034ed87e736e1fff1bb3352113941ff2c26aa18` |
| `go2_nominal_measured_plane_maze02_v1_attempt_001` | `79a68e2de1160d4a015ea7b118f515a21af3f1050028034978f9ab29ba01470a` |
| `go2_reactive_measured_plane_maze02_v1_attempt_001` | `d20747931ff557206698ad7f5ae8d46145d2af5317b12cc7ed62a772039172f6` |
| `go2_nominal_measured_plane_native_wait_v1_attempt_001` | `287e83fc4b3591a24db68afc5bb97bb10c13495e99f6eb112310b12495993e28` |
| `go2_reactive_measured_plane_native_wait_v1_attempt_001` | `b11b395e19eecaf1a7b8d7052b7762eea31db9e9a4266ae5640d935a02133c75` |

Reactive native root evidence:

| File | SHA-256 |
| --- | --- |
| `launch.json` | `fccd73bc30bc33a84e657d53738f310767b501ed6c8e8f1c53b1f54343502383` |
| `reactive_measured_plane_maze_02_audit.json` | `ec7529b950d8526e5829a663f48c7bf0d0dbdc0e0ea53e3f8916094c1168c951` |
| `reactive_measured_plane_maze_02_prefix_comparison.json` | `752353bb6dd8c136083d499622866543eae4e1540e9c64905c71037fcf66cb42` |
| `reactive_measured_plane_maze_02_worker_terminal.json` | `7786d7bc9872a811b66406b5d7cb836aa77e9d8921d399dece1b17bad5a810b7` |

All 2,592 reactive result-bound current sources and all 2,806 result-bound
artifacts were rehashed and matched. Exact result/worker, native-evaluation,
and prefix consistency checks passed. Parent PID 2934380, worker 2934551,
and reactive waiter 2922235 ended. No execution failure was reported. The
completed reactive waiter independently reports the full native artifact
roster verified and the actual physical prefix reconstructed. All 2,595
waiter-bound current sources and all four waiter-bound artifacts were also
rehashed and matched. Equality of the eight shared launch fields described
above was checked programmatically across all three arms.

For the earlier completion checks and detailed evidence identities, see
`go2_measured_plane_completed_native_result_2026-09-12.md` and
`go2_nominal_measured_plane_completed_native_result_2026-09-12.md`.

## Next decision

The original queued full-history timing replay still owns the CPU replay
slot. The chained tracking-controller replay follows through its existing
waiter. A candidate physical execution requires that replay's authenticated
actual intervention boundary; pairwise feature matches alone are insufficient.
No replacement native scene is running or being inferred from these results.

Before a broad comparison can establish usefulness, the system still needs a
completed round trip, independent-maze replication, matched perception across
new controller arms, JEPA/supervised and memory comparisons, realistic sensing
and execution timing, and bounded hardware evidence. The full goal remains
active and incomplete.
