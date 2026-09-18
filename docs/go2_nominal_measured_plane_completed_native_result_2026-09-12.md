# Nominal forecast native comparison: completed negative result

The nominal predictive controller completed its measured-plane maze 02 run
and raw audit. It stopped at decision frame 55 because no candidate met the
unchanged surface/nominal constraints, after ten infeasible-wait commands.
Visual tracking was still available. It made no cell crossings or arrivals.

This is one paired development-layout experiment against the completed
learned no-RGB direct controller, which reached the outbound goal but failed
on the return turnaround. Both arms remain predictive. This comparison does
not isolate planning on/off or demonstrate a general JEPA or memory benefit.

## Exact completed evidence

Artifact base:
`/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/navigation_development_artifacts_v1`.

Native root: `go2_nominal_measured_plane_maze02_v1_attempt_001`.

| File | SHA-256 |
| --- | --- |
| `result.json` | `79a68e2de1160d4a015ea7b118f515a21af3f1050028034978f9ab29ba01470a` |
| `launch.json` | `463ec812fb12e16f3b8543a28d69b9275774488ebc3759b6fd457837abc6d1e6` |
| `no_rgb_nominal_measured_plane_maze_02_worker_terminal.json` | `f100967d330a6f2d2775101d8a19b0089028b75b1f38676248dc41e78f470f60` |
| `no_rgb_nominal_measured_plane_maze_02_audit.json` | `12f9bfedffe1aaf7a96b449b1263e343f893ece86a1215bc727a8c8f001ab9f4` |
| `no_rgb_nominal_measured_plane_maze_02_prefix_comparison.json` | `2c7d9cdd8bf1e77c1f9f453944bf02dfe7cf76294c8cdec6480390afcbe3b0c3` |

Waiter root: `go2_nominal_measured_plane_native_wait_v1_attempt_001`.
Completed waiter result SHA-256:
`287e83fc4b3591a24db68afc5bb97bb10c13495e99f6eb112310b12495993e28`.

The native result status is `NOMINAL_MEASURED_PLANE_MAZE02_V1_COMPLETE`.
All 2,577 result-bound current sources and all 436 result-bound native
artifacts were rehashed and matched. The result's condition exactly matches
the worker terminal record, and its native evaluation matches the audit.
Parent 2933558, native worker 2933696, and nominal waiter 2919853 had ended.
No execution-failure result was present. The scientific negative is retained.

## Result and failure mechanism

- 66 observations, 65 completed commands, 4,000 physics samples; ten terminal
  zero commands after the controller terminal decision.
- No goal arrival, outbound or return crossing, route retracing, or round trip.
- Zero recorded native contact samples; no physical or acquisition stop.
- Strict physical visibility and raw sensor/controller/nominal-forecast
  replay pass. No hard measurement-failed frames.
- Zero actual learned-model forward calls, as required for the nominal arm.
- Actual four-observation shared prefix and 900 shared physics samples match
  the registered comparison. Frame 3 changes the learned left arc
  `[0.16, 0, 0.45]` to nominal forward `[0.2, 0, 0]`; the boundary command was
  physically completed. Later trajectories are not a common-prefix replay.
- Observation/control median 931.7922085 ms, p95 1151.48065175 ms;
  all 66 samples exceed the 100 ms command interval. Physics was paused during
  computation. This does not establish real-time execution.

Inspection of the authenticated complete decision stream shows:

| Frame | Current floor-registered pose | Selected action | Phase-admissible candidates | Hold nominal cell distance |
| --- | --- | --- | --- | --- |
| 44 | Present, frame 44 | right arc | 4 | 0.45477667009564005 m |
| 45 | Present, frame 45 | none; wait begins | 0 | 0.44880215643605625 m |
| 55 | Present, frame 55 | none; terminal | 0 | 0.444552932651919 m |

The nominal disk radius is 0.45 m. At frames 45 and 55, all six nominal
connector checks fail, including hold and both turns; all six surface checks
report no possible intersection. Frames 45–54 accumulate ten consecutive
infeasible waits. Frame 55 reaches eleven consecutive infeasible observations
and records `NO_PHASE_CANDIDATE_SATISFYING_SURFACE_AND_NOMINAL_CONSTRAINTS`.
This identifies the recorded constraint failure without treating the nominal
disk as a physical collision certificate or changing the frozen gates.

## Next running comparison

The existing reactive waiter automatically launched
`go2_reactive_measured_plane_maze02_v1_attempt_001` against the exact completed
nominal waiter result above. Launch SHA-256:
`fccd73bc30bc33a84e657d53738f310767b501ed6c8e8f1c53b1f54343502383`.
Parent PID 2934380, creation time 1789172301.33, matched its launch command
and was live; native worker PID 2934551, creation time 1789172349.84, was
running. All 2,592 reactive launch source bindings were rehashed and matched,
as did the live parent boot/PID/creation-time/command identity. The actual case
directory is `reactive_measured_plane_maze_02`; its timing stream advanced
through tick 105. This is progress evidence, not a completed reactive result.

The full-history performance replay remains active separately. The candidate
tracking-controller replay is still queued behind that CPU replay. Neither
pending result is inferred from the nominal outcome.
