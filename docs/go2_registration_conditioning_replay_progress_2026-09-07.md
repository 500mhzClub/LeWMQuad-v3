# Whole-stream registration diagnosis: completed

The subsequent goal turn completes replay57556 exit0 (52ce89) and independently
verifies its686 sources, predecessor inputs/native bindings,35 output bindings,
terminal result and complete joined sensor/evaluation rows (42433 exit0,be999b).
All12,138 frame/arm observations and7,383 original decisions are covered.
See the [completed result and next experiment](go2_registration_conditioning_replay_result_2026-09-07.md).
Result SHA256: `dc392027aa0b9e65566b436257485369f2cdf468daa80acc43908d436415d28e`.
Replay57556 is no longer live; do not restart it. The collector25963 and its
l02 child2071161 remain live (777664), reaching25/120 prechecks (041c0f).
The implementation/launch narrative below is historical, not current job state.

The previous goal turn completed the matched-study executable, its 3,230-test
regression and the reviewed 786-source definition. This turn makes concrete
progress on the separate local-execution prerequisite: instrument the unchanged
visual observer and replay complete recorded streams before evaluating error.
No new recorded model fit, controller modification, gate change or physical
simulation is introduced.

## Implemented evidence path

`lewm/rgbd_registration_trace_development.py` observes actual calls/returns/failures
of the unchanged matcher, registration function and primary/alternative candidate
function. It records stage-wise populations, converged candidate diagnostics,
original rejections and instrumented component timings. The tracer neither
replaces estimator functions nor returns diagnostic candidates to control.
It refuses another active tracer, restores tracing on exit and exposes its own
faults instead of silently turning them into ordinary tracking failures.

The conditioning report distinguishes conditional algebra from reliability.
With rotation fixed, the translation normal matrix is N*I even for spatially
degenerate matches. Scatter and angular sensitivity remain separately reported.
Tests demonstrate that common point bias can produce wrong translation with
zero residual, and verify gyro lever-arm sensitivity by finite differences.
Full rank, low residual and many points are not calibrated uncertainty or a
reason to accept the five-cell failures.

`scripts/replay_go2_registration_conditioning_v1.py` authenticates the completed
inner-arrival and persistent-intent recordings plus the original paired balanced
replay estimates. It covers all nine specified estimator streams / 12,138
frame-arm observations, preserving failed and latched frames. Original poses
must exactly reproduce physical decisions; both persistent-intent frontends
must also reproduce their complete earlier pose/selection/failure streams.
All sensor estimates are saved and hash-bound before native coordinates are
parsed. A separately guarded evaluator then reports actual pose/candidate error,
including converged rejected candidates, without changing their qualification.

Available-frame and all-frame timing populations are separate; matching and
registration timings include instrumentation overhead. These are not isolated
hardware latency or realtime results. This replay overlaps other CPU work and
does not alter that limitation.

## Tests and preflight

- Initial trace tests 8120 complete exit 0: 17 passed in 2.25 s.
- First combined test run 56548 reports 28 passed / one failed. The failed test
  wrongly assumed exactly zero drift from the unchanged float32 image/flow
  fixture; observed drift is approximately 1.3183e-9 m. The test now checks the
  exact original observer's measured error, not an invented zero or a loosened
  estimator gate. No recorded data or physical acceptance threshold changed.
- Focused 17097 completes exit 0: **61 passed in 9.81 s**, including 17 new trace
  tests, 12 replay/evaluator tests and the existing rigid/multi-reference tests.
  Cases include both original/balanced observers, exact outputs, terminal
  failures, unavailable matching, stale intermediate fits, trace restoration,
  sensor-before-native ordering, original-pose mismatch, clocks and retained
  storage/source/native failures. Synthetic results are not physical recovery.
- Read-only preflight 9075 completes exit 0. It authenticates 8,286 inner and
  14,217 persistent-intent input bindings, validates 686 replay source bindings
  and checks a 24,047,474-byte launch definition against the existing metadata
  allowance. It also verifies the unchanged 786-source matched-study definition
  and confirms the replay output was absent before launch.
- Full explicit 241-file regression 48254 completes exit 0: **3,259 passed in
  278.98 s**, including all 29 new trace/replay tests. It is not a live job.

Bound sources at launch:

- trace helper: `141461bb9b2e6567211c8eecece6ae00eac48113b996afa1b2d10db99ce4734f`
- replay executable: `8eecf90376c0584e4b254b109faf27f38c89963d316c878451d4c7f58fdc11e2`
- trace tests: `cca0de5626e212544f9704978e7e0dd612221f37aeb9861ab38da77799f80fb5`
- replay tests: `ceddc964a58a333c29957a5c4eca9b244645b1c2cb624e177962002c65640dd1`
- protocol: `a207abe97710432cb20d3f11958250db5c7e0fab813ad186086ecd54905cd3ed`

## Live work and next action

Replay handle **57556**, PID **2070961**, is independently confirmed live and
has started inner/nominal-left frame 0. Output:
`/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/navigation_development_artifacts_v1/go2_registration_conditioning_replay_v1_attempt_001`.
Launch SHA-256:
`23dbfff75d651323a443a62b8ea319d343f6c6a4f536126749b825ee86fe1b71`.
It binds 686 sources. Read-only 292d9f rechecks every binding unchanged and
confirms its result, failure and complete-sensor-phase marker are still absent.
The three inner-arrival sensor streams subsequently complete: 1,138/1,014/576
frames, with 1,128/1,004/566 exact original decisions. Available-pose counts
remain 1,127/1,003/576; original failures at 1,127 and 1,003 are preserved.
All 2,698 recorded inner-arrival decisions are reproduced, not recovered.
Live output 373f85 reaches persistent-intent nominal-left frame 0 for both
original and balanced observers; later a00a2a reaches frame1000 with1001
available poses in each. Do not restart or edit its bound
sources while it runs. A missing observation is not a terminated replay.

The original collection supervisor **25963** remains live as PID2063013. Its
l01 child exits 0 after the completed 120-case audit, and the supervisor fully
verifies l01 before automatically launching l02. The new child is PID2071161,
with launch SHA `530fbb478586af052da3e2452ea7d4b8a478d4b360380ba507dd6b6af4ba1dea`;
its source map is exactly the original765 collection bindings. l01's verified
receipt file has SHA `526905f89159d0922075267b09bb72df42f9e6595d54fe48f70d419da906892a`,
and terminal-audit SHA is
`97be7fd5e47b0a100f809e2c68f143d1a5a429b589850d5a1ce441af95d756a0`.
Read-only 9d8b35 verifies the actual launch/audit files against that receipt.
It records120eligible/115complete schedules/5physical terminals,22positive
contact horizons,100exact nonreference prefixes,20eligible/action,no exclusions
and one retained strict boundary-visibility failure. These are two completed
training layouts, not a complete study or43independent contacts across them.
Do not launch a competing collector/auditor or alter the771 live bindings.
Later live output2657e5 reaches7/120 l02 raw prechecks.

Regression48254 is terminal successful. Observe the exact replay57556 and
collector25963 handles and the latter's l02 child, not the exited l01 PID.
When replay57556 is terminal, inspect its actual failure/result and verify all
declared sources/inputs/outputs before drawing conclusions. Full-stream rejected
candidate errors may motivate a distinct qualification or estimator experiment;
they do not authorize retrospective gate relaxation or a physical-recovery claim.
The existing low-friction dynamics failure is separate from visual support loss.

Once the original collection supervisor is truly successful and terminal, bind
its exact result SHA and run the already-reviewed matched study through its
full-cohort gates. Its definition remains
`3a1b0d78ba9bb4cc1854bc4d1079a26f21ffa57652a7664aaff6af650318956b`;
do not train on the live partial cohort. Reliable execution, online rollout and
memory/backtracking, whole novel-maze missions, deployment-valid sensing,
real-time operation and bounded hardware evidence remain unfinished. The latest
physical return result remains0/3; no unit test or replay completes that goal.
