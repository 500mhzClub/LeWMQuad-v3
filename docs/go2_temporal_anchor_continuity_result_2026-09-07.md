# Temporal-anchor continuity recovers three brief tracking interruptions

The fixed candidate returns **7,433/7,433 poses**, versus **7,400/7,433** for the
original. Three single-frame measured-increment bridges each rejoin a retained
anchor on the next observation. All7,400 shared pose dictionaries are exactly
unchanged;33 additional frames become available and none are lost.

This is positive evidence on reused development recordings, not independent
validation, continued physical execution, a learned-policy improvement or maze
navigation success. The latest actual simulated room-return result remains0/3.
Do not repeat or relabel the two failed flow-initialization experiments.

## Verified result

| Stream | Frames | Original available | Candidate available | Bridge frame / anchor rejoin | Bridge absolute / incremental position error (mm) |
| --- | ---: | ---: | ---: | --- | ---: |
| Inner left |1,138|1,127|1,138|1127 /1128|10.083 /0.688|
| Inner right |1,014|1,003|1,014|1003 /1004|8.305 /0.161|
| Inner low friction |576|576|576|None|Unavailable: no bridge|
| Intent left |2,413|2,413|2,413|None|Unavailable: no bridge|
| Intent right |1,820|1,809|1,820|1809 /1810|6.718 /0.316|
| Intent low friction |472|472|472|None|Unavailable: no bridge|

The33 recovered frames have mean8.413mm and maximum10.083mm position error.
There are7,400 both-available frames,33 candidate-only frames, zero original-only
frames and zero neither-available frames. The three bridges themselves are only
three observations—not33 independent recovery events. Each recovery prevents
the original terminal latch from suppressing the following ten recorded poses.

Rejoin disagreement is0.748/0.163/1.337mm. It is a correlated consistency check,
not an absolute-error estimate: absolute errors at the bridge observations are
substantially larger. The synthetic common-depth-bias counterexample still
applies. The10-frame operational bridge allowance was not exercised naturally;
all three recorded bridges last only one frame.

On shared frames, pose dictionaries and paired error summaries match exactly.
Across the expanded population, inner-right maximum error increases from8.963mm
to9.248mm because new observations are scored. The overall candidate maximum is
11.275mm; it is an empirical development error, not a calibrated bound.

Candidate available-frame observer medians are53.0–62.9ms versus38.2–46.0ms for
the original, with a candidate maximum168.8ms. These are observer-only timings
under concurrent collection, excluding packet loading and control. They do not
establish a100ms full-loop deadline; earlier full-loop timing already exceeded
that deadline. The extra registration has a real compute cost.

## Completion and independent verification

- Replay86816, PID2088443, terminates exit0 (8a4cdc). The process is subsequently
  absent (508f3c). All six sensor streams are saved before native evaluation;
  no estimator, acceptance threshold or input source changed during execution.
- Independent verifier31101 terminates exit0 (37da63). It authenticates701
  sources, predecessor raw/native/original-witness bindings and27 output
  artifacts, then independently reconstructs all7,433 sensor/evaluation/original
  row joins, clocks, first failures, availability, anchor/bridge provenance,
  complete bridge spans and absolute/consecutive-estimate error distributions.
  It uses separate quaternion rotation calculations, not the runner's angle or
  continuity-audit functions. Output hashes verify again afterward.
- Read-only2b4c42 additionally checks all7,400 complete shared pose dictionaries
  exactly and reconstructs the33 recovered-frame position errors.
- The runner/runtime full regression27186 passed3,396 tests across247 explicit
  files in318.33s. Its durable JUnit SHA is
  `1d84946225948481a807da6feeb67167a06c09400ffe7de72bf23e5332465fbd`.
- The separately implemented verifier's initial17 tests pass; after a direct
  stream-helper sensor-phase guard, final37823 passes18 tests in7.74s. These
  test corruption, frame/reference identities, denominators, clocks, population
  length, math and native-access ordering. They are additional focused tests,
  not a new combined3,414-test regression invocation.

Read-only verification is reproducible with the declared development interpreter:
`scripts/verify_go2_temporal_anchor_continuity_v1.py --result-sha256 4cf2877bf4a400794c0a5c481773bb830c8e7d2f01210582b46e32cc8e144ebe`.
Verifier source SHA:
`b1c9f7885c52d5ec12b1a4abc0eaf4e1d76a08cd1bab006bb56a5ede4d05d296`.
Verifier-test SHA:
`044548e2e106e80408ceebf09338db813a9514bc2e08c16b58fe4c08401db6f4`.
Comparison tolerances1e-12m/1e-10rad concern independent numerical calculations,
not changed physical acceptance or sensor uncertainty.

Output root:
`/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/navigation_development_artifacts_v1/go2_temporal_anchor_continuity_v1_attempt_001`.
Launch SHA:
`d774dddf1b012dc0f69116d7e11e0f3307c0c5e549e807ef65e227b0d240b143`.
Complete sensor-phase SHA:
`8fca9fcda7ac3c14e335b69222e2e6126e8fe36a629444018e6871645f7f4574`.
Result SHA:
`4cf2877bf4a400794c0a5c481773bb830c8e7d2f01210582b46e32cc8e144ebe`.

## What this establishes—and does not

Retained-anchor failure need not mean that current relative motion is
unobservable. In these three interruptions, the previous image supplies a
qualified short-baseline measurement and an older anchor becomes usable again
on the next frame. That is a different capability from changing the initializer
for long-baseline correspondence, whose two completed candidates failed early.

The recordings were generated by the original controller, including its stop/
drain behavior after failure. They cannot show the observations or dynamics
that would result from continuing the turn under this candidate. Nor do three
one-frame recoveries validate long unanchored motion, new appearances, sensor
bias/dropouts, self-occlusion, hardware calibration or real-time execution.
The lower-friction control failure remains a dynamics problem despite tracking.

## Next concrete work

1. **Build and freeze an independent observation challenge before adoption.**
   Inspect the existing raw RGB-D/gyro acquisition and validated renderer/session
   interfaces; reuse them without changing frozen sources. Acquire genuinely
   different scene/start observations with both turn directions, maze-scale
   turns, translation and stopping, including the lower-friction condition.
   Verify actual scene/initial-state/prefix differences, not merely new seed
   labels. Run original and frozen continuity observers in shadow on identical
   sensor tapes. Predeclare the complete population, artifact/resource bounds,
   stop-only supervision and task-derived accuracy criteria before collection.
   Preserve the active independent-learning collector and avoid a competing
   collection launch until its resource/scheduling constraints are resolved.
2. **Challenge the right failure modes.** Distinguish unavailable old anchors
   from missing/invalid current RGB-D: this method addresses the former only.
   Include longer anchor-loss spans, contradictions, repeated appearance,
   depth/gyro error and current-sensor dropout with fixed declared perturbations.
   Label these as stress tests unless sensor distributions are calibrated from
   hardware. Keep unknown/error accumulation explicit. Do not choose the bridge
   duration or weaken checks using these exposed outcomes.
3. **Measure the full computation path, then test fresh closed-loop execution.**
   Profile acquisition, feature extraction, both registrations, planning and
   control separately. Preserve measurement timestamps and count missed
   deadlines; paused physics is not real-time proof. A separately frozen fresh
   nominal/lower-friction turn-and-return test is needed before physically
   executed branch choice/backtracking. Retain every hold/home failure and the
   old0/3 result. Tracking recovery alone cannot fix a wrong dynamics model.
4. **Finish the independent JEPA experiment already in progress.** The original
   supervisor has verified l00–l03:480 eligible collection trials across four
   layouts. It has started l04 with unchanged765 child-source bindings. Complete
   all12 receipts, then run the reviewed36-fit JEPA/supervised/input-ablation
   study with its unchanged786-source definition. Require useful prediction
   against the empirical action/time baseline before claiming a learned
   contribution. Subsequently separate predictive-training, online-rollout and
   memory effects using matched sensors, gait, execution and budgets.

Independent novel-maze/training-seed evidence, actual memory/backtracking/home
association, deployment-valid sensing and bounded hardware evidence remain
required. This positive replay result advances a prerequisite, not the full goal.
