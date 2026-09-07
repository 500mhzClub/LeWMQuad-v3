# Completed registration replay: support loss, not demonstrated pose recovery

The fixed nine-stream replay completes successfully with the original estimators,
gates and failures unchanged. It reproduces all 7,383 recorded physical decisions
and all 9,410 previously saved persistent-intent frontend observations exactly.
All 12,138 frame/arm observations are retained, including 558 unavailable poses.
This is diagnostic progress on local execution, not a new physical return or
evidence of JEPA/navigation benefit. The latest simulated room return stays 0/3.

## Verified completion

- Replay session 57556 terminates exit 0 (52ce89); PID 2070961 is absent in the
  subsequent process check. Do not restart it.
- Read-only verification 42433 terminates exit 0 (be999b). It verifies all 686
  source bindings, predecessor native/input bindings, balanced-replay witnesses,
  all 35 declared output bindings and the terminal result. It also joins every
  sensor/evaluation row in order, checks complete per-stream denominators and
  checks all nine summary files against the terminal result.
- Launch SHA-256:
  `23dbfff75d651323a443a62b8ea319d343f6c6a4f536126749b825ee86fe1b71`.
- Complete sensor-phase marker SHA-256:
  `1e34a13607b97f0285b8ff70ed76b3937e940eb6b58c9eafd367a366502f00aa`.
- Result SHA-256:
  `dc392027aa0b9e65566b436257485369f2cdf468daa80acc43908d436415d28e`.
- Exact output root:
  `/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/navigation_development_artifacts_v1/go2_registration_conditioning_replay_v1_attempt_001`.

All sensor estimates were persisted and bound before native coordinates entered
the separate evaluator. No new simulation, fit, control decision, gate change,
reinitialization, reference substitution or post-failure extrapolation occurred.
The preceding 241-file regression remains 3,259 passed; no source code changed
during this completion/analysis turn.

## Whole-stream outcomes

Errors below are three-dimensional position errors against the recorded native
pose in the initial body frame, in millimetres. They are maxima over *available*
poses, not guarantees over the missing frames or future operation.

| Recorded stream | Frontend | Available / all frames | First unavailable frame | Maximum available error, mm |
| --- | --- | ---: | ---: | ---: |
| Inner left | Original | 1127 / 1138 | 1127 | 10.341 |
| Inner right | Original | 1003 / 1014 | 1003 | 8.963 |
| Inner low friction | Original | 576 / 576 | None | 5.225 |
| Intent left | Original | 2413 / 2413 | None | 9.630 |
| Intent left | Balanced | 1888 / 2413 | 1888 | 9.173 |
| Intent right | Original | 1809 / 1820 | 1809 | 11.275 |
| Intent right | Balanced | 1820 / 1820 | None | 8.998 |
| Intent low friction | Original | 472 / 472 | None | 4.276 |
| Intent low friction | Balanced | 472 / 472 | None | 3.456 |

The balanced frontend still loses 514 available observations overall on the
paired intent recordings (4,180 versus 4,694). Its smaller maximum error on a
shorter, failure-truncated left trajectory is not an accuracy improvement on
matched rows. Never select the frontend by trajectory identity or native error.

There are 11,617 candidate calls: 11,571 qualified, 29 grid-only rejected and
17 other rejected. Every grid-only rejected candidate converges; maximum error
is 10.764 mm. Only two other-rejected candidates converge, with maximum error
6.585 mm; the other 15 have no converged diagnostic pose and must not disappear
from the denominator. Maximum qualified-candidate error is 11.275 mm. Candidate
calls, correlated frames and independent physical trials are different units.

## What changed our understanding

The final primary candidates show substantial matched support despite rejection:

| Failure | Unique mutual matches → flow → paired depth → consensus | Final grid cells, reference/current | Primary diagnostic error, mm |
| --- | --- | --- | ---: |
| Inner left, frame 1127 | 161 → 104 → 95 → 74 | 5 / 6 | 9.702 |
| Inner right, frame 1003 | 126 → 67 → 63 → 48 | 6 / 5 | 8.184 |
| Balanced intent left, frame 1888 | 147 → 88 → 87 → 68 | 5 / 7 | 6.787 |
| Original intent right, frame 1809 | 92 → 48 → 48 → 24 | 6 / 5 | 6.585 |

For the first three, fraction and displacement gates pass. The last also fails
the inlier-fraction gate (0.50 versus 0.60 required): a grid-only change would
not rescue that primary candidate. Across these examples, flow/consensus remove
more matches than depth lifting. This localizes where support is lost; it does
not establish whether discarded matches were correct or why flow rejected them.

The evidence now directly shows that the latest rejected nominal candidates
were not grossly wrong on these recorded frames. It does **not** show that they
would remain correct after new keyframes, altered robot motion or further turns.
The estimator's latched failure also means no counterfactual continued tracking
was tested. These reused, ideal-sensor room recordings contain no large-error
converged diagnostic candidate with which to validate rejection specificity.
They are unsuitable by themselves for selecting a replacement confidence gate.

For correct static correspondences and fixed rotation, translation information
is N*I, so the occupied-cell count is not an algebraic translation-rank test.
But this does not validate correspondences or bound common error. Grid-rejected
candidates have gyro-angle translation sensitivity of 2.320–3.665 m/rad. A
0.01-rad common orientation perturbation can therefore have a first-order
translation sensitivity of roughly 23–37 mm. This is an illustrative conditional
sensitivity, not a measured gyro error or calibrated uncertainty bound.

Instrumented available-frame observer medians span 51.1–67.7 ms; registration
component medians are about 21–22 ms. These are traced offline observer calls
overlapping collection, not the full observation/control pipeline or real-time
hardware evidence. The earlier 143–147 ms paused-physics control measurements
remain unresolved. This evaluator reports position, not a complete orientation,
calibration, timing or sensor-robustness qualification.

## Next execution experiment

Pursue a distinct gyro-seeded temporal correspondence path, initially outside
control. The hypothesis is that usable image support is being lost by requiring
mutual descriptor detections in both frames before tracking. Reference features
can instead seed bidirectional image tracking using measured relative gyro
rotation and observed depth, without an executed-command displacement prior.
This is a hypothesis to test, not a concluded cause or a validated replacement.

1. Implement the bounded correspondence path in new files. Preserve ordinary
   RGB/depth/clock validation, pixel-domain checks, duplicate suppression,
   bidirectional tracking, depth consistency and existing rigid-consensus/grid/
   increment gates. Do not modify frozen observers or add native-pose input.
   Log failures and support at every stage. A rotation-only seed is imperfect
   under translation and must not be treated as a measured correspondence.
2. Test known image/depth transforms and negative cases: repeated textures,
   occlusion/disocclusion, insufficient texture, depth discontinuities, wrong
   associations, common depth bias, gyro perturbations and timestamp mismatch.
   Retain false acceptance, unavailable-pose and error counts separately. The
   existing six-cell gate is itself not an uncertainty certificate.
3. Replay complete original development streams with a fixed candidate policy,
   keeping every failure and evaluating only after sensor estimates are saved.
   Add orientation error and paired common-frame error reporting. These streams
   remain development diagnostics, not independent validation of the new method.
4. Before closed-loop adoption, freeze the method and evaluate additional
   scene/sensor challenges not used to choose it, with real metric error,
   availability and uninstrumented end-to-end latency. If the correspondence
   change fails, retain it and reconsider complementary geometric constraints;
   do not substitute a retrospective lower threshold or per-scene frontend.
5. Only then run a new local physical-execution experiment. Keep the separate
   low-friction dynamics failure in scope: better tracking alone cannot solve it.

In parallel, continue the original 12-layout collector. At verification 777664,
supervisor PID 2063013 and l02 child PID 2071161 remain live; stdout 041c0f has
reached 25/120 l02 prechecks. l00/l01 remain the two completed audited layouts.
The 36-fit matched study must still wait for all 12 successful receipts and use
the reviewed 786-source definition, not a partially collected training cohort.
Its result must establish whether JEPA, RGB and history help relative to matched
baselines before claiming learned online planning benefit.

Useful online rollout, memory/backtracking, whole unfamiliar-maze missions,
deployment-valid sensing, real-time operation and bounded hardware evidence
remain unfinished. No conclusion here changes benchmark custody or final roles.
