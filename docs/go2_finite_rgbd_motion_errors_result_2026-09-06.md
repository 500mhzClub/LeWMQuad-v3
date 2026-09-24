# Finite motion errors: causal RGB contribution, physical coverage still missing

Completed 16 fixed sensor-error members on each of the three separately recorded
shadow-motion appearance arms. All 162 nominal frame outputs matched their original
recordings exactly, including the neutral arm's terminal failure. Blanking RGB
reproduced that neutral fusion/failure history exactly in every arm. This is causal
evidence for RGB point constraints in this development estimator, not a learned
navigation-policy or JEPA result. No new physics or navigation was executed.

The other important finding is a data limitation: all 124 actual-body surface
queries had zero observed floor-footprint coverage. These short motion recordings
cannot validate underbody clearance, even when their numerical reference-gap errors
are small. The next useful step is new sustained, supervised development motion
into previously observed floor, followed by turns and braking—not more attempts to
extract a clearance certificate from these same short recordings.

## Implemented and verified

New source: `lewm/finite_rgbd_error_members_development.py`.
Protocol: [finite motion errors V1](go2_finite_rgbd_motion_errors_development_v1_2026-09-06.md).
Runner: `scripts/probe_go2_finite_rgbd_motion_errors_development_v1.py`.
Output: `.generated/go2_finite_rgbd_motion_errors_development_v1_attempt_001`.

Each finite member owns the actual unchanged RGB-D/gyro/point/plane/ray-memory
pipeline. Range perturbations retain float32 and validity; overlapping gyro/force
histories share timestamp-consistent biases; the initial prior changes once; blank
RGB retains paired image/depth provenance. Members terminate independently and are
never reinvoked after failure. Original commands, joints, camera and scene are unchanged.

Eight new tests passed, including exact nominal-owner equivalence, weak-direction
prior response, cross-rate/history consistency, terminal sibling isolation, RGB
rejection, deterministic represented noise and un-clipped range faults. Eighty-two
focused tests passed. Full regression: **2,071 tests in 166 explicit files**, 193.71 s.

The separate accounting analysis checked all 2,592 member status rows, terminal
history preservation, evaluation/admission alignment and 124 physical queries.
Its comparisons use only explicitly shared admitted rows; an early stop is not
credited as a lower-error full trajectory. It also verified exact blank-RGB versus
neutral-nominal fusion/failure histories. Native scoring occurred only after each
arm's predictions were persisted. The native scorer itself was not independently
reimplemented in this turn; the underlying physical traces have the prior raw audit.

## Main findings

| Condition | Neutral | Repeated texture | Distinctive texture |
| --- | ---: | ---: | ---: |
| Nominal admitted frames / 54 | 28 | 54 | 54 |
| Nominal maximum admitted position error | 4.888 mm | 2.441 mm | 2.734 mm |
| Nominal point-complemented intervals | 0 | 18 | 18 |
| Nominal inertial-fallback intervals | 13 | 0 | 0 |
| Blank-RGB admitted frames / 54 | 28 | 28 | 28 |
| Negative gyro-bias maximum admitted position error | 5.333 mm | 10.299 mm | 9.998 mm |

All neutral members stopped. Most exhausted the unchanged proxy budget at 4.3 s;
positive independent-noise and positive combined members stopped one observation
earlier, at 4.2 s. Positive noise changed a depth-rank decision. Its lower admitted
maximum position error is therefore not evidence of a better full-motion estimator.

In each textured arm all fifteen non-blank members completed all 54 observations.
Blanking RGB removed the complementary point constraints and exactly reproduced
the neutral arm's 28 admitted frames, subsequent failure and non-reinvocation.
The repeated and distinctive arms share identical physics, so this is a matched
appearance intervention on one development layout, not three layout replications.
The relevant point tracker is engineered, not JEPA-trained.

The ±1-mrad/s gyro-bias cases reached about 5.30 mrad orientation error at the end
of the textured tapes; their maximum position errors reached 10.30/10.00 mm.
This points to attitude/calibration error as a priority for future validation;
it does not justify estimating a new safe multiplier from these maxima.

Post-anchor force-Y bias changed neutral inertial-fallback position estimates by
up to 8.888 mm relative to nominal, but did not change positions in the fully
point/depth-constrained textured members. Initial-velocity-Y perturbations had
zero position response in these recordings: the early full-rank observation
removed that velocity contribution before subsequent weak motion. The separate
synthetic test shows a nonzero response when weakness starts immediately. Thus
these physical recordings still do not validate initial-prior uncertainty during
an initially weak observation sequence.

Gyro/noise/combined members changed point-support or rank/status decisions in some
frames. Exact correspondence identities are not fully observable, so unchanged
summary signatures are not a smoothness guarantee. No derivatives or covariance
were computed, and these sixteen members do not span a continuous error set.

## Physical-query limits

Initial surfaces were retained per admitted member and queried at observations
0, 27 and 53 only while that member was alive. Actual measured joints and estimated
body poses were used, not virtual forward configurations. All 124 queries retained
zero observed underbody coverage. Their zero additional point-error setting is
finite-member isolation, not an uncertainty allowance usable by navigation.

Native scoring compared body placement against the SAME member's measured reference
plane. Those small pose-only reference-gap discrepancies are not actual-floor
calibration, evidence of common surface identity or clearance through unseen floor.
Neither these finite members nor the 0.1-mm surface range hypothesis covers the
full missing camera/kinematic/timing/slip/false-match error population. The original
two ambiguous calves and failed fresh-maze mission remain unresolved.

## Identities and next action

The launch binds 495 sources and 6,645 inputs plus native/OpenCV dependencies. The
accounting analysis separately binds its source and exact output artifacts; all
bindings were verified unchanged. Per-arm offline replay wall times were about
37.6, 91.5 and 79.1 s, overlapping regression work. These are not deployment timings.

- Launch SHA-256: `0c0cd27c661b14e1dfb6e5eb606929ae1329700931cd203239a6374065064558`.
- Result SHA-256: `d8146a104bb5408234679ebf77cc592e30fc1b5883f33d78821787460f14299e`.
- Shared-row analysis SHA-256: `6263944993c1c881d8ba9bb709bafd84db536ecf3d046f61c1eff2ccd7e13534`.

Next freeze a new supervised development motion protocol with sustained forward
travel sufficient to place the full body on previously observed floor, explicit
turn/brake segments and stopping tails, plus an initially weak-depth condition.
Use separate declared fitting/validation trials and retained failures. Sensing and
prediction remain deployment-valid; native poses/contacts only supervise collection
and score results. This is action-response and uncertainty-data collection, not
permission inferred from an unvalidated prospective controller envelope.

Evaluate relative body/surface error directionally with shared pose/history errors,
not only a sum of isotropic global radii. Add calibrated camera/kinematic/timing
uncertainty before relying on positive gap margins. Integrate one consistent
floor/non-floor evidence consumer only after validation. Then complete genuine
discovery/backtracking/return, real-time execution, matched JEPA/supervised/geometric
comparisons with actual multistep rollout and memory contributions, independent
layouts/seeds/robustness and bounded real-platform evidence. The goal remains active.
