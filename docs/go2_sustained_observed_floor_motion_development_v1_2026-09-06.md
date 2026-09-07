# Sustained observed-floor motion V1: supervised data collection

Purpose: obtain actual sustained motion into previously observed floor, turn/brake
response and an initially weak-depth condition. Prior short recordings supplied
zero underbody coverage in all 124 queries. Do not call numerical extrapolation
clearance. This protocol collects fitting and reserved development-validation
trials; it does not fit a model, validate a prospective safety envelope, select
navigation commands, train JEPA or solve a maze.

## Fixed scene, trials and stimulus

Two separately instantiated trials: fit (physics seed 2026090622, appearance seed
2026090624, spawn x/y/yaw [-.5,-.3,0]) and validation (2026090623, 2026090625,
[-.5,.3,.04]). Both spawn at body z=.375 and use the unchanged learned gait,
checkpoint actuator gains, collision geometry and 2-ms/20-ms/100-ms physical/
policy/command clocks. The wall is centred [4.5,0,.3], size [.08,16,.6], yaw0;
the physical floor is unchanged, with a separate 32-m visual floor. Distinctive
appearance affects only visual surfaces, not collision or materials. Declared
body/capture workspace is [-8,8]². This is one simple development geometry,
not independent novel mazes or a generalization experiment.

The wide wall and floor are intended to give only two independent surface-normal
directions initially; actual sensor-estimated ranks must be reported. Do not
assert weak depth from geometry alone or alter the scene if it is not observed.
Likewise, sustained travel is intended to put all body footprints on previously
observed floor; verify coverage from actual recording before claiming success.

After fifteen settling ticks (anchor 1.5 s), execute these fixed 10-Hz segments:

| Segment | Ticks | [forward m/s, lateral m/s, yaw rad/s] |
| --- | ---: | --- |
| Initial observation hold | 10 | [0,0,0] |
| Sustained forward | 150 | [.12,0,0] |
| Forward brake | 10 | [0,0,0] |
| Left turn | 50 | [0,0,.25] |
| Left brake | 10 | [0,0,0] |
| Post-turn forward | 30 | [.10,0,0] |
| Post-turn brake | 10 | [0,0,0] |
| Right turn | 50 | [0,0,-.25] |
| Right brake | 10 | [0,0,0] |

Then five extra zero-command tail ticks. Total 330 stimulus +5 tail ticks,
expected 17,500 physics samples and 336 RGB-D observations per complete trial.
No adaptive extension, command escalation, restart or replacement attempt. Retain
incomplete physical trials. Native stops terminate that trial's physics immediately;
they do not trigger a recovery tail or renewed actuation.

## Interfaces and stopping

Reuse the reviewed actual RGB/depth renderer, 50-Hz body sensing, 500-Hz gyro,
per-step contact attribution and fixed-camera/native identity checks through a
NEW explicit initializer, never a global patch of a predecessor's pack/builder.
Verify exact specification before native construction. The initial setup region
is used only for initial evaluator admission and never passed to navigation or
propagated as a map. Preserve measured-foot identity and exact checkpoint gains.

External native supervision enforces inherited disallowed-contact/fall/tilt stops,
non-foot ground-contact exclusion, base speed<=.3 m/s and the declared capture
domain at every active 2-ms sample. These guards supervise data collection, not
a deployable policy. Known scene and native states are not estimator inputs.

The original ShadowObserver runs without commanding the robot. Record its original
proxy-budget/registration failures and never invoke a failed owner again. Physics
may continue on the predeclared externally supervised stimulus after a shadow
failure, as in the predecessor collection; this is not navigation continuing after
its safety gate failed. Do not reset its accumulated point/pose scales. The long
weak-depth exposure may legitimately exhaust its uncalibrated point-error budget.

Collect all raw contacts, physical states, commands, sensor histories, paired
RGB/depth/native-depth images, camera clocks, native/visual identities, native
guard rows, segment timing and shadow statuses. Validation data must not be used
to fit model coefficients or choose a clearance multiplier. Acquisition integrity
can be audited independently; freeze any later fitting procedure before scoring
validation performance.

## Freeze, outputs and required follow-through

Before physical launch, bind protocol, scene definition, new initializer/session,
collector and focused tests plus their narrow import closure to the preceding
finite-error source/input identities. Verify native/OpenCV and completed predecessor
accounting. Use an exclusive root:
`.generated/go2_sustained_observed_floor_motion_development_v1_attempt_001`.
The fit and validation directories are the two predeclared trials, not retries.
Write explicit partial artifacts and a terminal infrastructure-failure record on
unexpected errors. A physics stop is a recorded trial outcome, not an infrastructure
retry. Verify sources before each trial and after acquisition. No source edits
after launch, whole-tree export, sealed access, GPU training or hardware actuation.

After acquisition, independently audit raw sensor/contact/command/geometry/timing
evidence; compare initial observed depth rank, actual translation and turns,
stopping tails and observed full-body floor footprints. Do not use native pose to
repair a failed causal estimator. Geometry-only evaluator queries can diagnose
whether the scene supplies floor evidence, but are not causal navigation results.
Insufficient travel, coverage, turning, sensing or physical integrity stays a failure.

Then fit/validate prospective motion and directional shared relative-error models
using only the declared fitting role; account for camera/kinematic/timing errors.
Retain uncertainty stops. Complete common floor/non-floor integration, actual
exploration/backtracking/return, real-time operation, matched JEPA/supervised/
geometric and genuine multistep/memory comparisons, independent layouts/seeds/
robustness and bounded hardware evidence. This collection cannot close the goal.
