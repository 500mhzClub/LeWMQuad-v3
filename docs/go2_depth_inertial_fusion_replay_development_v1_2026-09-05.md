# Depth–inertial weak-subspace replay V1

This is a new offline development experiment, not a retry, rescore, or rescue of
any predecessor navigation attempt. The original outcomes stay unchanged.

## Fixed population and question

Replay the complete sensor histories of both trials from each of the completed
measured-region, measured-line/integral, release-aware, observable-hold, and
depth-floor-hold V1 studies: 10 trajectories and 3,610 observations. These are
repeated development layouts, not independent evaluation. They include weak
lateral and vertical constraints as well as fully constrained turning motion.

Feed the fusion kernel only original RGB/body policy packets and their previously
audited causal depth/gyro observer records. It may not access raw world poses,
future commands, wall geometry, or the evaluator. This experiment does not
recompute depth registration; its exact predecessor audit is the observation
identity witness. The live depth-wrapper contract is tested separately.

The question is whether the fixed weak-subspace integration preserves accurately
constrained motion and supplies useful short missing components across the whole
recorded population, not only the six-frame feasibility example. Ground truth
is loaded separately for scoring after all estimates for a trajectory exist.
The original trajectories end at their recorded stops: replay cannot establish
continued-navigation success after a hypothetical intervention.

## Fixed estimator and checks

The kernel preserves original rank/status and fills only the weak subspace with
velocity initialized from previous full depth motion and causal accelerometer/
gyro integration. It rejects missing velocity, invalid histories, identity/time
faults, nonorthogonal subspaces, and increments beyond 0.15 m per 100 ms. Full
depth recovery resets the consecutive weak duration, not accumulated pose error.

Declare before replay: 0.5-mm depth-step scale, 5-mm/s initial velocity scale,
0.02-m/s² unmodelled acceleration scale, 0.00001-rad orientation-step scale,
multiplier 3, and 8-cm position-scale budget. These are **uncalibrated development
proxies**, not covariance, coverage probabilities, guaranteed bounds, or safety
certificates. No navigation consumer may infer authority from the budget flag.

Report every frame, original rank, predicted versus constrained kind, estimated
position/step, independently measured error, declared scale, scale exceedance,
and budget status. The fixed descriptive acceptance checks are: all frames
processed, maximum step error ≤1 cm, maximum/final position error ≤5 cm, no error
above the declared position proxy, and no budget exhaustion. Report per-trajectory
checks and all failures; aggregate success is not a hardware/generalization claim.

## Analytic falsification and remaining limits

Tests cover constant velocity, acceleration, rotated subspaces, rank-one/two
constraints, immutable input evidence, history faults, absent initialization,
weak-duration budget exhaustion, and nonshrinking error on recovery. A deliberate
0.2-m/s² bias exceeds the acceleration assumption and demonstrates that actual
error can exceed the proxy while its budget flag remains true. Retain this
counterexample: nominal replay cannot validate an unknown-bias safety bound.

Before control integration, uncertainty must influence transported ray evidence
and stopping, and bias/gravity/attitude assumptions need an explicit operating
envelope or additional observation. Also address gait translation during turns;
successful observer replay alone does not resolve that navigation defect.

## Custody and execution

Use only the five exact unsealed roots named by the runner. Bind inherited source
and terminal identities, all newly imported local source/test/protocol paths, and
each consumed artifact before processing. Recheck bindings at completion. Write
one fresh named output, retain terminal failures, and never overwrite old runs.
No GPU, training, hardware, sealed benchmark, or navigation execution occurs.
