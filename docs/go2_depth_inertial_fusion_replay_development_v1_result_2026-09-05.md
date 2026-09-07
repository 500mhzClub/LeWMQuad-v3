# Depth–inertial fusion V1: complete replay, limited accuracy evidence

The new offline replay completed all 10 trajectories / 3,610 observations.
Independent verification passed all 3,610 error, depth-component, and position-
composition checks. All fixed descriptive replay checks pass. **This does not
qualify the uncertainty model or establish continued navigation.**

The preceding status-only goal turn made no implementation progress. This turn
implemented and tested fusion, completed the fixed replay, independently checked
its reductions, and found a velocity-initialization defect that changes the next
implementation. The overall scientific goal remains active and unachieved.

## What was implemented and verified

The separate kernel combines audited causal depth/gyro motion constraints with
RGB/body sensor packets. Full depth constraints remain unchanged. Weak components
are explicitly inertial predictions; original rank and missing full displacement
remain intact. Clock/identity/history faults latch. Accumulated position-error
proxies do not shrink on rank recovery. No navigation consumer is installed.

Test session 73814 passed 1,182 tests across 110 files in 79.02 s. Focused session
32858 passed 32 checks, including the new kernel, evaluator and existing depth
observer tests. Independent-verifier session 31716 passed seven further tests,
including deliberately corrupted metrics, components, positions and labels.
These seven were run separately, not as a claimed new full-suite result.

Preflight 50549 passed 322 source / 3,660 consumed-artifact bindings and the fixed
10-trajectory / 3,610-frame population. Earlier preflight 60999 rejected the
inherited exact external URDF identity under the repository-relative verifier;
the runner used the existing narrow URDF-aware verifier before any launch/output.
Replay 24603 completed, exit 0. Verification 72250 passed, exit 0. No process from
this study remains live. All launched sources and results remain unchanged.

## Complete recorded population

| Development study | Frames | Weak intervals | Worst position error |
| --- | ---: | ---: | ---: |
| Measured region | 511 | 6 | 0.835 mm |
| Measured line/integral | 782 | 0 | 0.939 mm |
| Release aware | 864 | 6 | 7.501 mm |
| Observable hold | 402 | 0 | 0.577 mm |
| Depth-floor hold | 1,051 | 6 | 1.296 mm |

Across all trials, worst step error is 1.27121 mm and worst/final position error
is 7.50116 mm. There are no position-proxy exceedances or proxy-budget stops in
this population. All 18 weak intervals occur in three six-frame stop/release
sequences. Seven trajectories never require a predicted weak component. The
ten trajectories reuse two development layouts and are not independent samples
of novel-maze generalization. Original whole-task success remains 0/2 in the
latest physical simulation; no original failed outcome is rescored as success.

The verifier uses a separate quaternion quadratic-form implementation for
physical error reduction. It checks every constrained component, position
composition, rank/prediction label, uncertainty monotonicity, frame population,
per-frame error and aggregate check. It does not claim a new depth-registration
replay: those records are bound to their completed predecessor raw audits.

## Why the uncertainty model is not validated by these passes

The analytic 0.2-m/s² bias case exceeds the declared 0.02-m/s² acceleration
assumption and causes actual error beyond the position proxy even while the
budget flag remains true. Neither bias observability nor calibrated coverage is
established. Initial force magnitude alone also does not prove stationary gravity.

Additional read-only physical-reference diagnostics expose nominal limitations:

- Diagnostic 9eb6e3 compares the estimated interval-average acceleration with
  measured physical velocity change. Errors exceed the declared 0.02-m/s² scale
  on 94 of 3,600 intervals, reaching 0.03976 m/s². All 18 actual weak intervals
  stay below 0.02 m/s², but that does not establish a general operating envelope.
- Diagnostic 00cfa9 compares endpoint velocity with the physical reference. On
  1,527 of 3,582 full-depth intervals, error exceeds the assumed 5 mm/s scale;
  the maximum is 47.794 mm/s. At the three entries into weak motion, errors are
  0.526, 34.342 and 6.705 mm/s. Thus the velocity assumption is already exceeded
  at two actual weak entries, despite no aggregate position-proxy exceedance.

These are scales/proxies, not asserted deterministic bounds. The diagnostics
nonetheless rule out treating them as validated covariance or a safety bound.
The accumulated depth proxy can conceal a weak velocity assumption in a short
position-error check. Do not merely enlarge a constant and declare calibration.

## Identified implementation defect and bounded feasibility evidence

V1 approximates endpoint velocity as displacement/T plus mean acceleration*T/2.
That expression assumes constant acceleration within the interval. The five
causal 20-ms accelerometer samples already contain timing information discarded
by the mean; the gait is not constant-acceleration over 100 ms.

For acceleration a_i constant within each 20-ms bin, T=0.1 s and bin midpoint
t_i, endpoint velocity is displacement/T + sum(a_i*dt*t_i/T). The five endpoint
correction weights are [0.002, 0.006, 0.010, 0.014, 0.018] s. Forward displacement
from an initialized velocity is v_start*T + sum(a_i*dt*(T-t_i)), with weights
[0.0018, 0.0014, 0.0010, 0.0006, 0.0002] s². Rotate individual measured force
samples consistently, remove the declared gravity hypothesis, and retain both
moments rather than replacing them with one mean.

Read-only probe 37940 applied this causal moment calculation to all frames of
the three weak trajectories. Physical references were opened only after each
prediction pass. A preceding diagnostic command failed on a module-name typo
before processing inputs; it created no artifacts and is not a scientific run.

| Weak trajectory | V1 final error | Moment-probe final error | Endpoint velocity error before weakness, V1 → probe |
| --- | ---: | ---: | ---: |
| Measured-region south | 0.237 mm | 0.309 mm | 0.526 → 0.617 mm/s |
| Release-aware south | 7.501 mm | 2.515 mm | 34.342 → 4.561 mm/s |
| Depth-floor-hold north | 1.296 mm | 0.666 mm | 6.705 → 1.322 mm/s |

This is a posthoc feasibility probe, not a launched successor, independent
validation, calibrated uncertainty, or continued-navigation result. Preserve the
slight worsening in the first trace and all original V1 outputs.

## Next execution steps toward the actual scientific goal

1. Implement a separately named moment-aware fusion successor. Test early versus
   late acceleration with equal interval means, switched acceleration, rotating
   gravity, weak-subspace changes, unavailable initialization, dropout, and
   long weak intervals. Verify endpoint velocity and displacement against an
   independently integrated fine-time reference. Preserve the launched V1.
2. Explicitly validate an operating envelope for depth, velocity, bias and
   attitude error. Use new trajectories with extended weak geometry and varied
   gait phase/actions; the three short stop traces cannot establish it. Keep
   unknown-bias and prolonged-weakness failures as required negative controls.
3. Couple pose uncertainty to transport of historical ray evidence and stop
   decisions. Never feed a predicted component to the old rank-3-only memory
   as if it were observed. Current-view evidence and historical transport need
   different treatment; an uncalibrated budget flag is not clearance authority.
4. In the same navigation successor, account for turning-induced gait translation
   through observed-space positioning and action-conditioned prediction. Keep
   the nominal-volume and native-contact guards. Avoid an indefinite sequence
   of controller constants without a learning comparison.
5. Compare matched geometric/empirical dynamics, supervised and JEPA turning
   predictors on identical sensor/action/data budgets. Test changed action
   choices and physical outcomes, then resume full discovery/return experiments,
   memory and genuine multistep-rollout ablations, independent layouts/seeds,
   sensor robustness and bounded hardware tests when available.

## Exact output identities

Root: `.generated/go2_depth_inertial_fusion_replay_development_v1_attempt_001`.

- launch.json: `0704e85130bab1ae7be26e4e2909fdf1ae6b3f7bc0b04f50613afebd8696320e`
- result.json: `ad958582a364da3b04464159c29dadebb5501e4ccdafc687d45bd2d322d1974c`
- independent_verification.json: `f607a8ffa326181922a3b0231a353607a2aa23728f598c59576dfd0b0b070909`
