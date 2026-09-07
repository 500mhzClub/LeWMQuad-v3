# Sensor-error propagation must include depth registration

This work implements paired causal sensor-error propagation and tests it on
recorded development packets. It does not complete navigation, calibrate sensor
uncertainty, or authorize smaller clearance envelopes. The preceding user-facing
status turn was no progress; this implementation turn produces new code and a
counterexample that changes the required uncertainty model.

## Implementation

`lewm/correlated_moment_sensitivity_development.py` provides two explicit
development diagnostics. Each source column denotes the same unit-variance
latent variable throughout the history. Loadings specify its physical effect
on each sample. Reusing a source therefore retains temporal and cross-channel
correlations; different source columns assert zero cross covariance. Those are
declared modeling assumptions, not learned or calibrated distributions.

The conditional variant propagates paired perturbations through the existing
fast-gyro integration, specific-force transport, initial gravity normalization,
velocity initialization and moment-aware observed/weak-subspace updates. It
accepts explicit projected-depth and weak-basis perturbations while otherwise
conditioning on registration. The raw RGBD variant instead recomputes actual
surface eligibility, correspondence matching, translation registration, weak
directions and fusion from each perturbed raw packet. It also accepts explicit
range-error loadings and preserves depth quantization and missing pixels.

Both variants retain named pose snapshots and construct the current/stored joint
pose covariance from their shared source factors. The existing point-transport
Jacobian then includes the cross blocks. Sensor values and error loadings must
remain identical where histories overlap; fast and slow gyro samples must also
share their error factors. Sensor faults latch and disable retained queries. A
raw perturbation that changes accepted registration rank invalidates the local
smooth approximation and latches a fault. Stable rank alone is not proof of
differentiability, so finite-difference step checks remain necessary.

These are offline numerical reference calculations, not an efficient online
covariance estimator. No navigation consumer or clearance radius was changed.

## Actual-packet counterexample

Read-only diagnostic session 58330 processed 181 original north-trace packets
with the previously declared narrow-depth mask at ticks 80–139. Two explicitly
assumed sources were a body-y accelerometer bias with scale 0.02 m/s² beginning
at tick 80 and a shared yaw-gyro bias with scale 0.001 rad/s present throughout.
Initial gravity and all repeated fast/slow samples used their actual causal
histories. No physical trajectory was rerun or rescored.

The conditional calculation agreed at difference steps 0.001 and 0.0005 to
1.10e-11 in its pose-factor entries. Nevertheless, rerunning registration under
the gyro perturbation changed the position sensitivity substantially:

| Tick | Conditional gyro position factor, y | Raw-registration gyro position factor, y | Norm of omitted position response |
| --- | ---: | ---: | ---: |
| 79 | +5.31 mm | −8.80 mm | 14.52 mm |
| 93 | +6.59 mm | −9.74 mm | 16.66 mm |
| 140 | −9.57 mm | −33.31 mm | 23.93 mm |
| 180 | −8.52 mm | −38.17 mm | 29.77 mm |

These are response factors per declared unit latent source, not measured physical
position errors or confidence bounds. No perturbed registration rank changed.
Thus a stable rank and a numerically stable conditional derivative still miss
the coupled gyro/registration response. The conditional model cannot be used as
the complete pose-error model merely because its matrix is positive semidefinite.

## Raw-registration successor check

Session 7705 completed the same 181 declared diagnostic inputs using the raw RGBD
variant at both difference steps. Every nominal fusion record matched an
independently instantiated existing observer exactly. Its raw gyro response
reproduced the prior counterexample. Maximum difference-step disagreement was
1.103e-7 in pose-factor entries; this is local numerical evidence for these two
sources on this trajectory, not general linearization or covariance calibration.

For a query at body coordinates [1, 0, −0.3] m relative to the retained tick-79
view, the assumed two-source point standard deviations at tick 93 were about
[0.299, 17.449, 0.809] mm. Discarding the cross blocks changed them to
[5.002, 17.423, 0.842] mm: it increased two components but slightly decreased
one. Independent endpoint uncertainty is not universally conservative.
At tick 180, the two-source relative y standard deviation reached 362 mm.
These post-budget outputs are diagnostics; the existing required stop at tick
93 remains in force. No travel beyond the stop was approved.

The pair of raw diagnostics (two step sizes, five observers each) took a median
141.6 ms and maximum 205.5 ms per packet while the regression suite was running.
This excludes acquisition and normal control work and is not a runtime latency
claim. Depth-noise loadings were zero in this actual-packet probe; only the two
declared bias hypotheses were exercised. Depth-error propagation was separately
tested on synthetic room observations, not calibrated on this trace.

## Verification and remaining work

The focused suite passed 28 tests in session 29684. Tests include analytic
persistent-bias growth and recovery, initial gravity/bias confounding, gyro
cross-pose cancellation, shared versus independent depth errors, weak-basis
transport, real synthetic-room registration under gyro/range perturbations,
input preservation and latched fault contracts. The initial focused attempt
failed 18 cases because the new fixture passed a list to an existing helper
requiring a NumPy rotation array; that fixture was corrected before subsequent
passes. No predecessor source or result was edited.

Both diagnostic scripts verified all 333 predecessor source bindings and the
bound inputs/artifacts before and after processing. The scripts print their
diagnostics and do not create experiment directories or modify original results.
Full regression session 10919 passed 1,302 tests across 118 explicitly selected
files in 87.47 s, with no concurrent source edits. All test and diagnostic handles
are terminal; no navigation, training or experiment process is left running.

Next actions, in order:

1. Use raw paired propagation as the reference for an efficient online joint
   model that includes the registration response, shared sample noise, velocity,
   gravity and bias correlations. Do not reinstate fixed-registration covariance
   as if the counterexample had been resolved by changing only a scale.
2. Establish the actual sensor-error loading model and its validity range using
   separately declared development data and negatives: all bias axes, range
   quantization/scale/offset, temporally shared depth errors, changing normals,
   missing rays and rank transitions. Two assumed bias sources do not cover
   baseline registration bias or sensor noise and do not prove calibration.
3. Propagate the same sources into stored/current floor evidence, not only robot
   pose. Ground-normal, range and pose errors are coupled. Preserve observed
   footprint support and wall/hole negatives; neither unknown floor nor the
   existing 6-cm tolerance may be silently reclassified to obtain an arrival.
4. Finish full-loop timing and integrate the explicit fused-speed interface and
   dynamics-aware observation actions into a fresh navigation successor. Then
   run complete discovery/return missions with matched supervised/JEPA action
   predictors, memory ablation, genuine multistep rollouts and independent layouts.

Whole-task success remains 0/2. Learned-navigation, JEPA contribution, independent
maze and hardware requirements remain unproved. The full goal remains active.
