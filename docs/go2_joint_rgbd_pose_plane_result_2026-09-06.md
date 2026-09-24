# Joint RGB-D pose/plane diagnostic: implemented, linearization not reliable

The actual complementary RGB-D estimator now has a shared raw-error diagnostic,
including RGB point registration, depth registration, gyro/force history and the
initial-velocity prior. The diagnostic completed both fixed perturbation steps
with exact nominal replay, but its physical plane-gap factors were strongly
step-dependent. They must not replace the navigation controller's error envelopes.
No turn, new mission, trained navigation policy or JEPA comparison was executed.

This is progress toward the [current execution plan](go2_floor_factored_navigation_next_steps_2026-09-06.md),
not completion of its uncertainty-validation stage or the scientific goal.

## Implemented and verified

New estimator: `lewm/raw_complementary_rgbd_sensitivity_development.py`.
New physical plane relation: `lewm/paired_rgbd_physical_plane_development.py`.
Protocol: [joint pose/plane V1](go2_joint_rgbd_pose_plane_development_v1_2026-09-06.md).
Output: `.generated/go2_joint_rgbd_pose_plane_development_v1_attempt_001`.

At each of two finite-difference steps (.01 and .005), a nominal model and ten
signed perturbation models replayed the 219 saved mission observations through
23.3 s. The nominal fusion and raw depth motion matched the frozen controller
exactly. Each error source was shared consistently across time and affected all
relevant measurements; fast/slow gyro overlap and initial-prior identity were
checked. The five source-unit amplitudes were depth scale .001, depth offset
1 mm, yaw gyro bias 1 mrad/s, post-anchor force-Y bias .01 m/s² and initial
velocity-prior Y mean .01 m/s. Those are explicit diagnostic assumptions, not a
calibrated sensor distribution or complete source population.

Paired initial and terminal measured planes were queried for all 27 physical
primitives at the terminal sensed joint posture. Joint pose/plane factors were
kept separate from pose-only and plane-only factors. The latter two were also
combined under deliberately incorrect independence, solely as a comparator.
Nominal/source state, quantization and all original controller scales were retained.

Thirteen new estimator/relation tests passed; 75 focused tests passed including
predecessor covariance and floor geometry. The full regression passed 2,008 tests
in 161 explicit files (184.14 s). Nine subsequent numerical-comparison and
independent coordinate/sign tests passed. No combined 2,017-test run is claimed.

## Scientific findings

Both step runs completed with no detected rank, point-status/support-count or
plane-seed changes. Actual correspondence identities are not fully exposed, so
this does not establish differentiability. The maximum nominal-position midpoint
remainders were 37.5 nm and 9.3 nm; small pose remainders did not imply stable
physical plane-gap sensitivities.

For the two still-ambiguous front lower-calf primitives, using the initial
observed plane:

| Quantity | Front left | Front right |
| --- | ---: | ---: |
| Joint source-unit gap scale, step .01 | 1.118 mm | 1.115 mm |
| Joint source-unit gap scale, step .005 | 1.636 mm | 1.583 mm |
| Gap-factor vector difference / larger vector norm | 71.1% | 68.1% |
| Pose-only factor step difference | 0.200 µm | 0.200 µm |
| Plane-only factor step difference | 1.163 mm | 1.079 mm |

These source-unit scales are NOT confidence bounds. At step .01, the incorrectly
independent scales were 1.084 and 1.069 mm: retaining the actual shared correlation
made the joint scales slightly **larger**, not smaller. Correlation cannot be
assumed to make a desired clearance decision easier.

The initial view had full physical floor-footprint coverage for all 27 primitives
in both finite-pair populations. The terminal view had none: it cannot see the
floor directly beneath the robot. Its numerical fitted-plane factors therefore
remain unsupported extrapolation diagnostics, not observed underbody clearance.

All depth intervals in this recorded mission were full rank. Consequently the
post-anchor force-Y and initial-velocity-Y sources had zero position factors here.
That is a property of this trajectory/constraint pattern, not proof that those
sources are harmless during weak-depth motion. The synthetic tests separately
exercise point-complemented and inertially predicted weak directions.

## Causal isolation of the unstable factor

`scripts/analyze_go2_joint_rgbd_plane_quantization_development_v1.py` separately
reconstructed the initial measured plane cell [377, 525] and its four original
range samples. It reproduced every saved plane-only factor for the two depth
sources at both steps, holding the nominal relative pose fixed.

With actual float32 rounding of perturbed ranges, the maximum plane-factor step
discrepancy across primitives was 1.320 mm. In an otherwise matched calculation
that left the perturbation arithmetic unrounded, it was 1.17e-12 m. The two front
lower-calf discrepancies fell from 1.163/1.079 mm to 8.0e-13/3.1e-13 m.

Thus quantization interacting with the three-point plane normal is the dominant
cause of this local factor's instability in the tested cell. This is a diagnostic
counterfactual only: the runtime sensor was not upgraded to float64, no output
was overwritten, and the unrounded answer was not selected for clearance.
An unchanged discrete branch label and small pose remainder are insufficient
checks on differentiating a quantized sensor/plane pipeline.

## Evidence identities and limits

The 483-source/6,138-input/native/OpenCV closure was verified before and after
the fixed diagnostic. The separately bound numerical analysis checked every
saved nominal fusion again, covariance-factor accounting, primitive/source/clock
alignment and both-step physical relations. The rounding diagnostic independently
reconstructed the saved plane-only factors.

SHA-256 identities:

- Launch: `5361a89c73762d789cc431c3b4a35c24c440916bb1af17bf9720c22dde1d808a`.
- Result: `12cd0cba81d4600f5777bbc811500cc80172f8e475a31343f35c50dbf35f6fe6`.
- Numerical comparison: `45a722a379e12d54791def074b1dc052ffe739d949d3bf8c3e1ba924f47e915b`.
- Quantization diagnosis: `fb9e54331a4f3e548c0cb50f9c94d7840f94e692ed05dfd92b9bcbd77f03ef01`.

The two runs took about 240 and 233 s of wall time. They are offline diagnostics;
the first also overlapped CPU regression work. They establish neither online
latency nor a new independent experimental trial. Independent per-pixel/frame
noise, camera intrinsics/extrinsics, joint/kinematic error, correspondence failures,
appearance corruption and future action response remain outside this five-source
model. All earlier mission failures and the two unresolved physical primitives
remain unchanged.

## Next action: finite-amplitude, quantization-aware evidence

Do not tune the finite-difference step or covariance multiplier until the failed
configuration passes. Propagate explicitly declared finite-amplitude shared and
independent errors through the actual quantizer and estimator, retaining validity,
rank, match and stopping changes. Validate predictions against separate physical
reference measurements on independent development motion, including weak-depth
and point-rejection intervals. The distribution/bounds are hypotheses until that
validation succeeds; never substitute this small five-source covariance for the
complete sensor-error population.

In a separate new source, assess a well-conditioned multi-pixel measured-plane
estimate and/or direct quantization-aware plane bounds. Preserve observed support,
discontinuities and plane-family identity; do not smooth across an obstacle,
infer unseen ground, or use the simulator's ground plane. Validate finite-amplitude
gap behavior and footprint coverage, not merely another small derivative. If a
plane cannot be stably bounded, retain unknown rather than grant contact.

The future-gait/braking validation, complete discovery/backtracking/return tasks,
matched JEPA versus supervised/geometric baselines, genuine multistep rollout,
memory ablations, independent layouts/seeds/robustness and hardware evidence remain
required. This numerical negative result does not redefine the ultimate goal.
