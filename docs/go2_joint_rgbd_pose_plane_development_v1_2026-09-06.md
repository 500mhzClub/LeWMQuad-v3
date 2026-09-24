# Joint raw RGB-D pose/plane sensitivity V1

One fixed saved-data diagnostic, not training, calibration or a physics run.
The previous physical-configuration adapter and all failed missions remain frozen.
The question is whether an actual shared raw error affects motion and measured
floor together, and whether finite-difference answers are stable across two steps.

Replay the same 219 saved mission sensor frames, ending at 23.3 s, through the
actual `RGBDInertialState` complementary model. Verify the nominal fusion and raw
depth motion against the saved controller on every frame. For each source use
the same signed error throughout history and both gyro sampling rates. Include
the supplied initial velocity mean as a distinct uncertainty source, not a sensor.

Fixed named sources and one-source-unit loadings:

- depth scale: .001 times each valid measured optical range;
- depth offset: .001 m at every valid range;
- gyro yaw bias: .001 rad/s at all fast/slow timestamps, including initialization;
- specific-force Y bias: .01 m/s² only after the 1.5 s sensor anchor;
- initial-velocity-prior Y mean: .01 m/s, constant throughout the episode.

Five named sources assert a unit independent source basis; temporal and channel
correlation within each source is preserved. These amplitudes and independence
assumptions are not a calibration model or complete error population. RGB pixel
appearance, camera intrinsics/extrinsics, joint error, slips and other sensor
faults are not covered by these five sources. Depth noise does affect RGB point
lifting as well as plane registration; both must be recomputed for every pair.

Run finite-difference steps .01 and .005, in that order: nominal plus ten signed
perturbed RGB-D estimators per step. Preserve float32 range quantization. Unknown
ranges have zero loading. A perturbation that violates validity or faults the
actual estimator terminates and is retained; do not clip it into a passing pair.
Preserve rank/point-status/inlier-count changes as explicit categorical warnings.
Unchanged counts do not establish unchanged correspondence identities or smoothness.

Retain the initial and terminal paired raw observations/poses. For each retained
view, independently select the actual median-row-major eligible plane cell in
each perturbed frame. Query all 27 unpadded primitive minimum gaps for the
terminal sensed joint posture at zero relative translation/yaw. Report the joint
pose-and-plane signed-gap factor, pose-only and plane-only factors, their
deliberately incorrect independent variance sum, paired midpoint remainders,
plane-seed changes and full perturbed floor-footprint coverage. Floor-family
query hypotheses remain .002 normal, .001 up, and 1 mm plane-offset error.

Finite pairs and first-order moments do not bound the continuous error set. No
scale multiplier, calibrated probability, physical permission or substitute for
future-gait validation is produced. No covariance is installed in the controller.
Keep unsupported/no-plane numerical relations explicit as null, not zero.
Compare both steps after completion; do not pick the more favourable one to
clear the two ambiguous front lower-calf shapes.

Bind the previous configuration launch/result/reference audit and artifacts,
inherited source/input/native/OpenCV identities, and new protocol/runner/source/
synthetic tests recursively before launch. Fresh exclusive root:
`.generated/go2_joint_rgbd_pose_plane_development_v1_attempt_001`.
Save both full step results or a terminal failure without overwrite/retry. This
is already-seen development data and has zero independent experimental trials.
The full navigation, matched JEPA/rollout/memory, independent-layout/seed and
hardware scientific goal remains unchanged and unachieved.
