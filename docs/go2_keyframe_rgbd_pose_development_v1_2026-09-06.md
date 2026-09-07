# Direct keyframe RGB-D pose V1: fitting-only recorded comparison

Question: can direct constraints to retained RGB-D views provide useful
longer-horizon relative pose without the predecessor's per-frame accumulated
2-mm point allowance? This is a NEW estimator diagnostic, not a resumed failed
observer, navigation controller, JEPA model, calibrated uncertainty or clearance
certificate. Preserve sustained V1's original 5.1-s failure and missing coverage.

## Frozen computation

Only `.generated/go2_sustained_observed_floor_motion_development_v1_attempt_001/fit`
sensor frames enter the new estimators. The 336-frame tape is fixed. No validation
frame is loaded, no new physics occurs, and no parameter is chosen after replay.
Inherited artifact integrity verification may hash prior validation artifacts;
that does not provide them to prediction, fitting or new scoring.

Use paired RGB/depth and unchanged causal 500-Hz gyro integration. For a retained
keyframe and current frame, use the predecessor's mutual SIFT 0.7 ratio matching,
duplicate-location rejection, forward/backward LK and depth lifting rules.
Register translation under observed gyro rotation from the directly paired 3D
points. Require at least 12 inliers, 60% inlier fraction, six image-grid cells in
both views, 20-mm residual and 1-pixel bidirectional reprojection agreement. The
returned mean and inlier set must stabilize within ten iterations. Reject a direct
reference displacement above 3 m or consecutive inferred displacement above
0.15 m. These are rejection rules, not proofs of correspondence correctness.

After an ACCEPTED pose, promote its view at relative translation >=0.4 m or
relative angle >=0.35 rad. Compose its pose and error allowance into an explicit
parent chain. Do not reset the global pose, gyro history or accumulated anchor
allowance. An unsupported direct match or malformed packet terminates that
member; never try another keyframe, restart, fill gaps or silently fall back after
failure. The initial view defines an identity coordinate anchor, not an observed
translation; blank RGB should fail at the first attempted registration.

## Conditional error model, not fitted uncertainty

Assume each inlier is the same static physical point in both images, exact camera
intrinsics/extrinsics, <=0.5-pixel error in each image coordinate and <=0.25-mm
optical-depth error at the true matched point, including interpolation. Assume
the total gyro/integration angular-error rate is bounded by 0.001 rad/s. These
are explicit development hypotheses, not established physical sensor bounds.
Missing timing/calibration/slip/false-match populations remain unqualified.

For optical ray coordinates r=(u+0.5-cx,v+0.5-cy)/f, depth z, depth allowance ez
and coordinate allowance ep, use the lifted point radius

`sqrt(sum((ez*abs(r) + (z+ez)*ep/f)^2) + ez^2)`.

For every actual inlier, sum reference/current point radii and the rotation
operator allowance `2*sin(min(theta,pi)/2)*norm(current_body_point)`; average
those radii because translation is the actual equal-weight inlier mean. Do not
divide by sqrt(N): errors may be fully correlated. Compose global translation
radius as `anchor_radius + local_radius + rotation_operator(anchor_theta)*norm(t)`.
Common orientation before a retained reference cancels from its relative rotation;
the local angular hypothesis depends on reference-to-current elapsed time, while
global orientation and anchor translation history remain accumulated.

This is a distinct conditional model, not a recalibration of the old 80-mm proxy
budget. Output its radii and any crossings of that numerical value descriptively;
do not treat its continued production of estimates as safety-qualified operation.
No floor/ray consumer or actuation uses these outputs.

## Fixed comparisons and scoring

Five members: nominal, blank RGB, gyro-Z biases +0.001 and -0.001 rad/s, and
independent represented depth noise +0.0001 m. Use the frozen timestamp-consistent
perturbation helper and paired RGB hashes. The depth allowance is a hypothesis
for total rendering/interpolation/noise error, not a claim inferred from the
previous observed maximum. Each member terminates independently. Compare the
nominal predecessor ShadowObserver replay exactly against all 336 saved outputs,
including terminal and not-reinvoked records.

Persist ALL new predictions, failures and keyframe parent chains before loading
native pose traces for scoring. Report admitted counts, failure time/reason,
keyframes, actual position/orientation error and conditional-radius exceedances.
Use only common admitted rows for direct old/new error comparisons; longer new
histories cannot make the original failed navigation successful.

At each saved matched inlier, use native poses ONLY for a separate necessary
consistency test: the two lifted points transformed through true relative pose
should agree within their summed point hypotheses. A violation invalidates the
joint static-correspondence/point-error assumptions for that pair; a pass does
not independently prove them. Report these violations even if averaged pose
error fits its radius. Do not select another threshold, seed or inlier rule.

Bind protocol, model, tests, runner and narrow source closure before replay to
the sustained V1 acquisition/audit identities. Use an exclusive output directory
`.generated/go2_keyframe_rgbd_pose_development_v1_attempt_001`. Preserve failures
and never overwrite or retry this attempt. Synthetic tests cover direct transform
conventions, point-error box corners, correlation, composition, keyframe history,
image/packet rejection and terminal behavior. Recorded outcomes remain exploratory
fitting-role evidence; new independent validation and full-body observed motion
are still necessary before prospective control or JEPA comparisons.
