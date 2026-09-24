# Joint RGB-D pose V1: matched rotation-source comparison on fitting data

Question: does image/depth estimation of rotation reduce the observed gyro-bias
translation drift, while preserving useful nominal tracking? This is a new
development estimator comparison, not JEPA, learned control, independent
validation, navigation, error calibration or hardware execution.

## Fixed methods and controls

Both new modes share source, camera/sensor contracts, cached keyframe SIFT/LK
matching, robust proposal selection, pruning, six-cell acceptance and support-
aware promotion. Joint mode estimates a proper rotation and translation from
paired RGB-D points. Gyro mode uses the same procedure but fixes rotation to
the causal gyro reference-to-current estimate and fits translation only. Compare
both to the unchanged support-aware V1 results on SAME INPUTS. The new matched
gyro mode controls for changes in robust fitting, rather than attributing the
entire change from the old mean estimator to rotation estimation alone.

Use all-point fitting plus 128 deterministic three-point proposals per frame,
seeded by [2026090631, frame]. Joint fitting uses centred cross-covariance SVD
with a proper-rotation determinant correction; gyro fitting uses the supplied
proper rotation and centroid translation. Both require second centred-scatter
RMS >=0.02 m and second/first ratio >=0.05 in both point sets. Planar but
noncollinear points are admissible; a plane is not equivalent to a line.
Conditioning is NOT calibrated covariance or a pose-error bound.

Score proposals using the existing <=20-mm 3D residual and <=1-pixel bidirectional
reprojection checks. Select maximum inlier count, then minimum summed inlier
squared residual, with deterministic proposal order for exact ties. Refit and
monotonically remove inconsistent selected points until no more are removed;
do not re-add pruned points or retry after terminal failure. Require >=12
inliers, >=60% fraction and >=6 occupied cells in each image. Keep <=3-m
reference displacement and <=0.15-m consecutive position increment. Also reject
consecutive image-pose rotation above0.20 rad and reference image/gyro rotation
disagreement above0.10 rad. These are diagnostic sanity limits, not calibrated
uncertainty. Gross disagreement terminates the member without a fallback.

Promote accepted references at the unchanged0.4-m/0.35-rad motion thresholds or
<=7 image-grid cells. Joint-mode global orientation and position compose through
the image-derived reference chain. Gyro integration continues independently
from the original anchor: its relative estimate is a consistency monitor in
joint mode, not a fitted bias correction, blend, global reset or replacement for
lost image evidence. In matched gyro mode it remains the rotation estimator.
Never promote a failed frame or restart/relocalize a failed member.

## Scope and frozen cases

Replay only all336 frames from the sustained V1 **fitting** recording. Five
unchanged cases in both modes: nominal, blank RGB, gyro-Z+0.001/-0.001 rad/s,
and independent depth noise+0.1 mm using the prior timestamp-consistent helper.
Keep the known malformed-depth range crossing as a negative; do not fix its
mask or resume the model. No validation frame is loaded. Inherited integrity
hashing may include past validation artifacts, but they do not enter prediction,
fitting or new scoring. No coefficient, threshold or seed selection after replay.

Persist all ten models' predictions, failures, correspondences, conditioning,
gyro-disagreement traces and parent chains before native pose is loaded for
scoring. Report success/failure counts, promotions, full and common-admitted
position/orientation errors, and same-input comparisons against the new gyro
control and prior support-aware estimator. Blank RGB must not produce fabricated
motion. Report exact equality or differences between nominal and gyro-bias
joint-mode geometry histories: invariance alone is expected from the gyro's
monitor-only role and is NOT evidence of accuracy or general sensor fusion.
Native errors determine whether the image pose is useful.

Do not reuse the previous gyro-derived uncertainty radius for image-derived
rotation. Both new modes explicitly output unknown position/orientation error
bounds; fit residuals, singular values and a successful trace cannot authorize
floor clearance or future gait. Deriving and independently validating appropriate
uncertainty remains necessary after selecting a useful estimator.

Freeze source, tests, protocol and narrow source closure against exact support-
aware V1 launch/result/prediction/evaluation/audit identities. Exclusive output:
`.generated/go2_joint_rgbd_rigid_pose_development_v1_attempt_001`. No overwrite,
retry, resume, new physics, GPU training, navigation or hardware authority.
Preserve all predecessors and this attempt's negatives.

Then obtain sufficient real simulated travel into observed floor, validate local
relative error and prospective gait/braking on new trials, integrate one floor/
non-floor interface and a timed whole-maze exploration/return baseline, and test
JEPA predictive training, actual multistep rollout and memory contributions on
independent layouts/seeds/robustness before bounded real-platform experiments.
