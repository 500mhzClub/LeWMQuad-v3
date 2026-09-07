# Fixed RGB-D correspondence diagnostic V1

Before inspecting recorded RGB correspondence outcomes, fix one diagnostic using
the preserved A/B acquisitions. No physics, failed-controller resume, refit,
threshold search or changes to their source/results. B is already development
evidence, not an independent test of this new observer.

SIFT up to600 features, mutual two-neighbour descriptor ratio<0.7; deduplicate
both feature locations on a half-pixel grid (orientation copies are not extra
point observations); pyramidal
Lucas–Kanade initialized at the descriptor match,21x21 window,3 pyramid levels,
30 iterations/0.01 termination; forward/backward error<=0.5 pixels and refinement
within1 pixel of the independent descriptor match. RGB is converted to grayscale.
OpenCV CPU threads1. No random sampling or native-pose initialization.

Lift both image endpoints with bilinear aligned optical depth and the existing
half-pixel-centre/mount calibration. All four depth neighbours must be valid;
reject spread>0.02m+0.01*mean depth. Integrate the original500Hz body gyro with the
unchanged causal orientation implementation. Point displacement is
p_previous - R_previous_from_current*p_current, including camera lever arm.
Use componentwise median initialization then three mean/inlier refinements:
3D residual<=0.02m and bidirectional reprojection<=1pixel. Require>=12 inliers,
>=60% lifted-match support,>=6 occupied cells in a4x3 grid in each image, and
translation norm<=0.15m per100ms. No fit/clipping or fallback command odometry.

An accepted estimate is conditional RGB-D point-correspondence rank3, not a
change to the original plane-depth rank or a calibrated uncertainty bound.
Static scene and correct correspondence remain assumptions. Rejections remain
unobserved; do not fill the lost component or reset global uncertainty.
Synthetic tests cover textured planar tangent motion, zero commands, blank/
repeated/unrelated/occluded images, invalid/discontinuous depth, mount/rotation
conventions, coverage/outliers and causal packet/calibration/identity failures.

Replay A and B only through each original terminal controller decision (A5.7s,
B5.3s), no tail extension. Save every estimate/rejection, clocks/hashes, original
depth rank, correspondence counts, error and timing; compare original depth
observable projection and new translation to native relative displacement in
an evaluation-only second calculation. Separately report the exact B weak
interval4.1–5.3s, including failure counts. No successful subset presented as a
complete observer or independent validation. Whole-task0/2 remains unchanged.

Freeze source/input/native/OpenCV binary identities in a fresh one-shot output
`.generated/go2_rgbd_correspondence_motion_diagnostic_v1_attempt_001` before replay.
Reject overwrite; verify bindings before/after. Follow the post-B whole-mission
plan after this causal measurement question, not another static-floor search.
