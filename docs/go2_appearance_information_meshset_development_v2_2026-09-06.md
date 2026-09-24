# Appearance V1 camera-readback infrastructure correction

MeshSet V1 built all native geometry successfully and rendered the first RGB/
depth pair, then stopped before saving those arrays or invoking the observer.
Its readback assertion compared OpenGL and optical camera axes directly.
Preserve this terminal failure, its launch, five PLY surfaces and native identity,
along with the original unsupported-PLY failure. Neither supplies a scientific
appearance result. This is a separate explicit development correction, not a
resume, overwrite or scientific parameter search.

New output `.generated/go2_appearance_information_meshset_development_v2_attempt_001`.
Only the readback comparison changes: multiply the native transform on the right
by diag(1,-1,-1,1), as Genesis's own Camera.render_pointcloud does. Retain the
1e-6 absolute tolerance, zero relative tolerance, original set_pose arguments,
rendered images/depth, optical ray checks, scientific conditions and frozen
observer. No input or output pose is supplied to the motion estimator.

Two synthetic nontrivial camera poses passed this conversion in a native scene;
the separate RGB/depth render used the expected single-sample framebuffer and
zero physics steps. Pure tests reject a convention mismatch, real1mm position
change and malformed/nonfinite matrices. AST comparison checks every scientific
function unchanged except this explicit assertion conversion.

All appearance V1 and MeshSet V1 conditions remain fixed: three arms, seed271828,
cell0.125m, camera indices25–38, original body/gyro tape, collision/visual geometry,
no robot or physics steps, no threshold tuning and sensor-first scoring.
Verify source/input/native/OpenCV identities before and after one acquisition.
Bind both failed attempts; retain all artifacts on any new failure. Independently
reconstruct saved sensor predictions and cross-arm depth after completion.

This is still a seen-motion information assay, not independent physical motion,
calibrated uncertainty, navigation, JEPA improvement or hardware evidence.
Then validate recovered information on fresh motions, integrate the continuous
observer, complete discovery/marker/return and run matched predictive-training,
real multistep-rollout and memory experiments on independent layouts.
