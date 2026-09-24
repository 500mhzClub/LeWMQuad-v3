# Appearance V1 mesh-loader infrastructure correction

The original V1 attempt is terminal: Genesis rejected its first visual PLY
before scene build, rendering, or physics. Preserve its launch, failure and
partial floor witness, and every launched source. This is an explicit new
development correction under the autonomous navigation goal, not a resume or
reclassification of that attempt. The original no-retry rule remains intact.

New output: `.generated/go2_appearance_information_meshset_development_v1_attempt_001`.
Retain all scientific conditions in the original appearance V1 protocol:
neutral/repeated/distinctive arms, seed271828, cell size0.125m, B camera
indices25–38, original body/gyro measurements, fixed observer thresholds,
collision primitives, visual surfaces, zero physics steps, no robot, same
native geometry/camera/depth checks and sensor-first scoring order.

Only the visual loader changes: serialize the same PLY exclusively, load it
with Trimesh without processing, verify vertices (float32), faces and RGBA
exactly, then supply that mesh through Genesis MeshSet instead of its file
extension dispatcher. No mesh alignment, decimation, convexification, collision
or coordinate transformation is enabled. No frozen native source is edited.

Before this correction, a synthetic 0.5m patch successfully passed native
MeshSet geometry/color readback and scene construction with zero physics steps
(terminal diagnostic25216). Pure tests cover all three arms, exact serialization,
visual-only flags, existing-path/symlink rejection and unchanged scientific
functions. Bind original terminal evidence plus inherited source/input/native
and OpenCV identities before and after the single corrected acquisition.

Audit all saved predictions independently from RGB-D/gyro packets. Compare
cross-arm depth and physical geometry, retaining discrepancies and every
rejection/error. No acceptance gate is relaxed after looking at results.
This tests information availability on previously seen motion, not independent
odometry, sensor-fusion calibration, a navigation success or a JEPA advantage.
Successful information recovery leads to fresh motion validation, integration
and complete discovery/marker/return tests with matched learning/planning/memory
ablations; the whole scientific goal remains unchanged.
