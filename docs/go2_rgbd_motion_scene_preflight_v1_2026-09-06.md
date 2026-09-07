# Fresh robot-scene native preflight V1

One zero-step, no-render native construction check before fresh physical motion.
Output `.generated/go2_rgbd_motion_scene_preflight_v1_attempt_001`.
This is not another B camera-pose assay. Use a6m-square enclosure with a new
offset angled partition and spawn(-.25,-.2,.375), heading0.27rad. Physics/topology
seed2026090606, independently specified appearance seed2026090607. No controller,
gait checkpoint, RGB-D estimates, commands, physics steps or navigation results.

Construct one reference scene with the frozen bounded primitive builder, then
new neutral/repeated/distinctive robot scenes. The new path keeps collision
plane/Box/URDF entities and creates separate visual-only meshes. Compare every
native robot and environment collision shape's data, link name, world pose,
friction and solver parameters exactly with reference. Exclude only global
geom/link indices, which move when visual-only entities are reordered. Across
new arms these indices and all native visual triangle identities must also
remain equal. All expected visual coordinates/colors are validated on exclusive
PLY serialization. Six environment collision primitives and nonempty actual Go2
geometry are required. Preserve partial artifacts and terminal failure on error.

The inherited static mesh checker covers the environment; override its
robot_present=False statement explicitly because these scenes contain a robot.
Read back the actual robot rather than claiming equality from an unchanged URDF
path. No scene.step or camera.render call is permitted. This checks construction,
not actual dynamics, gait gains, sensor timing, uncertainty or motion validity.

Bind current new fusion/memory/scene sources and tests, inherited437-source
closure and sensor/native identities, and successful appearance evidence before
and after. Do not edit old sources or prior outputs. New fusion hypotheses have
no runtime defaults; synthetic accounting values are not promoted to calibrated
errors. Follow with source-bound fresh bidirectional walking/turning/braking and
shadow fusion under explicit bounded supervision, then independent validation
and continuous whole-task discovery/marker/return. Full scientific goal remains.
