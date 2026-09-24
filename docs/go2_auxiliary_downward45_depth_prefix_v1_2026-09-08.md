# Native 45-degree auxiliary-camera prefix V1

Capture 18 observations at frames 0 through 17 while executing the exact first
17 zero/turn commands from the completed JEPA bounded reobservation mission on
development layout `family_episode_039`. Stop before its first translating
command at tick 17. This is fixed-command sensor characterization, not online
navigation or a recovery attempt.

Preserve the complete robot-visible scene, physics, gait, gains, friction, primary
camera, draw order, lazy primary-before-auxiliary acquisition and exact paired
physical sample. Change only auxiliary downward pitch from 30 to the fixed
45-degree candidate at the same [0.35, 0, 0.08]-m mount. Retain 640-by-480 native
RGB/depth, the 0.2-to-5-m public depth mask and evaluator-only link segmentation.
Restore the primary camera transform exactly after every auxiliary capture.

Authenticate the preceding native mission, successful 30-degree capture and fixed
45-degree geometry characterization. Freeze all sources and inputs before the
exclusive `go2_auxiliary_downward45_depth_prefix_v1_attempt_001` root. Use one
fresh CPU scene and one numerical thread. Recheck hardware, 32 GiB available RAM
and a 1-GiB output allowance above the 40-GiB reserve, following the demonstrated
robot-visible capture workload. Preserve failures without retry/resume.

Require exact physical, ideal sensor, gyro and policy prefixes against the
original native mission. Record primary RGB equality. Run the original primary
raw and metric/visibility audits and the same auxiliary raw auditor with its
explicit 45-degree calibration. Keep complete robot-population and segmentation
checks; body-occluded samples cannot pass the static-surface visibility gate.
Report all measurement outcomes, native camera matrices, valid masks, robot
pixel counts and acquisition wall time.

Reconstruct complete retained auxiliary floor patches from actual new depth,
using original observed primary poses only as a retrospective diagnostic. Check
all twelve previously identified whole-foot squares after every captured frame,
preserving the full 44-mm size and each source tick. This does not qualify a
new observer, controller, hardware mount or realistic timing. Reverify all
sources and artifacts afterward. Controller adoption requires a separate
explicit calibrated public packet and integration test.
