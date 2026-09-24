# Dual-camera motion admission and floor registration replay

Use one new DualCameraVisualMotion and JointFloorRegistration instance from
frame0 through all1881 observations of the ninth completed episode. Bind the
completed continuous observer result
540f63243f74b63e90dadeaa8e7aebde936bb7880d49f0c77ffc5e8affc59eea
and every referenced input/source. Require every accepted raw pose field,
continuity record, camera selection, reference selection and overlap receipt
to reproduce that completed observer. Use the explicit dual-camera accessor
before the unchanged joint floor registration and its existing accessor.
Before the first auxiliary use at1870, require the registered pose and full
floor-registration receipt to match the original controller recording.

The motion wrapper adds actual auxiliary calibration/pixel/depth bindings and
current/historical camera-choice semantics. It preserves terminal failure,
returns no current pose after its timestamp, and cannot feed an unvalidated
auxiliary pose into registration. The separately prepared settled controller
uses this path while preserving mission, map, residual and planner logic.
This replay does not execute that controller, a learned model or new commands.

Save any first admission/registration failure and keep terminal state. Evaluate
raw and registered pose errors with native state only after processing all
observations. Report full coverage, exact comparisons, errors and processing
time without calling them calibrated bounds, navigation or hardware evidence.
The original stop/drain trajectory is not a prospective recovered return.

Exclusive root go2_dual_camera_registered_replay_v1_attempt_001. Run
scripts/replay_go2_dual_camera_registered_v1.py --preflight-only before launch.
One CPU process/numerical thread beside the ongoing tenth-episode audit, no
new scene;4GiB available RAM and1GiB output above40GiB reserve. Record CPU/GPU,
RAM/storage and competition. Verify all source/input hashes before and after.
Preserve the completed observer source, all frozen attempts and all failures.
