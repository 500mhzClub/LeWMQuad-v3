# Bounded/aligned floor Go2 interface V1

One new robot-level interface run, not a maze mission or replacement for an old
experiment. Preserve all prior negatives. Use the source-verified32m visual and
separate physical plane with the original Go2 URDF, checkpoint, gains, camera,
non-floor scene construction, command clipping, low-level gait and sensor loops.

An explicit new initialization class imports a new builder; no global monkeypatch
or frozen-source modification. C3 session ordering retains the existing native
contact,50Hz sensor and500Hz gyro wrappers. Only the collision plane is a ground
contact object; the visual-only plane has no collision geometry and is excluded
from contact identities. Existing calf/foot native support-group semantics are
preserved for the physical stop, NOT newly established contact permissions.

Scene: original5m command-probe arena geometry, fresh identifier and seed
2026090601. Pack-declared body workspace[-4,4]² plus fixed mount and all image
corner rays at optical depth<=5m must fit inside[-16,16]² with1cm margin. Actual
capture frusta are also checked, in evaluator/acquisition code only. Nothing
outside the visual support becomes free. Native paired-plane geometry and wall
identities must match construction. No evaluator pose/contact enters policy
sensor packets; the original ideal simulated RGBD/body/gyro modality assumptions
remain explicit and are not hardware calibration.

Single environment, CPU, no viewer. Configure the existing checkpoint gains
before stepping. Record15 zero-command ticks of settling (750physics steps),
then five ticks each of zero, forward(0.2,0,0), left(0,0,0.3), right(0,0,-0.3),
and zero:25ticks/1250steps. Each command lasts100ms, same slew/gait control.
Capture after settling and each subsequent tick:26paired640x480 RGBD frames.
Keep native contacts, actual body/joint trace, sensor histories, actuator and
geometry identities. A disallowed-contact or body-stability stop is terminal;
retain partial evidence, do not retry or change the tape.

Fresh raw audit: verify source/input/artifact/native hashes, sample counts and
clocks, unchanged gains, actual physical-only ground identities, reconstruct
every native contact classification, verify policy packets from stored sensor
histories and validate floor-frustum coverage. Compare native depth against
actual floor0 and recorded wall-box geometry at pixel centres, stride8 and2cm
box-edge exclusion, physical range0.22..4.98m. Report floor and wall errors
separately and whether ALL captured-frame assessed rays meet1mm. This is a
bounded integrated rendering check, not an all-scene noise bound.

At every recorded physics step, compute exact nominal URDF physical-primitive
minimum z from actual evaluator body pose and joint positions. Report four foot
sphere gaps separately from every non-foot primitive, including settle/active
minima and force/contact populations. These are evaluator contact-dynamics
diagnostics; no new penetration tolerance is selected or permitted. A positive
interface outcome requires the complete tape, no native disallowed contact,
unchanged actuator configuration, correct acquisition identities and depth
checks. It does NOT require zero foot penetration during modelled dynamic
contact. Unlike the previous sphere assay, there is no additional hidden
whole-impact1mm contact gate. The old sphere failure remains unchanged.

One-shot directory `.generated/go2_bounded_floor_robot_interface_development_v1_attempt_001`.
Bind343 predecessor paths plus the seven new builder/init/session/test/runner/
auditor/protocol paths before execution. Bound predecessor launch/result/audit
and existing gait identities; only the existing gait is loaded. No navigation
checkpoint, JEPA training, sealed/held-out data, repeat, resume or replacement
execution. An existing directory prevents a second launch. Complete the raw
audit and retain any negative result before proceeding to contact-aware control.

The ultimate goal still requires reliable complete discovery/return, online
memory, matched supervised/JEPA predictive and genuine multistep-rollout tests
across independent layouts/seeds/robustness and bounded hardware evidence when
available. This interface assay cannot establish any of those outcomes.
