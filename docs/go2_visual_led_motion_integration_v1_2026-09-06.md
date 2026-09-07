# Visual-led motion-evidence interface integration

New implementation around the unchanged rigid RGB-D/gyro observers. This is an
integration check on exposed development tapes, not a repeat scientific model
comparison, physical trial, fit, validation selection or navigation qualification.

Use both audited 226-frame friction recordings. For each mode, pass only the
original RGB-D/body/fast-gyro packets to the frozen observer. Separately adapt
the existing live rolling-support diagnostic into explicit typed same-episode,
same-acquisition, ideal-vector contact velocity samples. Input acquisition has
zero simulated delay; this is not a measured hardware publication clock. Do not
use native pose, contacts, wall labels, environment geometry or command-integrated
position in the interface. Reading the stream's declared acquisition identity
from specification metadata is not reading evaluator geometry for inference.

Optional contact windows comprise the six endpoint-inclusive 50Hz samples over
the visual interval. Rotate velocities to their fixed gyro anchor before
trapezoidal integration, then compare visual/contact displacement in the current
body frame. If any velocity is unavailable, retain None and all missing times.
Malformed optional windows produce a contact-specific rejection, not visual
failure. No contact weight, slip threshold, zero-motion imputation or terrain
permission is introduced. The fixed gyro-anchor convention remains an upstream
acquisition assumption, not an independently calibrated orientation channel.

Observe both models at every 100ms image timestamp and query the interface at
the four intermediate 20ms timestamps. Intermediate pose is unobserved; expose
the last visual pose as historical only. This version does not asynchronously
integrate gyro between image updates, and must not claim a current rotation or
translation there. It is intended first for a 10Hz navigation boundary, not a
50Hz current-pose controller. Processing latency is explicitly unaccounted for;
recorded decision clocks are not real execution timestamps.

Save interface evidence before checking against the frozen visual-only output
witness. Require exact preservation of pose/provenance fields for all observations
and explicit unobserved status for every intermediate query. Keep all failure
outputs. Synthetic tests cover missing, stale, delayed, cross-episode and malformed
contact windows, visual terminal failures, clock order, immutable snapshots and
fixed-frame contact integration. Run focused and full explicit regression lists.

Bind all source/input/native identities before the exclusive output
`.generated/go2_visual_led_motion_integration_v1_attempt_001`. No source export,
new simulation, protected benchmark access or model training. Future physical
integration still requires justified near-field/footfall/body/braking evidence,
valid optical mounts, genuinely fresh physical conditions and stop-only native
supervision. Full JEPA/memory/multistep/maze/hardware aims remain outstanding.
