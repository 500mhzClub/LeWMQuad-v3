# Pulse native outcome targets: development derivation V1

Derive labels for every window in the completed185-window pulse pairing index,
using only its bound predecessor coupled-room physics traces. No running
intent-return outputs, fitting, training, control, new physics or generalization
claim. Exclusive output `.generated/go2_pulse_native_targets_v1_attempt_001`.
Freeze this protocol, source and focused tests at launch. Validate explicit
source/input/index/trace hashes before and after derivation; retain every window.

The target is actual start-body XY displacement and projected relative yaw
from R_start^T R_target, not commanded displacement or world-yaw subtraction
under tilt. Both transforms use normalized native xyzw quaternions. Exact
target offsets remain .5,1,1.5,2 and2.2/2.5 s. Unknown slots have absent labels.
No native pose or outcome label enters the current RGB/body encoder or plan.

Verify the actual native requested-command sequence against the declared pulse
plus20 zero ticks at every2 ms sample, using canonical float32 request identity
and the already audited tape-completion flag. This does not equate requested
commands with actual base velocities. A changed/missing prefix censors affected
motion/safe-contact targets. An observed disallowed-native-contact event before
command divergence remains a positive cumulative contact label through later
known-plan horizons even if execution stops. Events after divergence are not
evidence for the original plan; noncontact termination means unknown, not safe.
If contact precedes departure, retain the window with unavailable new-outcome
labels. Contact here is not every foot-ground force or a calibrated terrain-risk
probability. No post-contact motion target is used.

Native motion/contact availability is separate from future RGB availability
and from visual tracking success. A missing target image does not destroy an
observed native endpoint; conversely an observed raw image need not certify
collision-free motion. Every matched baseline must get the same actual-image
population, with native labels exclusively on the target side. Report motion,
contact and positive-contact counts, image/motion mask differences and all
censoring; do not discard difficult windows to improve averages.

Focused synthetic tests cover partial endpoints, stops, contact at/between
boundaries, divergence before contact, missing images, prior contact, malformed
traces and full3D coordinate-frame invariance. Actual derivative labels do not
establish predictive quality. The remaining adapter work is matched partial-time
training losses/tensor integration and adequate independent scene/action/state
coverage with frozen splits. If these room traces have no positive contact
events, they cannot establish learned collision-risk discrimination.
