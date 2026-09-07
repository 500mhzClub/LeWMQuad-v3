# RGBD V1: acquisition works, fixed geometric check fails

Two actual-render trials completed, and the full raw audit passed. All ten
frames failed the predeclared5-mm depth check. This is not a qualified metric
depth input for navigation. The sources, protocol, raw data and failed metric
outcomes remain unchanged; audit PASS means faithful evidence, not sensor success.

## Implemented and exercised

The new modality is separate from unchanged RGB/body packets. It binds current
RGB pixels, episode, optical-depth calibration and acquisition/availability/
decision clocks; rejects privileged or malformed fields; marks nonfinite or
out-of-range depth unknown; retains four causal frames; and projects valid
depth into body-frame surface points without claiming surrounding free space.
Both probes used actual camera rasterization, not analytic policy-side raycasts.

Collector77987 completed two cases, exit0. Full audit88750 passed both, exit0:
1,900 native physics/fast samples,190 ordinary sensor samples and ten actual
RGB/depth pairs. It checks the entire inherited RGB/body evidence, added native
depth, sanitized depth/masks, calibration, fast histories, command/contact/gain
accounting, exact detector replay and independent geometric outcomes.
Visible-marker RGB detection remained5/5; occluded detection0/5. No contact or
body stop occurred. Only zero commands were executed.

Final preflight99963 passed253 source,189 input, two gait and five explicitly
bound installed-renderer paths. Full48708 passed1,055 tests across92 files in
75.33 s. The first targeted run92959 had28 passing tests and one analytical
specification failure: a close occluder covers the whole view and cannot also
provide background rays. That criterion was clarified before actual rendering;
the visible case requires background and both reject invalid background-as-free.

## Actual failure evidence

On the first visible frame, all1,835 evaluated ground rays exceeded5 mm error
against the collision-floor reference; maximum17.253 mm. Most wall/panel rays
were accurate to tens of micrometres. Three rays expected to reach an interior
far wall returned near-panel depths instead (about1.03 m discrepancy), and four
nominal background rays returned finite foreground depth. Across all five visible
frames, maximum error ranged1.0339–1.0392 m.

On the first occluded frame, all446 evaluated ground rays exceeded5 mm, maximum
8.273 mm, while all4,237 occluder rays passed. Across five occluded frames,
maximum error ranged8.265–8.490 mm. No panel rays were visible in that case,
as expected. Thus the failure is not a uniform units or focal-length scale error.

## Source-backed diagnosis, not rescoring

The installed `genesis/utils/mesh.py:create_plane` makes a collision box whose
top is z=0 but shifts its visual plane vertices to z=-.005 m. The rigid entity
loader uses the returned visual and collision meshes separately. Our evaluator
assumed the visual floor coincided with the physical plane. The rendered sensor
therefore sees a different surface from that reference even when depth is valid.
This must be explicit in any collision-aware use; silently treating rendered
ground as exact physical ground would be wrong.

The installed pyrender renderer uses a multisample framebuffer for combined
RGB/depth rendering and resolves its depth buffer. In contrast, its DEPTH_ONLY
path binds the single-sample framebuffer and disables multisampling. Its
offscreen adapter selects DEPTH_ONLY when depth is requested without RGB.
The combined path therefore does not establish an exact pixel-centre depth
contract at silhouettes. This is a source-supported explanation of the observed
foreground/background discrepancies, not yet a demonstrated corrected capture.

Read-only diagnostic51748 computed the per-surface and silhouette results above.
Diagnostic31147 found inferred visible-floor z between-.004400 and-.002817 m
under centre-ray unprojection, rather than the source-defined-.005 m. Merely
substituting the visual plane into that diagnostic still leaves up to13.366 mm
residual in the visible case. The floor-reference correction alone is therefore
insufficient. Neither diagnostic writes artifacts, changes metrics or reruns physics.

Additional installed source identities inspected after the run:

- `utils/mesh.py`: `084105dc71a9ae390c6dd794e3ca2ccc1abd5acfffc78eec78a3e26a9fabc0eb`.
- `engine/entities/rigid_entity/rigid_entity.py`: `05c16043c05acb129f2e54723bc2d619686a1e00cfe2f6419b0e68521c940649`.
- `ext/pyrender/offscreen.py`: `54d6ad33b0ca7d9fe0219f57e5990a276524dd0c1ff072cd323b86ecbac07a56`.

## Next concrete change

Create a separately specified acquisition successor. Render RGB using its existing
path, then render depth-only without an intervening physics step or camera change.
Verify native single-sample framebuffer configuration and exact common capture
time/pose. Record actual visual-floor mesh vertices/pose and collision-plane
identity rather than assuming they coincide. Check depth against observed visual
surfaces under unchanged5-mm tolerance, and report the physical/visual ground
offset separately; that is a different prospective measurement claim, not a
rescoring of V1. No runtime wall labels, floor truth or simulator pose enter the
sensor packet. Preserve the original V1 failure and all RGB-only task results.

Once the acquisition semantics are actually verified, connect observed boundary,
relative-motion and clearance state to continuous exploration/discovery/return.
Do not proliferate stationary benchmarks instead of implementing that connection.
Added ideal depth still needs a real deployment counterpart and calibration;
noise/latency/holes, hidden side/rear space, independent mazes and matched JEPA
comparisons remain open. The full scientific objective is not achieved.

Root: `.generated/go2_rgbd_observation_development_v1_attempt_001`.
Launch SHA-256: `50d842c6bae1537e6b79874e50235c7eea84740b7c03cb0d361c14898e2459e1`.
Result SHA-256: `8063448c31d013a0e0cfc56aa4f87c1e3dfb1256016a355f5a80105c93a5441a`.
Full audit SHA-256: `e52424ec2b58291dfd43b97af5c2f896daf2beae3d447cb9261b98000bb701b4`.
All ten new source/test/protocol paths are bound; no collector or audit remains running.

Post-document guard77011 passes253 source,189 input, two gait, five native
renderer bindings and three terminal identities, exit0. No live verification
process remains either.
