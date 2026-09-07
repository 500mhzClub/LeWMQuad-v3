# Appearance information recovered; full navigation still unproved

The fixed RGB-D point observer recovers the thirteen previously unobserved
transitions when actual rendered surface appearance contains texture. Original
B plane-depth ranks remain2 for all thirteen; the original B controller failure
and original recorded RGB-D0/80 result remain unchanged.

| Actual rendered arm | Accepted / pairs | Mean translation error | Maximum translation error | Inliers per accepted pair |
| --- | --- | --- | --- | --- |
| Neutral | 0/13 | unavailable | unavailable | none |
| Repeated checker | 13/13 | 0.245mm | 1.042mm | 90–123 |
| Distinctive grayscale cells | 13/13 | 0.265mm | 1.249mm | 105–133 |

All three arms use the same14 camera poses from B4.0–5.3s, the same original
co-timed body/gyro measurements, fixed geometry and frozen RGB-D observer rules.
New rendered sensor packets have separate counterfactual identities. Native
camera poses place the renderer and score predictions only; they do not enter
the observer. This is not new physical motion or an independent navigation trial.

## What the result establishes—and does not

The independent audit reconstructed all42 sensor predictions (including three
initial anchors) exactly before loading native motion for scoring. Every one of
the42 depth arrays is bit-identical to its same-pose counterpart in the other
arms. Physical geometry/material/solver readbacks are equal across arms. Saved
PLY vertices, faces and colors match their declared generators, and native
visual triangle multisets match those saved surfaces. Independent interior
ray intersections agree with rendered depth within0.028361mm. No physics step,
robot simulation, controller resume, training or hardware actuation occurred.

Inspected distinctive/repeated images show the intended grayscale floor/wall
patterns, not semantic markers or location/goal codes. Neutral frames have zero
keypoints; checker frames600 and distinctive frames304–349. This supports a
visual-information explanation for the fixed observer's textureless failure.
It does not show that distinctive texture is necessary or that repetitive
appearance is harmless: small adjacent-frame motion and depth/gyro consistency
can make this checker trackable. These13 overlapping transitions are not13
independent motion/layout trials. Appearance realism, lighting, moving objects,
occlusion, blur, sensor noise and perceptual aliasing remain unvalidated.

Mean observer-only times were51.95ms neutral,74.29ms repeated and64.00ms
distinctive, measured concurrently with regression testing. These exclude full
capture/control execution and are not isolated latency or real-time evidence.
Neither this millimetre-scale sample error nor inlier residual is a calibrated
deployment uncertainty bound. No covariance reset, depth-rank relabelling or
navigation admission follows from an accepted point estimate.

## Infrastructure failures retained

Original appearance V1 stopped at its first PLY because Genesis's mesh-file
dispatcher does not support that extension. No scene build/render occurred.
The separate MeshSet V1 correction built native geometry and rendered the first
RGB/depth pair, then stopped before saving the arrays or invoking the observer:
its camera assertion compared native OpenGL axes directly with optical axes.

MeshSet V2 uses exactly the same PLY through Trimesh's unprocessed load and
Genesis's supported in-memory MeshSet interface. It checks optical readback
using native_transform @ diag(1,-1,-1,1), matching Genesis's own point-cloud
conversion, retaining the original1e-6 tolerance and original set_pose arguments.
Both failed outputs and all launched source/protocol/input files remain frozen.
No scientific parameter or matching threshold changed between these attempts.

## Verification and identities

Synthetic native MeshSet25216 passed geometry/color roundtrip and zero-step
construction. Native camera86392 passed two nontrivial pose conversions and
single-sample rendering with zero physics steps. Focused7475 passed30tests;
camera-inclusive38563 passed34. Full6679 passed1846tests/150 explicit files in
153.44s; final81297 passed1850tests/151 files in155.19s. No tested source changed
during either run. Preflight39451 verified437sources/4658inputs plus inherited
native identities and the exact OpenCV5.0.0 binary. Acquisition56107 and
independent audit18152 both ended exit0. All handles are terminal.

Successful output:
`.generated/go2_appearance_information_meshset_development_v2_attempt_001`.

- Launch: `4e7b4bf77c3449d77bae09d7426a40e3b320409bc937afc2eace475f5ad5efb7`.
- Result binding112 artifacts: `b0689df36aaf202058cd5d85ba88c0cee34391def6496b3259d2e72072e43b4f`.
- Independent audit: `4b8315e32a51eb79bdd632f083ea6535c9bac7cfe432d46ef07145650c83edb4`.
- Auditor source: `d858a761ccbc30dfa52d07f2d8df788872f69d361f7e0725de3dd7f3c66afc77`.

Original V1 launch/failure:
`bfb933ccaf1eeece784e6d35c0a0e48519662c28f53ebcade0b1d0474b790362` /
`bd7ee928e917cf98270b388fae5f3afff2b9215c496725a645ece8840667d53e`.
MeshSet V1 launch/failure:
`4e362976161de1041c2a8c498d97ff55dc39f555c3b0177e25c11be658a310ed` /
`9ff5ed99f11850a3e4b1d5d7d884130289ffec40c274c751933dd94e844bd220`.
Their partial geometry witnesses are retained and bound by the successor.

## Next decision

Stop appearance tuning on B. Implement and test explicitly sourced sensor fusion,
then use a fresh bounded physical motion/layout to challenge it and validate
dropout handling. Integrate into continuous full discovery/marker/return rather
than building further matched-pose demonstrations. The concrete work sequence is
in [the next-execution plan](go2_rgbd_fusion_and_fresh_motion_next_steps_2026-09-06.md).
Full-maze0/2, no demonstrated JEPA advantage and no hardware qualification remain.
