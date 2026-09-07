# RGB-D correspondence V1: synthetic positive, recorded appearance negative

The new causal RGB-D point observer passes its synthetic motion/negative tests,
but accepts **0/80 recorded A/B image pairs**, including **0/13** in B's weak
motion interval. An independent sensor-only replay reproduces every estimate
and rejection. No matching threshold, old observer, A/B result or physical trace
was changed. This is not a recovered navigation observer or JEPA result.

## Evidence and cause

| Recording | Pairs | Accepted | Maximum keypoints in a current frame | Maximum mutual matches | Maximum depth-lifted matches |
| --- | ---: | ---: | ---: | ---: | ---: |
| A through its terminal decision | 42 | 0 | 16 | 4 | 2 |
| B through its terminal decision | 38 | 0 | 6 | 0 | 0 |
| B weak interval, 4.1–5.3 s | 13 | 0 | 0 | 0 | 0 |

An inspected B frame (`rgb_0026.png`) shows broad nearly uniform floor/wall
regions without distinctive interior texture. The rejection happens before
the 3-D consistency/coverage tests: B never obtains a mutual descriptor match.
It is not evidence that a looser residual or grid-coverage threshold would help.
The zero-keypoint claim applies to B's weak interval, not every B frame.

Source confirms that `scripts/bounded_floor_physical_init_development.py`
constructs neutral floor/wall materials and calls the bounded scene builder with
`render_robot=False, apply_textures=False`. Keeping the moving robot out of
camera matching is useful; disabling appearance texture prevents this test from
demonstrating texture-based RGB motion recovery. Uniform visible plane interiors
provide no distinctive point identity along their surfaces. This is an observed
limitation of these scenes and this fixed observer, not a proof that every
possible RGB method must fail or that RGB is useless for maze navigation.

The existing `apply_textures=True` branch in
`lewm_genesis/lewm_genesis/bounded_scene_builder_development.py` is not an
appearance-only switch: when a texture is available it replaces a collision
`Box` with a convexified collision `Mesh`. Do not use that branch to claim a
geometry/physics-matched visual-information comparison without separate proof.

## Implemented and tested

The observer uses mutual-ratio SIFT matches (deduplicated physical image points),
forward/backward pyramidal LK refinement, valid bilinear RGB-aligned depth,
the original causal500Hz gyro rotation, camera lever arm, robust point
translation, bidirectional reprojection and distributed support. An accepted
estimate would be conditional point-correspondence motion, never a relabelled
plane-depth rank, native pose, command-based observation or calibrated bound.
Image rejection retains an unobserved transition; packet faults latch.

Tests include textured planar tangent translation with zero commands, blank/
repeated/unrelated/occluded images, depth jumps/unknowns, known rotation and camera
mount conventions, wrong matches, insufficient spatial support, clock/episode/
calibration/image-binding errors and privileged-field rejection. The textured
synthetic displacement is recovered within1mm. This is an interface/algorithm
positive, not independent physical validation.

The replay writes each complete sensor-prediction tape before loading native
pose for scoring. All80 recorded transitions remain rejected, so accepted-only
error summaries are null, not zero. Mean observer-only time is49.25ms on A and
49.09ms on B, excluding file loading and the rest of the navigation loop. It
does not solve the previously measured >100ms moving-loop latency.

## What changes next

Implement the [controlled appearance experiment](go2_rgb_information_and_whole_mission_next_steps_2026-09-06.md):
new rendered observations with independently seeded, nonsemantic surface
appearance, unchanged collision primitives/materials and verified visual-depth
surface alignment, alongside textureless and repeated-texture controls. Keep the
V1 matching rules frozen for that comparison. Do not add unique goal/location
markers as a shortcut, recolor old recordings, or rerun/relabel B as successful.

If the appearance-controlled observation supplies the missing direction, test it
on a fresh motion/layout and connect it to the continuous full-mission observer
with explicit evidence provenance and uncertainty. If it does not, retain missing
directions and use observation actions or justify additional deployment-valid
sensing. The task remains complete discovery/marker/return and matched JEPA,
online-rollout and memory contributions, not an ever-growing collection of local
perception tests. Whole-task0/2 and existing JEPA negatives remain unchanged.

## Verification and immutable artifacts

Focused98642 passed17tests before feature-location deduplication and the stronger
repeated-checkerboard fixture; focused17279 passed17 after both. Full regression
50326 passed1,833tests across148 explicit files in152.26s. First preflight48907
rejected the symlinked environment binary as an ordinary input before output
creation. After that check and all tests terminated, only the new runner's
dependency binding was corrected to one exact installed OpenCV5.0.0 binary;
the common source/symlink guard and estimator rules stayed unchanged.

Preflight96611 passed425sources/4,641inputs,16 inherited native bindings plus
the exact OpenCV binary. Diagnostic27604 completed once, exit0. Independent
reconstruction32233 reproduced43A/39B sensor frames, exit0. Source/input/native
bindings were verified before/after; no physics, GPU training, failed-controller
resume, sealed access, old source edits or hardware actuation occurred.

Output `.generated/go2_rgbd_correspondence_motion_diagnostic_v1_attempt_001`:

- Launch: `8a725bfe6fbdbd7220962d5195282660c479ed16350ea09c32c4be74064111a4`.
- Result: `b5aae296ed45dd565b27318d1d42dca06340684de25557dfe3dc1dd7776ad32b`.
- Sensor reconstruction audit: `3888530bcbf246a4e1b6f0135c6f8cc8881001e2687f9dd037875b7329aa6f2f`.

The result binds both sensor tapes and all scored rows. Every launched source,
protocol, input and output remains frozen. The overall goal stays active.
