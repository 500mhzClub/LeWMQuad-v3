# Recorded anchor-loss sequence after the height intervention

Read-only inspection of the completed native height result
32edbb748e04e18816e0b0fb265f465706ac8c8684849d73a091321f79a3b07f,
under go2_partial_floor_height_maze01_scoped_verification_pilot_v1_attempt_001.
The result and its bound full_jepa_partial_floor_height_maze_01 decision stream
were authenticated before inspection; the stream was rehashed afterward.
Decision-stream SHA-256:
a2516a79dc2bba4e00cbde225b2b49d64d7cf74952d29dd8df8037e3a9960b2c.
Inspection48763 exited0; compact aggregate50395 exited0. No model, scene,
counterfactual pose or alternative action outcome was evaluated.

The last qualified retained-reference measurement occurs at observation634,
using auxiliary-camera reference633. Observation633 had been retained after
14 inliers from62 reference features (22.58% overlap). Observation634 has
18 inliers from34 reference features (52.94% overlap), so the unchanged half
overlap rule does not retain it. Its pose is already anchor-qualified, but
promoted_keyframe is false. This is a recorded retention decision, not proof
that retaining every view would improve navigation or bound accumulated error.

Observations635 through644 use ten successive measured auxiliary increments,
with bridge_frames increasing exactly1 through10. The first requests left arc
[.16,0,.45]; the following nine request left turn[0,0,.45]. The retained reference
population stays626 through633. Across635 through645 inclusive, all88 attempted
retained-reference registrations in each camera report insufficient rigid-pose
matches (176 recorded rejections total). This count describes final recorded
camera selections, not every internally repeated attempt during fallback.

At645 both direct-corner-flow incremental pairs from644 qualify:31 valid depth
pairs in the primary camera and33 in the auxiliary camera. Their pair failures
are null. The fallback still fails because the ten-frame bridge is exhausted
without a retained anchor. Current pose is absent; accepted controller tick
remains644 and the command is zero with SENSOR_OR_MODEL_FAILURE. These are
qualified incremental measurements, not an accepted current absolute pose.

The source corroborates the recorded sequence:
lewm/dual_camera_anchor_pose_development.py only remembers non-bridge poses
when motion/support/overlap promotion triggers;
lewm/overlap_retention_joint_observer_development.py retains an otherwise
unpromoted anchor-qualified view when2*inliers <= reference_features;
lewm/temporal_anchor_continuity_development.py never promotes a bridge pose
and rejects an eleventh bridge increment. The direct-flow fallback changes
only immediately preceding-frame association and cannot restore an older
retained reference by that route.

Next investigate retaining recent already anchor-qualified observations and
matching current observations to retained references. Any proposed change must
have a separate causal replay and fresh physical experiment, preserve qualified
conflict vetoes and registration gates, and measure accumulated pose error and
actual navigation outcomes. Do not tune the overlap threshold merely to include
52.94%, extend the bridge budget, retroactively relabel a bridge, or infer a
successful return from this recorded sequence. No controller change was made.
