# Frozen full-stream multi-reference replay V1

This is an offline replay of the three completed, independently raw-audited
coupled-return sensor streams. It issues no commands, launches no physics or
training, and cannot establish a counterfactual recovered return. Preserve
the original 0/3 outcomes and 12/13 native-passing local holds.

Fixed inputs: `.generated/go2_coupled_room_return_v1_attempt_001`, launch
`be4267ab90f122e18ca8ef8f260cacc2150cdb6aed57f568b6170863c41e0fbf`, result
`b5ac72bf6ad35df56d99f0dc9be00ae19208d091e805aeaea4501e8d0152cf98`, raw audit
`2d25a7ce77d4c7c5cf3688f50c6a5f17d872243b10d4bf613554ebca6c97c099`.
Replay all actual frames, including recorded failure drains, in order
nominal_left, nominal_right, lower_friction_left. No invented future frames.

MultiReferenceRGBDPose preserves the primary reference while it passes the
unchanged RGB-D registration and consecutive pose-increment gates. On rejection,
try the remaining members of the eight most recent accepted reference frames,
with their copied visual poses and common integrated-gyro frame. Each candidate
must pass the exact original pair and increment gates. Reject if no candidate
passes or if qualified alternatives disagree by more than .02 m or .10 rad.
These consistency thresholds are diagnostic, not calibrated uncertainty.
Choose qualified, mutually consistent alternatives by descending minimum grid
coverage, inlier fraction, inlier count, ascending residual, then recency.
Never rank by native error. Promote the current qualified observation after a
fallback. Retain the original motion/support promotions otherwise. No pose
reset, gate relaxation, contact-based translation, extrapolation or retry after
an actual terminal observer failure. Feature storage is bounded to eight
references; pose-only history remains recorded as before.

The explicit cached reader validates the same sensor manifests, tensor fields,
frame paths and packet contracts as the existing readers. It caches immutable
body/gyro histories and returns copied packet arrays; it reads no native pose,
geometry, future-command tape or evaluation data. Focused equivalence tests
compare against the original readers at ordinary and failure frames. Verify
all input/source byte bindings before and after replay so caching does not hide
changed input bytes. Cache loading is an offline efficiency measure, not a
claim of online or deployment-valid sensor availability.

Require exact equality with the original saved current poses before the first
new fallback wherever original observations exist. Record all frame estimates,
first terminal failure, fallback choices and qualified/rejected alternatives.
Only AFTER the stream's estimator outputs are produced, load native poses to
evaluate available translation error. Report all unavailable frames and any
new terminal failure, not just the two previously inspected failure frames.
Replaying original executed commands/sensor trajectories is not replaying the
counterfactual commands or trajectory of a new closed-loop controller.

Freeze the new persistent return-intent implementation and focused tests with
this source review, but do not run that mission on old sensor tapes. Its goals
would change commands. It preserves one corner approach heading across clipped
legs, uses travel heading for clipped home legs and home heading0 only on the
actual final-home target. Full original pose/hold criteria and no-home-claim
semantics remain. Model-matched synthetic returns are source tests, not physics.

Exclusive output `.generated/go2_multi_reference_recorded_replay_v1_attempt_001`.
No overwrite, retry or source edits after launch; terminal failure is retained.
No claim of calibrated pose uncertainty, traversability, hardware, JEPA benefit
or full-goal success follows from this replay. A fresh, separately frozen
closed-loop test and the independent-layout maze/JEPA/memory experiment remain.
