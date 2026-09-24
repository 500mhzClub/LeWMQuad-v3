# RGB-D physical-configuration evidence V1: saved terminal diagnostic

New source-only consumer and one saved-sensor diagnostic. No physics, mission
resume, threshold relaxation, model fit, held-out access or hardware actuation.
Preserve the audited first fresh-maze failure. This does not authorize a turn.

Replay exactly its 219 decisions, stopping at the terminal observation 23.3 s,
before the five tail observations. Use one `DepthProposalNavigation` and its
existing RGB-D state owner. Verify every replayed decision against the bound
original. Do not send evaluator world pose, native geometry state, topology or
marker location to the adapter. Use the calibrated URDF primitive identities,
the owner's retained raw depth/up/pose/error frames and current sensed joints.

Prepare one fixed median-row-major eligible plane cell per immutable retained
frame. Bind depth, up, pose, clock and uncertainty metadata; never integrate the
owner again. A frame-relative upward plane is a hypothesis, not a ground label.
Keep complete non-floor beam evidence, full observed floor footprint, unpadded
physical gap bounds and exact four-foot contact candidates separate. Retain
covered penetration, incompatible plane witnesses and non-floor conflicts.
Missing views remain unknown and no initial setup prism is available.

Keep existing global-endpoint-sum transport radii unchanged. Explicit additional
endpoint-error hypothesis 5 mm, normal error .002, up error .001, plane-offset
error 1 mm and depth-range error 1 mm. These are uncalibrated development
hypotheses, not values fitted to the small errors in this run. Retain 4 cm
non-floor geometry padding, separately from unpadded physical ground gaps.

Query zero translation with the terminal sensed joint posture at gravity-axis
yaws 0,+30,-30,+60,-60,+90,-90 degrees, in that fixed order. These are supplied
configurations, not a gait, continuous yaw sweep, action sequence or predicted
future posture. Every result keeps `navigation_action_permitted=False` and
`future_gait_qualified=False`, even if all non-floor primitives are clear.
Check the complete zero-yaw result against independent reference floor/beam
implementations. Save every full result, summary, timing and provenance. Check
that no query changed the owner's fusion/gyro state or the controller terminal.

Before the recorded diagnostic, run synthetic tests for same-owner/no-reintegration,
unknown/missing views, wall conflicts, elevated planes and penetration, exact
foot roles, invalid/stale/modified bindings, proper transforms and reference
agreement. Reuse predecessor tests for the geometric primitives and aggregation.

Fresh exclusive output:
`.generated/go2_rgbd_physical_configuration_evidence_development_v1_attempt_001`.
Bind original launch/result/audit, every original output, inherited sources and
inputs/native/OpenCV plus this protocol, new consumer/runner/tests recursively.
Check identities before and after. Retain an infrastructure failure without
overwriting it. This diagnostic can identify residual evidence gaps; it cannot
establish maze success, future-gait safety, real-time operation, JEPA benefit,
generalization or hardware readiness. The full scientific objective is unchanged.
