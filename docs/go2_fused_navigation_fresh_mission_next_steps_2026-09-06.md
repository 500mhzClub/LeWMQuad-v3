# Next: complete sensor-controlled maze missions

Update: the first fresh actual mission has now been executed and independently
audited. It failed on floor/pose-envelope turn evidence after a 2.137 m path, not
on a sensor fault. See the [audited result](go2_fresh_fused_maze_audited_result_2026-09-06.md)
and [current next steps](go2_floor_factored_navigation_next_steps_2026-09-06.md).
The historical starting state below describes the pre-mission integration stage.

Starting evidence: [fused navigation integration](go2_fused_navigation_integration_result_2026-09-06.md).
The full consumer now uses shared RGB-D/inertial state; measured-depth proposals
address the old green-floor assumption. Recorded proposals were not executed.
The final scientific goal is unchanged and unachieved; original whole-task0/2
and all negative results remain authoritative.

## 1. Fresh complete-task scene and physical executor

Implement a new initializer accepting an explicit construction-only maze pack.
Do not invoke `AppearancePhysicalInit` unchanged: its `fresh_pack()` fixes the
short-motion arena regardless of most supplied geometry. Reuse its gait loading,
gain readback and raw sensor/contact wrappers through a new source path; validate
that supplied wall boxes, spawn and native shapes actually match the new pack.

Use independently specified development layouts with at least a genuine junction,
wrong branch/backtracking opportunity and initially occluded marker, not the seen
short-motion arena or reused old north/south fixtures. Keep physics separate from
independently seeded appearance. Preserve the red/blue marker's visual identity:
the current appearance generator grayscales *all* boxes. A new marker-aware visual
adapter must leave collision geometry/materials unchanged, verify visual-depth
alignment, and check marker absence in initial RGB and detectability at appropriate
visible viewpoints. Marker position and topology remain evaluator-only.

Construct one DepthProposalNavigation per mission, initially with episodic route
hypotheses. Feed only captured RGB/depth/body/joints/fast gyro and actual command
acknowledgements. Never supply world pose, cell/goal coordinates, scene labels or
layout-derived commands. Ensure sensor epoch/initial velocity prior is identical
from startup to terminal tail. Do not recover a failed observer or reset error.

Before launch, freeze the scene, source/native/input inventory, physical command
limits, mission time/leg budgets, marker/return criteria and stopping procedure.
Issue controller commands, not a predetermined tape. Persist every partial tick,
command acknowledgement, raw contact/pose sample, sensor frame, decision, ledger,
intervention and terminal tail. Native guards are external simulation supervision,
not deployment sensors. Retain each stop/failure, never silently resume/retry.

## 2. Close execution-safety and uncertainty gaps without hiding failures

Floor extensions are not free body/foot trajectories. The inherited approach rule
does not validate a full future gait sweep; its braking velocity is an estimate,
not a stopping-distance guarantee. Bind any prospective forward/turn/braking
envelope to measured rays and explicitly evaluated body/foot response hypotheses.
Observed conflicts veto clearance; missing rays stay unknown. A finite supplied
starting region may cover startup only and must not become a hidden maze map.
Preserve actual nonfoot/obstacle contact and fall termination in every2ms sample.
Do not call simulation development envelopes hardware safety certificates.

Retain the current uncalibrated 80mm budget in its frozen consumers. Point-error
hypotheses accumulate linearly; many weak-plane intervals can exhaust it before a
mission finishes even while nominal tracking is good. Do not raise the threshold,
fit it to the observed maximum error or reset global uncertainty to permit motion.
If this prevents useful missions, implement a separate, justified local-relative
error consumer: transport shared sensor/anchor errors with their correlations,
retain global history and uncertainty, and test common-error cancellation against
independent noisy motion, biases, calibration offsets, aliasing and false matches.
Local relative consistency is not verified global home identity. Preserve the
earlier covariance/sensitivity work but re-derive it for the actual new point
constraints rather than copying depth-only assumptions.

Run a declared complete mission and independently audit it, even if the outcome
is a scientifically useful failure. Use the first causal failure to choose the
next repair; do not substitute an open-loop tracking assay for mission execution.

## 3. Establish the claimed outcome and isolate the JEPA contribution

Require actual discovery of an initially hidden marker, physically verified return
to the starting region, and a measured stopping tail without intervention/reset.
Report collisions/falls, false marker/home declarations, uncertainty stops, route
length, time, energy/effort proxies and full-loop latency. Keep controller terminal
candidates separate from evaluator-confirmed success. Online memory must handle
visited branches/return and reject false loop closures, not merely retain images.

Once a common system completes development missions, freeze shared sensors,
controller, low-level gait, datasets and compute budgets for geometry-only,
supervised and JEPA comparisons. Factor predictive training from actual online
no/one/multistep action-sequence rollout and from memory-off/on. Training seeds and
independently generated layouts are experimental units; repeated windows are not.
Appearance/depth/IMU/occlusion robustness and calibrated uncertainty need separate
development/validation evidence. Genuine independent final evaluation requires
external custody; never touch the protected legacy tests or treat V4 as final.

Finally resolve real-sensor calibration, synchronization, gait/URDF warnings and
deployment latency, then perform bounded hardware work when access exists. No
current simulation result establishes physical-platform readiness.
