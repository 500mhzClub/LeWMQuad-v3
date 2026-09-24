# Prospective native nominal-reentry intervention V1

Run `scripts/run_go2_nominal_reentry_maze_pilot_v1.py --prefix-result-sha256 HASH`
once at exclusive root `go2_nominal_reentry_maze_pilot_v1_attempt_001`, binding
completed prefix result
`8ece4c85f9226265b5c6a425ff589071024738c815513cb0f0d34063e02bb77b`.
The first native failure and readout are authenticated through that prefix's
exact input bindings and remain unchanged.

## Scientific change and assignment

The only controller change is the explicit nominal-clearance reentry rule in
`lewm/nominal_clearance_reentry_development.py`, as specified and tested in
`docs/go2_nominal_reentry_maze_prefix_v1_2026-09-08.md`. At the first predicted
intervention, observation 407, the original zero wait becomes a left turn. Its
forecast never gets closer than the already violated current nominal clearance
over all eight segments and predicts positive first-step clearance gain; its
original surface and phase checks pass. This is an experimental policy exception;
the original 0.45 m veto stays recorded, and no physical improvement is guaranteed.

Execute only `full_jepa_novel_maze_00` with unchanged
`seed_2026091001_full_jepa`, corrected state
`4f796afdf32c2c6dbcd1d5981834a7b17be86e2efdafd9427b2d80240def0ca6`.
This reuses development maze 0, its initial state, scene/appearance seeds and
coordinate-only outbound/home instructions. It is not a new independent maze
evaluation. No fitting, checkpoint selection, camera/raster/gait change, pose
integration, native-state feedback, global radius change or phase expansion is
performed. All new collector/audit identities are explicit separate files.

The shared 3,000-interval outbound/return budget, three warmups, ten terminal
zeros, original 4 cm observed arrival region and ten completed quiet intervals
remain. Observation, map/history/residual persistence, sensor failures, mission
deadline and terminal latching are inherited. The original physical setup/contact
and visibility checks remain required; no predecessor attempt is resumed.

## Verification

First, collect a fresh native scene and preserve every compressed decision,
requested/applied command, native trace, primary/auxiliary image/depth and audit
receipt. Next, independently reconstruct every public packet and complete
decision with a fresh unchanged model, and audit native dispatch/slew, clock,
contacts, setup, raw sensor/pixel metrics and physical visibility. Apply the
unchanged independent outbound/return dwell and actual reverse-edge evaluator.
Every failed gate remains a failure.

Additionally compare the two native attempts through the prospectively bound
intervention observation: the first 408 observed frames and first 21,100 physics
samples must match exactly. Compare full public primary/body/fast/auxiliary
packets, observer/map/mission receipts, raw forecasts and original constraints.
Requests before 407 must match; request 407 must equal the prefix's new request.
No equality claim extends to outcomes after the changed command. Prefix failure
prevents attribution and is preserved with generated artifact identities.

Two scope tests verify that collector changes are limited to controller/status
identity and the explicit storage envelope, and that raw auditing and goal
verification remain unchanged while the reused layout is correctly labeled.
Six prefix-comparison tests exercise matched observations through intervention
and rejection of changed physics, forecast, public packet, earlier command or
missing observation. These do not replace native execution/audit.

## Explicit resource change and limits

Assess hardware and live jobs before launch. Use one CPU native scene, one
thread per numerical library, at least 32 GiB available RAM and 15-second resource
monitoring. Current space no longer meets the predecessor's 12 GiB collection
allowance plus 1 GiB persistence headroom above 40 GiB reserve. This new attempt
therefore declares **11 GiB collection allowance plus the same 1 GiB persistence
headroom and 40 GiB reserve**. The original collector is not changed.

The smaller allowance may censor this attempt earlier for storage; it is not a
claim that every 3,000-interval trajectory fits. Stop if volume consumption since
collection start reaches 11 GiB or fewer than 41 GiB remains. Other writers count
against the conservative volume check. Persist and report any storage stop; do
not change the budget or delete dependencies to force completion. The resource
envelope is an admission/monitoring rule, not an enforced OS memory limit.

Record receipt-write and complete iteration timing. Physics remains paused during
computation, with ideal body/gyro and zero acquisition-latency assumptions. No
real-time, hardware, general navigation or learned-planning/memory advantage is
claimed. Matched direct/reactive/non-predictive cases and new layouts remain
outstanding even if this recovery produces a successful reused-maze round trip.
