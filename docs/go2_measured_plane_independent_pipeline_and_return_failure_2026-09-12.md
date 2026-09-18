# Revised-perception independent pipeline and return failure

Recorded 2026-09-12 Europe/London (2026-09-11 23:15 UTC). Development evidence;
the overall navigation goal remains incomplete. No independent scene was run.

## Implemented and checked

The separate measured-plane study now has its current-observation planning
comparator, fresh controller factory, and composed native collector/full raw
audit. It retains the original eight independent layouts, four arm assignments,
cyclic arm order, and exact corrected model identities. The new case names,
perception implementation, 4,000 navigation ticks, 4,014 observation limit and
14 GiB per-case collection allowance are separate from the original study.
No predecessor or live-run source was changed.

The collector and audit retain the predecessor function code. Their private
globals select the new factory/contract and the already tested extended sensor,
decision-stream, command and renderer dependencies. Cooperative session
inheritance retains the independent physical initializer. Scene specification,
private physical evaluator, contact/visibility checks and public mission
boundary remain the original independent-study implementations.

Tests completed:

- 21 tests for measured-plane current-observation planning and the independent
  study/factory: passed, 12.49 s (session 88906). This recovers the result lost
  from the previous test-session output.
- 18 new pipeline tests: passed, 10.50 s (session 1354). All four arms complete
  a synthetic 4,014-observation episode, 4,013 commands, 201,400 physical-sample
  slots, and ten zero-command drain intervals. Complete decision replay and
  real raw command auditing reject late decision, command-role, applied-command
  and timing corruption. Session tests reach the extended acquisition boundary
  and reject observation 4,014 before primary capture.

The full-budget tests substitute physical acquisition, sensor reconstruction
and learned inference. They prove orchestration and audit wiring, not new native
sensor validity or navigation. The separate comparator tests exercise actual
synthetic images and model calls. No outcome claim follows from either suite.

This is source preparation, not a frozen execution protocol. The study still
requires completed measured-plane comparison review and full input/resource
admission before launch. Its initial allowance is 520 GiB plus 32 GiB available
RAM; remaining capacity must be remeasured after pending jobs.

Scope limits are explicit: one training seed, no repeated appearance condition,
no nominal predictive arm and no no-RGB direct reference arm in this 32-case
roster. Reactive is a whole-method comparison. The current-observation arm
restricts planning cells but retains localization, contact, residual, mission
and model history; it is not fully memoryless. Additional evidence remains
necessary to distinguish all contributions in the overall goal.

## Current native collection: negative, final audit pending

Artifact root:
`/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/navigation_development_artifacts_v1/go2_measured_plane_dispatch_recovery_v1_attempt_001`.

The collection log and case `no_rgb_direct_measured_plane_maze_02/result.json`
now record:

- 3,124 observations, 3,123 completed commands, 156,900 physical samples.
- An observed outbound arrival at frame 3,062; physical verification pending.
- First terminal controller decision at frame 3,113:
  `SENSOR_OR_MODEL_FAILURE`, with `same-episode current visual evidence required`.
- Ten terminal zero-command intervals completed; no collection-level physical
  or acquisition stop. The last mission receipt remains RETURN, with one arrival
  and no verified round trip.

The collected case result SHA-256 is
`922cf4a1134eb5458010e12f89ba65fe427b839476837ab035d8ceace4181680`.
This binds the provisional collection summary, not a completed raw audit.

At inspection, parent 2916106 (creation 1789162140.42) and worker 2916239
(creation 1789162190.15) were live. Worker source executes full raw auditing
after collection/persistence/artifact checks; its decision-stream descriptor
was read-only. The parent result and worker-terminal result were not yet
present. The stable timing tail at tick 3,123 therefore did not indicate an
ongoing scene stalled on that command.

Selected recorded decisions 3,112 and 3,113 were inspected from the collected
compressed stream. This was inspection of recorded evidence, not independent
sensor reconstruction; the final worker audit must still reproduce it.

At 3,112, primary frame-to-frame registration remained available but all eight
retained references failed with insufficient rigid-pose matches. The observer
used its tenth measured incremental bridge frame. At 3,113 the primary and
auxiliary continuity records both reported `MEASURED_BRIDGE_BUDGET_EXHAUSTED`,
with incremental measurements available and no qualified retained anchor.
Primary increment support was 135 inliers; auxiliary support was 45. No camera
was selected for an admitted current pose. This explains the controller's
generic missing-current-visual-evidence failure.

This identifies a reference-retention/reacquisition problem during the return
turn as the next diagnostic target. It does not establish a validated fix,
permit longer unanchored integration, or justify relaxing the existing gates.
The earlier full-history success of measured-plane perception on a different
recorded trajectory did not establish success on this changed closed-loop path.

## Continuing work

1. Let the same live worker finish full raw sensor/controller/physical audit;
   preserve the negative result and any audit failure separately.
2. Keep the nominal and reactive waiters and full-history timing waiter tied
   to their original exact owners and completed-result checks. Do not restart
   a job because an observation times out.
3. Inspect the reference histories and paired observations around the new
   return-turn failure using the completed raw artifact identities. Validate
   any retention/reacquisition change separately before a new native run.
4. Review the resulting native comparisons before freezing a prospective
   independent population. Do not count the earlier three never-dispatched
   diagnostic scenes as completed.

## Prepared source identities

These hashes identify the current local files; they do not freeze or authorize
an execution protocol.

| Path | SHA-256 |
| --- | --- |
| `lewm/measured_plane_current_observation_planning_controller_development.py` | `d6d71909eddae6281d4a191a518ed88dbf303df3d41a68ea70dab48df646dbd9` |
| `lewm/measured_plane_independent_round_trip_study_development.py` | `8c24597cd9e2e93c51d465a52d0e7299a414f9c8db263a0c5573fc8d006f0bba` |
| `scripts/measured_plane_independent_controller_factory_development.py` | `0c93665d421687c79d477cd04e56623f04195cb008b5f448028347607b2f3b64` |
| `scripts/measured_plane_independent_multiarm_pipeline_development.py` | `e054e260d8de9cc04c1d7199109f00af0ddaa8205301795ef60d256ae44cad03` |
| `lewm/tests/test_measured_plane_current_observation_planning_development.py` | `e439d212874b7527ec86bd68e97a47ad0cb4ba7904bcf9ba89c75b0341ebff8e` |
| `lewm/tests/test_measured_plane_independent_study_development.py` | `ae9b0a111c7279612a63512b82cb5b90a06e3a985a66385d770aa4550d354292` |
| `lewm/tests/test_measured_plane_independent_pipeline_development.py` | `8f5ca02b9c481f4cff59fb9a8a6c215c2bfc1ff63fe1305c72c68ae93b0d3eef` |
