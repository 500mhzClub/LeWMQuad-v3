# Current-position coverage-view experiment

The preceding coverage filter repeatedly blocked travel to a distant viewpoint
while the requested patch was projectable from the measured current position.
The complete failure and replay evidence are recorded in
`go2_coverage_translation_view_2026-09-17.md`.

This single exposed-maze trial changes viewpoint preference: first consider
turning at the actual measured current position, using calibrated cameras and
the observed obstacle map. Existing turning eligibility and all clearance,
reserve, stopping, dispatch and translation-coverage checks remain. No floor
under the current robot is declared observed. Projection does not resolve the
request; a mapped floor or obstacle observation must do so. A fresh aligned
view that remains unknown is recorded and excluded from immediate repetition
within 0.10 m; the existing alternate-viewpoint search then applies.

Nine focused tests passed in 1.99 seconds, including the inherited coverage
guards, current-position viewing without a floor connector, recorded-obstacle
occlusion and fallback after fresh unknown evidence. Model weights, six
candidates, exposed layout, sensors, CPUs, timing and 4800-tick budget remain.

Launcher: `scripts/run_go2_current_position_coverage_view_development.py`.
Plan: `docs/go2_current_position_coverage_view_plan_2026-09-17.json`.
Output base: `/mnt/steam_drive/LeWMQuad-v3/navigation_development_artifacts_v1`.
Root: `go2_current_position_coverage_view_supervised_rollout_noise_2mm_native_layout01_4800_v1_attempt_001`.
Existing references and model/training inputs stay at their original paths.

First launch exited before recording creation because the inherited path
validator still named the original storage base. Its frozen plan and failure
are preserved in `go2_current_position_coverage_view_prelaunch_path_failure_2026-09-17.json`.
The launcher now binds the unchanged validator to the designated output base.
No native execution occurred in that prelaunch failure. The updated source
bindings were frozen before retry. Native launch then acknowledged in session
9763, owner PID 4095278, CPUs 8–15,24–31, using the secondary recording disk.

Completed failure: owner exited 1 after a registration-stage `Full()` exception
at 4466 acquired camera frames and 22330 command intervals. Full native archival
completed. No goal or home arrival, no disallowed contacts, 4465 registered
poses, maximum position error 8.90 mm. This is a pipeline failure, not ordinary
budget exhaustion. There is no successful navigation claim.

The in-place-view mechanism was exercised: 28 plans requested those views,
selecting 27 pure turns and one hold. Three recorded visits ended in actual
observation of their requested patches. Four completed visits overall observed
a patch; two fresh aligned views remained unknown and triggered alternate-view
search. Five plan receipts explicitly report a resolved patch (a request can
resolve before a visit starts). Coverage rejected 17 translations across this
trajectory, versus 1174 in the preceding failure. These asynchronous trajectories
are different; those counts do not establish a matched causal navigation gain.

Only 356/1106 plans were on time. From frame 1600 onward, all 710 recorded plans
were late, selecting 540 forward and 170 left-arc proposals. None had every
candidate nominally blocked, and none was rejected by the coverage filter.
All 14330 command intervals in that final segment were zero requests with
`NO_ON_TIME_PLAN`. The navigation stall therefore preceded the queue exception
and was sustained by missed planning deadlines.

Median routing wall time grew from 9.05 ms in frames 0–599 to 70.76 ms in
frames 1600–4465. In the latter interval model forward time was median 8.78 ms,
total action selection 39.90 ms, and observation-to-plan completion 320 ms
against the unchanged 300-ms deadline. Profile components are nested, not
additive. The final in-flight route call took 1.053 s. The narrower routing
hotspot still needs isolation using saved inputs.

Mapping accepted frame 4464 while planning was processing 4452 with 4456/4460
waiting. Source order and recorded stages therefore indicate that registration
overflowed the two-slot planning queue. The original exception only says
`Full()` and did not name its destination; preserve that inferential distinction.
Subsequent clock-closed faults followed shutdown. The temporarily static live
log was later identified as failure archival, not a proven native deadlock.
An attempted read-only debugger attachment failed; no live stacks were obtained.

Posthoc native wall clearance at planning samples stayed at least 0.49643 m;
none crossed the 0.45-m circular nominal margin. That is separate from the
zero-contact result and is not articulated-body safety certification.

Evaluate this secondary-storage run using
`scripts/read_go2_current_position_coverage_view_development.py`; it binds the
shared forecast evaluator's unchanged root validator to this output base.
The launcher's inherited `--evaluate` first saved physical/treatment summaries
but stopped at the old forecast output-base check; the dedicated reader then
completed evaluation without changing those results. The deadline readout is
`scripts/read_go2_current_position_coverage_deadlines_development.py`.
Receipts: `current_position_coverage_view_readout_v1.json` and
`planning_deadline_diagnosis_v1.json` in the failure root. Full failure depth
and all other evidence remain; the recording disk has about 49 GiB available.

The saved routing computation has now been isolated to repeated fine-goal
connectivity search; the exact-cache intervention and its comparison are in
`go2_cached_fine_connectivity_2026-09-17.md`. This is
development on an exposed failure; no fresh-layout, JEPA-contribution,
real-time or hardware claim follows. The broader navigation goal remains open.
