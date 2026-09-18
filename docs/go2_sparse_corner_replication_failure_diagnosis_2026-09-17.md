# Replication failure diagnosis and selected-turn memory correction

The frozen replication remains 14/15 successful round trips, with its
command-history layout-1 failure preserved. The following is post-hoc
development analysis and a new intervention, not a replacement batch result.

Exact input root:
`/mnt/steam_drive/LeWMQuad-v3/navigation_development_artifacts_v1/go2_sparse_corner_replication_06_command_history_noise_2mm_native_layout01_4800_v1_attempt_001`.

The unchanged public-sensor tracker replay reproduced all 1921 recorded raw
poses exactly and rejected frame 1921. At rejection the primary camera had
zero detected corners; the auxiliary camera had 150 selected features but
failed the unchanged correspondence/gyro-consensus criteria. Eight recent
references did not qualify. A subsequent diagnostic stored-view probe also
reproduced every recorded pose: none of eight old views met the existing
revisit eligibility rule, and none of their sixteen single-camera candidate
fits qualified. Increasing old-view search cadence alone is unsupported by
this evidence. No tracker threshold or feature budget was changed.

Outputs in the input root:
`tracking_failure_replay_v1/result.json` and
`tracking_failure_view_probe_v1/result.json`.
Each replay took about 210 seconds on one sequential CPU worker.

The decision replay reconstructs map axes from the initial public force
history and matches all recorded recovery-heading errors. It reproduces all
480 recorded actions and local-memory flags. Local memory recorded 21
visually interrupted left-turn attempts. At 45 subsequent non-recovery
planning steps, clearance handling selected that failed left turn while the
right turn passed the existing clearance checks and had no matching failure.
The memory initiation checked `before_memory_filter_action` (right) instead
of the clearance-selected action (left), preventing the intended alternative.

`lewm/selected_route_turn_memory_development.py` adds a new memory implementation
that checks the selected failed turn in this case. Existing locality, target,
measured-progress, recovery, clearance, generation and completion rules remain.
The frozen predecessor implementation is unchanged. Six focused tests passed.
Isolated probes from the original memory states change exactly the 45 identified
decisions, beginning at frame 984, to the forecast-clear right turn. They do
not propagate hypothetical state or establish a navigation outcome.

Outputs: `turn_memory_failure_replay_v1.json` and
`turn_memory_selected_turn_probe_v1.json` in the failure root.

One live command-history pilot is prepared on the same exposed layout with
this memory correction alone. Plan:
`docs/go2_selected_route_turn_memory_pilot_attempt02_plan_2026-09-17.json`.
Launcher: `scripts/run_go2_selected_route_turn_memory_pilot_development.py`.
Models, tracking, mapping, six candidates, sensor noise, timing, affinity and
4800-tick budget remain fixed. This tests the diagnosed controller interaction;
it cannot establish fresh-layout reliability, a JEPA advantage or hardware
readiness. The original failure remains in the comparison regardless of the
new outcome.

The first launch exited before simulation at launch-metadata writing because
the reused runner requires a closure-free writer. Its initial plan, source
snapshot and failure record remain preserved in the attempt-001 root. The
writer was moved to module scope; controller source and scientific settings
are unchanged. The corrected launch uses attempt 002 and a separate plan.

## Completed pilot: successful navigation, memory correction unexercised

Native session 33757 (owner PID 4181724) and evaluator session 70492 both
exited zero. The independent evaluator verified a **299.80 simulated-second
round trip with zero contacts and zero pipeline faults**. Tracking covered
2992 camera pairs. Goal/home arrival frames were 2404/2989, with maximum
physical one-second dwell distances 23.24/13.45 mm; quiet arrivals passed.
All eight unique return edges reversed outbound edges (ten outbound), with
zero invalid graph transitions. Live navigation took 391.97 wall seconds.
712 of 741 plans were on time (96.09%), with 29 late.

There were **zero local turn-memory selections**. The six visual-recovery
plans concerned only the initial survey's trigger; the later repeated recovery
loop did not arise on this asynchronous trajectory. Thus the success does not
establish that the correction prevents the original failure. Keep the original
failure and frozen batch counts unchanged. Do not append attempts simply to
obtain an exercised or favorable outcome. The diagnosed decision defect and
its focused tests are supported; its closed-loop benefit remains unproven.

On 698 overlapping 700-ms executed windows, applied command-history XY RMSE
was 8.28 mm, unused neural 9.32 mm and pose-command 6.67 mm. Applied yaw RMSE
was 0.691 degrees versus unused neural 0.827 degrees. This provides no new
evidence for a learned-prediction advantage.

Result: `docs/go2_selected_route_turn_memory_pilot_result_2026-09-17.json`.
Full pilot root:
`/mnt/steam_drive/LeWMQuad-v3/navigation_development_artifacts_v1/go2_selected_route_turn_memory_command_history_noise_2mm_native_layout01_4800_v1_attempt_002`.
Its raw recording remains retained. All processes from this experiment have
finished. The goal remains incomplete: prospective reliability, isolated
persistent-map contribution, learned-prediction benefit and realistic
sensing/timing remain open. Subsequent experiments should report mechanism
activation alongside navigation outcomes rather than infer a benefit from
success alone.
