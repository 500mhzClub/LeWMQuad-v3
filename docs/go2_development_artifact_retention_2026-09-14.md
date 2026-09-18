# Development artifact retention

September 18: end the old full-depth success-reference pin for the workspace
`go2_hold_relative_recovery_pulse_round_trip_native_layout06_4800_v1_attempt_001`.
Its native owner exited 0, both physical arrivals passed, and its timing,
tracking and treatment-activation analyses are complete. No current training
or pending raw replay uses it. The newer full sparse-corner JEPA/ablation pair,
reactive reference, command-history failure and all four return-routing-memory
recordings remain untouched. They now retain the current navigation evidence.
Retire only the 8,124 exact primary/auxiliary depth NPZ leaves from this old
success; preserve every non-depth file, including RGB, body/gyro, physics,
commands, results and diagnoses. Exact historical sensor replay will require
regeneration and is not promised to reproduce its closed-loop trajectory.
Receipt: `.generated/depth_retirement_old_layout06_success_2026-09-18/`.
Completed: 8,124 leaves retired, reclaiming 6,673,707,008 allocated bytes
(6.22 GiB). All 8,160 non-depth file hashes were unchanged. Free workspace
space afterward was 7,416,713,216 bytes (6.91 GiB). A `depth_retention.json`
marker records the intentionally unavailable historical depth replay.

September 18: deduplicated 96 visual-mesh paths in the completed workspace
balanced-start collection. Replaced 86 byte-identical copies with hard links,
recovering 286,580,736 allocated bytes (273.3 MiB). All 96 paths and hashes
were preserved. No RGB, physics, commands, checkpoints or depth changed.
Receipt: `.generated/balanced_start_mesh_deduplication_2026-09-18/`.
The mesh files share inodes now and must remain immutable.

September 17: retired only the 232 unused depth arrays (194,335,299 bytes)
from root-volume `go2_direct_visual_feedback_pilot_v1_attempt_001/case_00`.
This completed, analysed success exactly reproduces the physical trajectory
of the fully retained visual-arrival case 0. No depth replay or fit is pending.
All RGB, physics, commands and results remain, along with every failed trial.
Receipt: `.generated/depth_retirement_duplicate_direct_feedback_success_2026-09-17/`.

Prospective RGB-only policy for `go2_fresh_visual_goal_comparison_v1_attempt_001`:
retain every RGB frame, physics/contact trace, requested/applied command, body/
gyro record, source/model identity and outcome for successes and failures.
Do not write unused native/processed depth arrays. Depth is not an input to
either controller in this comparison. The existing separate render sequence
is preserved and its depth output discarded; no historical data is removed.
Each recording has `recording_policy.json`. The short native check reproduced
all eleven original reference RGB frames exactly. This changes future logging,
not the retention of any preceding failed trial.

September 17: retired 232 unused depth arrays (193,930,370 bytes) from the
completed, analysed, contact-free visual-arrival pilot `case_01` only, under
the dedicated experiment-volume `go2_dense_visual_arrival_pilot_v1_attempt_001`.
Keep case 0 in full, all RGB/physics/commands/results from both successes, and
all preceding failures. The two successes execute identical commands and have
identical physical outcomes; no fit or pending replay uses this duplicate depth.
Receipt: `.generated/depth_retirement_duplicate_visual_arrival_2026-09-17/`.
Exact historical case-1 depth replay requires regeneration.

Retire diagnosed, superseded rectangle-map depth and the completed unexercised
memory pilot before the return-only routing-memory comparison, September 17.
Exact dedicated-volume roots:

- `go2_frozen_readout_navigation_01_command_history_noise_2mm_native_layout01_4800_v1_attempt_001`
- `go2_frozen_readout_navigation_03_supervised_rollout_noise_2mm_native_layout01_4800_v1_attempt_001`
- `go2_frozen_readout_navigation_04_untrained_noise_2mm_native_layout01_4800_v1_attempt_001`
- `go2_selected_route_turn_memory_command_history_noise_2mm_native_layout01_4800_v1_attempt_002`

The first three failures remain failures in the full four-arm population. Their
physical/forecast comparisons, targeted pixel diagnosis and corrected-geometry
verification are complete in `docs/go2_frozen_readout_navigation_2026-09-17.md`
and `docs/go2_polygon_floor_coverage_2026-09-17.md`. No raw replay or training
input is pending. Keep the full predecessor JEPA arm 2 as the original
rectangle-map failure reference; this explicitly ends the earlier all-four
depth pin after its diagnostic purpose is complete. The pilot's independent
evaluation and activation analysis are complete; no local memory was exercised
and no raw replay is pending. Preserve every outcome, failure record, diagnosis,
configuration, source/model identity, RGB/body/gyro, physics, command and pose
record. Keep current sparse-corner failure 6, original transfer tracking failure,
exercised-memory pair 11/12 and reactive reference 15 in full, along with all
other unresolved inputs. Only inventoried regular single-link depth NPZ leaves
are retired; these four historical full-sensor replays become unavailable.
Receipts: `.generated/depth_retirement_return_memory_headroom_2026-09-17/`.
Completed: 34,814 exact leaves retired, reclaiming 11,205,955,584 allocated
bytes (10.44 GiB). All preserved top-level JSON hashes matched. Free space
afterward: 14,154,657,792 bytes. Every unsuccessful outcome remains recorded.

End the replication-layout-1 JEPA success depth pin after completed batch
analysis and failure diagnosis, September 17. Exact dedicated-volume root:
`go2_sparse_corner_replication_09_jepa_noise_2mm_native_layout01_4800_v1_attempt_001`.
Its physical, tracking, forecast, backtracking and memory analyses are complete;
no raw replay or training input is pending. The failure's exact tracker replay
and decision-memory replay identify the next intervention using the retained
command-history failure itself. Preserve that full failure (6), the current
exercised-memory JEPA/ablation pair (11/12), the final reactive reference (15),
every other failure and every non-depth record. This supersedes the earlier
success-9 diagnostic depth pin. Retire only inventoried regular single-link
primary/auxiliary depth NPZ leaves; exact success-9 sensor replay becomes
unavailable. Receipts:
`.generated/depth_retirement_sparse_corner_replication_09_2026-09-17/`.
Completed: 7318 leaves reclaimed 2,269,904,896 allocated bytes (2.11 GiB),
with preserved JSON hashes matched. Free space afterward: 5,209,505,792 bytes.

Apply the prospective per-run success policy to completed replication
assignment 14, September 17. Exact dedicated-volume root:
`go2_sparse_corner_replication_14_command_history_noise_2mm_native_layout02_4800_v1_attempt_001`.
Physical, tracking, forecast, backtracking and memory-execution analyses are
complete; no raw replay or training input is pending. Preserve every non-depth
record, the full failed command-history run 6 and JEPA diagnostic control 9,
the current JEPA/ablation memory pair 11/12, and every other failure. Retire
only inventoried regular single-link primary/auxiliary depth leaves; exact
sensor replay becomes unavailable. Receipts:
`.generated/depth_retirement_sparse_corner_replication_14_2026-09-17/`.
Completed: 5212 leaves reclaimed 1,587,924,992 allocated bytes (1.48 GiB),
with preserved JSON hashes matched. Free space afterward: 4,429,639,680 bytes.

Apply the prospective per-run success policy to completed replication
assignment 13, September 17. Exact dedicated-volume root:
`go2_sparse_corner_replication_13_supervised_rollout_noise_2mm_native_layout02_4800_v1_attempt_001`.
Physical, tracking, forecast, backtracking and memory-execution analyses are
complete; no raw replay or training input is pending. Keep all non-depth
evidence, current full JEPA/ablation memory pair 11/12, earlier full failure/
JEPA diagnostic pair 6/9, and every other failure. Remove only inventoried
regular single-link primary/auxiliary depth leaves; exact replay becomes
unavailable. Receipts:
`.generated/depth_retirement_sparse_corner_replication_13_2026-09-17/`.
Completed: 4738 leaves reclaimed 1,423,773,696 allocated bytes (1.33 GiB),
with preserved JSON hashes matched. Free space afterward: 4,801,986,560 bytes.

Retire completed preceding-layout reactive success depth after its analyses,
September 17. Exact dedicated-volume root:
`go2_sparse_corner_replication_07_reactive_feedback_noise_2mm_native_layout01_4800_v1_attempt_001`.
Its physical, tracking, selector, backtracking and forecast analyses are
complete; no raw replay or training input is pending. End its prior depth pin,
preserving every non-depth record. Keep same-layout command-history failure
6 and JEPA diagnostic control 9 in full, plus current exercised-memory JEPA
12 and ablation 11, and every other failure. Only inventoried regular
single-link primary/auxiliary depth leaves are retired; exact historical
sensor replay becomes unavailable. Receipts:
`.generated/depth_retirement_sparse_corner_replication_07_2026-09-17/`.
Completed: 4444 leaves reclaimed 1,304,010,752 allocated bytes (1.21 GiB),
with preserved JSON hashes matched. Free space afterward: 5,132,017,664 bytes.

Retire completed replication-layout-1 supervised success depth, September 17.
Exact dedicated-volume root:
`go2_sparse_corner_replication_10_supervised_rollout_noise_2mm_native_layout01_4800_v1_attempt_001`.
Its completed five-arm comparison and all physical, tracking, backtracking and
prediction analyses are recorded; no raw replay or training input is pending.
End its prior depth pin. Keep the same-layout command-history failure and
JEPA/reactive diagnostic controls, the final-layout ablation, and every other
failure in full. Preserve all non-depth records. Only inventoried regular
single-link primary/auxiliary depth leaves are retired; exact replay becomes
unavailable. Receipts:
`.generated/depth_retirement_sparse_corner_replication_10_2026-09-17/`.
Completed: 4606 leaves reclaimed 1,424,228,352 allocated bytes (1.33 GiB),
with preserved JSON hashes matched. Free space afterward: 5,426,475,008 bytes.

After the completed replication-layout-1 five-arm comparison, retire two
fully analysed success-depth recordings, September 17. Exact dedicated roots:

- `go2_sparse_corner_comparison_03_command_history_noise_2mm_native_layout00_4800_v1_attempt_001`
- `go2_sparse_corner_replication_08_jepa_no_route_turn_memory_noise_2mm_native_layout01_4800_v1_attempt_001`

The predecessor comparison and the current local-memory paired analysis are
complete. Neither current memory arm exercised a local memory selection; its
decisions, timing, poses and forecasts remain recorded. No raw replay or
training/fit input needs these two successes. This ends their earlier depth
pins. Keep current assignments 6 (failure), 7 (reactive), 9 (JEPA), and 10
(supervised) in full for diagnosis, along with every other failure. Preserve
all non-depth artifacts. Retire only inventoried regular single-link primary
and auxiliary depth NPZ leaves; exact sensor replay becomes unavailable.
Receipts: `.generated/depth_retirement_completed_sparse_corner_controls_2026-09-17/`.
Completed: 9936 leaves reclaimed 3,031,957,504 allocated bytes (2.82 GiB),
with preserved JSON hashes matched. Free space afterward: 5,723,889,664 bytes.

Retire completed predecessor JEPA and supervised success depth, September 17.
Exact dedicated-volume roots:

- `go2_sparse_corner_comparison_01_jepa_noise_2mm_native_layout00_4800_v1_attempt_001`
- `go2_sparse_corner_comparison_02_supervised_rollout_noise_2mm_native_layout00_4800_v1_attempt_001`

The full five-arm predecessor study and all per-run analyses are complete;
neither is a pending raw replay or training/fit input. This ends their earlier
depth pins. Keep replication assignments 6–9 in full as the current same-layout
diagnostic comparison and the predecessor command-history success as an older
same-arm counterpart for the failure. Preserve every failure and all non-depth
records. Remove only inventoried regular single-link primary/auxiliary depth
leaves; exact historical sensor replay becomes unavailable. Receipts:
`.generated/depth_retirement_superseded_sparse_corner_learned_successes_2026-09-17/`.
Completed: 9300 leaves reclaimed 2,802,749,440 allocated bytes (2.61 GiB),
with preserved JSON hashes matched. Free space afterward: 4,437,606,400 bytes.

Replace the predecessor no-local-memory success depth reference with completed
replication assignment 8, September 17. Retire only exact dedicated-volume root
`go2_sparse_corner_comparison_05_jepa_no_route_turn_memory_noise_2mm_native_layout00_4800_v1_attempt_001`
depth. The predecessor study and all analyses are complete; it is neither a
pending raw replay nor a training input. Keep new assignment 8 in full for the
same-layout comparison, assignments 6/7, every failure, and the predecessor's
remaining full records. This ends only its predecessor no-local-memory depth
pin. Preserve all non-depth evidence; remove only inventoried regular
single-link primary/auxiliary depth leaves. Exact sensor replay becomes
unavailable. Receipts:
`.generated/depth_retirement_superseded_sparse_corner_no_local_memory_2026-09-17/`.
Completed: 4698 leaves reclaimed 1,408,303,104 allocated bytes (1.31 GiB),
with preserved JSON hashes matched. Free space afterward: 4,394,123,264 bytes.

Replace the predecessor reactive success depth reference with the current
same-layout diagnostic pair, September 17. Retire only depth at exact root
`go2_sparse_corner_comparison_04_reactive_feedback_noise_2mm_native_layout00_4800_v1_attempt_001`
on the dedicated volume. Its five-arm study and all tracking, physical,
selector, backtracking and prediction analyses are complete, with no pending
raw replay or fit input. Keep replication assignments 6 (command-history
failure) and 7 (reactive success) in full for diagnosis, every other failure,
and the predecessor's other four full recordings. This explicitly revises the
predecessor full-population pin for its reactive success only. Preserve all
non-depth evidence. Only inventoried regular single-link primary/auxiliary
depth leaves are retired; exact original replay becomes unavailable. Receipts:
`.generated/depth_retirement_superseded_sparse_corner_reactive_2026-09-17/`.
Completed: 3846 leaves reclaimed 1,195,057,152 allocated bytes (1.11 GiB),
with preserved JSON hashes matched. Free space afterward: 4,819,869,696 bytes.

Release the completed replication-layout-0 JEPA success depth pin after its
five-arm layout analysis, September 17. Exact dedicated-volume root:
`go2_sparse_corner_replication_01_jepa_noise_2mm_native_layout00_4800_v1_attempt_001`.
All physical, tracking, backtracking and prediction analyses are complete;
no raw replay or training input is pending. Retain the original full five-arm
sparse-corner comparison and the new layout-1 command-history failure in full.
This explicitly ends only the new-layout assignment-1 success pin. Retire
inventoried regular single-link primary/auxiliary depth leaves; preserve every
non-depth record and failure. Exact historical sensor replay becomes
unavailable. Receipts:
`.generated/depth_retirement_sparse_corner_replication_01_2026-09-17/`.
Completed: 4678 leaves reclaimed 1,454,280,704 allocated bytes (1.35 GiB),
with preserved JSON hashes matched. Free space afterward: 5,224,710,144 bytes.

Apply the prospective replication policy to completed assignment 5:
`go2_sparse_corner_replication_05_jepa_no_route_turn_memory_noise_2mm_native_layout00_4800_v1_attempt_001`
on the dedicated volume. Its owner/evaluator exited zero; all physical,
tracking, backtracking, forecast and ablation analyses are complete. No raw
replay or training input is pending. Keep the preceding full five-arm
comparison, new-layout JEPA reference, every failure and all non-depth files.
Retire only inventoried regular single-link primary/auxiliary depth leaves;
exact sensor replay becomes unavailable. Receipts:
`.generated/depth_retirement_sparse_corner_replication_05_2026-09-17/`.
Completed: 5238 leaves reclaimed 1,627,275,264 allocated bytes (1.52 GiB);
preserved JSON hashes match. Free space afterward: 5,242,757,120 bytes.

Apply the same prospective replication policy to completed assignment 4:
`go2_sparse_corner_replication_04_reactive_feedback_noise_2mm_native_layout00_4800_v1_attempt_001`
on the dedicated volume. Its owner/evaluator exited zero; physical arrival,
tracking, forecast, selector and backtracking analyses are complete. No raw
replay or training/fit input is pending. Retire only inventoried regular
single-link primary/auxiliary depth leaves, preserve all non-depth evidence,
the preceding full comparison and new-layout JEPA reference, and every failure.
Exact sensor replay becomes unavailable. Receipts:
`.generated/depth_retirement_sparse_corner_replication_04_2026-09-17/`.
Completed: 5094 leaves reclaimed 1,539,358,720 allocated bytes (1.43 GiB);
preserved JSON hashes match. Free space afterward: 5,609,308,160 bytes.

Apply the prospective sparse-corner replication success-retention policy,
September 17, to these exact dedicated-volume roots:

- `go2_sparse_corner_replication_02_supervised_rollout_noise_2mm_native_layout00_4800_v1_attempt_001`
- `go2_sparse_corner_replication_03_command_history_noise_2mm_native_layout00_4800_v1_attempt_001`

Both owners and evaluators completed successfully. Their physical arrival,
backtracking, timing, tracking and executed-window forecast analyses are
complete in `docs/go2_sparse_corner_replication_2026-09-17.md` and saved
readouts. Neither is a pending raw replay or training/fit input. Preserve every
non-depth record, the full new-layout JEPA reference, the previous complete
five-arm comparison and every failure. Only inventoried regular single-link
primary/auxiliary depth NPZ leaves are retired. Exact sensor replay becomes
unavailable for these two completed successes. Receipts:
`.generated/depth_retirement_sparse_corner_replication_02_03_2026-09-17/`.
Completed: 9882 depth leaves reclaimed 3,003,097,088 allocated bytes
(2.80 GiB); preserved JSON hashes match. Free space afterward was
5,951,205,376 bytes. No failure or non-depth recording was retired.

Retire two superseded completed tracking-success depth recordings, September 17.
Exact roots on the dedicated navigation volume:

- `go2_sparse_corner_completion_jepa_noise_2mm_native_layout01_4800_v1_attempt_001`
- `go2_route_turn_memory_transfer_02_supervised_rollout_noise_2mm_native_layout01_4800_v1_attempt_001`

Their full saved tracking/support replays (where applicable), physical arrival,
backtracking and forecast analyses are complete. The pilot's corrected V2
corridor result is retained alongside its original metadata error. The complete
five-treatment sparse-corner comparison now supersedes these as successful
tracking references; keep all five current recordings and the original transfer
JEPA failure in full. No pending raw replay or training/fit input needs these
two successes. This review ends their earlier full-depth pins. Preserve every
non-depth artifact and every failure. Only inventoried regular single-link
primary/auxiliary depth NPZ leaves may be removed; exact historical sensor
replay becomes unavailable. Inventory and completion receipts:
`.generated/depth_retirement_superseded_tracking_successes_2026-09-17/`.
Completed: 11,054 depth leaves reclaimed 3,365,265,408 allocated bytes
(3.13 GiB); all preserved top-level JSON hashes match. Free space afterward
was 8,427,798,528 bytes. No failure recording or non-depth file was retired.

Retire the superseded interrupted-view pilot success depth, September 17.
Exact dedicated-volume root:
`go2_interrupted_view_replan_supervised_rollout_noise_2mm_native_layout01_4800_v1_attempt_001`.
Its physical, coverage, timing, forecast and repeatability analyses are complete
in `docs/go2_interrupted_view_replan_2026-09-17.md` and subsequent comparison
notes. The interrupted-view rule did not activate in that successful pilot.
No raw replay is pending and it is not a training/fit input. End its old
full-depth success-reference pin; retain the current sparse-corner success,
the full fresh-maze model comparison and every failure. Preserve all non-depth
files. Exact original sensor replay becomes unavailable. Receipt directory:
`.generated/depth_retirement_superseded_interrupted_view_success_2026-09-17/`.
Completed: 6120 leaves reclaimed 1,977,786,368 allocated bytes (1.842 GiB);
preserved JSON hashes match. Free space afterward: 8,246,665,216 bytes.

Retire four superseded completed successes after the sparse-corner pilot,
September 17. Exact roots on the dedicated navigation volume:

- `go2_polygon_floor_jepa_readout_noise_2mm_native_layout01_4800_v1_attempt_001`
- `go2_polygon_floor_repeatability_02_supervised_rollout_noise_2mm_native_layout01_4800_v1_attempt_001`
- `go2_polygon_floor_repeatability_03_supervised_rollout_noise_2mm_native_layout01_4800_v1_attempt_001`
- `go2_interrupted_route_turn_memory_jepa_noise_2mm_native_layout01_4800_v1_attempt_001`

Their geometry, repeatability, return, prediction and memory analyses are
complete in the corresponding polygon-floor/repeatability/interrupted-memory
notes. No further raw-depth replay is pending and none is a training/fit input.
This review ends their full-depth success pins. Keep the complete fresh-maze
JEPA failure and supervised success, the successful sparse-corner pilot, and
all failed counterparts. These retain current tracking, geometry, memory and
comparison evidence. Preserve every outcome and all non-depth files; only
inventoried regular single-link primary/auxiliary depth NPZ leaves are retired.
Exact historical sensor replay of these four successes becomes unavailable.
Inventory and receipts:
`.generated/depth_retirement_superseded_polygon_memory_successes_2026-09-17/`.
Completed: 26,486 exact depth leaves retired; 8,283,889,664 allocated bytes
(7.72 GiB) reclaimed. Preserved top-level JSON hashes match. No failure
recording was retired. The dedicated volume had 11,587,952,640 bytes free.

Retire two superseded view-replan repeatability success recordings, September 17.
Exact roots on the dedicated navigation volume:
`go2_view_replan_repeatability_01_jepa_noise_2mm_native_layout01_4800_v1_attempt_001`
and
`go2_view_replan_repeatability_02_supervised_rollout_noise_2mm_native_layout01_4800_v1_attempt_001`.
Their fixed comparison, physical evaluations, backtracking and executed-window
forecast analysis are complete in `docs/go2_view_replan_repeatability_2026-09-17.md`.
The later bias/readout studies are complete and neither success is a raw-depth
training input. No raw replay of either success is pending. This review ends
their old success-reference pins: the full polygon-floor pilot, all four
polygon-floor repetitions, and the new exercised route-turn-memory success
retain the current geometry, model, recovery and comparison evidence. Keep both
older failed counterparts in full, along with every other active failure.
Only inventoried native primary/auxiliary depth NPZ leaves are retired. Preserve
all RGB, trajectory, model/source identities, commands, scores and outcomes.
Exact historical full-sensor replay of these two successes becomes unavailable.
Inventory and receipts: `.generated/depth_retirement_superseded_view_replan_successes_2026-09-17/`.
Completed: 12,782 exact depth leaves retired, 3,950,735,360 allocated bytes
(3.68 GiB) reclaimed. Preserved top-level JSON hashes match and both roots have
`DEPTH_RETIRED` markers. No failure recording was retired. The dedicated volume
had 8,355,549,184 bytes (7.78 GiB) free afterward.

Next native recordings may use the dedicated project directory
`/mnt/steam_drive/LeWMQuad-v3/navigation_development_artifacts_v1`, created
September 17 with 55,800,016,896 bytes free. This is local recording storage,
not a change to the source workspace. Existing recordings and model/training
inputs remain at their recorded paths; nothing was moved. Each new launcher
must name its actual output base and continue reading frozen reference/model
inputs from their original locations. The current coverage-view attempt stays
on RecoveryStorage through archival. Apply the same retention policy and sealed
exclusions on the additional volume; do not delete unrelated drive contents.

Retire completed superseded success-reference depth, September 17.
Exact root: `go2_recovery_survey_transfer_limited_survey_jepa_noise_2mm_native_layout01_4800_v1_attempt_001`.
The four-run survey transfer comparison and its turn-cycle, physical, forecast, timing and backtracking analyses are complete in docs/go2_recovery_survey_transfer_2026-09-16.md. The new full turn-cycle and clearance-entry failures supersede this old raw turning reference. No raw replay is pending.
This review ends the older full-depth success pin for this exact root. It is
not a current training or correction-fit input. Keep the full host-clock and
publication-fix successes, all current failure/debugging inputs and every
outcome and non-depth file. Only inventoried regular single-link native depth
leaves are retired. Exact original depth replay becomes unavailable.
Inventory: `.generated/depth_retirement_completed_survey_transfer_reference_2026-09-17/`.
Completed: 8596 leaves reclaimed 2990039040 allocated bytes; all non-depth identities and JSON hashes matched.

Retire completed superseded success-reference depth, September 17.
Exact root: `go2_nogil_recovery_limited_survey_jepa_noise_2mm_native_layout00_4800_v1_attempt_001`.
The startup-deferral repair and its fixed transfer/comparison work are complete. Current full corrected-runtime and exposed failure recordings retain the same startup mechanism for debugging. No raw replay of this predecessor is pending.
This review ends the older full-depth success pin for this exact root. It is
not a current training or correction-fit input. Keep the full host-clock and
publication-fix successes, all current failure/debugging inputs and every
outcome and non-depth file. Only inventoried regular single-link native depth
leaves are retired. Exact original depth replay becomes unavailable.
Inventory: `.generated/depth_retirement_completed_startup_repair_reference_2026-09-17/`.
Completed: 5858 leaves reclaimed 1839263744 allocated bytes; all non-depth identities and JSON hashes matched.

Retire completed superseded success depth, September 17.
Exact root: `go2_post_repeatability_transfer_learned_native_layout02_4800_v1_attempt_001`.
The fixed eight-run comparison and its physical, controller and trajectory diagnoses are complete in docs/go2_post_repeatability_transfer_preparation_2026-09-14.md. This successful case exercised no heading-recovery event. Preserve all four failed reactive counterparts and every comparison result.
End its superseded full-depth success pin. No sensor replay is pending, and
this root is not a current training or correction-fit input. Current full
repair references and every active failure remain preserved. Retire only
inventoried regular single-link native depth leaves; exact original depth
replay becomes unavailable. Inventory: `.generated/depth_retirement_superseded_clean_transfer_success_2026-09-17/`.
Completed: 6084 leaves reclaimed 1771094016 allocated bytes; all non-depth identities and JSON hashes matched.

Retire completed superseded success depth, September 17.
Exact root: `go2_declared_floor_gap_learned_noise_2mm_native_layout00_4800_v1_attempt_001`.
The completed injected-gap execution and post-gap recovery diagnosis are documented in docs/go2_declared_floor_gap_2026-09-15.md. Preserve all gap-timing and recovery receipts, the failed reactive companion and every non-depth record.
End its superseded full-depth success pin. No sensor replay is pending, and
this root is not a current training or correction-fit input. Current full
repair references and every active failure remain preserved. Retire only
inventoried regular single-link native depth leaves; exact original depth
replay becomes unavailable. Inventory: `.generated/depth_retirement_superseded_declared_floor_gap_success_2026-09-17/`.
Completed: 6332 leaves reclaimed 1888907264 allocated bytes; all non-depth identities and JSON hashes matched.

Retire completed diagnosed pose_command fresh-maze replication success depth, September 17.
Exact root: `go2_publication_repaired_replication_pose_command_noise_2mm_native_layout03_4800_v1_attempt_001`.
Owner exited; physical outcome, contacts, forecast and added-delay receipts
were evaluated. Verified success and deadline crossings are retained.
The exercised publication-fix reference, prior active references and all active failures remain full. No raw replay is pending; preserve all non-depth evidence, timing records and
original tracking-loss inputs. No alternative navigation success is inferred.
Only inventoried regular single-link native depth leaves retired.
Inventory: `.generated/depth_retirement_publication_repaired_replication_assignment15_2026-09-17/`.
Completed: 2702 leaves reclaimed 808472576 allocated bytes; all non-depth identities and JSON hashes matched.

Retire completed diagnosed supervised_rollout fresh-maze replication success depth, September 17.
Exact root: `go2_publication_repaired_replication_supervised_rollout_noise_2mm_native_layout03_4800_v1_attempt_001`.
Owner exited; physical outcome, contacts, forecast and added-delay receipts
were evaluated. Verified success and deadline crossings are retained.
The exercised publication-fix reference, prior active references and all active failures remain full. No raw replay is pending; preserve all non-depth evidence, timing records and
original tracking-loss inputs. No alternative navigation success is inferred.
Only inventoried regular single-link native depth leaves retired.
Inventory: `.generated/depth_retirement_publication_repaired_replication_assignment14_2026-09-17/`.
Completed: 3276 leaves reclaimed 991100928 allocated bytes; all non-depth identities and JSON hashes matched.

Retire completed diagnosed jepa fresh-maze replication success depth, September 17.
Exact root: `go2_publication_repaired_replication_jepa_noise_2mm_native_layout03_4800_v1_attempt_001`.
Owner exited; physical outcome, contacts, forecast and added-delay receipts
were evaluated. Verified success and deadline crossings are retained.
The exercised publication-fix reference, prior active references and all active failures remain full. No raw replay is pending; preserve all non-depth evidence, timing records and
original tracking-loss inputs. No alternative navigation success is inferred.
Only inventoried regular single-link native depth leaves retired.
Inventory: `.generated/depth_retirement_publication_repaired_replication_assignment13_2026-09-17/`.
Completed: 2694 leaves reclaimed 816988160 allocated bytes; all non-depth identities and JSON hashes matched.

Retire superseded completed matched-supervised success depth, September 17.
Exact root: `go2_matched_training_supervised_rollout_heading_release_native_layout02_4800_v1_attempt_001`.
The twelve-trial comparison and its physical, forecast and trajectory analyses
completed in docs/go2_matched_motion_residual_controls_2026-09-14.md.
End any old success-reference pin; the current corrected-cohort supervised
result and retained current repair references supersede it. No raw replay is
pending. Preserve the failed companions,
every comparison and all non-depth evidence. This older recording is not an
active raw-depth training or replay input. Current startup/publication repair
references and all active failures remain full. Exact historical depth replay
becomes unavailable. Only inventoried regular single-link native depth leaves
are retired. Inventory: `.generated/depth_retirement_superseded_matched_supervised_layout02_2026-09-17/`.
Completed: 6164 leaves reclaimed 1778184192 allocated bytes; all non-depth identities and JSON hashes matched.

Retire completed diagnosed reactive_feedback fresh-maze replication success depth, September 17.
Exact root: `go2_publication_repaired_replication_reactive_feedback_noise_2mm_native_layout03_4800_v1_attempt_001`.
Owner exited; physical outcome, contacts, forecast and added-delay receipts
were evaluated. Verified success and deadline crossings are retained.
The exercised publication-fix reference, prior active references and all active failures remain full. No raw replay is pending; preserve all non-depth evidence, timing records and
original tracking-loss inputs. No alternative navigation success is inferred.
Only inventoried regular single-link native depth leaves retired.
Inventory: `.generated/depth_retirement_publication_repaired_replication_assignment12_2026-09-17/`.
Completed: 3692 leaves reclaimed 1114759168 allocated bytes; all non-depth identities and JSON hashes matched.

Retire completed diagnosed supervised_rollout fresh-maze replication success depth, September 17.
Exact root: `go2_publication_repaired_replication_supervised_rollout_noise_2mm_native_layout02_4800_v1_attempt_001`.
Owner exited; physical outcome, contacts, forecast and added-delay receipts
were evaluated. Verified success and deadline crossings are retained.
The exercised publication-fix reference, prior active references and all active failures remain full. No raw replay is pending; preserve all non-depth evidence, timing records and
original tracking-loss inputs. No alternative navigation success is inferred.
Only inventoried regular single-link native depth leaves retired.
Inventory: `.generated/depth_retirement_publication_repaired_replication_assignment10_2026-09-17/`.
Completed: 2916 leaves reclaimed 855142400 allocated bytes; all non-depth identities and JSON hashes matched.

Retire completed diagnosed jepa fresh-maze replication success depth, September 17.
Exact root: `go2_publication_repaired_replication_jepa_noise_2mm_native_layout02_4800_v1_attempt_001`.
Owner exited; physical outcome, contacts, forecast and added-delay receipts
were evaluated. Verified success and deadline crossings are retained.
The exercised publication-fix reference, prior active references and all active failures remain full. No raw replay is pending; preserve all non-depth evidence, timing records and
original tracking-loss inputs. No alternative navigation success is inferred.
Only inventoried regular single-link native depth leaves retired.
Inventory: `.generated/depth_retirement_publication_repaired_replication_assignment09_2026-09-17/`.
Completed: 3682 leaves reclaimed 1118760960 allocated bytes; all non-depth identities and JSON hashes matched.

Retire completed diagnosed reactive_feedback fresh-maze replication success depth, September 17.
Exact root: `go2_publication_repaired_replication_reactive_feedback_noise_2mm_native_layout02_4800_v1_attempt_001`.
Owner exited; physical outcome, contacts, forecast and added-delay receipts
were evaluated. Verified success and deadline crossings are retained.
The exercised publication-fix reference, prior active references and all active failures remain full. No raw replay is pending; preserve all non-depth evidence, timing records and
original tracking-loss inputs. No alternative navigation success is inferred.
Only inventoried regular single-link native depth leaves retired.
Inventory: `.generated/depth_retirement_publication_repaired_replication_assignment08_2026-09-17/`.
Completed: 3550 leaves reclaimed 1084100608 allocated bytes; all non-depth identities and JSON hashes matched.

Retire completed diagnosed instantaneous fresh-maze replication success depth, September 17.
Exact root: `go2_publication_repaired_replication_instantaneous_noise_2mm_native_layout02_4800_v1_attempt_001`.
Owner exited; physical outcome, contacts, forecast and added-delay receipts
were evaluated. Verified success and deadline crossings are retained.
The exercised publication-fix reference, prior active references and all active failures remain full. No raw replay is pending; preserve all non-depth evidence, timing records and
original tracking-loss inputs. No alternative navigation success is inferred.
Only inventoried regular single-link native depth leaves retired.
Inventory: `.generated/depth_retirement_publication_repaired_replication_assignment07_2026-09-17/`.
Completed: 3830 leaves reclaimed 1151565824 allocated bytes; all non-depth identities and JSON hashes matched.

Retire completed diagnosed pose_command fresh-maze replication success depth, September 17.
Exact root: `go2_publication_repaired_replication_pose_command_noise_2mm_native_layout02_4800_v1_attempt_001`.
Owner exited; physical outcome, contacts, forecast and added-delay receipts
were evaluated. Verified success and deadline crossings are retained.
The exercised publication-fix reference, prior active references and all active failures remain full. No raw replay is pending; preserve all non-depth evidence, timing records and
original tracking-loss inputs. No alternative navigation success is inferred.
Only inventoried regular single-link native depth leaves retired.
Inventory: `.generated/depth_retirement_publication_repaired_replication_assignment06_2026-09-17/`.
Completed: 3214 leaves reclaimed 948391936 allocated bytes; all non-depth identities and JSON hashes matched.

Retire superseded completed heading/survey success depth, September 17.
Exact root: `go2_survey_reposition_matched_jepa_native_layout01_4800_v1_attempt_001`.
The completed comparisons and full execution diagnoses are documented in
docs/go2_heading_recovery_fixed_repeatability_2026-09-14.md and
docs/go2_survey_clearance_reposition_hypothesis_2026-09-14.md.
The successful repeat exercised no heading recovery; the survey-success
reposition event and first trajectory divergence are already diagnosed.
End any superseded success-reference pin; no sensor replay is pending.
Preserve the failed companions,
every comparison and all non-depth evidence. This older recording is not an
active raw-depth training or replay input. Current startup/publication repair
references and all active failures remain full. Exact historical depth replay
becomes unavailable. Only inventoried regular single-link native depth leaves
are retired. Inventory: `.generated/depth_retirement_superseded_heading_survey_success01_2026-09-17/`.
Completed: 8122 leaves reclaimed 2469621760 allocated bytes; all non-depth identities and JSON hashes matched.

Retire superseded completed heading/survey success depth, September 17.
Exact root: `go2_heading_recovery_repeatability_rep1_layout00_4800_v1_attempt_001`.
The completed comparisons and full execution diagnoses are documented in
docs/go2_heading_recovery_fixed_repeatability_2026-09-14.md and
docs/go2_survey_clearance_reposition_hypothesis_2026-09-14.md.
The successful repeat exercised no heading recovery; the survey-success
reposition event and first trajectory divergence are already diagnosed.
End any superseded success-reference pin; no sensor replay is pending.
Preserve the failed companions,
every comparison and all non-depth evidence. This older recording is not an
active raw-depth training or replay input. Current startup/publication repair
references and all active failures remain full. Exact historical depth replay
becomes unavailable. Only inventoried regular single-link native depth leaves
are retired. Inventory: `.generated/depth_retirement_superseded_heading_survey_success00_2026-09-17/`.
Completed: 7780 leaves reclaimed 2410418176 allocated bytes; all non-depth identities and JSON hashes matched.

Retire superseded completed shared-recovery transfer success depth, September 17.
Exact root: `go2_shared_recovery_transfer_reactive_noise_2mm_native_layout00_4800_v1_attempt_001`.
The ten-run comparison and its diagnoses were completed in
docs/go2_shared_recovery_transfer_2026-09-15.md. End the old temporary
layout-0 controller-reference and layout-1 supervised-success pins: their
comparison, forecast, timing and exercised-recovery diagnoses are complete,
and the startup/publication repair references supersede them. No further raw
replay is pending. Preserve the failed companion,
every comparison and all non-depth evidence. This older recording is not an
active raw-depth training or replay input. Current startup/publication repair
references and all active failures remain full. Exact historical depth replay
becomes unavailable. Only inventoried regular single-link native depth leaves
are retired. Inventory: `.generated/depth_retirement_superseded_shared_recovery_success00_2026-09-17/`.
Completed: 4476 leaves reclaimed 1345114112 allocated bytes; all non-depth identities and JSON hashes matched.

Retire superseded completed gyro-estimator comparison success depth, September 17.
Exact root: `go2_current_plane_matched_training_jepa_noise_2mm_native_layout01_4800_v1_attempt_001`.
The live probe and the full 3574-frame historical estimator replay completed in
docs/go2_gyro_coherent_floor_constraint_2026-09-15.md. Their physical,
forecast and tracking results are preserved; no raw replay is pending.
Preserve the failed companion,
every comparison and all non-depth evidence. This older recording is not an
active raw-depth training or replay input. Current startup/publication repair
references and all active failures remain full. Exact historical depth replay
becomes unavailable. Only inventoried regular single-link native depth leaves
are retired. Inventory: `.generated/depth_retirement_superseded_gyro_comparison_success01_2026-09-17/`.
Completed: 7148 leaves reclaimed 2087596032 allocated bytes; all non-depth identities and JSON hashes matched.

Retire superseded completed gyro-estimator comparison success depth, September 17.
Exact root: `go2_gyro_coherent_floor_jepa_noise_2mm_native_layout03_4800_v1_attempt_001`.
The live probe and the full 3574-frame historical estimator replay completed in
docs/go2_gyro_coherent_floor_constraint_2026-09-15.md. Their physical,
forecast and tracking results are preserved; no raw replay is pending.
Preserve the failed companion,
every comparison and all non-depth evidence. This older recording is not an
active raw-depth training or replay input. Current startup/publication repair
references and all active failures remain full. Exact historical depth replay
becomes unavailable. Only inventoried regular single-link native depth leaves
are retired. Inventory: `.generated/depth_retirement_superseded_gyro_comparison_success00_2026-09-17/`.
Completed: 4284 leaves reclaimed 1278476288 allocated bytes; all non-depth identities and JSON hashes matched.

Retire completed diagnosed instantaneous fresh-maze replication success depth, September 17.
Exact root: `go2_publication_repaired_replication_instantaneous_noise_2mm_native_layout01_4800_v1_attempt_001`.
Owner exited; physical outcome, contacts, forecast and added-delay receipts
were evaluated. Verified success and deadline crossings are retained.
The exercised publication-fix reference, prior active references and all active failures remain full. No raw replay is pending; preserve all non-depth evidence, timing records and
original tracking-loss inputs. No alternative navigation success is inferred.
Only inventoried regular single-link native depth leaves retired.
Inventory: `.generated/depth_retirement_publication_repaired_replication_assignment03_2026-09-17/`.
Completed: 7326 leaves reclaimed 2336083968 allocated bytes; all non-depth identities and JSON hashes matched.

Retire superseded completed plane-coverage success depth, September 16.
Exact root: `go2_current_plane_coverage_noise_2mm_native_layout02_4800_v1_attempt_001`.
The four-layout comparison and its diagnoses were completed in
docs/go2_noisy_floor_coverage_gap_2026-09-15.md. Preserve the failed companion,
every comparison and all non-depth evidence. This older recording is not an
active raw-depth training or replay input. Current startup/publication repair
references and all active failures remain full. Exact historical depth replay
becomes unavailable. Only inventoried regular single-link native depth leaves
are retired. Inventory: `.generated/depth_retirement_superseded_plane_coverage_layout02_2026-09-16/`.
Completed: 6134 leaves reclaimed 1805033472 allocated bytes; all non-depth identities and JSON hashes matched.

Retire superseded completed plane-coverage success depth, September 16.
Exact root: `go2_current_plane_coverage_noise_2mm_native_layout01_4800_v1_attempt_001`.
The four-layout comparison and its diagnoses were completed in
docs/go2_noisy_floor_coverage_gap_2026-09-15.md. Preserve the failed companion,
every comparison and all non-depth evidence. This older recording is not an
active raw-depth training or replay input. Current startup/publication repair
references and all active failures remain full. Exact historical depth replay
becomes unavailable. Only inventoried regular single-link native depth leaves
are retired. Inventory: `.generated/depth_retirement_superseded_plane_coverage_layout01_2026-09-16/`.
Completed: 7090 leaves reclaimed 2199855104 allocated bytes; all non-depth identities and JSON hashes matched.

Retire superseded completed plane-coverage success depth, September 16.
Exact root: `go2_current_plane_coverage_noise_2mm_native_layout00_4800_v1_attempt_001`.
The four-layout comparison and its diagnoses were completed in
docs/go2_noisy_floor_coverage_gap_2026-09-15.md. Preserve the failed companion,
every comparison and all non-depth evidence. This older recording is not an
active raw-depth training or replay input. Current startup/publication repair
references and all active failures remain full. Exact historical depth replay
becomes unavailable. Only inventoried regular single-link native depth leaves
are retired. Inventory: `.generated/depth_retirement_superseded_plane_coverage_layout00_2026-09-16/`.
Completed: 6796 leaves reclaimed 2014687232 allocated bytes; all non-depth identities and JSON hashes matched.

Retire completed diagnosed jepa fresh-maze replication success depth, September 16.
Exact root: `go2_recovery_repaired_replication_jepa_noise_2mm_native_layout00_4800_v1_attempt_001`.
Owner exited; physical outcome, contacts, forecast and added-delay receipts
were evaluated. Verified success and deadline crossings are retained.
End the temporary first-success depth pin because this cohort was interrupted and its completed success analyses are finished. Preserve the original exposed and fresh-maze repair references and both failures from this interrupted cohort in full. No training input depends on this recording. No raw replay is pending; preserve all non-depth evidence, timing records and
original tracking-loss inputs. No alternative navigation success is inferred.
Only inventoried regular single-link native depth leaves retired.
Inventory: `.generated/depth_retirement_interrupted_cohort_success_reference_assignment01_2026-09-16/`.
Completed: 3510 leaves reclaimed 1046126592 allocated bytes; all non-depth identities and JSON hashes matched.

Retire completed diagnosed supervised_rollout fresh-maze replication success depth, September 16.
Exact root: `go2_recovery_repaired_replication_supervised_rollout_noise_2mm_native_layout00_4800_v1_attempt_001`.
Owner exited; physical outcome, contacts, forecast and added-delay receipts
were evaluated. Verified success and deadline crossings are retained.
The first JEPA success in this repaired cohort remains full, together with prior active references and failures. No raw replay is pending; preserve all non-depth evidence, timing records and
original tracking-loss inputs. No alternative navigation success is inferred.
Only inventoried regular single-link native depth leaves retired.
Inventory: `.generated/depth_retirement_recovery_repaired_replication_assignment02_2026-09-16/`.
Completed: 4468 leaves reclaimed 1334792192 allocated bytes; all non-depth identities and JSON hashes matched.

Retire completed diagnosed enabled_rep2 unexercised heading-release success depth, September 16.
Exact root: `go2_turn_release_repeatability_enabled_rep2_jepa_noise_2mm_native_layout01_4800_v1_attempt_001`.
Owner exited; physical outcome, contacts, forecast and added-delay receipts
were evaluated. Verified success and deadline crossings are retained.
This completed execution had no eligible heading-release decision. Its physical, forecast, timing and backtracking analyses are complete. The fixed comparison continues using preserved plans, requests, physical states and per-run readouts; no historical depth replay is pending. Full outcomes and all non-depth evidence remain. Keep the earlier exposed and fresh-maze JEPA repair references and all active failures full. No raw replay is pending; preserve all non-depth evidence, timing records and
original tracking-loss inputs. No alternative navigation success is inferred.
Only inventoried regular single-link native depth leaves retired.
Inventory: `.generated/depth_retirement_unexercised_heading_repeatability_assignment04_2026-09-16/`.
Completed: 3444 leaves reclaimed 1058398208 allocated bytes; all non-depth identities and JSON hashes matched.

Retire completed diagnosed disabled_rep2 unexercised heading-release success depth, September 16.
Exact root: `go2_turn_release_repeatability_disabled_rep2_jepa_noise_2mm_native_layout01_4800_v1_attempt_001`.
Owner exited; physical outcome, contacts, forecast and added-delay receipts
were evaluated. Verified success and deadline crossings are retained.
This completed execution had no eligible heading-release decision. Its physical, forecast, timing and backtracking analyses are complete. The fixed comparison continues using preserved plans, requests, physical states and per-run readouts; no historical depth replay is pending. Full outcomes and all non-depth evidence remain. Keep the earlier exposed and fresh-maze JEPA repair references and all active failures full. No raw replay is pending; preserve all non-depth evidence, timing records and
original tracking-loss inputs. No alternative navigation success is inferred.
Only inventoried regular single-link native depth leaves retired.
Inventory: `.generated/depth_retirement_unexercised_heading_repeatability_assignment03_2026-09-16/`.
Completed: 3502 leaves reclaimed 1073922048 allocated bytes; all non-depth identities and JSON hashes matched.

Retire completed diagnosed disabled_rep1 unexercised heading-release success depth, September 16.
Exact root: `go2_turn_release_repeatability_disabled_rep1_jepa_noise_2mm_native_layout01_4800_v1_attempt_001`.
Owner exited; physical outcome, contacts, forecast and added-delay receipts
were evaluated. Verified success and deadline crossings are retained.
This completed execution had no eligible heading-release decision. Its physical, forecast, timing and backtracking analyses are complete. The fixed comparison continues using preserved plans, requests, physical states and per-run readouts; no historical depth replay is pending. Full outcomes and all non-depth evidence remain. Keep the earlier exposed and fresh-maze JEPA repair references and all active failures full. No raw replay is pending; preserve all non-depth evidence, timing records and
original tracking-loss inputs. No alternative navigation success is inferred.
Only inventoried regular single-link native depth leaves retired.
Inventory: `.generated/depth_retirement_unexercised_heading_repeatability_assignment02_2026-09-16/`.
Completed: 3868 leaves reclaimed 1153097728 allocated bytes; all non-depth identities and JSON hashes matched.

Retire completed diagnosed enabled_rep1 unexercised heading-release success depth, September 16.
Exact root: `go2_turn_release_repeatability_enabled_rep1_jepa_noise_2mm_native_layout01_4800_v1_attempt_001`.
Owner exited; physical outcome, contacts, forecast and added-delay receipts
were evaluated. Verified success and deadline crossings are retained.
This completed execution had no eligible heading-release decision. Its physical, forecast, timing and backtracking analyses are complete. The fixed comparison continues using preserved plans, requests, physical states and per-run readouts; no historical depth replay is pending. Full outcomes and all non-depth evidence remain. Keep the earlier exposed and fresh-maze JEPA repair references and all active failures full. No raw replay is pending; preserve all non-depth evidence, timing records and
original tracking-loss inputs. No alternative navigation success is inferred.
Only inventoried regular single-link native depth leaves retired.
Inventory: `.generated/depth_retirement_unexercised_heading_repeatability_assignment01_2026-09-16/`.
Completed: 4866 leaves reclaimed 1444020224 allocated bytes; all non-depth identities and JSON hashes matched.

Retire completed diagnosed survey_transfer_repaired unexercised heading-release success depth, September 16.
Exact root: `go2_no_early_heading_release_survey_transfer_repaired_jepa_noise_2mm_native_layout01_4800_v1_attempt_001`.
Owner exited; physical outcome, contacts, forecast and added-delay receipts
were evaluated. Verified success and deadline crossings are retained.
Both intended ablations were unexercised and their analyses are complete. First command/deadline differences, full outcomes and all non-depth evidence remain. Keep the earlier exposed and fresh-maze JEPA repair references and all active failures full. No raw replay is pending; preserve all non-depth evidence, timing records and
original tracking-loss inputs. No alternative navigation success is inferred.
Only inventoried regular single-link native depth leaves retired.
Inventory: `.generated/depth_retirement_unexercised_heading_release_assignment02_2026-09-16/`.
Completed: 3508 leaves reclaimed 1078489088 allocated bytes; all non-depth identities and JSON hashes matched.

Retire completed diagnosed replication_original unexercised heading-release success depth, September 16.
Exact root: `go2_no_early_heading_release_replication_original_jepa_noise_2mm_native_layout01_4800_v1_attempt_001`.
Owner exited; physical outcome, contacts, forecast and added-delay receipts
were evaluated. Verified success and deadline crossings are retained.
Both intended ablations were unexercised and their analyses are complete. First command/deadline differences, full outcomes and all non-depth evidence remain. Keep the earlier exposed and fresh-maze JEPA repair references and all active failures full. No raw replay is pending; preserve all non-depth evidence, timing records and
original tracking-loss inputs. No alternative navigation success is inferred.
Only inventoried regular single-link native depth leaves retired.
Inventory: `.generated/depth_retirement_unexercised_heading_release_assignment01_2026-09-16/`.
Completed: 5646 leaves reclaimed 1664655360 allocated bytes; all non-depth identities and JSON hashes matched.

Retire superseded completed success depth, September 16: `go2_live_local_feature_depth_noise_0mm_native_layout02_4800_v1_attempt_001`.
 Its completed comparison, physical outcome and diagnoses remain. End its older
 full-depth reference pin; no current training/fit or pending sensor replay uses
 this recording. All failed companions and every non-depth artifact remain.
 Current exposed/fresh-maze JEPA repair references and turning failures stay full.
 Exact historical depth replay becomes unavailable. Only inventoried regular
 single-link native depth leaves are retired. Inventory: `.generated/depth_retirement_superseded_success_references_2026-09-16/go2_live_local_feature_depth_noise_0mm_native_layout02_4800_v1_attempt_001/`.
Completed: 6116 leaves reclaimed 1771642880 allocated bytes; all non-depth identities and JSON hashes matched.

Retire superseded completed success depth, September 16: `go2_floor_reacquisition_reactive_noise_2mm_native_layout03_4800_v1_attempt_001`.
 Its completed comparison, physical outcome and diagnoses remain. End its older
 full-depth reference pin; no current training/fit or pending sensor replay uses
 this recording. All failed companions and every non-depth artifact remain.
 Current exposed/fresh-maze JEPA repair references and turning failures stay full.
 Exact historical depth replay becomes unavailable. Only inventoried regular
 single-link native depth leaves are retired. Inventory: `.generated/depth_retirement_superseded_success_references_2026-09-16/go2_floor_reacquisition_reactive_noise_2mm_native_layout03_4800_v1_attempt_001/`.
Completed: 3798 leaves reclaimed 1161936896 allocated bytes; all non-depth identities and JSON hashes matched.

Retire superseded completed success depth, September 16: `go2_wall_reactive_floor_reacquisition_layout01_4800_v1_attempt_001`.
 Its completed comparison, physical outcome and diagnoses remain. End its older
 full-depth reference pin; no current training/fit or pending sensor replay uses
 this recording. All failed companions and every non-depth artifact remain.
 Current exposed/fresh-maze JEPA repair references and turning failures stay full.
 Exact historical depth replay becomes unavailable. Only inventoried regular
 single-link native depth leaves are retired. Inventory: `.generated/depth_retirement_superseded_success_references_2026-09-16/go2_wall_reactive_floor_reacquisition_layout01_4800_v1_attempt_001/`.
Completed: 4212 leaves reclaimed 1257525248 allocated bytes; all non-depth identities and JSON hashes matched.

Retire superseded completed success depth, September 16: `go2_recent_reference_refresh_reactive_native_layout03_4800_v1_attempt_001`.
 Its completed comparison, physical outcome and diagnoses remain. End its older
 full-depth reference pin; no current training/fit or pending sensor replay uses
 this recording. All failed companions and every non-depth artifact remain.
 Current exposed/fresh-maze JEPA repair references and turning failures stay full.
 Exact historical depth replay becomes unavailable. Only inventoried regular
 single-link native depth leaves are retired. Inventory: `.generated/depth_retirement_superseded_success_references_2026-09-16/go2_recent_reference_refresh_reactive_native_layout03_4800_v1_attempt_001/`.
Completed: 4106 leaves reclaimed 1248833536 allocated bytes; all non-depth identities and JSON hashes matched.

Retire superseded completed success depth, September 16: `go2_recent_reference_refresh_reactive_native_layout00_4800_v1_attempt_001`.
 Its completed comparison, physical outcome and diagnoses remain. End its older
 full-depth reference pin; no current training/fit or pending sensor replay uses
 this recording. All failed companions and every non-depth artifact remain.
 Current exposed/fresh-maze JEPA repair references and turning failures stay full.
 Exact historical depth replay becomes unavailable. Only inventoried regular
 single-link native depth leaves are retired. Inventory: `.generated/depth_retirement_superseded_success_references_2026-09-16/go2_recent_reference_refresh_reactive_native_layout00_4800_v1_attempt_001/`.
Completed: 5436 leaves reclaimed 1662033920 allocated bytes; all non-depth identities and JSON hashes matched.

Retire superseded completed success depth, September 16: `go2_current_plane_heading_reactive_noise_2mm_native_layout01_4800_v1_attempt_001`.
 Its completed comparison, physical outcome and diagnoses remain. End its older
 full-depth reference pin; no current training/fit or pending sensor replay uses
 this recording. All failed companions and every non-depth artifact remain.
 Current exposed/fresh-maze JEPA repair references and turning failures stay full.
 Exact historical depth replay becomes unavailable. Only inventoried regular
 single-link native depth leaves are retired. Inventory: `.generated/depth_retirement_superseded_success_references_2026-09-16/go2_current_plane_heading_reactive_noise_2mm_native_layout01_4800_v1_attempt_001/`.
Completed: 4524 leaves reclaimed 1386663936 allocated bytes; all non-depth identities and JSON hashes matched.

Retire the completed nearby-panorama baseline layout-0 success depth, September 16.
Exact root: `go2_nearby_panorama_baseline_native_layout00_4800_v1_attempt_001`.
All four missions, physical evaluations, both comparisons and frontier-survey
diagnoses were completed on September 14. End its older depth reference pin.
No pending sensor replay or current training/fit input uses this recording.
Preserve the failed companions and every comparison, outcome, survey diagnosis,
source identity, RGB/body/gyro, physics, pose and command record.
Keep both the exposed JEPA repair reference and the new fresh-maze layout-1
repair success in full. Only inventoried regular single-link native depth
leaves are retired; exact historical depth replay becomes unavailable.
Inventory: `.generated/depth_retirement_superseded_nearby_panorama_success_2026-09-16/`.
Completed: 9516 leaves reclaimed 2949894144 allocated bytes; all non-depth identities and JSON hashes matched.

Retire completed diagnosed supervised_rollout fresh-maze replication success depth, September 16.
Exact root: `go2_nogil_replication_supervised_rollout_noise_2mm_native_layout01_4800_v1_attempt_001`.
Owner exited; physical outcome, contacts, forecast and added-delay receipts
were evaluated. Verified success and deadline crossings are retained.
End the earlier full-depth pin for this first success now that all twenty fixed missions and their scientific readouts are complete. The later JEPA recovery-limited survey success remains the full current reference. All six replication failures remain full. This recording is not a training or fit input, and no raw replay is pending; preserve all non-depth evidence, timing records and
original tracking-loss inputs. No alternative navigation success is inferred.
Only inventoried regular single-link native depth leaves retired.
Inventory: `.generated/depth_retirement_superseded_nogil_replication_reference_assignment06_2026-09-16/`.
Completed: 4956 leaves reclaimed 1509855232 allocated bytes; all non-depth identities and JSON hashes matched.

Retire completed diagnosed limited_survey fresh-maze survey-transfer success depth, September 16.
Exact root: `go2_recovery_survey_transfer_limited_survey_jepa_noise_2mm_native_layout00_4800_v1_attempt_001`.
Owner exited; physical outcome, contacts, forecast and added-delay receipts
were evaluated. Verified success and deadline crossings are retained.
The earlier repair JEPA reference and active failures remain full. Route latency profiles, forecasts, RGB and physical backtracking are preserved. No raw replay is pending; preserve all non-depth evidence, timing records and
original tracking-loss inputs. No alternative navigation success is inferred.
Only inventoried regular single-link native depth leaves retired.
Inventory: `.generated/depth_retirement_survey_transfer_assignment02_2026-09-16/`.
Completed: 4742 leaves reclaimed 1292877824 allocated bytes; all non-depth identities and JSON hashes matched.

Retire completed diagnosed original fresh-maze survey-transfer success depth, September 16.
Exact root: `go2_recovery_survey_transfer_original_jepa_noise_2mm_native_layout00_4800_v1_attempt_001`.
Owner exited; physical outcome, contacts, forecast and added-delay receipts
were evaluated. Verified success and deadline crossings are retained.
The earlier repair JEPA reference and active failures remain full. Route latency profiles, forecasts, RGB and physical backtracking are preserved. No raw replay is pending; preserve all non-depth evidence, timing records and
original tracking-loss inputs. No alternative navigation success is inferred.
Only inventoried regular single-link native depth leaves retired.
Inventory: `.generated/depth_retirement_survey_transfer_assignment01_2026-09-16/`.
Completed: 8132 leaves reclaimed 2353315840 allocated bytes; all non-depth identities and JSON hashes matched.

Retire completed diagnosed supervised_rollout exposed-maze recovery-limited survey success depth, September 16.
Exact root: `go2_nogil_recovery_limited_survey_supervised_rollout_noise_2mm_native_layout00_4800_v1_attempt_001`.
Owner exited; physical outcome, contacts, forecast and added-delay receipts
were evaluated. Verified success and deadline crossings are retained.
The matched JEPA repair success remains retained in full. No raw replay is pending; preserve all non-depth evidence, timing records and
original tracking-loss inputs. No alternative navigation success is inferred.
Only inventoried regular single-link native depth leaves retired.
Inventory: `.generated/depth_retirement_recovery_limited_survey_supervised_2026-09-16/`.
Completed: 5932 leaves reclaimed 1852432384 allocated bytes; all non-depth identities and JSON hashes matched.

Retire completed diagnosed jepa nogil-drawing planning-latency stress depth, September 16.
Exact root: `go2_nogil_drawing_plus20ms_jepa_noise_2mm_native_layout03_4800_v1_attempt_001`.
End the earlier full-depth pin for this renderer-fix reference now that the
20-run fresh-maze replication is complete. The fresh-maze supervised success
and recovery-limited-survey JEPA success remain full; all current failure
recordings remain full. This older run is not a current fit or pending replay
input. Owner exited; physical outcome, contacts, forecast and added-delay receipts
were evaluated. Budget failure or success and deadline crossings are retained.
No raw replay is pending; preserve all non-depth evidence, timing records and
original tracking-loss inputs. No alternative navigation success is inferred.
Only inventoried regular single-link native depth leaves retired.
Inventory: `.generated/depth_retirement_nogil_drawing_jepa_2026-09-16/`.
Completed: 4164 leaves reclaimed 1137614848 allocated bytes; all non-depth identities and JSON hashes matched.

Retain the first recovery-limited initial-survey success, September 16.
Exact root: `go2_nogil_recovery_limited_survey_jepa_noise_2mm_native_layout00_4800_v1_attempt_001`.
Both physical arrivals and the treatment/backtracking readouts passed. Keep its
full depth as the current repair reference. Also keep the preceding prompt-hold
tracking failure in full. No original failed outcome is replaced.

Retire completed diagnosed pose_command fresh-maze replication success depth, September 16.
Exact root: `go2_nogil_replication_pose_command_noise_2mm_native_layout03_4800_v1_attempt_001`.
Owner exited; physical outcome, contacts, forecast and added-delay receipts
were evaluated. Verified success and deadline crossings are retained.
No raw replay is pending; preserve all non-depth evidence, timing records and
original tracking-loss inputs. No alternative navigation success is inferred.
Only inventoried regular single-link native depth leaves retired.
Inventory: `.generated/depth_retirement_nogil_replication_assignment20_2026-09-16/`.
Completed: 3894 leaves reclaimed 1153597440 allocated bytes; all non-depth identities and JSON hashes matched.

Retire completed diagnosed supervised_rollout fresh-maze replication success depth, September 16.
Exact root: `go2_nogil_replication_supervised_rollout_noise_2mm_native_layout03_4800_v1_attempt_001`.
Owner exited; physical outcome, contacts, forecast and added-delay receipts
were evaluated. Verified success and deadline crossings are retained.
No raw replay is pending; preserve all non-depth evidence, timing records and
original tracking-loss inputs. No alternative navigation success is inferred.
Only inventoried regular single-link native depth leaves retired.
Inventory: `.generated/depth_retirement_nogil_replication_assignment19_2026-09-16/`.
Completed: 4598 leaves reclaimed 1306886144 allocated bytes; all non-depth identities and JSON hashes matched.

Retire completed diagnosed jepa fresh-maze replication success depth, September 16.
Exact root: `go2_nogil_replication_jepa_noise_2mm_native_layout03_4800_v1_attempt_001`.
Owner exited; physical outcome, contacts, forecast and added-delay receipts
were evaluated. Verified success and deadline crossings are retained.
No raw replay is pending; preserve all non-depth evidence, timing records and
original tracking-loss inputs. No alternative navigation success is inferred.
Only inventoried regular single-link native depth leaves retired.
Inventory: `.generated/depth_retirement_nogil_replication_assignment18_2026-09-16/`.
Completed: 4740 leaves reclaimed 1331945472 allocated bytes; all non-depth identities and JSON hashes matched.

Retire completed diagnosed reactive_feedback fresh-maze replication success depth, September 16.
Exact root: `go2_nogil_replication_reactive_feedback_noise_2mm_native_layout03_4800_v1_attempt_001`.
Owner exited; physical outcome, contacts, forecast and added-delay receipts
were evaluated. Verified success and deadline crossings are retained.
No raw replay is pending; preserve all non-depth evidence, timing records and
original tracking-loss inputs. No alternative navigation success is inferred.
Only inventoried regular single-link native depth leaves retired.
Inventory: `.generated/depth_retirement_nogil_replication_assignment17_2026-09-16/`.
Completed: 4090 leaves reclaimed 1232969728 allocated bytes; all non-depth identities and JSON hashes matched.

Retire completed diagnosed instantaneous fresh-maze replication success depth, September 16.
Exact root: `go2_nogil_replication_instantaneous_noise_2mm_native_layout03_4800_v1_attempt_001`.
Owner exited; physical outcome, contacts, forecast and added-delay receipts
were evaluated. Verified success and deadline crossings are retained.
No raw replay is pending; preserve all non-depth evidence, timing records and
original tracking-loss inputs. No alternative navigation success is inferred.
Only inventoried regular single-link native depth leaves retired.
Inventory: `.generated/depth_retirement_nogil_replication_assignment16_2026-09-16/`.
Completed: 4212 leaves reclaimed 1278709760 allocated bytes; all non-depth identities and JSON hashes matched.

Retire completed diagnosed supervised_rollout fresh-maze replication success depth, September 16.
Exact root: `go2_nogil_replication_supervised_rollout_noise_2mm_native_layout02_4800_v1_attempt_001`.
Owner exited; physical outcome, contacts, forecast and added-delay receipts
were evaluated. Verified success and deadline crossings are retained.
No raw replay is pending; preserve all non-depth evidence, timing records and
original tracking-loss inputs. No alternative navigation success is inferred.
Only inventoried regular single-link native depth leaves retired.
Inventory: `.generated/depth_retirement_nogil_replication_assignment15_2026-09-16/`.
Completed: 3238 leaves reclaimed 970362880 allocated bytes; all non-depth identities and JSON hashes matched.

Retire completed diagnosed jepa fresh-maze replication success depth, September 16.
Exact root: `go2_nogil_replication_jepa_noise_2mm_native_layout02_4800_v1_attempt_001`.
Owner exited; physical outcome, contacts, forecast and added-delay receipts
were evaluated. Verified success and deadline crossings are retained.
No raw replay is pending; preserve all non-depth evidence, timing records and
original tracking-loss inputs. No alternative navigation success is inferred.
Only inventoried regular single-link native depth leaves retired.
Inventory: `.generated/depth_retirement_nogil_replication_assignment14_2026-09-16/`.
Completed: 3476 leaves reclaimed 1043599360 allocated bytes; all non-depth identities and JSON hashes matched.

Retire completed diagnosed reactive_feedback fresh-maze replication success depth, September 16.
Exact root: `go2_nogil_replication_reactive_feedback_noise_2mm_native_layout02_4800_v1_attempt_001`.
Owner exited; physical outcome, contacts, forecast and added-delay receipts
were evaluated. Verified success and deadline crossings are retained.
No raw replay is pending; preserve all non-depth evidence, timing records and
original tracking-loss inputs. No alternative navigation success is inferred.
Only inventoried regular single-link native depth leaves retired.
Inventory: `.generated/depth_retirement_nogil_replication_assignment13_2026-09-16/`.
Completed: 4142 leaves reclaimed 1269964800 allocated bytes; all non-depth identities and JSON hashes matched.

Retire completed diagnosed instantaneous fresh-maze replication success depth, September 16.
Exact root: `go2_nogil_replication_instantaneous_noise_2mm_native_layout02_4800_v1_attempt_001`.
Owner exited; physical outcome, contacts, forecast and added-delay receipts
were evaluated. Verified success and deadline crossings are retained.
No raw replay is pending; preserve all non-depth evidence, timing records and
original tracking-loss inputs. No alternative navigation success is inferred.
Only inventoried regular single-link native depth leaves retired.
Inventory: `.generated/depth_retirement_nogil_replication_assignment12_2026-09-16/`.
Completed: 3306 leaves reclaimed 986312704 allocated bytes; all non-depth identities and JSON hashes matched.

Retire completed diagnosed pose_command fresh-maze replication success depth, September 16.
Exact root: `go2_nogil_replication_pose_command_noise_2mm_native_layout02_4800_v1_attempt_001`.
Owner exited; physical outcome, contacts, forecast and added-delay receipts
were evaluated. Verified success and deadline crossings are retained.
No raw replay is pending; preserve all non-depth evidence, timing records and
original tracking-loss inputs. No alternative navigation success is inferred.
Only inventoried regular single-link native depth leaves retired.
Inventory: `.generated/depth_retirement_nogil_replication_assignment11_2026-09-16/`.
Completed: 3596 leaves reclaimed 1063743488 allocated bytes; all non-depth identities and JSON hashes matched.

Retire redundant depth from the completed older no-RGB direct adapter budget failure, September 16.
Exact case: `go2_all_phase_adapter_maze02_matched_native_v1_attempt_001/all_phase_no_rgb_direct_residual_maze_02`.
The completed six-case cohort and raw sensor/model/command audits remain.
This superseded 3,000-tick paused-physics mission reached the outward goal
at frame 2935, but did not return before the budget. It had zero contacts,
no hard measurement failures, and passed strict camera visibility checks.
The preserved physical/audit readouts retain those successes and failures.
No current training, fit or pending sensor replay uses this recording's depth.
Retire only its regular single-link depth leaves; preserve all RGB, physics,
body/gyro/commands, decisions, models, result/audit records and other cases.
Current replication failures and retained successful references stay full.
Complete historical raw-depth replay of this old case becomes unavailable.
Inventory: `.generated/depth_retirement_old_adapter_no_rgb_direct_2026-09-16/`.
Completed: 9042 leaves reclaimed 8564809728 allocated bytes; all non-depth identities and JSON hashes matched.

Retire completed diagnosed reactive_feedback fresh-maze replication success depth, September 16.
Exact root: `go2_nogil_replication_reactive_feedback_noise_2mm_native_layout01_4800_v1_attempt_001`.
Owner exited; physical outcome, contacts, forecast and added-delay receipts
were evaluated. Verified success and deadline crossings are retained.
No raw replay is pending; preserve all non-depth evidence, timing records and
original tracking-loss inputs. No alternative navigation success is inferred.
Only inventoried regular single-link native depth leaves retired.
Inventory: `.generated/depth_retirement_nogil_replication_assignment09_2026-09-16/`.
Completed: 7702 leaves reclaimed 2365104128 allocated bytes; all non-depth identities and JSON hashes matched.

Retire completed diagnosed instantaneous fresh-maze replication success depth, September 16.
Exact root: `go2_nogil_replication_instantaneous_noise_2mm_native_layout01_4800_v1_attempt_001`.
Owner exited; physical outcome, contacts, forecast and added-delay receipts
were evaluated. Verified success and deadline crossings are retained.
No raw replay is pending; preserve all non-depth evidence, timing records and
original tracking-loss inputs. No alternative navigation success is inferred.
Only inventoried regular single-link native depth leaves retired.
Inventory: `.generated/depth_retirement_nogil_replication_assignment08_2026-09-16/`.
Completed: 4988 leaves reclaimed 1519288320 allocated bytes; all non-depth identities and JSON hashes matched.

Retire completed diagnosed pose_command fresh-maze replication success depth, September 16.
Exact root: `go2_nogil_replication_pose_command_noise_2mm_native_layout01_4800_v1_attempt_001`.
Owner exited; physical outcome, contacts, forecast and added-delay receipts
were evaluated. Verified success and deadline crossings are retained.
No raw replay is pending; preserve all non-depth evidence, timing records and
original tracking-loss inputs. No alternative navigation success is inferred.
Only inventoried regular single-link native depth leaves retired.
Inventory: `.generated/depth_retirement_nogil_replication_assignment07_2026-09-16/`.
Completed: 7584 leaves reclaimed 2416848896 allocated bytes; all non-depth identities and JSON hashes matched.

Retain the first physically verified fresh-maze renderer-fix success, September 16.
Exact root: `go2_nogil_replication_supervised_rollout_noise_2mm_native_layout01_4800_v1_attempt_001`.
Keep its complete depth as the successful fresh-maze replay reference. The five
layout-0 replication tracking failures and the original successful renderer-fix
JEPA reference remain fully retained. Other completed diagnosed replication
successes may retire redundant depth under the existing policy, preserving all
non-depth evidence and marking historical full-depth replay unavailable.

Retire completed diagnosed supervised_rollout nogil-drawing planning-latency stress depth, September 16.
Exact root: `go2_nogil_drawing_plus20ms_supervised_rollout_noise_2mm_native_layout03_4800_v1_attempt_001`.
Owner exited; physical outcome, contacts, forecast and added-delay receipts
were evaluated. Budget failure or success and deadline crossings are retained.
No raw replay is pending; preserve all non-depth evidence, timing records and
original tracking-loss inputs. No alternative navigation success is inferred.
Only inventoried regular single-link native depth leaves retired.
Inventory: `.generated/depth_retirement_nogil_drawing_supervised_rollout_2026-09-16/`.
Completed: 3198 leaves reclaimed 936153088 allocated bytes; all non-depth identities and JSON hashes matched.

Retire redundant depth from the completed older full-direct adapter failure, September 16.
Exact case: `go2_all_phase_adapter_maze02_matched_native_v1_attempt_001/all_phase_full_direct_residual_maze_02`.
The completed six-case cohort, raw audits, model/command replay and failure
readouts remain intact. This superseded 3,000-tick failed mission is not a
current training, fit or pending full sensor-replay input. Preserve its strict
primary-camera visibility failure and the exact depth inputs at frames
1317–1321, including failed frame 1320. All non-depth case/parent evidence,
models, physics, RGB, commands and outcomes remain. Only the other single-link
depth leaves are retired; complete historical replay becomes unavailable.
Current tracking/contact failures and the new successful renderer-fix reference
remain fully retained.
Only inventoried regular single-link native depth leaves retired.
Inventory: `.generated/depth_retirement_old_adapter_direct_2026-09-16/`.
Completed: 9027 leaves reclaimed 6877310976 allocated bytes; all non-depth identities and JSON hashes matched.

Retire completed diagnosed jepa isolated-forecast planning-latency stress depth, September 16.
Exact root: `go2_isolated_forecast_plus20ms_jepa_noise_2mm_native_layout03_4800_v1_attempt_001`.
Owner exited; physical outcome, contacts, forecast and added-delay receipts
were evaluated. Budget failure or success and deadline crossings are retained.
No raw replay is pending; preserve all non-depth evidence, timing records and
original tracking-loss inputs. No alternative navigation success is inferred.
Only inventoried regular single-link native depth leaves retired.
Inventory: `.generated/depth_retirement_isolated_forecast_jepa_2026-09-16/`.
Completed: 9610 leaves reclaimed 3004506112 allocated bytes; all non-depth identities and JSON hashes matched.

Retire completed diagnosed supervised_rollout shared-history planning-latency stress depth, September 16.
Exact root: `go2_shared_history_plus20ms_supervised_rollout_noise_2mm_native_layout03_4800_v1_attempt_001`.
Owner exited; physical outcome, contacts, forecast and added-delay receipts
were evaluated. Budget failure or success and deadline crossings are retained.
No raw replay is pending; preserve all non-depth evidence, timing records and
original tracking-loss inputs. No alternative navigation success is inferred.
Only inventoried regular single-link native depth leaves retired.
Inventory: `.generated/depth_retirement_shared_history_supervised_rollout_2026-09-16/`.
Completed: 9610 leaves reclaimed 3210100736 allocated bytes; all non-depth identities and JSON hashes matched.

End the older full-input supervised seed-1401 maze-0 depth reference pin, September 16.
Exact root: `go2_neural_rgb_transfer_seed_2026091401_full_supervised_rollout_noise_2mm_native_layout00_4800_v1_attempt_001`.
The complete 36-run RGB/no-RGB study and physical/forecast comparisons are
saved. This successful predecessor is not a current training, fit or pending
sensor-replay input. End its earlier full-depth reference pin, preserving the
matched no-RGB tracking failure, current JEPA failure and all non-depth evidence.
The outcome population is unchanged; this success no longer supports exact
historical depth replay.
Only inventoried regular single-link native depth leaves retired.
Inventory: `.generated/depth_retirement_older_supervised_reference_2026-09-16/`.
Completed: 3804 leaves reclaimed 1130270720 allocated bytes; all non-depth identities and JSON hashes matched.

Retire completed components planning-profile diagnostic depth, September 16.
Exact root: `go2_live_planning_profile_jepa_plus20ms_layout03_800_v1_attempt_001`.
Owner exited; physical outcome, contacts and forecasts were evaluated.
Diagnostic records, including invalid profiler evidence where present, are retained.
No raw replay is pending; preserve all non-depth evidence, timing records and
original tracking-loss inputs. No alternative navigation success is inferred.
Only inventoried regular single-link native depth leaves retired.
Inventory: `.generated/depth_retirement_planning_profile_components_2026-09-16/`.
Completed: 1610 leaves reclaimed 533442560 allocated bytes; all non-depth identities and JSON hashes matched.

Retire completed diagnosed jepa shared-history planning-latency stress depth, September 16.
Exact root: `go2_shared_history_plus20ms_jepa_noise_2mm_native_layout03_4800_v1_attempt_001`.
Owner exited; physical outcome, contacts, forecast and added-delay receipts
were evaluated. Budget failure or success and deadline crossings are retained.
No raw replay is pending; preserve all non-depth evidence, timing records and
original tracking-loss inputs. No alternative navigation success is inferred.
Only inventoried regular single-link native depth leaves retired.
Inventory: `.generated/depth_retirement_shared_history_jepa_2026-09-16/`.
Completed: 9610 leaves reclaimed 3087732736 allocated bytes; all non-depth identities and JSON hashes matched.

Retire completed stage planning-profile diagnostic depth, September 16.
Exact root: `go2_live_planning_stage_profile_jepa_plus20ms_layout03_800_v1_attempt_001`.
Owner exited; physical outcome, contacts and forecasts were evaluated.
Diagnostic records, including invalid profiler evidence where present, are retained.
No raw replay is pending; preserve all non-depth evidence, timing records and
original tracking-loss inputs. No alternative navigation success is inferred.
Only inventoried regular single-link native depth leaves retired.
Inventory: `.generated/depth_retirement_planning_profile_stage_2026-09-16/`.
Completed: 1610 leaves reclaimed 529608704 allocated bytes; all non-depth identities and JSON hashes matched.

Retire completed call planning-profile diagnostic depth, September 16.
Exact root: `go2_live_planning_call_profile_jepa_plus20ms_layout03_800_v1_attempt_001`.
Owner exited; physical outcome, contacts and forecasts were evaluated.
Diagnostic records, including invalid profiler evidence where present, are retained.
No raw replay is pending; preserve all non-depth evidence, timing records and
original tracking-loss inputs. No alternative navigation success is inferred.
Only inventoried regular single-link native depth leaves retired.
Inventory: `.generated/depth_retirement_planning_profile_call_2026-09-16/`.
Completed: 1610 leaves reclaimed 539287552 allocated bytes; all non-depth identities and JSON hashes matched.

Retire completed diagnosed supervised_rollout 1-ms-switch planning-latency stress depth, September 16.
Exact root: `go2_planning_switch1ms_plus20ms_supervised_rollout_noise_2mm_native_layout03_4800_v1_attempt_001`.
Owner exited; physical outcome, contacts, forecast and added-delay receipts
were evaluated. Budget failure or success and deadline crossings are retained.
No raw replay is pending; preserve all non-depth evidence, timing records and
original tracking-loss inputs. No alternative navigation success is inferred.
Only inventoried regular single-link native depth leaves retired.
Inventory: `.generated/depth_retirement_planning_switch1ms_supervised_rollout_2026-09-16/`.
Completed: 9610 leaves reclaimed 3128606720 allocated bytes; all non-depth identities and JSON hashes matched.

Retire completed diagnosed jepa 1-ms-switch planning-latency stress depth, September 16.
Exact root: `go2_planning_switch1ms_plus20ms_jepa_noise_2mm_native_layout03_4800_v1_attempt_001`.
Owner exited; physical outcome, contacts, forecast and added-delay receipts
were evaluated. Budget failure or success and deadline crossings are retained.
No raw replay is pending; preserve all non-depth evidence, timing records and
original tracking-loss inputs. No alternative navigation success is inferred.
Only inventoried regular single-link native depth leaves retired.
Inventory: `.generated/depth_retirement_planning_switch1ms_jepa_2026-09-16/`.
Completed: 9610 leaves reclaimed 3166474240 allocated bytes; all non-depth identities and JSON hashes matched.

Retire completed diagnosed supervised_rollout planning-latency stress depth, September 16.
Exact root: `go2_planning_latency_plus20ms_supervised_rollout_noise_2mm_native_layout03_4800_v1_attempt_001`.
Owner exited; physical outcome, contacts, forecast and added-delay receipts
were evaluated. Budget failure or success and deadline crossings are retained.
No raw replay is pending; preserve all non-depth evidence, timing records and
original tracking-loss inputs. No alternative navigation success is inferred.
Only inventoried regular single-link native depth leaves retired.
Inventory: `.generated/depth_retirement_planning_latency_supervised_rollout_2026-09-16/`.
Completed: 9610 leaves reclaimed 3204026368 allocated bytes; all non-depth identities and JSON hashes matched.

Retire completed diagnosed jepa planning-latency stress depth, September 16.
Exact root: `go2_planning_latency_plus20ms_jepa_noise_2mm_native_layout03_4800_v1_attempt_001`.
Owner exited; physical outcome, contacts, forecast and added-delay receipts
were evaluated. Budget failure or success and deadline crossings are retained.
No raw replay is pending; preserve all non-depth evidence, timing records and
original tracking-loss inputs. No alternative navigation success is inferred.
Only inventoried regular single-link native depth leaves retired.
Inventory: `.generated/depth_retirement_planning_latency_jepa_2026-09-16/`.
Completed: 9610 leaves reclaimed 3152003072 allocated bytes; all non-depth identities and JSON hashes matched.

Retire the completed original-controller layout-3 repeat success depth, September 16.
Exact root: `go2_visual_recovery_original_control_jepa_noise_2mm_native_layout03_4800_v1_attempt_001`.
Owner exited; physical arrivals, contacts, forecasts and actual original
controller treatment were evaluated. Timing-related command differences and
all three comparison outcomes are saved; no raw depth replay is pending.
Preserve all non-depth evidence and original tracking-failure recordings.
Only inventoried regular single-link native depth leaves retired.
Inventory: `.generated/depth_retirement_visual_recovery_control_success_2026-09-16/`.
Completed: 4334 leaves reclaimed 1266860032 allocated bytes; all non-depth identities and JSON hashes matched.

Retire the completed visual-recovery-hold layout-3 success depth, September 16.
Exact root: `go2_visual_recovery_dispatch_hold_jepa_noise_2mm_native_layout03_4800_v1_attempt_001`.
Owner exited; physical arrivals, contacts, forecasts and actual treatment were
evaluated. No recovery trigger occurred, so no repair benefit is claimed and
no raw replay is pending. Preserve all non-depth evidence and original failure
recordings. Only inventoried regular single-link native depth leaves retired.
Inventory: `.generated/depth_retirement_visual_recovery_hold_success_2026-09-16/`.
Completed: 4640 leaves reclaimed 1348276224 allocated bytes; all non-depth identities and JSON hashes matched.

Retire completed diagnosed baseline reactive_feedback/layout-3 depth, September 16.
Exact root: `go2_persistent_visual_baselines_reactive_feedback_noise_2mm_native_layout03_4800_v1_attempt_001`.
Owner exited and persistence, physical/selector/model/dispatch/forecast
evaluations completed. The outcome and applicable failure diagnosis are saved;
no raw replay is pending. Preserve every non-depth record, including failures,
RGB, commands, poses, timing, forecasts and physics. This does not claim an
unobserved counterfactual success or resolve unisolated map/dynamics causes.
Only inventoried regular single-link native depth leaves retired.
Inventory: `.generated/depth_retirement_completed_baseline_reactive_feedback03_2026-09-16/`.
Completed: 3732 leaves reclaimed 1137950720 allocated bytes; all non-depth identities and JSON hashes matched.

Retire completed diagnosed baseline reserved_off/layout-3 depth, September 16.
Exact root: `go2_persistent_visual_baselines_reserved_off_noise_2mm_native_layout03_4800_v1_attempt_001`.
Owner exited and persistence, physical/selector/model/dispatch/forecast
evaluations completed. The outcome and applicable failure diagnosis are saved;
no raw replay is pending. Preserve every non-depth record, including failures,
RGB, commands, poses, timing, forecasts and physics. This does not claim an
unobserved counterfactual success or resolve unisolated map/dynamics causes.
Only inventoried regular single-link native depth leaves retired.
Inventory: `.generated/depth_retirement_completed_baseline_reserved_off03_2026-09-16/`.
Completed: 9610 leaves reclaimed 2671222784 allocated bytes; all non-depth identities and JSON hashes matched.

Retire the completed older persistent-routing layout-0 success depth, September 16.
Exact root: `go2_routing_memory_persistent_native_layout00_4800_v1_attempt_001`.
This ends its earlier full-depth reference pin. The fixed eight-run memory
study, physical arrivals, actual routing-scope comparison and inspected
trajectory figures are complete. No active depth replay or fit requires this
recording; current controller comparisons use their own recordings. Retain the
entire eight-outcome population, all scope checks, poses, RGB, body/gyro,
commands, forecasts and physics, and all reduced-memory failure evidence.
Retire only inventoried regular single-link native depth leaves. This does not
claim learned internal-memory attribution or availability of exact historical
sensor replay. Current JEPA tracking loss, original cadenced tracking loss and
direct-contact failure sensors remain full.
Inventory: `.generated/depth_retirement_completed_routing_memory_reference00_2026-09-16/`.
Completed: 6212 leaves reclaimed 1862594560 allocated bytes; all non-depth identities and JSON hashes matched.

Retire completed diagnosed baseline instantaneous/layout-3 depth, September 16.
Exact root: `go2_persistent_visual_baselines_instantaneous_noise_2mm_native_layout03_4800_v1_attempt_001`.
Owner exited and persistence, physical/selector/model/dispatch/forecast
evaluations completed. The outcome and applicable failure diagnosis are saved;
no raw replay is pending. Preserve every non-depth record, including failures,
RGB, commands, poses, timing, forecasts and physics. This does not claim an
unobserved counterfactual success or resolve unisolated map/dynamics causes.
Only inventoried regular single-link native depth leaves retired.
Inventory: `.generated/depth_retirement_completed_baseline_instantaneous03_2026-09-16/`.
Completed: 4892 leaves reclaimed 1458515968 allocated bytes; all non-depth identities and JSON hashes matched.

Retire completed diagnosed baseline reserved_off/layout-2 depth, September 16.
Exact root: `go2_persistent_visual_baselines_reserved_off_noise_2mm_native_layout02_4800_v1_attempt_001`.
Owner exited and persistence, physical/selector/model/dispatch/forecast
evaluations completed. The outcome and applicable failure diagnosis are saved;
no raw replay is pending. Preserve every non-depth record, including failures,
RGB, commands, poses, timing, forecasts and physics. This does not claim an
unobserved counterfactual success or resolve unisolated map/dynamics causes.
Only inventoried regular single-link native depth leaves retired.
Inventory: `.generated/depth_retirement_completed_baseline_reserved_off02_2026-09-16/`.
Completed: 9610 leaves reclaimed 3187372032 allocated bytes; all non-depth identities and JSON hashes matched.

Retire completed diagnosed baseline instantaneous/layout-2 depth, September 16.
Exact root: `go2_persistent_visual_baselines_instantaneous_noise_2mm_native_layout02_4800_v1_attempt_001`.
Owner exited and persistence, physical/selector/model/dispatch/forecast
evaluations completed. The outcome and applicable failure diagnosis are saved;
no raw replay is pending. Preserve every non-depth record, including failures,
RGB, commands, poses, timing, forecasts and physics. This does not claim an
unobserved counterfactual success or resolve unisolated map/dynamics causes.
Only inventoried regular single-link native depth leaves retired.
Inventory: `.generated/depth_retirement_completed_baseline_instantaneous02_2026-09-16/`.
Completed: 3996 leaves reclaimed 1228275712 allocated bytes; all non-depth identities and JSON hashes matched.

Retire completed diagnosed baseline reactive_feedback/layout-2 depth, September 16.
Exact root: `go2_persistent_visual_baselines_reactive_feedback_noise_2mm_native_layout02_4800_v1_attempt_001`.
Owner exited and persistence, physical/selector/model/dispatch/forecast
evaluations completed. The outcome and applicable failure diagnosis are saved;
no raw replay is pending. Preserve every non-depth record, including failures,
RGB, commands, poses, timing, forecasts and physics. This does not claim an
unobserved counterfactual success or resolve unisolated map/dynamics causes.
Only inventoried regular single-link native depth leaves retired.
Inventory: `.generated/depth_retirement_completed_baseline_reactive_feedback02_2026-09-16/`.
Completed: 3724 leaves reclaimed 1111363584 allocated bytes; all non-depth identities and JSON hashes matched.

Retire completed diagnosed baseline instantaneous/layout-1 depth, September 16.
Exact root: `go2_persistent_visual_baselines_instantaneous_noise_2mm_native_layout01_4800_v1_attempt_001`.
Owner exited and persistence, physical/selector/model/dispatch/forecast
evaluations completed. The outcome and applicable failure diagnosis are saved;
no raw replay is pending. Preserve every non-depth record, including failures,
RGB, commands, poses, timing, forecasts and physics. This does not claim an
unobserved counterfactual success or resolve unisolated map/dynamics causes.
Only inventoried regular single-link native depth leaves retired.
Inventory: `.generated/depth_retirement_completed_baseline_instantaneous01_2026-09-16/`.
Completed: 5108 leaves reclaimed 1559965696 allocated bytes; all non-depth identities and JSON hashes matched.

Retire completed diagnosed baseline reactive_feedback/layout-1 depth, September 16.
Exact root: `go2_persistent_visual_baselines_reactive_feedback_noise_2mm_native_layout01_4800_v1_attempt_001`.
Owner exited and persistence, physical/selector/model/dispatch/forecast
evaluations completed. The outcome and applicable failure diagnosis are saved;
no raw replay is pending. Preserve every non-depth record, including failures,
RGB, commands, poses, timing, forecasts and physics. This does not claim an
unobserved counterfactual success or resolve unisolated map/dynamics causes.
Only inventoried regular single-link native depth leaves retired.
Inventory: `.generated/depth_retirement_completed_baseline_reactive_feedback01_2026-09-16/`.
Completed: 3766 leaves reclaimed 1147236352 allocated bytes; all non-depth identities and JSON hashes matched.

Retire completed diagnosed baseline reserved_off/layout-1 depth, September 16.
Exact root: `go2_persistent_visual_baselines_reserved_off_noise_2mm_native_layout01_4800_v1_attempt_001`.
Owner exited and persistence, physical/selector/model/dispatch/forecast
evaluations completed. The outcome and applicable failure diagnosis are saved;
no raw replay is pending. Preserve every non-depth record, including failures,
RGB, commands, poses, timing, forecasts and physics. This does not claim an
unobserved counterfactual success or resolve unisolated map/dynamics causes.
Only inventoried regular single-link native depth leaves retired.
Inventory: `.generated/depth_retirement_completed_baseline_reserved_off01_2026-09-16/`.
Completed: 9610 leaves reclaimed 2119110656 allocated bytes; all non-depth identities and JSON hashes matched.

Retire completed diagnosed baseline reactive_feedback/layout-0 depth, September 16.
Exact root: `go2_persistent_visual_baselines_reactive_feedback_noise_2mm_native_layout00_4800_v1_attempt_001`.
Owner exited and persistence, physical/selector/model/dispatch/forecast
evaluations completed. The outcome and applicable failure diagnosis are saved;
no raw replay is pending. Preserve every non-depth record, including failures,
RGB, commands, poses, timing, forecasts and physics. This does not claim an
unobserved counterfactual success or resolve unisolated map/dynamics causes.
Only inventoried regular single-link native depth leaves retired.
Inventory: `.generated/depth_retirement_completed_baseline_reactive_feedback00_2026-09-16/`.
Completed: 9610 leaves reclaimed 2996178944 allocated bytes; all non-depth identities and JSON hashes matched.

Retire completed current-baseline layout-0 instantaneous and reserved-off depth,
September 16. Exact roots:
`go2_persistent_visual_baselines_instantaneous_noise_2mm_native_layout00_4800_v1_attempt_001`
and `go2_persistent_visual_baselines_reserved_off_noise_2mm_native_layout00_4800_v1_attempt_001`.
Both owners exited and fully persisted. Physical/selector/model/dispatch and
executed forecast evaluations completed. Instantaneous ranking passed its
round trip; reserved-off failed with 1,090 consecutive reserve-blocked holds.
Its first/last physical and stored clearances are saved. Underlying map-error
cause and safe turn-only escape are not established; no raw replay is pending.
All non-depth comparison inputs, including every failed outcome and diagnostic,
remain. Current JEPA tracking-loss and original direct-contact sensors remain
full. Only inventoried regular single-link native depth leaves retired.
Inventory: `.generated/depth_retirement_completed_baselines_instant_reserved00_2026-09-16/`.
Completed: 13580 leaves reclaimed 4110422016 allocated bytes; all non-depth identities and JSON hashes matched.

Retire the diagnosed original short-pulse instantaneous/layout-1 survey-stall
depth, September 16. Exact root:
`go2_short_pulse_navigation_instantaneous_noise_2mm_native_layout01_4800_v1_attempt_001`.
This ends its earlier full-depth pin. The fourteen-mission study, physical
outcome, survey/clearance diagnosis and complete comparison figures are saved.
No further raw replay is pending: the failure was a persistent predicted
clearance rejection during the initial panorama, with 972 blocked plans.
Retain every non-depth record and its failed outcome. Current JEPA/layout-3
tracking-loss sensors and original direct-contact sensors remain full. Retire
only inventoried regular single-link native depth leaves. Inventory:
`.generated/depth_retirement_diagnosed_old_instantaneous01_2026-09-16/`.
Completed: 9610 leaves reclaimed 3488686080 allocated bytes; all non-depth identities and JSON hashes matched.

Retire completed direct/layout-3 depth, September 16. Exact root:
`go2_persistent_visual_learning_comparison_direct_noise_2mm_native_layout03_4800_v1_attempt_001`.
Owner exited, persistence and physical/model/dispatch/forecast evaluations
completed. No raw-depth replay pending; all non-depth results, RGB, commands,
poses, timing, forecasts and physics retained for comparisons. Only inventoried
regular single-link native depth leaves retired. Inventory: `.generated/depth_retirement_completed_learning_direct03_2026-09-16/`.
Completed: 4564 leaves reclaimed 1338556416 allocated bytes; all non-depth identities and JSON hashes matched.

Retire completed direct/layout-2 depth, September 16. Exact root:
`go2_persistent_visual_learning_comparison_direct_noise_2mm_native_layout02_4800_v1_attempt_001`.
Owner exited, persistence and physical/model/dispatch/forecast evaluations
completed. No raw-depth replay pending; all non-depth results, RGB, commands,
poses, timing, forecasts and physics retained for comparisons. Only inventoried
regular single-link native depth leaves retired. Inventory: `.generated/depth_retirement_completed_learning_direct02_2026-09-16/`.
Completed: 3732 leaves reclaimed 1105235968 allocated bytes; all non-depth identities and JSON hashes matched.

Retire completed jepa/layout-2 depth, September 16. Exact root:
`go2_persistent_visual_learning_comparison_jepa_noise_2mm_native_layout02_4800_v1_attempt_001`.
Owner exited, persistence and physical/model/dispatch/forecast evaluations
completed. No raw-depth replay pending; all non-depth results, RGB, commands,
poses, timing, forecasts and physics retained for comparisons. Only inventoried
regular single-link native depth leaves retired. Inventory: `.generated/depth_retirement_completed_learning_jepa02_2026-09-16/`.
Completed: 3196 leaves reclaimed 927588352 allocated bytes; all non-depth identities and JSON hashes matched.

Retire completed jepa/layout-1 depth, September 16. Exact root:
`go2_persistent_visual_learning_comparison_jepa_noise_2mm_native_layout01_4800_v1_attempt_001`.
Owner exited, persistence and physical/model/dispatch/forecast evaluations
completed. No raw-depth replay pending; all non-depth results, RGB, commands,
poses, timing, forecasts and physics retained for comparisons. Only inventoried
regular single-link native depth leaves retired. Inventory: `.generated/depth_retirement_completed_learning_jepa01_2026-09-16/`.
Completed: 3538 leaves reclaimed 1071513600 allocated bytes; all non-depth identities and JSON hashes matched.

Retire completed direct/layout-1 comparison depth, September 16. Exact root:
`go2_persistent_visual_learning_comparison_direct_noise_2mm_native_layout01_4800_v1_attempt_001`.
Owner exited; full persistence, physical goal/home verification, model/dispatch
treatment and executed forecast evaluations completed. Zero contacts or visual
tracking loss. No raw-depth replay pending; all inputs needed for the paired
comparison remain in non-depth records. Retire only inventoried single-link
regular native depth leaves, preserving every result and failure record.
Inventory: `.generated/depth_retirement_completed_learning_direct01_2026-09-16/`.
Completed: 4388 leaves reclaimed 1315475456 allocated bytes; all non-depth identities and JSON hashes matched.

Retire completed JEPA/direct layout-0 comparison depth, September 16. Exact roots:
`go2_persistent_visual_learning_comparison_jepa_noise_2mm_native_layout00_4800_v1_attempt_001`
and `go2_persistent_visual_learning_comparison_direct_noise_2mm_native_layout00_4800_v1_attempt_001`.
Both owners ended and physical/model/dispatch evaluations completed. JEPA's
round trip and direct's budget failure remain in the four-arm comparison and
inspected figure. Direct's near-goal hold preference and later return deadline
misses are diagnosed from preserved plans, requests and physical trajectories;
no raw replay is pending. This does not claim counterfactual success or isolate
the internal cause of planning latency. Retain every non-depth artifact,
including all failure evidence. Only inventoried regular single-link native
primary/auxiliary depth leaves are eligible. Inventory:
`.generated/depth_retirement_completed_learning_pair00_2026-09-16/`.
Completed: 12768 depth leaves reclaimed 4372336640 allocated bytes; all 12855 non-depth identities and 69 JSON hashes matched.

Retire completed fresh-layout-3 pair depth, September 16. Exact roots:
`go2_persistent_visual_transfer_supervised_rollout_noise_2mm_native_layout03_4800_v1_attempt_001`
and `go2_persistent_visual_transfer_pose_command_noise_2mm_native_layout03_4800_v1_attempt_001`.
Both owners ended, full persistence and physical/model/dispatch/forecast
evaluations passed. The final paired result, inspected trajectory figure and
complete eight-mission aggregate are saved. No raw replay is pending; the next
training-condition comparison uses retained outcomes and unchanged source/model
inputs. Keep every non-depth artifact. Retire only inventoried regular
single-link primary/auxiliary depth leaves. Inventory:
`.generated/depth_retirement_completed_fresh_pair03_2026-09-16/`.
Completed: 8,766 leaves reclaimed 2,548,940,800 allocated bytes. All 68 JSON
hashes and 8,852 non-depth identities matched; 6,659,575,808 bytes were free.

Retire completed fresh-layout-2 pair depth, September 16. Exact roots:
`go2_persistent_visual_transfer_supervised_rollout_noise_2mm_native_layout02_4800_v1_attempt_001`
and `go2_persistent_visual_transfer_pose_command_noise_2mm_native_layout02_4800_v1_attempt_001`.
Both owners ended, full persistence and physical/model/dispatch/forecast
evaluations passed, and the paired result, command-duration/recovery readout and
inspected trajectory figure are saved. No raw replay is pending. Preserve all
RGB, poses, commands, forecasts, physics, timing and outcomes. Retire only
inventoried regular single-link primary/auxiliary depth leaves. Inventory:
`.generated/depth_retirement_completed_fresh_pair02_2026-09-16/`.
Completed: 6,478 leaves reclaimed 1,917,796,352 allocated bytes. All 68 JSON
hashes and 6,564 non-depth identities matched; 7,249,731,584 bytes were free.

Retire the completed fresh learned/layout-1 success and older completed
rollout-selection-off/layout-0 failure depth, September 16. Exact roots:
`go2_persistent_visual_transfer_supervised_rollout_noise_2mm_native_layout01_4800_v1_attempt_001`
and `go2_rollout_selection_off_seed_2026091001_full_supervised_rollout_noise_2mm_native_layout00_4800_v1_attempt_001`.
The fresh pair's physical/treatment/forecast evaluations and inspected trajectory
comparison are complete. The older failed experiment and four-controller
comparison are complete: its 940 consecutive current-clearance hold plans,
native/stored clearances, full failure outcome and all non-depth records remain.
Its map-discrepancy cause is still unisolated; this retirement does not claim
otherwise. No raw replay is pending for either case. The newer survey-stall,
direct-contact and original cadenced visual-loss raw inputs remain full.
This ends the older rollout-off failure's full-depth pin. Retire only inventoried
regular single-link depth leaves. Inventory:
`.generated/depth_retirement_completed_fresh_learned01_and_old_off00_2026-09-16/`.
Completed: 13,704 leaves reclaimed 4,136,443,904 allocated bytes. All 66 JSON
hashes and 13,788 non-depth identities matched; 7,689,973,760 bytes were free.

Retire completed fresh-layout-1 pose/command success depth, September 16. Root:
`go2_persistent_visual_transfer_pose_command_noise_2mm_native_layout01_4800_v1_attempt_001`.
Owner exited and persistence plus physical/model/dispatch/forecast evaluations
passed. There were no contacts or tracking losses, no visual recovery triggers,
and two early terminal entries. All comparison inputs (RGB, poses, physics,
commands, forecasts, timing and outcomes) remain; the next learned arm and
paired analysis do not require raw-depth replay. No such replay is pending.
Only inventoried regular single-link depth leaves may be retired. Inventory:
`.generated/depth_retirement_completed_fresh_pose01_2026-09-16/`.
Completed: 3,982 leaves reclaimed 1,218,371,584 allocated bytes. All 34 JSON
hashes and 4,025 non-depth identities matched; 5,073,014,784 bytes were free.

Retire completed first fresh-layout comparison depth, September 16. Exact roots:
`go2_persistent_visual_transfer_supervised_rollout_noise_2mm_native_layout00_4800_v1_attempt_001`
and `go2_persistent_visual_transfer_pose_command_noise_2mm_native_layout00_4800_v1_attempt_001`.
Both owners ended, persistence and physical/model/dispatch evaluations passed,
and their paired outcomes, forecast errors, deadline/service-time readout and
trajectory figure are preserved. The control's return pause was directly
explained by 690 late plans out of 693 during frames 1200–3999; retained timing,
commands and forecasts suffice for that finding. The internal source of the
planning-service increase is not established, and no raw-depth replay is
pending. Preserve all non-depth records, both successes and the timing caveat.
Retire only inventoried regular single-link primary/auxiliary depth leaves.
Inventory: `.generated/depth_retirement_completed_fresh_pair00_2026-09-16/`.
Completed: 12,108 leaves reclaimed 3,890,495,488 allocated bytes. All 68 JSON
hashes and 12,194 non-depth identities matched; 5,358,825,472 bytes were free.

Retire completed fixed-controller exposed-pair depth, September 16. Exact roots:
`go2_persistent_local_visual_supervised_rollout_noise_2mm_native_layout00_4800_v1_attempt_001`
and `go2_persistent_local_visual_supervised_rollout_noise_2mm_native_layout01_4800_v1_attempt_001`.
Both owners exited successfully; physical arrivals, actual model/dispatch
treatment and recovery/terminal-mode readouts are complete. The combined
`go2_persistent_local_visual_fixed_pair_v1_attempt_001` preserves both full
outcomes and the inspected trajectory comparison. No raw replay is pending;
fresh-layout comparisons use the retained results and frozen controller/model.
This ends their earlier full-sensor pins. Preserve all RGB, commands, poses,
forecasts, feature receipts, physics and results. Keep unresolved contact,
survey-stall and original cadenced visual-loss inputs in full. Retire only
inventoried regular single-link primary/auxiliary depth leaves. Inventory:
`.generated/depth_retirement_completed_persistent_pair_2026-09-16/`.
Completed: 14,590 leaves reclaimed 4,544,667,648 allocated bytes. All 70 JSON
hashes and 14,678 non-depth identities matched; 6,192,930,816 bytes were free.

Retire superseded cadenced maze-1 success and diagnosed framewise maze-0 failure
depth, September 16. Exact roots:
`go2_cadenced_view_age_250ms_supervised_rollout_noise_2mm_native_layout01_4800_v1_attempt_001`
and `go2_framewise_visual_support_supervised_rollout_noise_2mm_native_layout00_4800_v1_attempt_001`.
Both owners ended, persistence/physical/treatment evaluation and development
comparisons are complete. Their non-depth evidence supplies the upcoming fixed
controller comparison; no raw replay is pending. Keep every outcome, feature
receipt, RGB/body/command/physics record and comparison. This supersedes their
full-depth pins. Retain the new persistent-controller round trip, original
cadenced visual-loss, direct-contact and survey-stall sensors in full. Only
inventoried regular single-link depth leaves are eligible. Inventory:
`.generated/depth_retirement_completed_cadenced_and_framewise_2026-09-16/`.
Completed: 8,242 leaves reclaimed 2,610,941,952
allocated bytes; all 68 JSON hashes and 8,328 non-depth identities match.

Retire diagnosed prefix-aware terminal attempt depth, September 16. Exact root:
`go2_prefix_aware_terminal_supervised_rollout_noise_2mm_native_layout00_4800_v1_attempt_001`.
Its tracking failure, physical/model/dispatch evaluations, exact 1,134-pose
replay and camera-cadence recovery readout are complete and preserved. No pending
raw replay uses it. Keep all RGB, poses, features, commands, forecasts and physics,
plus the newer framewise failure and original cadenced tracking-loss sensors.
This ends the prior full-depth pin for this diagnosed failure only. Inventory:
`.generated/depth_retirement_diagnosed_prefix_failure_2026-09-16/`.
Completed: 2,276 leaves reclaimed 760,795,136
allocated bytes; all 35 JSON hashes and 2,320 non-depth identities match.

End the diagnosed age-250 tracking-queue failure full-depth pin, September 16.
Exact root:
`go2_pipeline_age_250ms_supervised_rollout_noise_2mm_native_layout01_4800_v1_attempt_001`.
Exact original replay, candidate replay, floor/pose accuracy evaluation and
matched tracking-cost diagnosis are complete. Cadenced native tracking has since
completed maze 1 and a full maze-0 mission. No pending raw replay uses this queue
failure. Preserve its failure, all non-depth records, replay results and profiles.
Keep the distinct maze-0 visual-loss, direct-contact, survey-stall and cadenced
maze-1 success sensors. This supersedes earlier queue-failure full-depth pins.
Inventory: `.generated/depth_retirement_diagnosed_age_queue_2026-09-16/`.
Completed: 3,904 depth leaves reclaimed
1,264,144,384 allocated bytes; all 31 JSON hashes and
3,944 non-depth identities matched.

Retire the completed visual-support maze-0 near-miss depth, September 16.
Exact root:
`go2_visual_support_recovery_supervised_rollout_noise_2mm_native_layout00_4800_v1_attempt_001`.
Owner exit, full persistence, physical goal/dwell/contact evaluation, actual
model/dispatch treatment and terminal-command diagnosis are complete. Its
return failure, all recovery feature counts, RGB, poses, command histories,
forecasts and physics remain. The next terminal-control experiment uses these
retained records, not raw-depth replay. This ends the temporary full-depth pin
in its experiment journal. Keep the earlier tracking failure, contact failure,
survey stall, queue failure and cadenced maze-1 success sensors in full. Only
inventoried regular single-link depth leaves are eligible. Inventory:
`.generated/depth_retirement_diagnosed_visual_support_maze00_2026-09-16/`.
Completed: 9,610 depth leaves reclaimed
3,057,565,696 allocated bytes; all 35 JSON hashes and
9,654 non-depth identities matched.

End the remaining older current-reserve maze-0 full-depth pin, September 16.
Exact root:
`go2_current_reserve_terminal_feedback_seed_2026091001_full_supervised_rollout_noise_2mm_native_layout00_4800_v1_attempt_001`.
The completed reserve-deficit diagnosis, physical/treatment evaluations and
four-controller comparison remain. No pending sensor replay uses this older
case. Preserve its unsuccessful outcome and all non-depth files; retire only
inventoried regular single-link depth leaves. This supersedes the previous
maze-0 full-depth pin. Keep current contact, survey-stall, tracking-queue failure
and repaired-controller success recordings in full. Inventory:
`.generated/depth_retirement_diagnosed_reserve_maze00_2026-09-16/`.
Completed: 9,610 depth leaves reclaimed 2,103,701,504 allocated bytes. All
31 JSON hashes and 9,650 non-depth identities match; free space was
5,391,552,512 bytes after cleanup.

End the older current-reserve maze-1 full-depth pin, September 16. Exact root:
`go2_current_reserve_terminal_feedback_seed_2026091001_full_supervised_rollout_noise_2mm_native_layout01_4800_v1_attempt_001`.
Its completed reserve-deadlock diagnosis, physical/actual-treatment evaluation,
four-controller comparisons and inspected figures are preserved. The matching
maze-0 reserve-deadlock recording stays full. No pending raw-depth replay or fit
uses maze 1, and current learning uses retained RGB/body/commands and labels.
Keep this unsuccessful outcome and every non-depth artifact; retire only exact
regular single-link primary/auxiliary depth leaves. This supersedes the older
instruction to retain both reserve-failure depth recordings in full. Keep the
new age-bound tracking-queue failure, direct-contact and survey-stall sensors
in full. Inventory:
`.generated/depth_retirement_diagnosed_reserve_maze01_2026-09-16/`.
Completed: 9,610 leaves reclaimed 2,922,696,704 allocated bytes. All 31
preserved JSON hashes and 9,650 non-depth identities match; available space
was 5,742,755,840 bytes before the next native mission.

Retire diagnosed thread-switch maze-1 depth, September 16. Exact root:
`go2_thread_switch_1ms_supervised_rollout_noise_2mm_native_layout01_4800_v1_attempt_001`.
Owner exit, complete persistence, physical/treatment/forecast evaluation and
paired timing/dispatch diagnosis are complete. Keep its unsuccessful mission,
all non-depth evidence, and the distinct contact and initial-survey-stall
raw inputs. No pending raw-depth replay uses this timing case. Only exact
inventoried regular single-link primary/auxiliary depth leaves are eligible.
Full original sensor replay ends. Inventory:
`.generated/depth_retirement_diagnosed_thread_switch_2026-09-16/`.
Completed: 9,610 leaves reclaimed 3,141,812,224 allocated bytes. All 33
preserved JSON hashes and 9,653 non-depth identities match; available space
was 4,382,846,976 bytes before the age-bound experiment.

Retire diagnosed packed-observer maze-1 depth, September 16. Exact root:
`go2_obstacle_grouping_navigation_packed_supervised_rollout_noise_2mm_native_layout01_4800_v1_attempt_001`.
Physical/treatment and forecast evaluations, dispatch diagnosis, paired timing
comparison and the final public startup-planning profile are complete. No
pending raw-depth replay uses this case. Preserve its budget failure, every
non-depth record and profiles; retain the distinct direct-collision and
initial-survey-stall depth inputs. Retire only inventoried regular single-link
primary/auxiliary depth leaves. Full original sensor replay ends. Inventory:
`.generated/depth_retirement_diagnosed_packed_observer_2026-09-16/`.
Completed: 9,610 leaves reclaimed 3,174,973,440 allocated bytes. All 33
preserved JSON hashes and 9,658 non-depth identities match; available space
was 5,057,183,744 bytes before the thread-switch experiment.

Retire completed reactive maze-1 and diagnosed observer-grouping reference
depth, September 16. Exact roots:
`go2_short_pulse_navigation_reactive_noise_2mm_native_layout01_4800_v1_attempt_001`
and `go2_obstacle_grouping_navigation_reference_supervised_rollout_noise_2mm_native_layout01_4800_v1_attempt_001`.
The reactive success, both reactive-layout comparison and complete fourteen-run
study/figures are saved. The new reference has complete physical/treatment,
forecast, phase and dispatch evaluations: no arrivals or contacts, maximum pose
error 3.948 mm, all 726 latched windows caused by stale observations. Neither
case is used by a pending raw-depth replay or fit. The upcoming timing-pair
comparison uses retained outcomes, timing, requests, poses and physics records.
Keep every non-depth file, including the reference's unsuccessful outcome and
all RGB/body/command inputs. Keep the direct maze-0 collision and maze-1 survey
stall raw inputs in full. This ends the reactive reference's previous full-depth
pin. Retire only exact inventoried regular single-link depth leaves; original
full sensor replay ends for these two cases. Inventory:
`.generated/depth_retirement_completed_reactive_and_observer_reference_2026-09-16/`.
Completed: 14,760 leaves reclaimed 4,732,420,096 allocated bytes (5,150
reactive leaves and 9,610 reference leaves). All 65 preserved JSON hashes
and 14,844 non-depth identities match. Available space was 5,726,306,304
bytes before the packed-observer mission launched.

Retire diagnosed short-pulse maze-1 JEPA depth after the complete fourteen-run
comparison, September 16:
`go2_short_pulse_navigation_jepa_noise_2mm_native_layout01_4800_v1_attempt_001`.
Owner exit, complete persistence, physical outcome, actual treatment, forecast,
phase/dispatch analyses and complete comparison/trajectory figure are saved.
Keep its budget failure and every non-depth file. No contact or tracking failure
occurred; maximum position error was 4.558 mm. All 459 latched windows began
with stale observations; no persistent all-candidate clearance block occurred.
No pending raw-depth replay or fit uses this recording. Keep the maze-0 direct
collision, maze-1 survey-stall inputs and reactive maze-1 reference in full.
Only exact inventoried regular single-link depth leaves are retired; original
full sensor replay ends. Inventory:
`.generated/depth_retirement_diagnosed_short_pulse_jepa_maze01_2026-09-16/`.
Completed: 9,610 leaves reclaimed 3,139,473,408 allocated bytes. All 33
preserved JSON hashes and 9,653 non-depth identities match. Available space
was 4,745,621,504 bytes before the observer-grouping reference launched.

Retire diagnosed short-pulse maze-1 direct depth, September 16:
`go2_short_pulse_navigation_direct_noise_2mm_native_layout01_4800_v1_attempt_001`.
Owner exit, persistence, physical evaluation, actual treatment and forecast,
phase, dispatch and deadline-completion analyses are complete. Preserve its
budget failure and every non-depth record, including the five-controller
stale-completion comparison. There were no contacts or tracking failures;
maximum position error was 5.509 mm. All 790 latched windows began with stale
observations and every selected plan had an eligible candidate. No pending
raw-depth replay or fit uses this recording. Keep the distinct maze-0 direct
collision, maze-1 survey-stall inputs and reactive maze-1 full reference.
Only exact inventoried regular single-link depth leaves are retired; original
full sensor replay ends. Inventory:
`.generated/depth_retirement_diagnosed_short_pulse_direct_maze01_2026-09-16/`.
Completed: 9,610 leaves reclaimed 3,125,456,896 allocated bytes. All 34
preserved JSON hashes and 9,654 non-depth identities match. Available space
was 5,434,552,320 bytes before assignment 14 launched.

Retire diagnosed short-pulse maze-1 supervised depth, September 16:
`go2_short_pulse_navigation_supervised_rollout_noise_2mm_native_layout01_4800_v1_attempt_001`.
Owner exit, complete persistence, physical outcome, actual treatment,
forecast and phase/dispatch analyses are complete. Preserve its budget failure,
all results and all non-depth records. No contact or tracking failure occurred;
maximum position error was 5.876 mm. All 614 latched windows were traced to
stale observations; every plan had an eligible candidate. No pending raw-depth
replay or fit uses this case. Retain direct-collision and survey-stall depth
inputs and the reactive maze-1 full reference. Retire only exact inventoried
regular single-link primary/auxiliary depth leaves; full original sensor replay
ends. Inventory:
`.generated/depth_retirement_diagnosed_short_pulse_supervised_maze01_2026-09-16/`.
Completed: 9,610 leaves reclaimed 3,134,046,208 allocated bytes. All 33
preserved JSON hashes and 9,653 non-depth identities match. Available space
was 6,096,474,112 bytes before assignment 13 launched.

Retire completed, diagnosed short-pulse maze-1 pose-command depth, September 16:
`go2_short_pulse_navigation_pose_command_noise_2mm_native_layout01_4800_v1_attempt_001`.
Owner exit, complete persistence, physical outcome and actual treatment are
verified. Preserve its unsuccessful mission (budget exhaustion, no arrivals),
forecast, phase/frontier and dispatch readouts and all non-depth records. No
contact or tracking failure occurred; maximum position error was 4.679 mm.
All 629 latched windows began with stale observations; there was no persistent
clearance deadlock. No pending raw-depth replay or fit uses this case. Keep
the active direct-collision and survey-stall depth inputs and reactive maze-1
full reference. Retire only inventoried regular single-link primary/auxiliary
depth leaves; original full sensor replay ends. Inventory:
`.generated/depth_retirement_diagnosed_short_pulse_pose_command_maze01_2026-09-16/`.
Completed: 9,610 leaves reclaimed 3,180,675,072 allocated bytes. All 33
preserved JSON hashes and 9,653 non-depth identities match. Available space
was 6,771,920,896 bytes before assignment 12 launched.

End the completed, diagnosed short-pulse maze-1 command-history depth pin,
September 16:
`go2_short_pulse_navigation_command_history_noise_2mm_native_layout01_4800_v1_attempt_001`.
The failed mission outcome (budget exhaustion, no arrivals), actual treatment,
forecast, phase/frontier, dispatch and trajectory comparisons are preserved.
There was no contact or tracking failure; maximum position error was 6.770 mm.
All 616 latched windows were traced to stale obstacle observations. No pending
raw-depth replay or fit uses this recording. Keep every non-depth file and
all failure/outcome evidence; retain the direct collision, maze-1 survey stall
and maze-1 reactive success depth references. Only exact inventoried regular
single-link primary/auxiliary depth leaves may be retired. Full original
sensor replay ends. Inventory:
`.generated/depth_retirement_diagnosed_short_pulse_command_history_maze01_2026-09-16/`.
Completed: 9,610 depth leaves reclaimed 3,145,342,976 allocated bytes. All
33 preserved JSON hashes and 9,653 non-depth identities match. Available
space was 7,447,568,384 bytes while assignment 11 was active.

End the diagnosed short-pulse maze-0 JEPA depth reference pin, September 16:
`go2_short_pulse_navigation_jepa_noise_2mm_native_layout00_4800_v1_attempt_001`.
Its outbound arrival, return timeout, terminal-selection, executed-forecast,
dispatch and complete first-maze comparisons are saved. No pending raw-depth
replay or fit consumes it. Retain every outcome and non-depth input, including
RGB/body/commands, poses, physics and models. Keep the direct collision,
maze-1 survey stall and maze-1 reactive success depth references in full.
Retire only inventoried regular single-link primary/auxiliary depth leaves in
this exact recording. Original full sensor replay becomes unavailable.
Inventory: `.generated/depth_retirement_diagnosed_short_pulse_jepa_2026-09-16/`.
This supersedes earlier full-depth pins only for this recording.
Completed: 9,612 leaves retired, reclaiming 3,111,563,264 allocated bytes.
All 31 preserved JSON hashes and 9,652 non-depth identities match; available
space was 4,307,427,328 bytes before assignment 11 launched.

End the completed short-pulse maze-0 pose-command and reactive success depth
pins, September 16. Exact roots:
`go2_short_pulse_navigation_pose_command_noise_2mm_native_layout00_4800_v1_attempt_001`
and `go2_short_pulse_navigation_reactive_noise_2mm_native_layout00_4800_v1_attempt_001`.
Both physical round trips, actual treatments, terminal/timing analyses and the
complete first-maze comparison are saved; the two-maze reactive comparison is
also complete. There is no pending raw-depth replay or fit using either case.
All outcome records, models, public RGB/body/command inputs, poses, physics and
other non-depth files remain. Keep the new maze-1 reactive success as a full
reference, the maze-0 JEPA timeout, direct collision and older tracking-failure
comparison inputs. Retire only regular single-link primary/auxiliary depth
leaves enumerated in
`.generated/depth_retirement_completed_short_pulse_maze00_baselines_2026-09-16/`.
Original full sensor replay ends for these two successes. This supersedes
earlier pose-command/reactive full-depth pins only for these exact recordings.
Completed: 13,080 depth leaves retired, reclaiming 4,164,550,656 allocated
bytes. All 63 preserved JSON hashes and 13,161 non-depth identities match.
Available space was 8,990,797,824 bytes while assignment 9 was running.

End the full-depth pin for the diagnosed short-pulse instantaneous-ranking
maze-0 run, September 16:
`go2_short_pulse_navigation_instantaneous_noise_2mm_native_layout00_4800_v1_attempt_001`.
Physical arrival, actual treatment, terminal-selection, forecast and dispatch
analyses are complete. Its budget timeout had no contacts or tracking failure;
maximum registered-position error was 7.100 mm. All 2,521 return-phase latched
request intervals followed stale obstacle observations. No pending depth replay
or fit consumes this recording. Preserve its incomplete outcome and every
non-depth record; retain JEPA, the direct collision and the successful
pose-command comparator in full. Retire only exact inventoried regular
single-link primary/auxiliary depth leaves. Full original sensor replay becomes
unavailable. Inventory:
`.generated/depth_retirement_diagnosed_short_pulse_instantaneous_2026-09-16/`.
This supersedes earlier blanket pins only for this diagnosed recording.
Completed: 9,610 leaves retired, reclaiming 3,024,207,872 allocated bytes.
All 33 preserved JSON hashes and 9,652 non-depth identities match. Available
space afterward was 8,488,873,984 bytes while assignment 7 was running.

End the full-depth pin for the diagnosed short-pulse command-history maze-0
run, September 16:
`go2_short_pulse_navigation_command_history_noise_2mm_native_layout00_4800_v1_attempt_001`.
Its budget timeout, verified outbound arrival, forecast/terminal comparisons,
phase/frontier diagnosis and dispatch-stall attribution are complete. No
contacts or tracking failures occurred (maximum position error 5.035 mm).
No pending raw-depth replay or fit uses this recording. Preserve every outcome
and non-depth record, including RGB, body/gyro, commands, physics, poses and
forecasts. Keep JEPA, the direct collision (including its active perception
replay) and the successful pose-command comparator in full. Retire only
inventoried regular single-link primary/auxiliary depth leaves in this exact
native directory. Original full sensor replay becomes unavailable. Inventory:
`.generated/depth_retirement_diagnosed_short_pulse_command_history_2026-09-16/`.
This supersedes earlier blanket pins only for this diagnosed recording.
Completed: 9,610 depth leaves retired, reclaiming 3,086,675,968 allocated
bytes; all 32 JSON hashes and 9,651 preserved non-depth identities match.
Free space was 8,960,761,856 bytes while assignment 6 persisted its recording.

End the full-depth pin for the completed short-pulse supervised maze-0 run,
September 16: `go2_short_pulse_navigation_supervised_rollout_noise_2mm_native_layout00_4800_v1_attempt_001`.
Its budget timeout, verified outbound arrival, terminal-selection delay,
executed forecast comparisons and unused contact-head readout are saved.
There were no contacts or tracking failures (maximum position error 6.954 mm).
No pending raw-depth replay or fit uses this recording. Preserve its incomplete
mission outcome and every non-depth record, including RGB, body/gyro, commands,
physics, poses and forecasts. Keep the new JEPA comparison, direct collision
and pose-command successful baseline depth in full. Retire only inventoried
regular single-link primary/auxiliary depth leaves in this exact native
directory; original full sensor replay becomes unavailable. Inventory:
`.generated/depth_retirement_diagnosed_short_pulse_supervised_2026-09-16/`.
This supersedes earlier blanket pins only for this diagnosed recording.
Completed: 9,610 depth leaves retired, reclaiming 3,132,776,448 allocated
bytes; all 32 preserved JSON hashes and 9,651 non-depth identities match.
Free space afterward was 6,049,976,320 bytes, before assignment 6.

Retire only unshared depth leaves from 43 completed family/switch training
recordings, September 16. The exact cases and leaves are enumerated in
`.generated/depth_retirement_completed_training_unique_depth_2026-09-16/inventory.json`.
Eligibility requires complete command execution, no physical/acquisition stop,
no measurement failure, passed strict visibility checks and no failed pairs in
the completed local-motion feature derivation. Keep the first eligible full
recording per source/geometry, every failed recording and every shared hard
link. No shared file is unlinked or rewritten.

The current neural fits/readouts use retained RGB, policy/body/command histories
and labels; the local-motion comparator uses its already derived features.
No pending raw-depth replay uses these selected successful cases. Preserve all
features, models, labels, results, audit records, RGB, body/gyro, commands,
physics and other files. This ends full raw-depth retention only for the exact
single-link leaves in that inventory, not for the rest of either training
collection. Mark each affected case as a partial depth retirement: full original
sensor replay is unavailable for it, although current model inputs remain.
Completed: 3,996 leaves from 43 cases retired, reclaiming 3,274,797,056 allocated
bytes. All 3,785 preserved JSON hashes and 8,216 other-file identities match;
shared hard links remain untouched. Free space was 10,000,039,936 bytes while
assignment 4 was still running, before its recording persistence.

End the first-seed JEPA maze-0 full/no-RGB depth reference pin, September 16:
`go2_neural_rgb_transfer_seed_2026091001_full_jepa_noise_2mm_native_layout00_4800_v1_attempt_001`
and `go2_neural_rgb_transfer_seed_2026091001_no_rgb_jepa_noise_2mm_native_layout00_4800_v1_attempt_001`.
Both are completed, physically verified successes with finished paired,
forecast and terminal analyses and no pending depth replay. All non-depth
evidence and current neural inputs remain. The second-seed tracking failure
and its full-input supervised comparator remain the retained raw pair from
that cohort; every new short-pulse navigation recording remains full for
current diagnosis. Only regular single-link native primary/auxiliary depth
leaves are retired. Inventory:
`.generated/depth_retirement_superseded_first_jepa_reference_pair_2026-09-16/`.
Completed: 7,848 leaves retired, 2,488,610,816 allocated bytes reclaimed;
all 62 preserved JSON hashes and 7,924 non-depth identities match. Free space
before short-pulse assignment 4 was 6,733,127,680 bytes.

Retire the completed third-seed JEPA maze-0 full/no-RGB pair's depth,
September 16, ending its earlier coordinate-case full-depth pin:
`go2_neural_rgb_transfer_seed_2026091402_full_jepa_noise_2mm_native_layout00_4800_v1_attempt_001`
and `go2_neural_rgb_transfer_seed_2026091402_no_rgb_jepa_noise_2mm_native_layout00_4800_v1_attempt_001`.
Both round trips and all paired/coordinate/terminal analyses are complete;
current executed-motion readouts use RGB/body/commands/physics, not depth.
There is no pending raw-depth replay for these successes. Keep the first-seed
JEPA pair, the tracking failure/comparator, all new short-pulse incomplete or
failed recordings, every result and all non-depth evidence. Retire only flat
regular single-link `native/primary_depth_NNNN.npz` and
`native/auxiliary_depth_NNNN.npz` files. Exact original sensor replay ends for
this completed pair. Inventory:
`.generated/depth_retirement_completed_coordinate_reference_pair_2026-09-16/`.
Completed: 7,980 depth leaves retired, 2,556,452,864 allocated bytes reclaimed;
all 62 preserved JSON hashes and 8,056 non-depth identities match.

Retire the completed first-seed direct and supervised maze-0 RGB comparison
depth, September 16, ending their earlier full-depth pins. Exact roots are
`go2_neural_rgb_transfer_seed_2026091001_{variant}_{method}_noise_2mm_native_layout00_4800_v1_attempt_001`
for the four combinations of `variant` = `full`, `no_rgb` and `method` =
`direct`, `supervised_rollout`. All four have verified round trips, completed
forecast/paired/terminal analyses and no failure or pending depth replay.
Current prediction readouts use retained RGB/body/command/physics records.
Keep the first-seed JEPA pair, the second-seed tracking failure and comparator,
and the third-seed coordinate case in full. The new JEPA incomplete mission
also remains full. No model, training input, result or non-depth evidence is
removed. Exact original depth replay of these four completed successes ends.
Inventory: `.generated/depth_retirement_superseded_neural_rgb_direct_supervised_references_2026-09-16/`.
Completed: 14,008 leaves retired, 4,343,750,656 allocated bytes reclaimed.
All 124 preserved JSON hashes and 14,160 non-depth identities match. Free
afterward was 7,356,522,496 bytes, before the direct short-pulse mission.

Retire redundant depth from the completed 36-episode short-pulse training
collection, September 16. Keep `pulse_episode_000` (transfer) and
`pulse_episode_003` (train) in full under
`go2_short_pulse_learning_v1_attempt_001`; retire only flat regular
`depth_NNNN.npz` and `native_depth_NNNN.npz` leaves from the other 34 episodes.
Every episode completed without physical/acquisition stops. The matched fits
and transfer/executed-motion analyses are complete; the neural training loader
uses retained RGB, policy histories and observation metadata, not depth.
Preserve all labels, physics, body/gyro, command records, models, results and
non-depth files. No pending depth replay uses these episodes. Historical full
sensor replay becomes unavailable for the 34 depth-retired episodes; future
regeneration is separate from retention of the original recording.
Inventory: `.generated/depth_retirement_completed_short_pulse_learning_2026-09-16/`.
Completed: 1,768 leaves retired, 1,148,162,048 allocated bytes reclaimed.
All 1,705 preserved JSON hashes and 2,754 non-depth file identities match.

Retire the completed mission-coordinate maze-1 pair's depth, September 16:
`go2_mission_coordinate_original_seed_2026091402_full_jepa_noise_2mm_native_layout01_4800_v1_attempt_001`
and `go2_mission_coordinate_consistent_seed_2026091402_full_jepa_noise_2mm_native_layout01_4800_v1_attempt_001`.
Both verified round trips, physical/model/input/coordinate evaluations, paired
comparisons and terminal diagnoses are complete. No current fit or pending
depth replay consumes them. Preserve all non-depth evidence, active training
inputs, selected pilot references and the tracking failure/comparator. Only
the inventoried regular primary/auxiliary depth leaves under these two exact
`native/` directories are retired. Exact historical depth replay is lost.
Inventory: `.generated/depth_retirement_completed_mission_coordinate_layout01_2026-09-16/`.
Completed: 6,400 depth leaves retired, 1,955,631,104 allocated bytes reclaimed;
all 64 preserved JSON hashes verified. Free space immediately afterward was
5,639,331,840 bytes, before the new short-pulse navigation recording.

Retire completed mission-coordinate maze-0 pair depth:

- `go2_mission_coordinate_original_seed_2026091402_no_rgb_jepa_noise_2mm_native_layout00_4800_v1_attempt_001`
- `go2_mission_coordinate_consistent_seed_2026091402_no_rgb_jepa_noise_2mm_native_layout00_4800_v1_attempt_001`

Both physical/model/input/coordinate evaluations, paired comparison, inspected
figures and terminal diagnosis are complete. The corrected run's residual
hold-prediction error is quantified using retained forecasts, observed poses
and matched physics; no pending raw-depth replay or fit uses either recording.
Keep all non-depth evidence, the original pilot's full coordinate-case depth,
the tracking failure/comparator and the active maze-1 follow-up. Retire only
inventoried regular primary/auxiliary depth leaves under these exact `native/`
directories. Exact historical sensor replay becomes unavailable.
Inventory: `.generated/depth_retirement_completed_mission_coordinate_layout00_2026-09-16/`.
Completed: 7,818 leaves retired, reclaiming 2,412,302,336 allocated bytes.
All 64 preserved JSON hashes match; both roots record replay unavailability.
Free afterward: 6,551,023,616 bytes, as assignment 3 finished recording.

Retire the four completed seed-2026091402 supervised-rollout success depths:

- `go2_neural_rgb_transfer_seed_2026091402_full_supervised_rollout_noise_2mm_native_layout00_4800_v1_attempt_001`
- `go2_neural_rgb_transfer_seed_2026091402_no_rgb_supervised_rollout_noise_2mm_native_layout00_4800_v1_attempt_001`
- `go2_neural_rgb_transfer_seed_2026091402_full_supervised_rollout_noise_2mm_native_layout01_4800_v1_attempt_001`
- `go2_neural_rgb_transfer_seed_2026091402_no_rgb_supervised_rollout_noise_2mm_native_layout01_4800_v1_attempt_001`

The 36-run pilot is complete. All four physical and model/input evaluations,
both paired comparisons, inspected figures, terminal diagnostics and complete
pilot/third-seed aggregates are saved. Maze-0 exploration-command diagnosis
is also retained. No pending raw-depth replay or fit uses these successes.
Preserve all non-depth evidence, selected first-seed depth, the third-seed
JEPA maze-0 coordinate case and second-seed tracking failure/comparator.
Retire only inventoried regular primary/auxiliary depth leaves under the four
exact `native/` directories. Exact historical sensor replay becomes unavailable.
Inventory: `.generated/depth_retirement_completed_neural_rgb_seed1402_supervised_2026-09-16/`.
Completed: 13,680 leaves retired, reclaiming 4,186,632,192 allocated bytes.
All 124 preserved JSON hashes match; all four roots record replay loss.
Free afterward: 8,393,068,544 bytes. No pilot native owner remains active.

Retire completed seed-2026091402 direct-prediction maze-1 success depth:

- `go2_neural_rgb_transfer_seed_2026091402_full_direct_noise_2mm_native_layout01_4800_v1_attempt_001`
- `go2_neural_rgb_transfer_seed_2026091402_no_rgb_direct_noise_2mm_native_layout01_4800_v1_attempt_001`

Both physical and actual-input evaluations, paired comparison, inspected
figures and terminal-command/selection diagnoses are complete. No pending
depth replay or correction fit uses these successes. Preserve every non-depth
record, selected first-seed depth, the third-seed JEPA coordinate case, the
second-seed tracking failure/comparator and current supervised-rollout runs.
Retire only inventoried regular primary/auxiliary depth leaves under these
exact `native/` directories; exact historical sensor replay becomes unavailable.
Inventory: `.generated/depth_retirement_completed_neural_rgb_seed1402_direct_layout01_2026-09-16/`.
Completed: 6,326 leaves retired, reclaiming 1,899,053,056 allocated bytes.
All 62 preserved JSON hashes match, and both roots record replay loss. Free
afterward: 6,559,969,280 bytes, before assignment 35.

Retire completed seed-2026091402 direct-prediction maze-0 success depth:

- `go2_neural_rgb_transfer_seed_2026091402_full_direct_noise_2mm_native_layout00_4800_v1_attempt_001`
- `go2_neural_rgb_transfer_seed_2026091402_no_rgb_direct_noise_2mm_native_layout00_4800_v1_attempt_001`

Physical arrivals, actual model/input checks, paired comparison, inspected
figures and phase/terminal/stopping-projection diagnosis are complete. The
observed action-selection differences remain in forecasts, maps' clearance
readouts, commands and physics; no pending raw-depth replay or fit uses either
recording. Preserve all non-depth evidence and selected first-seed depth,
the third-seed JEPA maze-0 coordinate case and the second-seed tracking
failure/comparator. Exact historical sensor replay becomes unavailable.
Retire only inventoried regular primary/auxiliary depth leaves in these two
exact `native/` directories. Inventory:
`.generated/depth_retirement_completed_neural_rgb_seed1402_direct_layout00_2026-09-16/`.
Completed: 9,772 leaves retired, reclaiming 3,101,229,056 allocated bytes.
All 62 preserved JSON hashes match. Both roots record replay unavailability;
free afterward was 8,758,878,208 bytes, before assignment 32.

Retire completed seed-2026091402 JEPA maze-1 success depth:

- `go2_neural_rgb_transfer_seed_2026091402_full_jepa_noise_2mm_native_layout01_4800_v1_attempt_001`
- `go2_neural_rgb_transfer_seed_2026091402_no_rgb_jepa_noise_2mm_native_layout01_4800_v1_attempt_001`

Physical arrivals, actual model/input checks, paired comparison, inspected
figures and terminal-command/selection diagnosis are complete. The turn/arrival
interaction is preserved in forecasts, decisions, observations and physics;
neither root supplies a pending raw-depth replay or fit. Keep all non-depth
evidence, selected first-seed depth, the third-seed maze-0 coordinate case and
the second-seed tracking failure/comparator. Exact historical depth replay of
these two completed successes becomes unavailable. Retire only inventoried
regular primary/auxiliary depth leaves under each exact `native/` directory.
Inventory: `.generated/depth_retirement_completed_neural_rgb_seed1402_jepa_layout01_2026-09-16/`.
Completed: 7,782 leaves retired, reclaiming 2,384,027,648 allocated bytes. All
62 preserved JSON hashes match; both roots record replay unavailability. Free
afterward: 9,343,295,488 bytes while assignment 29 finished recording.

Retire flat depth for the superseded diagnosed stationary failure
`go2_all_phase_adapter_maze02_matched_native_v1_attempt_001/all_phase_no_rgb_supervised_rollout_residual_maze_02`.
Its completed raw audit/readout and September 11 result record 2,922 holds,
68 left turns, ten right turns, no translation, no arrivals/crossings and zero
contacts across 3,014 observations. Both strict camera checks and raw
sensor/model/command replay passed. This old paused-physics adapter is superseded
by subsequent scoring/recovery and continuous native comparisons; no pending
depth replay or current fit uses it. Its full decision stream, all results,
RGB/body/gyro/poses, physics, contact and audit records remain. Exact raw-depth
replay becomes unavailable. Other failed cases, including the full-direct
visibility failure, are outside this retirement scope. Only the exact regular
`depth_NNNN.npz`, `native_depth_NNNN.npz`, `auxiliary_depth_NNNN.npz` leaves are
eligible. Inventory:
`.generated/depth_retirement_old_adapter_no_rgb_supervised_failure_2026-09-16/`.
Completed: 9,042 leaves retired, reclaiming 6,122,881,024 allocated bytes. All
3,070 preserved JSON hashes and 9,113 non-depth file identities were unchanged
before updating the parent retention marker. The prior marker is saved in the
inventory, the new case marker records replay loss, and the other cases remain
untouched. Free afterward: 10,043,641,856 bytes, before assignment 28.

Retire completed seed-2026091401 supervised-rollout maze-1 success depth:

- `go2_neural_rgb_transfer_seed_2026091401_full_supervised_rollout_noise_2mm_native_layout01_4800_v1_attempt_001`
- `go2_neural_rgb_transfer_seed_2026091401_no_rgb_supervised_rollout_noise_2mm_native_layout01_4800_v1_attempt_001`

Physical arrivals, model/input checks, paired comparison, inspected trajectory
figures, terminal-approach analysis and the complete second-seed aggregate are
saved. Neither recording supplies a pending depth replay or model/correction
fit. Preserve all non-depth evidence, selected first-seed depth references,
the second-seed maze-0 tracking failure and its full-input comparison, and active
third-seed recordings. Exact historical depth replay of these two completed
successes becomes unavailable. Retire only inventoried regular
`native/primary_depth_NNNN.npz` and `native/auxiliary_depth_NNNN.npz` files.
Inventory: `.generated/depth_retirement_completed_neural_rgb_seed1401_supervised_layout01_2026-09-16/`.
Completed: 6,104 leaves retired, reclaiming 1,815,883,776 allocated bytes. All
62 preserved JSON hashes match. Both roots record retirement; other recordings
were untouched. Free afterward: 8,286,248,960 bytes during assignment 25.

End the older diagnosed failure's full-depth pin for the exact case
`go2_all_phase_adapter_maze02_matched_native_v1_attempt_001/all_phase_full_supervised_rollout_residual_maze_02`.
The completed result and raw audit are documented in
`docs/go2_all_phase_adapter_full_supervised_maze02_result_2026-09-10.md`:
2,931 hold and 69 turn selections, no translation, no arrivals, zero contacts,
and exhaustion of the 3,000-tick budget. The contact-horizon scoring diagnosis
is complete and subsequent native scoring/controller experiments supersede
this configuration. No pending depth replay or current model/correction fit
uses this case. It remains a failed mission in the original six-case cohort.

Retire only its flat regular `depth_NNNN.npz`, `native_depth_NNNN.npz` and
`auxiliary_depth_NNNN.npz` leaves. Preserve the entire parent cohort's results,
raw-audit reports, worker terminals and every non-depth case artifact, including
RGB, body/gyro/history, commands, decision stream, physics, contact records,
raster descriptions and identities. The other five cases are outside this
retirement scope. Preserve current/unresolved failures and selected current
depth references. Exact historical raw depth replay of this diagnosed case
becomes unavailable; historical audit results remain evidence of the completed
audit, not a claim that it can still be rerun. Inventory:
`.generated/depth_retirement_diagnosed_old_adapter_supervised_failure_2026-09-15/`.
Completed: 9,042 depth leaves retired, reclaiming 5,750,251,520 allocated bytes.
All target leaves had link count one. Every preserved parent/case JSON hash
and every inventoried non-depth file identity remained unchanged. Both the
case and parent now record the partial retirement; the other five cases remain
untouched. Free afterward: 9,663,979,520 bytes, before the second-seed no-RGB
supervised-rollout maze-0 recording.

Retire the completed seed-2026091401 direct-prediction maze-1 pair's depth:

- `go2_neural_rgb_transfer_seed_2026091401_full_direct_noise_2mm_native_layout01_4800_v1_attempt_001`
- `go2_neural_rgb_transfer_seed_2026091401_no_rgb_direct_noise_2mm_native_layout01_4800_v1_attempt_001`

Both physical/treatment evaluations, paired comparison, inspected figures and
terminal-approach diagnostics are complete; no pending depth replay or fit uses
either success. Preserve every outcome, timing record and all other non-depth
evidence, the selected first-seed depth population, active recordings and
unresolved failures. Exact historical depth replay of this completed pair
becomes unavailable. Inventory:
`.generated/depth_retirement_completed_neural_rgb_seed1401_direct_layout01_2026-09-15/`.
Completed: 6,218 depth leaves retired, reclaiming 1,861,869,568 allocated bytes;
all preserved JSON hashes match. Free afterward: 5,325,676,544 bytes, before
the second-seed full-input supervised-rollout maze-0 recording.

Retire the completed seed-2026091401 direct-prediction maze-0 pair's depth:

- `go2_neural_rgb_transfer_seed_2026091401_full_direct_noise_2mm_native_layout00_4800_v1_attempt_001`
- `go2_neural_rgb_transfer_seed_2026091401_no_rgb_direct_noise_2mm_native_layout00_4800_v1_attempt_001`

Both physical/treatment evaluations, paired comparison, inspected figures and
terminal/timing diagnostics are complete; neither supplies a pending depth
replay or fit. Preserve all non-depth scientific evidence, including every late
plan and simulator-lag result. Keep the first-seed direct maze-0 pair in full
as the selected depth reference, with the other selected first-seed pairs,
active recordings and unresolved failures. Exact historical depth replay of
this completed pair becomes unavailable. Inventory:
`.generated/depth_retirement_completed_neural_rgb_seed1401_direct_layout00_2026-09-15/`.
Completed: 9,850 depth leaves retired, reclaiming 3,105,484,800 allocated bytes;
all preserved JSON hashes match. Free afterward: 5,801,627,648 bytes, before
the second-seed no-RGB direct maze-1 recording.

Retire the completed seed-2026091401 JEPA maze-1 pair's depth after both
physical/treatment evaluations, paired comparison, inspected figures,
terminal-approach and phase-path/command diagnostics:

- `go2_neural_rgb_transfer_seed_2026091401_full_jepa_noise_2mm_native_layout01_4800_v1_attempt_001`
- `go2_neural_rgb_transfer_seed_2026091401_no_rgb_jepa_noise_2mm_native_layout01_4800_v1_attempt_001`

No pending depth replay or fit uses either success. Retain every result and
non-depth record, including the no-RGB run's long return and pure-turn commands;
the timing difference is descriptive and does not establish a causal RGB
forecast-accuracy benefit. Keep the selected first-seed maze-0 depth population,
active recordings and unresolved failures in full. Exact historical depth
replay of this completed pair becomes unavailable. Inventory:
`.generated/depth_retirement_completed_neural_rgb_seed1401_jepa_layout01_2026-09-15/`.
Completed: 8,566 depth leaves retired, reclaiming 2,800,566,272 allocated bytes;
all preserved JSON hashes match. Free afterward: 6,573,805,568 bytes, before
the second-seed full-input direct maze-0 recording. Return selection and
preferred-action clearance diagnostics also remain in the comparison root.

End the older completed fitted-motion maze-0 success's temporary depth pin:
`go2_shared_recovery_transfer_pose_command_noise_2mm_native_layout00_4800_v1_attempt_001`.
Both physical arrivals, actual control assignment, the complete five-controller
comparison, inspected figures and recorded stage/forecast summaries are finished.
No pending depth replay or model/correction fit uses this recording. Its retained
successful outcome remains evidence against claiming learned-model superiority;
preserve all non-depth records and the complete comparison. Retire only depth
leaves. Keep current neural-RGB recordings and selected first-seed references,
the reactive signed-recovery reference, wall-clock references and unresolved
failures in full. Exact historical depth replay of this older success becomes
unavailable. Inventory:
`.generated/depth_retirement_completed_shared_fitted_reference_2026-09-15/`.
Completed: 4,708 depth leaves retired, reclaiming 1,424,834,560 allocated bytes;
all preserved JSON hashes match. Free afterward: 5,048,111,104 bytes, before
the second-seed full-input JEPA maze-1 recording.

Retire the completed seed-2026091401 JEPA maze-0 pair's depth after both
physical/treatment evaluations, paired comparison, inspected figures and
terminal/timing diagnostics:

- `go2_neural_rgb_transfer_seed_2026091401_full_jepa_noise_2mm_native_layout00_4800_v1_attempt_001`
- `go2_neural_rgb_transfer_seed_2026091401_no_rgb_jepa_noise_2mm_native_layout00_4800_v1_attempt_001`

Neither supplies a pending depth replay or fit. Preserve the full seed-2026091001
JEPA maze-0 pair as the selected depth reference and all non-depth evidence from
both new successes, including the RGB run's 34 late plans and 4.35-s peak lag.
Timing cause is not established; retained stage/command/forecast records support
the descriptive comparison, and no real-time or isolated RGB benefit is claimed.
Preserve every outcome, active recording and unresolved failure. Exact depth
replay of these two completed successes becomes unavailable. Inventory:
`.generated/depth_retirement_completed_neural_rgb_seed1401_jepa_layout00_2026-09-15/`.
Completed: 10,310 depth leaves retired, reclaiming 3,266,469,888 allocated bytes;
all preserved JSON hashes match. Free afterward: 5,810,102,272 bytes, before
the second-seed no-RGB JEPA maze-1 recording.

Retire the completed seed-2026091001 supervised-rollout maze-1 pair's depth:

- `go2_neural_rgb_transfer_seed_2026091001_full_supervised_rollout_noise_2mm_native_layout01_4800_v1_attempt_001`
- `go2_neural_rgb_transfer_seed_2026091001_no_rgb_supervised_rollout_noise_2mm_native_layout01_4800_v1_attempt_001`

Both physical/treatment evaluations, paired comparison, inspected figures,
terminal-approach diagnosis and complete first-seed aggregation are finished.
Neither recording supplies a pending depth replay or fit. Keep the complete
maze-0 full/no-RGB pair for each of the three methods as the selected first-seed
depth population. Preserve all outcomes, trajectories, commands, RGB/body/gyro,
physics, forecasts and diagnoses for all 12 assignments, plus active recordings
and unresolved failures. Later cohort aggregation uses these non-depth records.
Exact historical depth replay of this maze-1 pair becomes unavailable. Inventory:
`.generated/depth_retirement_completed_neural_rgb_supervised_layout01_2026-09-15/`.
Completed: 6,002 depth leaves retired, reclaiming 1,808,887,808 allocated bytes;
all preserved JSON hashes match. Free afterward: 6,609,813,504 bytes during
the second-seed full-input JEPA maze-0 mission, before its recording finalized.

End the completed rollout-selection-off maze-1 success's temporary full-depth
pin and retire only its depth leaves:
`go2_rollout_selection_off_seed_2026091001_full_supervised_rollout_noise_2mm_native_layout01_4800_v1_attempt_001`.
Physical and actual-selection checks, the three-controller comparison, inspected
figures, terminal-approach diagnosis and subsequent four-controller comparison
are complete. Its 172-s near-home approach is explained by the retained command
and selection evidence; no pending sensor replay or fit uses this recording.
Keep its complete outcome, commands, RGB/body/gyro/poses, physics and diagnoses.
Keep the failed maze-0 counterpart in full, along with current neural-RGB
comparison references, unresolved failures and wall-clock references. Exact
historical depth replay of this success becomes unavailable. Inventory:
`.generated/depth_retirement_completed_rollout_off_success_2026-09-15/`.
Completed: 7,012 depth leaves retired, reclaiming 2,118,041,600 allocated bytes;
all preserved JSON hashes match. Free afterward: 7,073,542,144 bytes during
the no-RGB supervised-rollout maze-1 mission, before its recording finalized.

The fresh full/no-RGB supervised-rollout maze-0 pair now replaces this older
full-depth success reference:
`go2_shared_recovery_transfer_seed_2026091001_full_supervised_rollout_noise_2mm_native_layout00_4800_v1_attempt_001`.
Its physical evaluation, five-controller comparison and saved motion/heading
forecast readouts are complete, as are the subsequent controller-ablation
comparisons that used it. No pending sensor replay or model/correction fit
depends on it. End the older full-depth success pin and retire only depth
leaves. Keep both fresh supervised-rollout recordings, selected JEPA/direct
maze-0 pairs, active recordings, unresolved failures and remaining current
control references in full. Preserve all non-depth evidence. Exact historical
depth replay of this older success becomes unavailable. Inventory:
`.generated/depth_retirement_superseded_shared_supervised_reference_2026-09-15/`.
Completed: 4,908 depth leaves retired, reclaiming 1,450,397,696 allocated bytes;
all preserved JSON hashes match. Free afterward: 4,962,156,544 bytes, before
the no-RGB supervised-rollout maze-1 recording.

Retire the completed, diagnosed seed-2026091001 direct-prediction maze-1 pair's
depth after its physical/treatment evaluations, paired comparison, inspected
figures and terminal command diagnosis:

- `go2_neural_rgb_transfer_seed_2026091001_full_direct_noise_2mm_native_layout01_4800_v1_attempt_001`
- `go2_neural_rgb_transfer_seed_2026091001_no_rgb_direct_noise_2mm_native_layout01_4800_v1_attempt_001`

Neither supplies a pending sensor replay or fit. The slower full-input success
and all its terminal pulse/hold evidence remain in the retained non-depth
records; this is not removal of an unfavorable outcome. Preserve the direct
maze-0 pair in full as the selected depth comparison, alongside the JEPA maze-0
pair, active recordings, unresolved failures and current controller references.
Exact historical depth replay of this maze-1 pair becomes unavailable. The
cohort analysis uses retained non-depth records. Inventory:
`.generated/depth_retirement_completed_neural_rgb_direct_layout01_2026-09-15/`.
Completed: 7,158 depth leaves retired, reclaiming 2,194,759,680 allocated bytes;
all preserved JSON hashes match. Free afterward: 6,177,820,672 bytes, before
the full-input supervised-rollout maze-0 recording.

For the ongoing neural-RGB pilot, end the temporary full-depth pins for the
completed, diagnosed seed-2026091001 JEPA maze-1 pair:

- `go2_neural_rgb_transfer_seed_2026091001_full_jepa_noise_2mm_native_layout01_4800_v1_attempt_001`
- `go2_neural_rgb_transfer_seed_2026091001_no_rgb_jepa_noise_2mm_native_layout01_4800_v1_attempt_001`

Both physical round trips, input/correction assignments, paired comparison,
inspected figures and terminal-approach diagnoses are complete. Neither supplies
a pending sensor replay or fitting input. Preserve both complete maze-0 JEPA
recordings as the selected full-depth input comparison, all current direct
recordings and unresolved failures. Preserve all maze-1 results, commands,
RGB/body/gyro/poses, physics and diagnoses for the complete 36-assignment
analysis. That analysis uses retained non-depth records. Retire only the two
maze-1 recordings' depth leaves; exact historical depth replay becomes
unavailable. This follows the pilot's declared retirement after paired
comparisons. Inventory:
`.generated/depth_retirement_completed_neural_rgb_jepa_layout01_2026-09-15/`.
Completed: 7,210 depth leaves retired, reclaiming 2,227,023,872 allocated bytes;
all preserved JSON hashes match. Free afterward: 6,714,638,336 bytes while the
no-RGB direct maze-1 mission was active, before its recording was finalized.

The completed fresh-maze full/no-RGB direct-prediction pair replaces the older
full-depth direct success reference:
`go2_shared_recovery_transfer_seed_2026091001_full_direct_noise_2mm_native_layout00_4800_v1_attempt_001`.
Its physical evaluation, five-controller comparison and motion/timing diagnoses
are complete; it supplies no pending sensor replay or model/correction fit.
End this older success pin and retire only its depth leaves. Keep both fresh
direct recordings, all four fresh JEPA recordings, current controller references
and unresolved failures in full. Preserve every non-depth record. Exact depth
replay of this older success becomes unavailable. Inventory:
`.generated/depth_retirement_superseded_shared_direct_reference_2026-09-15/`.
Completed: 4,718 depth leaves retired, reclaiming 1,415,491,584 allocated bytes;
all preserved JSON hashes match. Free afterward: 4,495,900,672 bytes, before
the maze-1 no-RGB direct recording.

The completed fresh-maze full/no-RGB JEPA pairs now replace the older full-depth
JEPA success reference:
`go2_shared_recovery_transfer_seed_2026091001_full_jepa_noise_2mm_native_layout00_4800_v1_attempt_001`.
Its physical evaluation, five-controller comparison and motion/timing diagnoses
are complete; no pending sensor replay or model/correction fit uses it. End
this older success pin and retire only its depth leaves. Keep all four fresh
JEPA input-comparison recordings in full, the other current controller
references, unresolved failures and every non-depth record. Exact historical
depth replay of the older success becomes unavailable. Inventory:
`.generated/depth_retirement_superseded_shared_jepa_reference_2026-09-15/`.
Completed: 6,018 depth leaves retired, reclaiming 1,801,252,864 allocated bytes;
all preserved JSON hashes match. Free afterward: 5,831,958,528 bytes, before
the next direct-prediction recording.

During the fresh neural-RGB pilot, end the older full-depth pins for these two
diagnosed, superseded current-plane training-comparison failures:

- `go2_current_plane_matched_training_jepa_noise_2mm_native_layout00_4800_v1_attempt_001`
- `go2_current_plane_matched_training_direct_noise_2mm_native_layout01_4800_v1_attempt_001`

The JEPA run's terminal sample exceeded the unchanged full-3D speed limit
(0.301271 versus 0.3 m/s), with no contact or domain exit. Its physical trace,
command history and corrected speed diagnostic remain. The direct run's last
330 plans selected hold before the map filter despite all six candidates
passing clearance; its retained forecasts and score diagnosis identify the
contact-penalized utility mechanism. Neither has a pending sensor replay or
supplies training/correction inputs. Later shared-perception and contact-score
comparisons supersede their controller configuration. Retain every original
outcome, failure, RGB/body/gyro/pose record, command, physical trace and diagnosis;
both remain failed missions in the complete older cohort. Keep the older
unresolved JEPA layout-3 perception recording, current failure/debugging inputs
and all current comparison recordings in full. This is diagnosed-failure depth
retirement under the existing September-14 policy, not removal of failures.
Exact historical sensor replay becomes unavailable for these two roots.
Inventory: `.generated/depth_retirement_diagnosed_old_training_failures_2026-09-15/`.
Completed: 15,186 depth leaves retired, reclaiming 4,787,949,568 allocated bytes;
all preserved JSON hashes match. Free afterward: 8,591,929,344 bytes, with the
first fresh neural-RGB recording complete on disk.

After the completed four-controller predictive/current-feedback comparison,
retire the remaining two depth recordings from the completed stopping and
instantaneous-scoring success experiments:

- `go2_shadow_stopping_projection_seed_2026091001_full_supervised_rollout_noise_2mm_native_layout00_4800_v1_attempt_001`
- `go2_instantaneous_waypoint_score_seed_2026091001_full_supervised_rollout_noise_2mm_native_layout00_4800_v1_attempt_001`

Their physical evaluations, action/dispatch diagnoses and inspected comparison
plots are complete; neither is a training/fit or pending depth-replay input.
This ends their temporary full-depth pins. Retain the full five-controller
shared-recovery layout-0 references, the nominal rollout-off layout-1 success,
the unique wall-clock references and every failure, including both new reserved
terminal failures. Preserve all non-depth records. Exact historical depth
replay of these two successes becomes unavailable. Inventory:
`.generated/depth_retirement_completed_predictive_ablation_successes_2026-09-15/`.
Completed: 10,758 depth leaves retired, reclaiming 3,276,480,512 allocated bytes;
all preserved JSON hashes match. Free afterward: 5,127,708,672 bytes.

Retire three superseded stable-reference learned-success depth recordings:

- `go2_arrival_entry_priority_stable_reference_jit_floor_cached_fine_goal_lzma_pulse_learned_round_trip_native_layout04_4800_v1_attempt_001`
- `go2_arrival_entry_priority_stable_reference_jit_floor_cached_fine_goal_lzma_pulse_learned_round_trip_native_layout06_4800_v1_attempt_001`
- `go2_fresh_stable_reference_learned_round_trip_native_layout02_4800_v1_attempt_001`

Their physical evaluations, matched comparisons and diagnoses are complete.
They are not training/fit or pending depth-replay inputs. Current full controller
references supersede these old success pins. Preserve every failed counterpart,
all current references and all non-depth records. Exact historical depth replay
becomes unavailable. Inventory:
`.generated/depth_retirement_superseded_stable_reference_successes_2026-09-15/`.
Completed: 18,384 depth leaves retired, reclaiming 5,551,943,680 allocated bytes;
all preserved JSON hashes match. Free afterward: 8,214,523,904 bytes.

Retire the completed instantaneous-scoring layout-1 success depth:
`go2_instantaneous_waypoint_score_seed_2026091001_full_supervised_rollout_noise_2mm_native_layout01_4800_v1_attempt_001`.
Its physical/treatment evaluations, command-reversal diagnosis and inspected
comparison plots are complete. The forthcoming three-way comparison consumes
retained non-depth data. It is not a training/fit or pending depth-replay input.
This ends its temporary full-depth pin; instantaneous layout 0 remains full,
as do current full controller references and every failure. Preserve all
non-depth evidence. Exact historical depth replay becomes unavailable.
Inventory: `.generated/depth_retirement_completed_instantaneous_reference_2026-09-15/`.
Completed: 3,754 depth leaves retired, reclaiming 1,111,982,080 allocated bytes;
all preserved JSON hashes match. Free afterward: 5,270,454,272 bytes.

Before the learned-rollout-selection-off comparison, retire the remaining two
older heading-success references:

- `go2_heading_release_fresh_learned_round_trip_native_layout02_4800_v1_attempt_001`
- `go2_hold_relative_heading_recovery_native_layout00_4800_v1_attempt_001`

Both have independently verified round trips and zero contacts, completed
matched comparisons and diagnoses; neither exercised its added recovery rule.
They are not training/fit or pending depth-replay inputs. This ends their older
full-depth pins. Keep current full shared-recovery layout-0 controller references,
the first stopping-off recording, both instantaneous-score recordings, every
failure and all non-depth evidence. Exact historical depth replay becomes
unavailable. Inventory:
`.generated/depth_retirement_superseded_heading_references_2026-09-15/`.
Completed: 12,912 depth leaves retired, reclaiming 3,915,669,504 allocated bytes;
all preserved JSON hashes match. Free afterward: 7,753,383,936 bytes.

Retire the shared-recovery supervised layout-1 reference depth after its complete
five-controller and stopping-enforcement comparisons and dispatch diagnoses:
`go2_shared_recovery_transfer_seed_2026091001_full_supervised_rollout_noise_2mm_native_layout01_4800_v1_attempt_001`.
This ends its temporary depth pin for the same-controller planning comparison.
The instantaneous-scoring comparison consumes its retained results and commands,
not depth; it is not a training/fit or pending sensor-replay input. Preserve the
full layout-0 controller references, first full stopping-off reference, current
instantaneous recording, every failure and all non-depth evidence. Exact
historical depth replay becomes unavailable. Inventory:
`.generated/depth_retirement_completed_shared_supervised_reference_2026-09-15/`.
Completed: 3,330 depth leaves retired, reclaiming 973,193,216 allocated bytes;
all preserved JSON hashes match. Free afterward: 5,236,682,752 bytes.

Before the instantaneous-utility experiment, retire successful depth from:

- `go2_stopping_projection_transfer_learned_noise_2mm_native_layout02_4800_v1_attempt_001`
- `go2_shadow_stopping_projection_seed_2026091001_full_supervised_rollout_noise_2mm_native_layout01_4800_v1_attempt_001`

The older transfer's long-turn comparison is complete (148.84 s turn-only,
21.240 m path versus reactive's 43.14 s and 15.323 m); no depth replay remains
planned. This ends its long-turn depth pin. The shadow layout-1 comparison,
physical evaluation, dispatch-veto/recovery diagnosis and inspected plots are
also complete; shadow layout 0 remains the full ablation reference. This ends
the temporary pin on shadow layout 1. Neither is a training/fit input. Preserve
all results, non-depth data, current full learned/fitted/reactive references and
every failed recording. Exact historical depth replay becomes unavailable.
Inventory: `.generated/depth_retirement_completed_stopping_successes_2026-09-15/`.
Completed: 8,622 depth leaves retired, reclaiming 2,465,566,720 allocated bytes;
all preserved JSON hashes match. Free afterward: 6,176,104,448 bytes.

Retire depth from the completed fitted-motion batched-tracker followup
`go2_batched_consensus_pose_command_noise_2mm_native_layout02_4800_v1_attempt_001`.
Its physical, forecast and tracking-cost diagnosis is complete; current full
shared-recovery fitted layout 0 supersedes this success reference. It is not a
training/fit or pending depth-replay input. Preserve its non-depth records and
the original failed recording and exact replay in full. This ends only the
successful followup's full-depth pin; exact historical depth replay becomes
unavailable. Inventory:
`.generated/depth_retirement_superseded_batched_fitted_reference_2026-09-15/`.
Completed: 3,322 depth leaves retired, reclaiming 956,006,400 allocated bytes;
preserved JSON hashes match. Artifact storage afterward: 4,876,210,176 bytes.

Retire depth from the completed combined-tracker supervised and signed-view
reactive exposed successes, now superseded by the full shared-recovery layout-0
references and the full layout-1 supervised reference:

- `go2_combined_tracking_floor_recovery_seed_2026091402_full_supervised_rollout_noise_2mm_native_layout01_4800_v1_attempt_001`
- `go2_signed_veto_view_recovery_reactive_noise_2mm_native_layout01_4800_v1_attempt_001`

Both verified round trips with zero contacts and no runtime failure; their
diagnoses and comparisons are complete. Neither is a training/fit or pending
depth-replay input. This ends their earlier full-depth pins. Preserve every
non-depth record and every failure. Exact historical depth replay becomes
unavailable. Inventory:
`.generated/depth_retirement_superseded_shared_mechanism_references_2026-09-15/`.
Completed: 6,024 depth leaves retired, reclaiming 1,816,358,912 allocated bytes;
all preserved JSON hashes match. Artifact storage afterward: 6,093,180,928 bytes.

After the complete ten-assignment shared-recovery comparison and both trajectory
figures, retire redundant successful layout-1 depth for fitted motion, direct
prediction and JEPA:

- `go2_shared_recovery_transfer_pose_command_noise_2mm_native_layout01_4800_v1_attempt_001`
- `go2_shared_recovery_transfer_seed_2026091001_full_direct_noise_2mm_native_layout01_4800_v1_attempt_001`
- `go2_shared_recovery_transfer_seed_2026091001_full_jepa_noise_2mm_native_layout01_4800_v1_attempt_001`

Keep all five layout-0 full controller references, the layout-1 supervised
success for the next same-controller planning comparison, and the complete
layout-1 reactive failure. All three retiring success diagnoses, forecast
evaluations and shared-source comparisons are complete; they are not training,
fit or pending depth-replay inputs. Preserve every result and non-depth record;
exact historical depth replay becomes unavailable. Inventory:
`.generated/depth_retirement_completed_shared_recovery_successes_2026-09-15/`.
Completed: 10,056 depth leaves retired, reclaiming 2,977,984,512 allocated bytes.
Preserved JSON hashes match and no selected depth remains. Free storage
afterward: 4,279,541,760 bytes (just under four GiB).

After all five shared-recovery layout-0 controllers passed physical evaluation
and their complete comparison, retire the older multiseed layout-0 first-success
depth for these four assignments:

- `go2_multiseed_navigation_seed_2026091001_full_jepa_noise_2mm_native_layout00_4800_v1_attempt_001`
- `go2_multiseed_navigation_seed_2026091001_full_direct_noise_2mm_native_layout00_4800_v1_attempt_001`
- `go2_multiseed_navigation_seed_2026091001_full_supervised_rollout_noise_2mm_native_layout00_4800_v1_attempt_001`
- `go2_multiseed_navigation_pose_command_noise_2mm_native_layout00_4800_v1_attempt_001`

The five new shared-recovery layout-0 recordings are now the full controller
references; preserve every one through the second-layout comparison. Their
157 common source identities and shared settings match. The four older
successes are not training/fit or pending sensor-replay inputs; their diagnoses
and comparisons are complete. This supersedes the earlier full-depth pins for
these four roots only. Preserve all their non-depth records, every original
outcome and every failed recording in full. Exact historical depth replay of
these successes becomes unavailable. Inventory:
`.generated/depth_retirement_superseded_multiseed_references_2026-09-15/`.
Completed: 17,816 depth leaves retired, reclaiming 5,602,234,368 allocated bytes;
preserved JSON hashes match and no selected depth remains. Artifact storage
afterward: 10,085,244,928 bytes free, before the second-maze fitted run.

The next storage release, after the active JEPA shared-recovery owner finishes,
retires depth from five fully evaluated older successes:

- `go2_multiseed_navigation_seed_2026091401_full_supervised_rollout_noise_2mm_native_layout01_4800_v1_attempt_001`
- `go2_multiseed_navigation_pose_command_noise_2mm_native_layout01_4800_v1_attempt_001`
- `go2_stopping_projection_transfer_learned_noise_2mm_native_layout00_4800_v1_attempt_001`
- `go2_stopping_projection_transfer_pose_command_noise_2mm_native_layout01_4800_v1_attempt_001`
- `go2_stopping_projection_transfer_reactive_noise_2mm_native_layout01_4800_v1_attempt_001`

All five have verified round trips and no terminal failure; their comparisons
and diagnoses are complete. They are not training/fit or pending replay inputs.
This ends their earlier full-depth pins, including the first-success pins in
the older stopping-projection cohort. Keep full seed-1001 layout-0 multiseed
references for all three neural methods, fitted multiseed layout 0, the current
combined-tracker supervised and signed-reactive references, and every current
shared-recovery transfer recording. These retain full examples of the current
controller mechanisms. Every failed recording remains full. Preserve all five
older successes' results, comparisons, plots and non-depth sensor/physics data;
exact historical depth replay becomes unavailable. The five selected sets
contain 16,330 leaves and 4,894,138,368 allocated bytes by the initial inventory.
Retirement completed after the JEPA owner exited: 16,330 depth leaves and
4,894,138,368 allocated bytes reclaimed. Preserved JSON hashes match and no
selected depth remains. Storage afterward: 9,832,386,560 bytes free. Inventory:
`.generated/depth_retirement_superseded_transfer_references_2026-09-15/`.

Before the shared-recovery fresh-maze comparison, retire depth from the four
completed cached/original-tracker diagnostic successes:

- `go2_cached_floor_moments_seed_2026091402_full_direct_noise_2mm_native_layout01_4800_v1_attempt_001`
- `go2_cached_floor_moments_seed_2026091401_full_jepa_noise_2mm_native_layout01_4800_v1_attempt_001`
- `go2_serial_original_tracker_seed_2026091402_full_direct_noise_2mm_native_layout01_4800_v1_attempt_001`
- `go2_serial_original_tracker_seed_2026091401_full_jepa_noise_2mm_native_layout01_4800_v1_attempt_001`

Their physical evaluations, timing diagnosis and four-way comparison are
complete; neither caching nor the lighter-workload controls demonstrated a
native cache advantage. They are not training/fit inputs or pending depth
replays. This ends their previous full-depth pins. Keep the complete original
failures and their exact estimator replays, original controller references,
the latest combined-tracker supervised success and first signed-reactive
success. Retain every non-depth artifact, comparison and plot. Exact historical
depth replay of these four successes becomes unavailable.
Inventory: `.generated/depth_retirement_completed_tracking_controls_2026-09-15/`.
Completed: 12,574 primary/auxiliary depth leaves retired, reclaiming
3,755,048,960 allocated bytes. All preserved JSON hashes match and no selected
depth remains. Artifact storage has 8,795,389,952 bytes free afterward.

Retire depth from the superseded, independently verified success
`go2_progress_rejoining_learned_round_trip_native_layout03_4800_v1_attempt_001`.
Its gyro-carry, gyro-conditioned and consensus replay diagnoses are complete
and retained. It is not a current sensor-replay, training or correction-fit
input. Current first full learned/fitted/reactive successes remain, together
with every failed mission. This ends the older successful mission's full-depth
pin. All results, replay diagnoses, figures, RGB, poses, body/gyro, commands and
physics remain; exact historical depth replay becomes unavailable.
Inventory: `.generated/depth_retirement_superseded_progress_rejoining_success_2026-09-15/`.
Completed: 9,504 depth leaves retired, 15,667,560,448 allocated bytes reclaimed,
all preserved JSON hashes unchanged. No data was moved to the Steam drive.

For the reactive wall-clock mission, retire depth only from the completed
60-second GC-diagnosis prefix
`go2_async_camera_wall_deadlines_layout01_600_v1_attempt_003`.
Its timing/GC diagnosis is complete and does not require depth replay. Keep
its source witnesses, all timings, physics, RGB, poses and other JSON; keep the
newer GC-deferred prefix and every full wall mission/failure. This ends only
that prefix's earlier full-depth pin. It had no arrivals because it was a
bounded timing prefix, not a full navigation assignment; its owner exited 0.
Inventory: `.generated/depth_retirement_completed_gc_timing_prefix_2026-09-15/`.
Completed: 1,202 depth leaves, 351,580,160 allocated bytes reclaimed, preserved
JSON hashes unchanged. Exact historical sensor replay is unavailable.

For the full wall-clock physics-cost follow-up, retire redundant analyzed
success depth from `go2_routing_memory_persistent_native_layout01_4800_v1_attempt_001`.
This ends its earlier depth pin; persistent layout 0 remains the full reference
and all four reduced-memory failures remain full. This run is not a pending
replay or training/fit input. All non-depth artifacts and their recorded hashes
remain unchanged. Exact historical sensor replay becomes unavailable.
Inventory: `.generated/depth_retirement_redundant_memory_reference_2026-09-15/`.
Completed: 7,050 depth leaves, 2,153,340,928 allocated bytes reclaimed;
5,573,283,840 bytes free afterward. Preserve all current wall-clock recordings.

Before the final fitted-motion pair, retire depth from two superseded successful
probes whose physical/forecast evaluations and paired diagnoses are complete:

- `go2_planned_stopping_projection_learned_noise_2mm_native_layout03_4800_v1_attempt_001`
- `go2_committed_camera_view_pose_command_noise_2mm_native_layout01_4800_v1_attempt_001`

The current transfer retains full first-success references for learned, fitted
and reactive control using these mechanisms. Neither older probe is a training,
fit or pending sensor-replay input. This explicitly ends those two earlier
full-depth pins. Keep all results, non-depth data, all failures and active
comparison recordings. Inventory:
`.generated/depth_retirement_superseded_view_stopping_successes_2026-09-15/`.

Completed: 9,144 depth leaves retired, reclaiming 2,662,809,600 allocated bytes
(2.480 GiB). Preserved JSON hashes match and both roots carry markers. Free
artifact storage afterward was 8,173,080,576 bytes, before the final pair.

First two stopping-projection transfer layouts are fully compared and diagnosed.
Under the existing per-layout rule, retire depth from fitted-motion layout 0
and learned layout 1:

- `go2_stopping_projection_transfer_pose_command_noise_2mm_native_layout00_4800_v1_attempt_001`
- `go2_stopping_projection_transfer_learned_noise_2mm_native_layout01_4800_v1_attempt_001`

Keep the first full success of each controller (learned 0, fitted motion 1,
reactive 1), the complete reactive-0 failure and all running/new layouts.
Both retiring runs have physical evaluations, actual-treatment checks, all
applicable forecast evaluations, paired behavior/view analyses and checked
trajectory figures. They are not training/fit or pending sensor-replay inputs.
All results and non-depth records remain. Inventory:
`.generated/depth_retirement_completed_transfer_first_layouts_2026-09-15/`.

Completed: 7,416 depth leaves retired, reclaiming 2,254,450,688 allocated bytes
(2.100 GiB). All preserved top-level JSON hashes match; both roots have markers
and no selected depth remains. Free space immediately afterward was
9,589,841,920 bytes while the fourth pair was recording.

Latest review during the fixed three-controller transfer: retire depth from
two completed, analyzed frontier-policy successes, superseded by the current
committed-view and stopping-projection references:

- `go2_arrival_conditioned_frontier_contact_learned_noise_2mm_native_layout00_4800_v1_attempt_001`
- `go2_arrival_conditioned_frontier_contact_disabled_noise_2mm_native_layout00_4800_v1_attempt_003`

Both passed independent round-trip checks; their original comparison and
view-start-repair comparison are complete. Neither is a training/correction-fit
input nor a pending sensor-replay input. This ends the old full-depth pin for
these two successes only. Keep all failures, current transfer recordings,
current full references, and all results, RGB, poses, commands, gyro/body and
physics. Exact historical sensor replay becomes unavailable. Routine authorized
inventory: `.generated/depth_retirement_superseded_frontier_successes_2026-09-15/`.

Completed: 14,340 exact depth leaves retired, reclaiming 4,212,301,824 allocated
bytes (3.923 GiB). Preserved top-level JSON hashes match and both roots carry
retirement markers. Free artifact storage afterward was 12,214,534,144 bytes.

Latest review for the fixed three-controller fresh-maze comparison: release
depth from six completed, analyzed, superseded round-trip successes:

- `go2_post_training_transfer_supervised_rollout_noise_2mm_native_layout00_4800_v1_attempt_001`
- `go2_contact_score_ablation_pose_command_xy_learned_noise_2mm_native_layout00_4800_v1_attempt_001`
- `go2_yaw_source_ablation_command_noise_2mm_native_layout00_4800_v1_attempt_001`
- `go2_combined_perception_motion_learned_noise_2mm_native_layout01_4800_v1_attempt_001`
- `go2_combined_perception_motion_pose_command_noise_2mm_native_layout00_4800_v1_attempt_001`
- `go2_combined_perception_motion_pose_command_noise_2mm_native_layout03_4800_v1_attempt_001`

These are neither training/correction-fit inputs nor pending sensor-replay
inputs. Preserve every study's outcomes, summaries, forecast/scope analyses,
poses, commands, RGB, gyro/body and native physics. Every failed mission remains
in full. Keep the committed-view control success, local-reference goal-only
run and stopping-projection verified round trip in full as current references.
Keep combined control layout 1 and all active replay inputs. This explicitly
ends earlier full-depth pins for only the six named successes, including the
earlier entire-eight-run combined cohort pin. Exact historical sensor replay
becomes unavailable. Inventory:
`.generated/depth_retirement_superseded_motion_successes_2026-09-15/`.

Completed: 38,594 exact depth leaves retired, reclaiming 11,395,891,200 allocated
bytes (10.613 GiB). Every preserved top-level JSON hash matches, no selected
depth remains and all six roots carry markers. Free artifact storage afterward
was 14,612,459,520 bytes. No failed mission or active replay input was retired.

For the new three-controller transfer cohort, keep full recordings through each
layout's three-arm comparison and diagnosis. After that, retain at least the
first successful full recording for each controller and every failed mission;
other analyzed successes may retire depth under the same policy. This preserves
all reported outcomes while limiting accumulation of redundant success depth.

Latest review for the stopping-projection experiment: retire depth from the
completed, analyzed persistent-routing successes on layouts 2 and 3:
`go2_routing_memory_persistent_native_layout02_4800_v1_attempt_001` and
`go2_routing_memory_persistent_native_layout03_4800_v1_attempt_001`.
Their paired results and recorded scope checks are complete; neither is an
active sensor-replay or training/correction-fit input. Keep persistent layouts
0/1 and all four reduced-memory failures in full. Keep every current combined
perception, committed-view, local-reference and floor-recovery recording.
This explicitly ends full-depth retention for only the two named older
successes. All results, comparisons, poses, commands, physics, RGB and gyro/body
records remain; exact historical sensor replay becomes unavailable. Inventory:
`.generated/depth_retirement_completed_memory_successes_2026-09-15/`.

Completed: 12,264 exact depth leaves retired, reclaiming 3,553,701,888 allocated
bytes (3.310 GiB). All preserved top-level JSON hashes match, no selected files
remain and both roots have markers. Free artifact storage afterward was
4,922,871,808 bytes. All four reduced-memory failures and persistent layouts
0/1 remain full-depth references.

Latest review after both committed-view learned follow-ups: retire full depth
from these two superseded successful perception probes, whose standard outcomes
and paired analyses are complete:

- `go2_coherent_reference_refresh_learned_yaw_noise_2mm_native_layout01_4800_v1_attempt_001`
- `go2_camera_frontier_viewpoint_learned_yaw_noise_2mm_native_layout00_4800_v1_attempt_001`

Neither supplies training/correction fits or a pending sensor replay. Current
combined-motion and committed-view recordings retain these perception mechanisms
in full; retain all those recordings, the active local-view native run and both
local-reference replay inputs. Retain every failure and the earlier probe result,
comparison, RGB, body/gyro, physics, commands, poses and configuration files.
This review ends the earlier blanket full-depth pin for these two successes
only. Exact historical sensor replay becomes unavailable. Routine authorized
retirement inventory: `.generated/depth_retirement_superseded_perception_successes_2026-09-15/`.

Completed: 8,544 exact primary/auxiliary depth leaves retired, reclaiming
2,515,787,776 allocated bytes (2.343 GiB). Preserved top-level JSON hashes all
match, no selected depth remains, and both roots carry retirement markers.
Free artifact storage immediately afterward was 8,108,343,296 bytes.

Latest review after the complete combined-perception comparison and verified
committed-view follow-up: release depth from three older supervised transfer
successes to run the two learned-controller viewing follow-ups:

- `go2_post_training_transfer_supervised_rollout_noise_2mm_native_layout01_4800_v1_attempt_001`
- `go2_post_training_transfer_supervised_rollout_noise_2mm_native_layout02_4800_v1_attempt_001`
- `go2_post_training_transfer_supervised_rollout_noise_2mm_native_layout03_4800_v1_attempt_001`

All three have completed verified round trips and comparison/forecast analyses;
they are evaluation recordings, not model-training or correction-fit inputs.
No pending sensor replay uses them. Keep that study's supervised layout-0
success and all five failures in full. Keep all eight current combined-motion
recordings, the committed-view follow-up, active replay inputs, models, fits and
cached training/validation data. This explicitly ends the earlier full-depth
pin only for these three supervised successes. Preserve every outcome, RGB,
gyro/body, physics, command, pose, configuration and analysis record. Exact
historical sensor replay becomes unavailable; regeneration does not promise
the same asynchronous trajectory. Existing routine-retirement authorization
applies. Inventory and completion:
`.generated/depth_retirement_completed_supervised_transfer_successes_2026-09-15/`.

Completed: 21,180 exact depth leaves retired, reclaiming 6,219,743,232 allocated
bytes (5.793 GiB). All preserved top-level JSON hashes match, no selected depth
remains, and each root carries its retirement marker. Free artifact storage
afterward was 11,504,648,192 bytes. Current failed recordings, active replay
inputs, the successful viewing follow-up and all models/fits remain retained.

Latest review during the completed-seven/final-running combined-perception
comparison: release depth from the two completed contact-score layout-1
successes, now superseded as diagnostic references:

- `go2_contact_score_ablation_pose_command_xy_learned_noise_2mm_native_layout01_4800_v1_attempt_001`
- `go2_contact_score_ablation_pose_command_xy_disabled_noise_2mm_native_layout01_4800_v1_attempt_001`

Both have verified round trips and completed paired/contact/forecast analyses.
Neither is a training or correction-fit input or a pending sensor-replay input.
Keep the contact pilot's full layout-0 success and failure as representative
references. Keep all current combined-perception recordings, all yaw/frontier/
reference-refresh/camera probes, all models/fits and cached training inputs.
This explicitly ends the earlier blanket full-depth pin only for these two
contact successes. Their complete outcomes, RGB, gyro/body, physics, commands,
poses, configurations and analyses remain; exact historical asynchronous sensor
replay becomes unavailable. Existing user-authorized routine retirement applies.
Inventory and completion records:
`.generated/depth_retirement_completed_contact_successes_2026-09-15/`.

Completed: 12,770 exact depth leaves retired, 3,793,137,664 allocated bytes
(3.533 GiB) reclaimed. Every preserved top-level JSON hash matches; neither
selected root retains depth leaves. Both carry explicit retirement markers.
Artifact storage had 9,994,502,144 free bytes after cleanup. All current
combined-perception recordings and pending follow-up inputs remain retained.

Latest review after the completed yaw study and verified reference-refresh and
camera-viewpoint follow-ups: end full-depth retention for the seven successful
assignments in the completed XY-source comparison. Exact roots:

- `go2_pose_command_xy_ablation_supervised_rollout_learned_noise_2mm_native_layout01_4800_v1_attempt_001`
- `go2_pose_command_xy_ablation_supervised_rollout_learned_noise_2mm_native_layout02_4800_v1_attempt_001`
- `go2_pose_command_xy_ablation_supervised_rollout_learned_noise_2mm_native_layout03_4800_v1_attempt_001`
- `go2_pose_command_xy_ablation_supervised_rollout_pose_command_noise_2mm_native_layout00_4800_v1_attempt_001`
- `go2_pose_command_xy_ablation_supervised_rollout_pose_command_noise_2mm_native_layout01_4800_v1_attempt_001`
- `go2_pose_command_xy_ablation_supervised_rollout_pose_command_noise_2mm_native_layout02_4800_v1_attempt_001`
- `go2_pose_command_xy_ablation_supervised_rollout_pose_command_noise_2mm_native_layout03_4800_v1_attempt_001`

All seven independent round-trip evaluations, paired comparisons, executed XY
and yaw/score analyses are complete. These navigation recordings are not the
separate cached model-training or motion-fit inputs. No pending sensor replay
uses them. The newer full contact/yaw/frontier and perception/viewpoint probes
supersede them as active debugging recordings. Keep the learned-XY layout-0
failure in full; keep every newer contact/yaw/frontier/reference-refresh/camera
recording and every model, fit and training/validation input.

This explicitly ends the earlier eight-run XY comparison's full-sensor pin.
All eight outcomes and their RGB, body/gyro, physics, commands, poses, model and
configuration identities and analyses remain. Only the exact inventoried
primary/auxiliary depth NPZ leaves in these seven roots are retired. A new
simulation does not promise the exact historical asynchronous trajectory.
Existing routine-retirement authorization applies. Inventory and completion:
`.generated/depth_retirement_completed_xy_successes_2026-09-15/`.
The selected 47,680 leaves occupy 13.010 GiB allocated before retirement.

Latest review after the XY/contact controls and repaired frontier follow-up:
end the full-depth pin for seven successful assignments in the completed
sixteen-run coherent-perception transfer comparison. Exact roots:

- `go2_post_training_transfer_jepa_noise_2mm_native_layout00_4800_v1_attempt_001`
- `go2_post_training_transfer_jepa_noise_2mm_native_layout01_4800_v1_attempt_001`
- `go2_post_training_transfer_direct_noise_2mm_native_layout00_4800_v1_attempt_001`
- `go2_post_training_transfer_direct_noise_2mm_native_layout01_4800_v1_attempt_001`
- `go2_post_training_transfer_direct_noise_2mm_native_layout03_4800_v1_attempt_001`
- `go2_post_training_transfer_reactive_noise_2mm_native_layout02_4800_v1_attempt_001`
- `go2_post_training_transfer_reactive_noise_2mm_native_layout03_4800_v1_attempt_001`

All seven physical round trips and the complete sixteen-outcome analysis are
finished. They are evaluation trajectories, not model-training or correction-fit
inputs; the frozen fits predate them. No pending sensor replay uses these seven.
Keep the other nine full recordings: all five failures and all four supervised
reference successes. Keep the full XY-source, contact-score and frontier-probe
recordings, including the original frontier stall and both failed follow-ups.
These newer controls and repaired exploration policy supersede the selected
successes as current debugging inputs. No unresolved failed sensor input is
released by this review.

This explicitly revises the earlier sixteen-recording pin and ends full sensor
replay completeness for that development comparison. All sixteen outcomes,
RGB/gyro/body records, physics, commands, poses, models, configuration identities
and analyses remain. Retire only the exact seven roots' primary/auxiliary depth
NPZ leaves under the existing user-authorized policy. Record file inventories,
preserved JSON hashes and completion under
`.generated/depth_retirement_completed_transfer_successes_2026-09-15/`.

Completed: 41,014 exact depth leaves retired, reclaiming 12,093,153,280 allocated
bytes (11.263 GiB). Every preserved top-level JSON hash matches and no selected
depth leaf remains. All seven roots have explicit `DEPTH_RETIRED` markers;
their outcomes and all other artifact types remain available.

Latest review after the completed sixteen-run coherent-perception transfer
comparison: end the predecessor current-plane training-comparison depth pin
for seven successful, fully analyzed and superseded assignments. Exact roots:

- `go2_current_plane_matched_training_jepa_noise_2mm_native_layout02_4800_v1_attempt_001`
- `go2_current_plane_matched_training_direct_noise_2mm_native_layout02_4800_v1_attempt_001`
- `go2_current_plane_matched_training_direct_noise_2mm_native_layout03_4800_v1_attempt_001`
- `go2_current_plane_matched_training_supervised_rollout_noise_2mm_native_layout00_4800_v1_attempt_001`
- `go2_current_plane_matched_training_supervised_rollout_noise_2mm_native_layout01_4800_v1_attempt_001`
- `go2_current_plane_matched_training_supervised_rollout_noise_2mm_native_layout02_4800_v1_attempt_001`
- `go2_current_plane_matched_training_supervised_rollout_noise_2mm_native_layout03_4800_v1_attempt_001`

Their verified arrivals, full study summary, command comparisons and executed
forecast analyses are complete in
`docs/go2_current_plane_matched_training_noise_2026-09-15.md`. They are evaluation
trajectories, not model-training or motion-fit input datasets. Subsequent
four-controller trials with coherent perception supersede them as the current
full comparison population. No pending sensor replay reads these seven roots.
Keep the predecessor's four failed assignments and JEPA layout 1's long
coherent-perception replay input as full diagnostic references. Keep all sixteen
new transfer recordings, all eight active/planned XY-source assignments, all
model/fit inputs, and the current reactive-layout-0 replay recording.

This review explicitly ends full sensor replay completeness for that older
twelve-trial development comparison; every one of its twelve outcomes remains
reported. Retire only the seven roots' exact primary/auxiliary depth NPZ leaves.
Preserve all other artifacts, including RGB, gyro/body histories, physics,
commands, poses, source/configuration identities and every analysis/failure.
Inventory and retained-record bindings are recorded under
`.generated/depth_retirement_superseded_current_plane_successes_2026-09-15/`.
The existing user authorization for routine superseded depth applies.

Completed September 15: retired 52,334 exact depth leaves from these seven
roots, reclaiming 15,576,043,520 allocated bytes (14.51 GiB). All preserved
top-level JSON hashes match, and no matching depth leaves remain. Each root
has `depth_retention.json` and `DEPTH_RETIRED`, explicitly marking full sensor
replay unavailable. The inventory directory's `result.json` records completion;
34.77 GiB was free afterward. All other recordings and model/fit inputs named
above remain retained.

Additional active future-control input: retain
`go2_pose_command_motion_control_fit_v1_attempt_001/motion_fit.npz` and its
launch/result records, together with the original cached residual-study
training/validation NPZ files identified in that launch. This separate
pose/command predictor is not used in the running post-training transfer study.

Latest review (September 15, after the current twelve-run noisy training study):
end the older fixed-transfer full-depth pin for both arms on layouts 4 and 7.
Exact roots are `go2_progress_rejoining_learned_round_trip_native_layout04_4800_v1_attempt_001`,
`go2_fixed_transfer_preferred_reactive_round_trip_native_layout04_4800_v1_attempt_001`,
`go2_progress_rejoining_learned_round_trip_native_layout07_4800_v1_attempt_001`, and
`go2_fixed_transfer_preferred_reactive_round_trip_native_layout07_4800_v1_attempt_001`.
All four owners completed, and their clearance stalls / return-budget outcome
and physical arrival evaluations are recorded in
`docs/go2_continuous_controller_comparison_2026-09-13.md`. Their image-rotation
estimator has been superseded by the gyro-conditioned native studies. No pending
sensor replay or training/validation fit reads these four roots. Preserve the
full layout-5/6 pairs, including false arrivals and the registration failure,
as the older cohort's diagnostic references. Preserve all eight historical
outcomes and comparison reports. This explicitly ends full sensor completeness
for that older cohort; it does not remove failed trials from its population.

Retire only exact primary/auxiliary depth NPZ leaves from these four roots;
preserve RGB, gyro/body, physics, requests, poses, configurations and diagnoses.
Current noisy training/reactive comparisons, active failure replays, model/fit
inputs and the newer clean reference remain fully retained. Exact inventories
and preserved JSON identities are recorded in
`.generated/depth_retirement_completed_original_transfer_references_2026-09-15/`.

Completed: 38,442 exact depth leaves retired, 61,536,505,856 allocated bytes
(57.31 GiB) reclaimed. All preserved top-level JSON hashes match. Every retired
root has a `DEPTH_RETIRED` marker; approximately 60.27 GiB is free afterward.

The user authorized this policy and depth cleanup on September 14:
"do that, and tidy up depth data we dont need - we can always regeenrate it".
This replaces indefinite retention of every routine development depth recording.
It supersedes the pending three-recording recompression proposal; recompression
is no longer required to unblock experiments.

Keep every experiment's configuration, code/model identity, results, failure
diagnosis, commands, trajectory and evaluation. Retiring depth never removes an
unsuccessful run from reported populations or changes its outcome. Keep existing
RGB, body, gyro and physics records during this depth cleanup.

Keep full sensor recordings for active debugging and unresolved failures, selected
reference successes/failures, and complete matched comparison populations used
as scientific evidence. Preserve current training inputs and checkpoints. Sealed
benchmark material remains inaccessible and outside cleanup scope.

For routine development, retain full depth until analysis is complete. Thereafter
retain at most the three latest unpinned completed recordings of the current
experiment family; retire older diagnosed/superseded depth. Aim to keep this
unpinned working set below 100 GiB across volumes. This is a retention trigger,
not a new experiment admission gate: remove eligible old depth before collecting
more. Never automatically discard an active, unresolved or explicitly retained
recording to meet the target. Do not add repeated approval steps for routine
depth retirement within this user-authorized scope.

New paired-camera experiments should use the existing raw-only LZMA archive
writer; derived depth and validity arrays are reconstructed by PublicReplay.
Retire only exact primary/auxiliary depth NPZ paths in completed ordinary
development roots. Keep a filename/size inventory and a `depth_retention.json`
marker per retired run. A new simulation can regenerate data for a new experiment;
it is not guaranteed to reproduce the exact historical closed-loop trajectory.
Historical full-archive checks will no longer pass after intentional retirement;
leave their original records intact and consult the retention marker.

Current full-depth recordings explicitly retained include:

- The workspace cached fine-goal layout-4 stable-reference replay recording.
  The older RecoveryStorage raw-depth terminal-pulse layout-6 recording was
  reviewed for retirement below after its replay work completed.
- The workspace hold-relative layout-6 verified round trip.
- The fixed transfer cohort: all four progress-rejoining learned layout-4–7
  runs and all four fixed-transfer reactive layout-4–7 runs.
- Both plane-consensus layout-0 arms and both pair-local-plane layout-3 arms.
- The indexed-geometry layout-0 verified round trip and layout-1 speed failure;
  arc-recovery layout-2 success and progress-rejoining layout-3 success.
- The original reactive layout-0 perception failure, arc-recovery layout-3
  failure and clearance-preferred reactive layout-3 failure; the orthonormal-gyro
  successor to the diagnosed reused-flow gyro layout-6 rotation failure.
  The predecessor rotation-failure depth was reviewed for retirement below.

Other unreviewed artifact families are not implicitly approved for deletion by
this list. Revisit retained diagnostic examples when their scientific use ends;
final comparison populations remain complete.

## Completed cleanup

The initial cleanup retired depth from 27 superseded RecoveryStorage controller
development runs plus the workspace uncached fine-goal layout-4 run. All selected
result/failure/configuration/evaluation record hashes were unchanged afterward.
Every deleted file was an exact `primary_depth_<frame>.npz` or
`auxiliary_depth_<frame>.npz` leaf; all other file types remain.

Exact inventories, per-run reasons, original record hashes and completed results:
`.generated/depth_retirement_2026-09-14/manifest.json`, `files.jsonl`,
`result.json`, `workspace_manifest.json` and `workspace_result.json`.
The concise measured totals are in
`docs/go2_depth_retirement_result_2026-09-14.json`.

## Completed-diagnostic review after the repeatability study

The two selected diagnostic recordings below have finished their scientific
replay use and are superseded by subsequent native perception evidence. Retire
only their exact depth NPZ files under the existing user authorization; preserve
all other artifacts and the historical failure outcomes. This is a review of
completed diagnostic examples, not removal from a current comparison population.

- `go2_raw_depth_terminal_pulse_fine_goal_learned_round_trip_native_layout06_4800_v1_attempt_001`: the drift diagnosis, failed overlap-only replay and successful
  4,805-frame stable-reference replay are complete. Stable references have since
  been used in native matched and repeatability studies. The other independent
  stable-reference replay recording remains retained.
- `go2_reused_flow_gyro_fine_goal_learned_round_trip_native_layout06_4800_v1_attempt_001`: the arithmetic rejection was reproduced and the orthonormal-gyro
  replay completed all 3,019 frames. The numerical fix survived the retained
  successor native run and subsequent studies. Preserve both replay results,
  round-off diagnosis, original failure, physics and successor full recording.

Neither root is an active input, training input or unresolved debugging case.
The current four-repeat population, its turn-oscillation failure, current
matched training controls and new-maze comparison remain fully retained.
Exact deletion inventory and result will be recorded under
`.generated/depth_retirement_completed_diagnostics_2026-09-14/`.

The reviewed cleanup completed: 15,648 exact depth NPZ files retired,
19,423,781,200 content bytes and 19,455,455,232 allocated bytes (18.1193 GiB)
reclaimed. All recorded result/failure/diagnostic hashes verified unchanged.
Both roots now have `DEPTH_RETIRED` markers; full sensor replay is intentionally
unavailable, while all other artifacts remain. RecoveryStorage had about
36 GiB free afterward, sufficient for the current eight-run comparison.

## Completed terminal-priority probe review

Retire depth from the single predecessor probe
`go2_stable_reference_jit_floor_cached_fine_goal_lzma_pulse_learned_round_trip_native_layout04_4800_v1_attempt_001`.
Its incomplete home approach was diagnosed through executed-pulse forecast
errors and saved terminal selection at frame 3796. Both analyses are complete
and retained. Arrival-entry priority superseded that controller and has since
been exercised in the retained successor and multiple native comparisons.
This single predecessor probe is not part of the successor learned/reactive
comparison population; both successor arms remain fully retained. No further
depth replay of this diagnosed probe is required for the current experiments.

Preserve all results, commands, trajectories, source witnesses, RGB, body/gyro
and physics. Inventory only its 9,610 primary/auxiliary depth NPZ leaves under
`.generated/depth_retirement_terminal_probe_2026-09-14/`, and preserve the
original failed round-trip outcome. Current comparisons and unresolved
perception/oscillation failures remain untouched.

This retirement completed: 9,610 depth files removed, reclaiming 2,856,882,176
allocated bytes (2.6607 GiB). All original top-level JSON record hashes matched
afterward. The root now has its `DEPTH_RETIRED` marker and the inventory directory
contains the manifest, exact filename/size list and completed result.

## Completed early navigation probes reviewed before routing-memory study

Retire only depth from these five completed 1,800-tick development probes.
Their source-level diagnoses and saved analyses are complete, their treatments
have been superseded by retained native navigation studies, and none belongs
to the current model residual training/validation roster or a final comparison
population. Keep all RGB, physics, commands, poses, results and diagnostic files.

- `go2_reserve_recovery_lookahead_native_layout00_v1_attempt_001`: the close-wall
  stop was diagnosed as newly revealed geometry, despite accurate motion against
  the old map; the subsequent standoff treatment and later failure examples remain.
- `go2_motion_residual_standoff_native_layout00_v1_attempt_002`: prospective
  executed-motion accuracy and the newly revealed wall were measured and saved;
  it was excluded from residual fitting and validation. Retain the fit inputs
  and the current matched training/navigation populations.
- `go2_panoramic_frontier_motion_residual_native_layout00_v1_attempt_001`:
  its early stop preceded any frontier panorama. The initial-view geometry
  failure is diagnosed in the September 13 experiment record; later initial
  panoramas and current close-wall failures remain fully recorded.
- `go2_joint_camera_initial_panorama_native_layout00_v1_attempt_001`:
  tracking, initial/frontier panorama completion and pose accuracy were measured;
  the short budget ended while it was following a goal route. Later full-budget
  tracking/navigation evidence supersedes this short probe.
- `go2_continuous_connector_optional_plane_native_layout00_v1_attempt_001`:
  full saved-route and depth-point reconstruction completed the coarse-cell
  inflation diagnosis and fine-cell comparison. Current native studies retain
  the resulting fine-map implementation and full recordings.

The completed analyses are documented in
`docs/go2_feature150_memory_clearance_experiment_2026-09-13.md`,
`docs/go2_optional_plane_and_continuous_connector_2026-09-13.md` and
`docs/go2_fine_stored_obstacle_routing_2026-09-13.md`. Their original failures
remain reported. The three current residual-fit launch records explicitly
exclude all five candidates from their training/validation roots. Exact
filename/size inventories and preserved result/diagnostic hashes go under
`.generated/depth_retirement_early_probes_2026-09-14/`.

Completed: 18,052 exact depth NPZ leaves removed from those five roots,
reclaiming 26,878,832,640 allocated bytes (25.032 GiB). All recorded top-level
JSON hashes verified unchanged; each root has a `DEPTH_RETIRED` marker. Current
fit inputs, comparison populations and unresolved debugging recordings remain.

## Completed orthonormal-gyro diagnostic reviewed after memory ablations

The previously retained
`go2_orthonormal_gyro_predictive_hold_fine_goal_learned_round_trip_native_layout06_4800_v1_attempt_001`
has finished its diagnostic use. Its complete 4,805-frame run established that
the numerical rotation fix survived native execution; that implementation has
subsequently survived the fully retained matched, transfer and memory studies.
Its one predictive-hold activation and failure to settle at home were measured
and documented in `docs/go2_gyro_pose_and_fine_goal_navigation_2026-09-14.md`.
Terminal pulses and arrival-entry priority superseded that controller, with
their separate comparison populations retained. This single diagnostic is not
part of those matched populations, and none of the three current motion-residual
fit launch records includes it. There is no pending depth replay of this run.

Under the existing cleanup authorization, retire its 9,610 exact depth NPZ
leaves (15.861 GiB allocated), retaining all other artifacts and its unsuccessful
round-trip outcome. This review ends its earlier diagnostic retention; current
training, full comparison populations and unresolved failures remain retained.
Inventory and completion evidence go under
`.generated/depth_retirement_orthonormal_diagnostic_2026-09-14/`.

Completed: all 9,610 listed depth files were retired, reclaiming 15.861 GiB
allocated. All original top-level JSON record hashes remained unchanged.
The recording has its `DEPTH_RETIRED` marker; all non-depth artifacts remain.

## Completed early perception/command probes reviewed September 15

Retire only the exact depth NPZ leaves in five completed predecessor probes:

- `go2_feature150_60s_native_layout00_v1_attempt_001`

- `go2_independent_depth_60s_native_layout00_v1_attempt_001`

- `go2_continuous_commitment_20s_native_layout00_v1_attempt_001`

- `go2_conditioned_support_native_layout00_v1_attempt_001`

- `go2_fine_stored_obstacle_native_layout00_v1_attempt_001`

The two 60-second timing/feature probes and 20-second commitment probe
completed their timing, command and pose analyses; later retained native
comparisons supersede them. The conditioned-support failure was reproduced
and its 150/300-feature replays completed; the fine-stored-obstacle failure
was reproduced and its conditioned-support replay completed. These diagnoses
are preserved in the September 13 feature150/commitment, independent-depth,
development-support and fine-stored-obstacle documents and per-root results.
No further depth replay is pending for these five probes. They are not a final
comparison population; all current comparisons and unresolved examples remain
retained. The three active motion-residual fit launch records explicitly
exclude these five roots from training/validation. Original owner PIDs are
absent. Exact inventories and preserved top-level JSON hashes are recorded in
`.generated/depth_retirement_early_perception_probes_2026-09-15/`.

Completed: 3,630 depth files retired, reclaiming 4.321 GiB allocated. All recorded top-level JSON hashes remained unchanged; every root has a DEPTH_RETIRED marker.

## Unexercised local terminal probe reviewed September 15

Retire exact depth leaves from `go2_terminal_100ms_pulse_local_round_trip_layout06_v1_attempt_001`. Its 90-mm target was unobserved by the cameras, so the 905-frame run never exercised the proposed terminal pulse. The corresponding standard arm was not launched. This completed setup diagnosis is recorded in `docs/go2_terminal_translation_pulse_study_2026-09-14.md`; no depth replay is pending. The subsequent paired 650-mm study stays fully retained. This single superseded probe is excluded from current training/validation, and its owner is absent. Preserve all outcomes, diagnoses, RGB, body/gyro, physics and other artifacts. The exact inventory and preserved JSON hashes are in `.generated/depth_retirement_unexercised_terminal_probe_2026-09-15/`.

Completed: 1,810 files retired, 2.879 GiB allocated reclaimed; all original top-level JSON hashes unchanged.

## Completed local terminal-control pair reviewed September 15

The paired 650-mm-target, 90-second pulse/standard diagnostic has completed
its scientific use. Both controllers reached the local goal, neither returned
home, and the comparison established no pulse advantage. Timing, exact pulse
execution, arrival physics and the unobserved-home return cause were analyzed
in `docs/go2_terminal_translation_pulse_study_2026-09-14.md`. No depth replay
is pending. Full-maze navigation/terminal studies have since superseded this
local setup; it is not a current or final navigation, model, or memory
comparison population. End full-depth retention for both conditions together,
while preserving their historical comparison and all non-depth evidence.

Exact roots:

- `go2_terminal_100ms_pulse_goal065_local_round_trip_layout06_v1_attempt_001`

- `go2_terminal_standard400ms_goal065_local_round_trip_layout06_v1_attempt_001`

The existing user-authorized retirement of completed development recordings
applies here. Keep the comparison result, all per-run outcomes/diagnoses,
commands, trajectories, RGB, body/gyro and physics unchanged. Current training,
current/final navigation comparisons and unresolved failures stay fully
recorded. Both old owners are absent and all current fit launch records exclude
these two roots. Exact inventories and preserved result hashes are in
`.generated/depth_retirement_completed_local_terminal_pair_2026-09-15/`.

Completed: 3,624 exact depth leaves retired, 5.657 GiB allocated reclaimed. Both conditions carry DEPTH_RETIRED markers, and all recorded JSON/comparison hashes remained unchanged.

## Diagnosed predecessor noisy-floor failures, September 15

End full-depth retention for layouts 1–3 of the completed original live noisy
tracker study, exact roots
`go2_live_local_feature_depth_noise_2mm_native_layout01_4800_v1_attempt_001`,
`go2_live_local_feature_depth_noise_2mm_native_layout02_4800_v1_attempt_001` and
`go2_live_local_feature_depth_noise_2mm_native_layout03_4800_v1_attempt_001`.
These are diagnosed development failures superseded by the current independent
local-floor experiment, not final evaluation material. Every original floor
failure probe reproduced, and local floor fits were available on all selected
frames. No further depth replay of these three runs is pending. Keep original
noisy layout 0 as a fully recorded reference failure, all four clean recordings,
all current independent-floor recordings and all comparison results. This
review explicitly ends full-sensor completeness for the older eight-run
development comparison; its full outcome population remains unchanged.

The existing user authorization to retire unneeded regenerable depth applies.
Only exact primary/auxiliary depth NPZ leaves are eligible. Preserve all JSON
results/diagnoses, noise recipes, RGB, public body/gyro, commands and physics.
Current training/validation roots and checkpoints are excluded. Record exact
inventories, non-depth JSON identities and the historical comparison identity
under `.generated/depth_retirement_diagnosed_noisy_floor_2026-09-15/`. New
simulation is not guaranteed to reconstruct the historical asynchronous path;
the retained layout-0 recording still permits exact noisy-packet replay.

Completed: 28,830 exact depth leaves retired, reclaiming 9.704 GiB allocated.
All recorded top-level JSON hashes and the original eight-run comparison hash
remain unchanged. All three roots carry `DEPTH_RETIRED` markers.

The last original noisy layout-0 depth recording is subsequently reviewed for
retirement after the first live mapping pair completed. Its raw-candidate and
51-frame local-floor diagnoses are finished; no replay remains pending. The
fully retained independent-floor population and the new mapping layout-0
near-wall failure now provide the current reference recordings. The active
failure replay reads the new mapping recording, not this predecessor. Retire
only depth leaves in
`go2_live_local_feature_depth_noise_2mm_native_layout00_4800_v1_attempt_001`,
preserving all original diagnoses/outcomes and the clean comparison recordings.
The inventory and result are in
`.generated/depth_retirement_last_original_noisy_floor_2026-09-15/`.

Completed: 9,610 exact depth leaves retired, reclaiming 2.947 GiB allocated.
All recorded top-level JSON identities remained unchanged; the original noisy
layout-0 root now carries its `DEPTH_RETIRED` marker.

## Superseded independent-floor recording review

After the complete local-mapping experiment and both exact registration-failure
diagnoses, retire depth from independent-floor layouts 1–3, exact roots
`go2_live_local_floor_obstacle_noise_2mm_native_layout01_4800_v1_attempt_001`,
`go2_live_local_floor_obstacle_noise_2mm_native_layout02_4800_v1_attempt_001` and
`go2_live_local_floor_obstacle_noise_2mm_native_layout03_4800_v1_attempt_001`.
Their mapping-prefix counterfactuals reproduced all saved count witnesses and
completed their diagnostic use. No further replay of these three roots is
pending. Keep independent-floor layout 0 as a full reference, all four current
local-mapping recordings (including both reproduced near-wall failures), all
clean/reference/training recordings and checkpoints. This explicitly ends
full-sensor completeness of the superseded independent-floor development
comparison; its full result population and every non-depth artifact remain.
Exact inventories and preserved JSON identities are recorded under
`.generated/depth_retirement_superseded_independent_floor_2026-09-15/`.

Completed: 28,830 exact depth leaves retired, reclaiming 5.563 GiB allocated.
All recorded top-level JSON identities remained unchanged. The three roots
carry `DEPTH_RETIRED` markers; current local-mapping recordings remain intact.

## Superseded local-mapping layout 1/2 depth

After all four gyro-height-floor native runs completed and were evaluated,
retire only primary/auxiliary depth NPZ leaves in
`go2_live_local_floor_mapping_noise_2mm_native_layout01_4800_v1_attempt_001`
and `go2_live_local_floor_mapping_noise_2mm_native_layout02_4800_v1_attempt_001`.
The layout-2 prefix and exclusion diagnoses are complete. The new fully
recorded gyro-floor layout 2 reproduces the exploration failure with complete
floor availability; the new layout 1 also remains fully recorded. No pending
replay reads either retired predecessor. Preserve their outcomes, noise
recipes, poses, RGB/body/gyro, commands, physics and diagnoses; retain the
original local-mapping 0/3 registration failures and all current gyro-floor
recordings, training inputs and checkpoints. This ends full-depth retention
of the superseded local-mapping comparison, without changing its population.
Inventory and completion records go under
`.generated/depth_retirement_superseded_local_mapping_2026-09-15/`.

Completed: 19,220 exact depth leaves retired, reclaiming 5,079,216,128
allocated bytes (4.730 GiB). Preserved JSON identities all match; both roots
carry `DEPTH_RETIRED` markers. Artifact free space increased to about 8.1 GiB.

## Remaining diagnosed predecessor mapping/floor depth

After the cached run completed and the current-plane layout-2 round trip was
physically verified, retire only depth leaves from
`go2_live_local_floor_mapping_noise_2mm_native_layout00_4800_v1_attempt_001`,
`go2_live_local_floor_mapping_noise_2mm_native_layout03_4800_v1_attempt_001` and
`go2_live_local_floor_obstacle_noise_2mm_native_layout00_4800_v1_attempt_001`.
Their registration, candidate-selection, replay and routing-prefix diagnoses
are complete. No pending replay reads these roots. The full current gyro-floor
four-run cohort, cached layout 0 and current-plane layout 2 retain their depth
and provide the current failure/success inputs. Preserve every predecessor
outcome, noise recipe, physics/commands, RGB/body/gyro, replay result and
diagnostic. This ends remaining full-depth retention of the superseded
independent-floor and local-mapping comparisons, without changing either
outcome population. Current training/validation/checkpoint and clean/reference
inputs remain excluded. Inventory and result are recorded under
`.generated/depth_retirement_remaining_diagnosed_floor_2026-09-15/`.

Completed: 16,622 exact depth leaves retired, reclaiming 3,953,664,000 allocated
bytes (3.682 GiB). Preserved top-level JSON identities match and all three
roots carry `DEPTH_RETIRED` markers. Current comparison recordings remain intact.

## Completed early clean transfer comparison depth

Retire depth only from the six completed roots with exact naming
`go2_post_repeatability_transfer_{learned,reactive}_native_layout{00,01,03}_4800_v1_attempt_001`
(both arms, layouts 0,1,3). Their fixed clean comparison is complete; no
pending depth replay reads these roots. The later full-journey noise replays
used the separate persistent-routing recordings, which remain retained.
The stronger reactive launchers read predecessor launch JSON, which remains
unchanged; these six roots are not current training or validation inputs.

Keep both layout-2 transfer recordings as full representative clean success
and failure references. Keep every result and failure, comparison summary,
trajectory/physics/commands, RGB/body/gyro, configuration and model identity
from all eight original assignments. Current gyro-floor, cache, current-plane,
clean noise-control and persistent-memory recordings, model-fit inputs and
checkpoints remain untouched. This explicitly ends full-sensor completeness
for the older clean eight-run comparison; its complete outcome population
remains intact. Regeneration does not promise the exact historical asynchronous
trajectory. Exact six-root inventories, preserved JSON hashes and completion
are recorded under `.generated/depth_retirement_completed_clean_transfer_2026-09-15/`.

Completed: 46,000 exact depth leaves retired, reclaiming 14,191,996,928
allocated bytes (13.217 GiB). Every preserved top-level JSON identity matches;
all six roots carry `DEPTH_RETIRED` markers. About 20 GiB is now free, allowing
the remaining noisy trials to use the two established CPU groups concurrently.

## Superseded training-control depth review, September 15

The earlier twelve-run heading-release training-method comparison and its
per-layout physical evaluations, control binding checks, trajectory figures
and failure diagnoses are complete. The current noisy perception comparison
is fully retained, and the next training-method study uses frozen model/fit
files and the original training inputs rather than these navigation depth
recordings. Review the earlier whole-population depth pin as follows:
retain full layout 2 for all three methods, JEPA layout 0 turn-recovery failure
and supervised layout 3 hold failure. Retire depth only from JEPA layouts 1/3,
direct layouts 0/1/3 and supervised-rollout layouts 0/1 under
`go2_matched_training_<condition>_heading_release_native_layoutXX_4800_v1_attempt_001`.
All twelve outcomes remain in the historical comparison; RGB, gyro/body,
physics, configurations, poses, commands, model identities, reports and
diagnoses remain. Full raw sensor replay will intentionally become unavailable
for these seven records. This is not deletion of any training input, fit,
checkpoint, current noisy comparison or active debugging recording.
Exact inventories and hashes are under
`.generated/depth_retirement_superseded_training_controls_2026-09-15/`.

The reviewed seven-root retirement completed: 64,768 exact primary/auxiliary
depth NPZ files, 20,040,077,312 allocated bytes (18.664 GiB) reclaimed.
All preserved top-level JSON hashes verified unchanged. Exact files and sizes,
per-root hashes and result are saved in the stated inventory directory; every
retired run has a `depth_retention.json` marker. RecoveryStorage now has about
25 GiB free while the reacquisition native trial continues.

## Completed clean controls and cache-profile predecessor review

While the frozen current-plane training comparison runs, retire exact depth
leaves from these four completed ordinary development roots:

- `go2_live_local_feature_depth_noise_0mm_native_layout00_4800_v1_attempt_001`
- `go2_live_local_feature_depth_noise_0mm_native_layout01_4800_v1_attempt_001`
- `go2_live_local_feature_depth_noise_0mm_native_layout03_4800_v1_attempt_001`
- `go2_live_gyro_height_floor_noise_2mm_native_layout00_4800_v1_attempt_001`

The first two clean controls have completed physical speed-stop and command
transition diagnoses; those physics and command records remain. Clean layout
3 is a completed verified round trip. Keep clean layout 2 as the full sensor
reference. This review ends the earlier full-depth pin for these three clean
controls; all eight original clean/noisy outcomes remain preserved.

The gyro-floor layout-0 backlog was reproduced in a complete 1,050-pose
profile and an exact cached replay. Its prospective cached successor completed
the full budget without queue termination, and that successor and all current
cached navigation recordings remain retained. No further depth replay of this
completed profiling input is pending. Keep gyro-floor layouts 1–3, including
the layout-2 mapping diagnostic input. This ends only the earlier depth pin
for the diagnosed gyro-floor layout-0 predecessor.

None of these four roots supplies current training inputs, model corrections,
checkpoints, the active comparison or the unresolved gap-recovery diagnosis.
Preserve every outcome, diagnosis, configuration, recipe, RGB/body/gyro,
physics, command and pose record. Full historical sensor replay will become
unavailable; regeneration is not an exact asynchronous trajectory guarantee.
Inventory and completion records:
`.generated/depth_retirement_clean_controls_and_backlog_2026-09-15/`.

Completed: 15,954 exact depth files retired, reclaiming 4,722,442,240 allocated
bytes (4.398 GiB). Every preserved top-level JSON hash matches; all four roots
now carry `DEPTH_RETIRED` markers. Active trials and retained debugging inputs
remain intact.

## Additional completed predecessor references during the training comparison

Release depth from the following four completed references to make room for
the remaining six assignments of the current frozen training comparison:

- `go2_matched_training_jepa_heading_release_native_layout02_4800_v1_attempt_001`
- `go2_matched_training_direct_heading_release_native_layout02_4800_v1_attempt_001`
- `go2_heading_recovery_repeatability_rep1_layout01_4800_v1_attempt_001`
- `go2_heading_recovery_repeatability_rep2_layout01_4800_v1_attempt_001`

The older training comparison is complete and already partly depth-retired.
Keep its supervised layout-2 full recording as a representative, plus its
JEPA layout-0 and supervised layout-3 unresolved failure recordings. Current
matched training inputs and frozen corrections are separate, older designated
training sources; these navigation recordings are not fit inputs. All old
twelve-run outcomes, executed-forecast analyses and comparisons remain.

The two layout-1 repeatability runs completed verified round trips without
activating the new heading rule, and their timing/trajectory comparisons are
finished. Keep the full successful and failed layout-0 repetitions and both
original heading-recovery references. The failed repeated-turn diagnosis and
its depth remain available. All four repetition outcomes remain in the report.

This explicitly revises the earlier full-depth pins for these four paths.
Preserve all configurations, source/model identities, recipes, results,
failures, diagnoses, physics, commands, RGB/body/gyro and pose records. No
pending replay uses these selected inputs. No current noisy comparison or
active gap-recovery recording is retired. Exact inventories and completion:
`.generated/depth_retirement_completed_training_and_repeat_references_2026-09-15/`.

Completed: 28,102 exact depth leaves retired, reclaiming 8,385,695,744 allocated
bytes (7.810 GiB). Preserved top-level JSON hashes all match. Each selected
root has its explicit `DEPTH_RETIRED` marker; all other artifact types remain.

## Completed XY-success depth retirement

The reviewed seven-root cleanup completed: 47,680 exact depth leaves retired,
13,968,850,944 allocated bytes reclaimed (13.010 GiB). All preserved top-level
JSON hashes match; no selected depth remains. Each root has `depth_retention.json`
and `DEPTH_RETIRED`. The original learned-XY layout-0 failure, newer contact/yaw/
frontier/refreshed-reference/camera-viewpoint recordings, model/fit files and
training/validation inputs remain retained. Free artifact storage afterward:
21,056,053,248 bytes (19.611 GiB). Completion record:
`.generated/depth_retirement_completed_xy_successes_2026-09-15/result.json`.

Completed late-transfer success depth retirement before the first full wall-clock mission:
reactive layout 2 and fitted-motion layout 3 only. All four transfer comparisons
and applicable forecast/physical/treatment analyses are complete. Neither root is
a training, fit, or pending replay input. First full successes learned 0, fitted 1,
reactive 1, all failures, and learned-2 turning diagnosis remain full.
5,882 depth leaves retired; 1,765,478,400 allocated bytes reclaimed. All preserved
top-level JSON hashes match; RGB, body/gyro, physics, poses and commands remain.
Inventory: `.generated/depth_retirement_completed_transfer_late_successes_2026-09-15/`.

Before the fitted wall-clock control, retired only depth from completed analyzed
transfer learned layout 3 and reactive layout 3. All outcomes and non-depth data
remain, along with first full successes per controller, learned-2 turning diagnosis,
all failures and the first full wall-clock success. Neither retiring root supplies
a fit, training data or pending sensor replay. Retired 6,342 leaves / 1,973,878,784
allocated bytes; preserved top-level JSON hashes match. Inventory:
`.generated/depth_retirement_completed_transfer_remaining_successes_2026-09-15/`.


## Superseded early round-trip references

While the new multi-seed comparison runs, retired primary/auxiliary depth from
`go2_clearance_preferred_arc_recovery_learned_round_trip_native_layout02_v1_attempt_001`
and `go2_indexed_geometry_precise_goal_round_trip_native_layout00_v1_attempt_001`.
Both have verified goal/home arrivals, zero contacts, completed physical and
forecast/trajectory analyses and no active fit or replay dependency. This ends
any earlier full-depth retention of these historical first milestones. Their
results and all non-depth records remain; current full learned/fitted/reactive
references, training/correction inputs and every failed recording remain full.

Retired 10,508 exact depth leaves, reclaiming 16,576,876,544 allocated bytes
(15.438 GiB), leaving 33,876,242,432 bytes free during the first native pair.
All preserved top-level JSON hashes match. Exact inventory and result:
`.generated/depth_retirement_superseded_early_round_trip_successes_2026-09-15/`.
Both roots carry explicit depth-retirement markers; exact historical sensor
replay is no longer available, and regenerated asynchronous trajectories need
not match. No checkpoints, fits, RGB, body/gyro, commands or physics were removed.


## Superseded heading-success depth retirement, 2026-09-15

Retired depth only from these completed, analysed round trips:

- `go2_heading_release_fresh_learned_round_trip_native_layout01_4800_v1_attempt_001`
- `go2_heading_release_fresh_learned_round_trip_native_layout03_4800_v1_attempt_001`
- `go2_hold_relative_heading_recovery_native_layout01_4800_v1_attempt_001`

This ends their earlier full-depth pins. Keep heading-release layout 2 and
hold-relative layout 0 in full, all current references and every failure.
None of the retired recordings is a current training/fit or pending replay
input. Results, comparisons, figures, RGB, gyro/body, poses, commands and physics
remain; exact historical depth replay is unavailable.
Inventory: `.generated/depth_retirement_superseded_heading_successes_2026-09-15`.
Completed: 28244 depth files, 8688566272 allocated bytes reclaimed; preserved JSON hashes match.


## Completed multi-seed comparison success depth, 2026-09-15

Completed fixed 22-run comparison and all six inspected trajectory figures; redundant successful depth only. Preserve seed1001 layout0 for each learned method, first supervised success seed1401 layout1, both fitted controls, every failure, all models/fits and non-depth evidence. No retired recording is a training/fit or pending depth-replay input.
This ends earlier full-depth pins only for the eleven successes listed below.
Exact historical sensor replay becomes unavailable for those recordings.

- `go2_multiseed_navigation_seed_2026091401_full_jepa_noise_2mm_native_layout00_4800_v1_attempt_001`
- `go2_multiseed_navigation_seed_2026091401_full_direct_noise_2mm_native_layout00_4800_v1_attempt_001`
- `go2_multiseed_navigation_seed_2026091401_full_supervised_rollout_noise_2mm_native_layout00_4800_v1_attempt_001`
- `go2_multiseed_navigation_seed_2026091402_full_jepa_noise_2mm_native_layout00_4800_v1_attempt_001`
- `go2_multiseed_navigation_seed_2026091402_full_direct_noise_2mm_native_layout00_4800_v1_attempt_001`
- `go2_multiseed_navigation_seed_2026091402_full_supervised_rollout_noise_2mm_native_layout00_4800_v1_attempt_001`
- `go2_multiseed_navigation_seed_2026091001_full_jepa_noise_2mm_native_layout01_4800_v1_attempt_001`
- `go2_multiseed_navigation_seed_2026091001_full_direct_noise_2mm_native_layout01_4800_v1_attempt_001`
- `go2_multiseed_navigation_seed_2026091001_full_supervised_rollout_noise_2mm_native_layout01_4800_v1_attempt_001`
- `go2_multiseed_navigation_seed_2026091401_full_direct_noise_2mm_native_layout01_4800_v1_attempt_001`
- `go2_multiseed_navigation_seed_2026091402_full_jepa_noise_2mm_native_layout01_4800_v1_attempt_001`

Inventory: `.generated/depth_retirement_completed_multiseed_successes_2026-09-15`.
Completed: 45522 depth files, 14148472832 allocated bytes reclaimed; preserved JSON hashes match.


## Completed RGB-only goal diagnostic depth retirement, 2026-09-17

Retired 1,278 regular, single-link depth/native-depth NPZ arrays (0.872 GiB)
from completed contact-free branches 00-11 of
`go2_dense_goal_overshoot_branches_v1_attempt_001` and branches 00-01 of
`go2_dense_metric_late_turn_v1_attempt_001`, both under the dedicated
`/mnt/steam_drive/LeWMQuad-v3/navigation_development_artifacts_v1` root.
These RGB-only diagnostics and their follow-up scoring are complete. Active
fitting/evaluation uses no diagnostic depth. Preserve every non-depth record,
all failed trials, and all training data/checkpoints. Exact historical depth
replay now requires regeneration. The prior user instruction authorizes cleanup
of unused regenerable depth. Exact file names, bytes and hashes, plus completion,
are recorded in `.generated/depth_retirement_dense_goal_diagnostics_2026-09-17/`.
