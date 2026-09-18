# Combined perception and motion-prediction comparison

## Completed result

### Transient floor-rejection follow-up

The native follow-up has completed: 4,805 camera frames, 2,912 accepted floor
poses and 1,893 explicit floor rejections. There were no arrivals or contacts;
the mission exhausted its budget. Median/maximum accepted-pose errors were
12.842/24.320 mm and path length was 7.730 m. All four standard evaluations
are saved. Owner exit was 0 after 11:11.14 including recording, with no swap.

The recovery mechanism activated at frame 1362, published at 137.812 s. It
later released 45 times after valid streaks, but no nonzero command was
requested after that first rejection. The last nonzero request was 137.240 s.
Post-rejection requests comprise 10,885 floor holds, 998 missing-plan holds,
291 unavailable/stale-observation vetoes and 5,031 latched-window vetoes.
The saved `floor_reacquisition_activation_diagnostic_v1.json` records every
release frame and these counts. Recovery prevented a terminal registration
latch; it did not restore navigation. Retain this failed mission in full.

The comparison report and PNG/SVG are under
`go2_combined_floor_reacquisition_layout02_summary_v1_attempt_001`; 145 common
source bindings and shared settings match, with registration as the intervention.

The complete original layout-2 pose/command recording was replayed through
the existing `ReacquiringFloorRegistration`. All 994 original accepted raw and
registered pose pairs matched. Frame 994 was explicitly unavailable; frame 995
was accepted with unchanged measurement limits. All 996 captured frames were
consumed, 995 accepted, and the registration failure latch was clear at the end.
The recording ends there, so sustained recovery and the required four-frame
valid streak cannot be inferred. No controller, physics or native pose was used.
Output: the original run's `floor_reacquisition_replay_v1/result.json`.

Test this existing recovery behavior in one native follow-up on that same
exposed layout-2 pose/command assignment using
`scripts/run_go2_combined_floor_reacquisition_development.py`. Output root:
`go2_combined_floor_reacquisition_pose_command_noise_2mm_native_layout02_4800_v1_attempt_001`.
Keep its original tracker, camera-view policy, fitted pose/command XY, command
yaw, model/fits, sensor/physics settings and mission budget. Add only the existing
hold/reacquire registration/runtime: rejected poses do not reach mapping or
arrival detection, pending commands are vetoed, four accepted poses are required
before new planning, and the global budget keeps advancing. The independent
obstacle observer remains unchanged, including its veto when a camera has no
usable cloud. One fixed assignment; retain any failure or nonactivation. Measure
actual recovery commands and physical arrivals after owner exit.

This native follow-up is running in session 68393, owner PID 3642681, on CPUs
0–7,16–23 alongside the separate local-reference maze-3 experiment. Its actual
launch confirms pose/command XY, command yaw, reacquiring registration, original
views, no local-reference bank and one planned assignment. No final outcome is
available yet.

The layout-2 pose/command floor conflict has now been reproduced from the
retained delivered noisy pixels using the unchanged coherent-refresh tracker.
All 994 accepted raw poses matched before rejection at frame 994. Results are
under that run's `recorded_floor_conflict_replay_v1/`; the generalized diagnostic
is `scripts/diagnose_go2_current_plane_reactive_floor_conflict_development.py`
with `--tracker coherent_refresh` and the layout-2 root name.

At rejection the primary camera had zero valid pixels and the auxiliary had
176,548. The selected auxiliary floor population contained 2,463 points;
its second covariance eigenvalue was 0.000356906 m², below the unchanged
0.0004 m² full-plane extent requirement. Its own fitted-plane maximum residual
was 2.997257 mm. Against the height-adjusted transported normal, eight points
exceeded 3 mm, with maximum 3.036749 mm and RMS 1.533546 mm. The two normals
differed by 0.004890 rad. The floor anchor was only one frame old (993).

This localizes one failure to partial-view floor transport and differing normal
estimates, rather than a missing current raw visual pose. It does not prove
that these points belong to the physical floor or justify changing acceptance
limits. No controller or physics was executed, native pose was unused, and
all thresholds/candidate rules were unchanged. Replay took 113.55 seconds.

All eight fixed assignments completed and were independently evaluated without
runtime/model/fit changes. Every failure remains in the reported population.

| Motion source | Verified goals | Verified round trips | Disallowed contacts |
| --- | --- | --- | --- |
| Learned corrected XY + learned yaw | 1/4 | 1/4 | 0 |
| Fitted pose/command XY + command yaw | 3/4 | 2/4 | 0 |

The common 145 source bindings match across all four pairs; actual final scored
XY/yaw/contact channels match the assignments. All pair reports and trajectory
PNG/SVGs are saved. The final aggregate is
`go2_combined_perception_motion_four_layout_summary_v1_attempt_001/result.json`.
No learned-model advantage, general reliability, JEPA-specific benefit,
wall-clock qualification or hardware validation is established. The four mazes
are fresh within the explicit development registry, from the same maze family.

Final pose/command layout 3 completed verified goal/home dwells at frames
3069/3936: maximum native distances 35.856/26.023 mm, zero contacts, 3,938
accepted poses. Median/maximum position errors were 31.154/36.077 mm. It
travelled 30.643 m; 949 of 977 selected plans were on time. Owner wall time,
including recording, was 9:35.27. The paired learned layout-3 goal claim remains
rejected by physical evaluation.

The dominant unresolved problems are long approach/view/recovery cycles,
accumulated localization drift and partial-camera floor-reference rejection.
Proceed with the separately documented committed-view follow-up on exposed
layout 1. Keep all eight full recordings as current diagnostic inputs.

## Fixed design and collection history

Fix eight native assignments before collection: learned motion and pose/command
motion on each of four fresh development mazes. Both arms combine the accepted
400 ms reference-image refresh and camera-based frontier viewpoints, while
retaining the same noisy paired cameras, gyro, mapping, persistent routing memory,
six actions, terminal pulses, physical guards and 4,800-tick mission budget.
No training, correction refit or outcome-dependent maze selection occurs here.

The learned arm scores the frozen supervised network's XY after its existing
frozen residual correction, and its learned sine/cosine yaw. The simpler arm
scores the separate frozen pose/command XY fit and integrated-command yaw.
Contact scoring is disabled in both. Both arms compute the network and both
prediction alternatives. This is a motion-prediction comparison conditional on
predictive planning and persistent routing memory; it is not a model-free,
reactive, JEPA-training or memory ablation. Learned XY still includes a fitted
residual correction. The earlier evidence establishes no JEPA advantage.

Inventory: `docs/go2_combined_perception_motion_layout_inventory_2026-09-15.json`,
SHA-256 `b1ab545f649d0efc8a9f406418a5cc454da70a18aa434c029ce99ae45d0cef4e`.
Construction seed 2026091541 selects the first four eligible mazes among six
candidates. All four topologies and grid embeddings differ from one another and
from the explicit 68-layout development registry. They remain the same general
16-cell maze family; this does not establish out-of-family generalization or
sealed evaluation. Physics seeds are 2026097800–03; appearance 2026097900–03.
Only public goal/home mission values and sensed observations reach the policy;
scene structure belongs to native construction and evaluation.

Fixed first-source order by layout: learned, pose_command, pose_command, learned.
Start layout 0 learned and layout 1 pose_command concurrently, then their paired
counterparts on the same CPU groups after each owner finishes recording and
physical evaluation. Repeat on layouts 2/3 in the stated order. At most two
native owners including archiving run concurrently; layout index parity keeps
its original physical CPU group. Preserve every assignment and failure. Freeze
runtime/model/fit settings through this cohort; any intervention is separate.

Source: `lewm/combined_motion_source_development.py` and
`scripts/run_go2_combined_perception_motion_development.py`.
Arguments: `--layout-index 0|1|2|3 --motion-source learned|pose_command`.
Root pattern: `go2_combined_perception_motion_<source>_noise_2mm_native_layoutXX_4800_v1_attempt_001`.
Eleven focused motion-source, camera-view state and reference-refresh tests
passed in 1.90 s. The motion tests check actual final channels, input preservation
and independence from neural motion in the simpler control.

After each owner exits, independently evaluate physical arrivals, contacts,
failures and executed XY/yaw forecasts. Compare actual applied prediction
channels, fixed models/fits and shared runtime settings across pairs; retain
all eight outcomes. Existing seven-success XY depth retirement leaves about
19.6 GiB free before collection. Keep all current recordings; review only older
eligible depth if later batches need more room. Measured-simulation timing,
synthetic sensor noise and ideal gyro remain explicit limitations. Multiple
training seeds, wall-clock qualification and hardware evidence are outstanding.

The first assignments launched concurrently: learned layout 0 in session 91169,
owner PID 3628498, and pose_command layout 1 in session 27590, owner PID 3628497.
Their actual launch records confirm both camera-based viewpoints and accepted
reference refresh, learned XY/learned yaw versus fitted XY/command yaw, the
frozen fresh inventory hash and eight planned assignments. Public goals are
(2.6,2.6) and (3.9,2.6), respectively. The paired counterparts remain queued
until each corresponding owner finishes its archive and physical evaluation.
The first learned layout-0 owner subsequently exited with a physical speed
guard stop: final recorded 3-D speed 0.3012059054 m/s exceeds the unchanged
0.3 m/s guard. It recorded 244 camera pairs, 243 registered poses, no arrivals
and zero disallowed contact samples. Maximum registered position error was
2.237 mm. All four standard arrival/navigation/forecast evaluations are saved.
This failed assignment remains in the fixed cohort. Its layout-0 pose/command
counterpart launched in session 62501 after owner exit and evaluation; the
original layout-1 pose/command owner remains running in session 27590.
Keep runtime/model/fit sources unchanged through this fixed cohort.

## First three completed assignments

| Layout | Motion source | Verified goal | Verified return | Disallowed contacts | Outcome |
| --- | --- | --- | --- | --- | --- |
| 0 | learned | no | no | 0 | 3-D speed guard at 25.892 s |
| 0 | pose/command | yes, frame 1816 | yes, frame 2309 | 0 | completed |
| 1 | pose/command | yes, frame 4188 | no | 0 | mission budget exhausted |

All three owners exited and all four standard evaluations are saved. Layout-0
control's physical quiet-dwell maximum distances were 6.615 mm at the goal and
10.266 mm at home; maximum pose error was 5.959 mm. Its 561 selected plans
included 519 on-time plans. Owner wall time including recording was 5:34.78.
The learned layout-0 failure occurred during a 0.2 m/s forward command: terminal
horizontal speed was 0.284268 m/s and vertical velocity -0.099583 m/s, producing
0.301206 m/s total speed. No camera tracking failure or disallowed contact was
recorded. Planar XY/yaw forecasts do not represent this vertical gait motion.
The guard and failed outcome remain unchanged.

Layout-1 control's goal quiet-dwell maximum distance was 39.178 mm, within the
fixed 40 mm physical criterion. It ended 25.028 mm from home, without a completed
return dwell, so this is not a round trip. It travelled 34.156 m; median/maximum
pose errors were 17.129/23.832 mm. Of 480.44 s of completed command intervals,
264.42 s were turn-only, 168.22 s translating and 47.80 s zero commands.
Twenty frontier events ended with the requested patch observed; one ended
because the viewpoint route was no longer available. These records suggest
exploration/turning efficiency and accumulated localization error deserve
attention after the fixed cohort; they do not isolate the cause of the budget
failure. The 1,123 executed 700 ms windows had fitted-XY RMSE 5.824 mm; commanded
yaw RMSE was 2.299 degrees versus the saved neural alternative's 3.556 degrees.
These overlapping conditional forecast windows establish no alternative
navigation outcome. Owner wall time including recording was 12:06.92.

The completed layout-0 pair is in
`go2_combined_perception_motion_comparison_layout00_v1_attempt_001/result.json`.
Actual final scored XY/yaw/contact channels and common source/settings match
the assigned comparison. The analysis preserves the failed run's final partial
20 ms command separately from completed command durations; its partial physical
execution remains in the native trace. The pair includes a trajectory PNG/SVG.
One pair favors the simpler controller, but this is not a reliability or
statistical superiority result. Both arms still use predictive planning and
persistent routing memory; JEPA training is not isolated here.

Next live assignments: learned layout 1 (session 94579) and pose/command layout 2
(session 75746), using their original CPU groups. Three assignments remain
unstarted: learned layout 2, learned layout 3 and pose/command layout 3. Finish
these fixed assignments without changing runtime/model/fit settings.

## Five assignments evaluated

Learned layout 1 subsequently completed a physically verified round trip:
2,393 poses, goal frame 1793 and home frame 2391, zero disallowed contacts.
The goal/home quiet-dwell maximum distances were 17.506/16.312 mm; median/maximum
pose errors were 6.607/12.090 mm. It travelled 27.364 m, with 566 of 590 selected
plans on time. Owner wall time including recording was 6:00.20. Its matched
pair and PNG/SVG are saved under
`go2_combined_perception_motion_comparison_layout01_v1_attempt_001`.
The first two paired mazes therefore split the round-trip outcome: the simpler
predictor succeeds on layout 0 and learned motion succeeds on layout 1.
This is not evidence of superiority for either arm.

Pose/command layout 2 stopped with a registration-stage rejection:
`current measured candidate conflicts with transported floor reference`.
The outer owner error was `runtime stopped`; the registration fault is preserved
in `pipeline_faults.json`. It recorded 996 camera pairs, 994 accepted poses, no
arrivals and zero disallowed contacts. Maximum pose error before rejection was
8.781 mm. All four standard evaluations are saved; the complete failed sensor
recording remains available for diagnosis. Owner wall time was 2:52.88.

The layout-1 control's `long_camera_view_episode_v1.json` identifies one pending
view task from 57.9 to 270.3 seconds, targeting unknown cell (21,20) from known
viewpoint (1.375,0.625). During its 212.4 seconds, 531 selected plans included
321 approach plans, 107 directed-view plans and 103 translation-veto recovery
plans; 493 selected actions were pure turns. The patch eventually became
observed. `pose_drift_profile_v1.json` shows gradual mostly-X error growth from
roughly 2 mm to 20 mm during this interval, rather than one large registration
jump. These are saved-behavior diagnoses; neither establishes an alternative
trajectory or the cause of the repeated approach/view/recovery transitions.
Keep this full sensor input for the follow-up investigation.

Current live owners: learned layout 2 (session 16770) and learned layout 3
(session 67240). Only pose/command layout 3 remains unstarted. The first five
outcomes and the two completed pair comparisons are retained. No runtime,
model, fit or physical-guard changes have been made during the cohort.

## Remaining learned assignments and targeted diagnosis

Learned layout 3 recorded an observed goal at frame 2720, but independent native
physics rejects it: the entire goal dwell was 42.926–45.621 mm from the target,
outside the fixed 40 mm radius, despite an observed error of only 1.247 mm.
The run then failed registration during its attempted return with
`current measured candidate conflicts with transported floor reference`.
It recorded 3,355 camera pairs and 3,353 accepted poses, no verified arrival and
zero disallowed contacts. Median/maximum pose errors were 44.344/53.347 mm.
Owner wall time including recording was 9:00.80. All four evaluations are saved.
This is a localization-driven false arrival claim as well as a later tracking
failure; it must not count as a verified goal or successful navigation.

Learned layout 2 finished its 4,800-tick budget without any observed arrival.
The owner exited cleanly after 11:24.48 including recording. Its independent
evaluation follows. Pose/command layout 3, the final fixed assignment, launched
in session 6347 after the learned layout-3 owner exited and was evaluated.
All eight assignments have now launched; no runtime/model/fit changes occurred.

The pose/command layout-2 tracking replay matched all 994 recorded raw poses and
reproduced the exact floor-reference rejection at frame 994 (123.03 s replay).
`terminal_depth_coverage_v1.json` shows primary valid pixels falling from 106,050
at frame 990 to zero at 993–995; auxiliary valid pixels at the failure were
176,548. The delivered noisy packets were reconstructed and hash-checked. The
recording remains retained. This establishes failure under partial camera
coverage; it does not prove a corrected tracker or justify relaxing its checks.

The completed control layout-1 view-loop replay and a separate, unexecuted
successor are documented in `docs/go2_committed_camera_frontier_view_2026-09-15.md`.
It matched all 531 episode planning map counts across 671 mapping updates and
identified 53 view/approach gate switches, with zero aligned view plans. Five
focused tests of the separate committed-view successor pass. No current cohort
assignment imports or uses that successor; its live follow-up waits for the
fixed comparison to finish.

Learned layout 2 is now independently evaluated: 4,805 accepted poses, zero
arrivals and zero disallowed contacts. Median/maximum pose errors were
29.799/51.918 mm. It travelled 13.785 m; 1,176 of 1,200 selected plans were on
time. Both layout-2 assignments therefore failed before a verified goal, with
different terminal outcomes (learned budget exhaustion; control registration
rejection). The completed pair and trajectory figures are under
`go2_combined_perception_motion_comparison_layout02_v1_attempt_001`.

Seven assignments are evaluated, and only the final pose/command layout-3 owner
remains live. The four learned assignments yielded one verified goal/round trip
out of four; all observed-arrival claims are checked against native physics,
including the rejected layout-3 claim. Final control totals await its last run.
Pose drift above 50 mm on learned layouts 2 and 3 remains a material limitation
even where the tracker continued producing accepted poses.
