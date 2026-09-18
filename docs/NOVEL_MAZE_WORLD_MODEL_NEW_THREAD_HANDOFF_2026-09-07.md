# New-thread handoff: novel-maze navigation through a learned world model

Prepared 2026-09-07 for the user's explicitly requested new thread and goal.
This is an operational handoff, not a successful scientific result. Paths below
are relative to the repository unless written as absolute paths.

## 1. Paste this instruction into the new thread

> Work in `/home/andrewknowles/Workspace/LeWMQuad-v3`. Read `AGENTS.md` fully,
> then read `docs/NOVEL_MAZE_WORLD_MODEL_NEW_THREAD_HANDOFF_2026-09-07.md`.
> Set a new long-term goal using the goal statement in section 2, and iterate
> autonomously towards it. Preserve the local uncommitted work and all failed
> experiments. Start with the concrete next implementation in section 9;
> do not rerun completed diagnostics, restart the failed tracking experiment,
> repeat the unchanged learning study, or substitute more planning documents
> for implementation and experiments. Assess hardware before substantial jobs.
> Ignore obsolete BEV work, preserve sealed benchmark custody, and distinguish
> learned-world-model planning from a learned low-level gait or scripted motion.

The original thread has an unfinished goal marked historically `blocked`.
The available goal tools refused a new goal because the old one is unfinished,
and expose no reactivation/replacement operation. We did NOT falsely complete
it. This is why the user is moving to a new thread. Do not import the old goal's
blocked status into a new goal or spend another iteration trying to replace it
from the old thread. No token budget was requested.

## 2. New goal

> Demonstrate reliable closed-loop navigation of previously unseen mazes on the
> Go2 quadruped using a learned action-conditioned world model, with RGB and
> deployment-valid additional sensors, to evaluate candidate actions and select
> commands online. Establish continuous local execution, observation-grounded
> exploration, useful memory and physical backtracking, and verified goal-reaching
> on independent maze layouts. Compare the world-model planner with matched
> reactive and non-predictive baselines; distinguish the contributions of JEPA
> training, online predictive planning and persistent memory. Iterate through
> diagnosis, implementation, verification and prospective experiments, assessing
> hardware and useful parallelism before large jobs. Preserve every failure and
> sealed benchmark custody. Completion requires verified end-to-end navigation
> evidence, not component tests, prediction loss, fixed command tapes or
> retrospective rescoring. Declare the simulation/hardware validation level
> honestly; realistic sensing, timing and bounded real-platform evidence remain
> necessary for deployment claims.

An engineered planner using a learned world model can satisfy the model-based
navigation architecture; it is not an end-to-end learned policy. Do not impose
policy distillation or model-free RL as an unrequested prerequisite. Conversely,
merely attaching a JEPA encoder to a hand-coded controller does not demonstrate
world-model-based action selection. A JEPA advantage is an experimental
hypothesis, not a result we can guarantee or manufacture.

## 3. First five minutes: actual state and custody

- Repository: `/home/andrewknowles/Workspace/LeWMQuad-v3`.
- Branch: `jepa-spatial-world-model-nav`.
- Current HEAD: `97f78427bfa24d90dfa82ebf806ce6714d3023fd`
  (`Record approved environment removal and restored tracking capacity`).
- Remote recorded during cleanup: `git@github.com:500mhzClub/LeWMQuad-v3.git`.
  This handoff did not fetch or push. The prior cleanup turn verified its push.
- No tracked-file modifications were shown by the final status inspection.
  **Eleven new source/test/result/plan files are untracked**, listed with hashes
  in section 12. They are real completed local work, not disposable files.
  This handoff is an additional new file. A fresh clone alone will miss them.
- No Python workload was running at the handoff inspection. Old session IDs
  below are terminal historical identifiers, not handles to poll in a new thread.
- The local checkpoint is
  `.generated/navigation-development-staging.m6MDz1/autonomous_checkpoint.json`.
  Its top `tracking_terminal_precision_diagnostic_current_work` entry is current.
  Older entries contain stale live jobs, missing-space warnings and obsolete
  deletion-approval requests. Their explicit precedence labels matter. Some
  intermediate test/session wording even inside the top entry is historical;
  this handoff and the saved final reports give the latest completed state.
- Original scientific document found at
  `/home/andrewknowles/SAINTS_Year_1_Progression_Document_final-1.pdf`,
  SHA-256 `ad793ae83e98e43b790aabbc5682de0b1690b80f84bed17eeafee7ccb111be32`.
  This handoff located and hashed it, but did not newly reinterpret its text.
  The user originally called it the saints final progression document.

Read the actual 900-line `AGENTS.md`; this summary is not a replacement.
Last verified SHA-256:
`0a82a695585d2a06e3e287abd309dcdf0e3dcad76b7b8b66ad07b10f87973858`.
`.ignore` SHA-256:
`b8cec05a9d3a6f7917a042e0067ba33e2acfa03c2d29c8e698b365c0ae6350ba`.

Essential restrictions:

- Never open, parse, summarize, index or recursively search `sealed_test.json`,
  `sealed/`, or `sealed_*/`. Legacy V4 is development-only and permanently
  ineligible for final evaluation even though old sealed roles remain inaccessible.
- Ordinary discovery uses `rg`/`rg --files` respecting `.ignore`. No `rg -u`,
  `rg --no-ignore`, `git grep`, broad recursive grep or equivalent bypass.
  Tools not honoring `.ignore` need explicit protected-path exclusions.
- No whole-tree archive/export/copy/worktree materialization across custody.
  Narrow source-export exceptions in AGENTS are not runtime permissions.
  No source export is needed for the immediate work.
- Preserve frozen source, original attempts, negative outcomes and masks.
  Use distinct new analysis/implementation files, not runtime monkeypatches to
  make the old experiment pass. Do not relax a gate because a rejected example
  appears favorable. Apply justified changes prospectively or label post-hoc
  analysis explicitly.
- Local edits use `apply_patch`. No destructive Git commands. No extra deletion
  is authorized. No external messages, hardware motion or broad mutations are
  implied by the goal. Follow the new thread's applicable agent/tool rules;
  hardware parallelism is not by itself authorization for parallel sub-agents.

## 4. Environment, hardware and storage

Working interpreter:

```text
/home/andrewknowles/Workspace/LeWMQuad-v3/.generated/venvs/genesis_rocm_0_4_6_v1/bin/python
```

The corresponding installed environment is under:

```text
/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/venvs/genesis_rocm_0_4_6_v1
```

Run repository Python from the repo root with:

```text
PYTHONDONTWRITEBYTECODE=1
PYTHONHASHSEED=0
PYTHONPATH=.:lewm_genesis:lewm_worlds
OMP_NUM_THREADS=1
MKL_NUM_THREADS=1
OPENBLAS_NUM_THREADS=1
```

The historical native tracking experiment additionally fixed
`LIBGL_ALWAYS_SOFTWARE=1`, `PYOPENGL_PLATFORM=egl`, `EGL_DEVICE_ID=2`.
Do not blindly apply/change these for a future GPU experiment; establish and
freeze an appropriate numerical/device setup before that new run.

Hardware: Ryzen 9950X3D, 16 physical / 32 logical CPUs; about 91 GiB usable RAM;
Radeon AI PRO R9700 with 34,208,743,424 B VRAM, plus an approximately 2 GiB iGPU.
Latest substantial-work preflight observed 77 GiB RAM available, about 98% CPU
idle, discrete GPU 5% utilization and approximately 1.25 GB VRAM used. These are
historical measurements, not reservations. Recheck before large jobs.

Artifact volume last had about 106 GiB free. Workspace volume had about 20 GiB
free and was displayed as 100% occupied due to rounding. Home and RecoveryStorage
share the same approximately 1.8 TiB filesystem; the approximately 3.7 TiB workspace
is separate. Moving data within the home filesystem does not free space.

Standing user rule: read `docs/hardware_utilisation_rule_2026-09-07.md`.
Assess CPU topology/affinity, actual utilization, RAM, GPU/VRAM, competing work,
storage and throughput. Parallelize independent new workloads when beneficial;
benchmark compatibility and total throughput before choosing new device/concurrency.
The completed learning study used four CPU workers; the fixed native challenge
used one fresh worker at a time. Neither fully utilized this machine. Do not
alter a frozen attempt's concurrency to improve a utilization number.

Cleanup is DONE: `/home/andrewknowles/TinyQuadJEPA` was deleted after explicit user
approval and verified source preservation/push. It was an old environment, not
another unpreserved Git repository. Code-preservation commit verified remotely
before deletion: `0c4ae4d8c01dfa5217b1c95eba2e7d5b5a387950`. About 22.77 GiB was
reclaimed. Package inventory/activation scripts were preserved; deletion was
permanent, not a full wheel archive. Do not delete it again or delete other data.
See `docs/tinyquadjepa_environment_retirement_result_2026-09-07.md`.

## 5. Architecture: what already exists

### 5.1 Learned prediction, not a deployed high-level policy

The completed study uses `CumulativePulseRGBBodyJEPA`, not a generic pretrained
world model. Important source chain:

| Responsibility | Source |
| --- | --- |
| RGB/body/control observation encoder | `lewm/rgb_body_jepa_reference_development.py` |
| Four-packet temporal context | `lewm/temporal_rgb_body_jepa_development.py` |
| Partial action-block timing and latent rollout | `lewm/pulse_timed_rgb_body_jepa_development.py` |
| Cumulative contact semantics | `lewm/cumulative_pulse_contact_development.py` |
| Matched losses and optimizer | `lewm/cumulative_pulse_learning_development.py` |
| Position scaling | `lewm/pulse_position_scale_learning_development.py` |
| Input ablations | `lewm/independent_pulse_input_ablation_development.py` |
| Schedule training and prediction | `lewm/independent_pulse_study_runner_development.py` |
| Dataset admission/stream | `scripts/independent_rgb_body_study_data_development.py`, `scripts/independent_rgb_body_study_stream_development.py` |
| Snapshots | `scripts/cumulative_pulse_snapshot_development.py` |
| Original scientific settings | `scripts/run_go2_independent_pulse_matched_study_v1.py` |
| Completed four-worker orchestration | `scripts/run_go2_independent_pulse_parallel_study_v1.py`, `scripts/independent_pulse_parallel_fits_development.py` |
| Completed result reader | `scripts/read_go2_independent_pulse_parallel_science_v1.py` |

Actual model/data interface:

- Four chronological packets: RGB `(B,4,3,96,128)`, body histories
  `(B,4,20,63)`, control histories `(B,4,15,7)`.
- Per-packet encoder: RGB Conv2d channels 3→16→32→64→128, SiLU, adaptive 3×4
  pooling, 1536→128 projection; body GRU 63→64; control GRU 7→32; fuse
  224→latent width, SiLU, LayerNorm. Another GRU aggregates four packet embeddings.
- Trained latent width is 32, not the class default 128.
- Candidate action tensor `(B,8,5,3)`, boolean validity `(B,8,5)`: up to eight
  blocks of five 100 ms ticks. Unknown suffixes are masked zero padding, not
  assumed braking. Actual target offsets use cumulative valid ticks, including
  partial final blocks. Commands are normalized by `[.3,1.,.5]`; lateral commands
  are zero in this vocabulary.
- Timed residual latent transition: `(latent+20)→256→latent`. The 20 action-token
  inputs are 15 command values plus five validity indicators. Direct action GRU
  20→128; direct outcome MLP `(latent+129)→256→5`. Rollout outcome MLP
  `(2*latent+1)→256→5`. The heads represent four motion components and one
  contact-rate component; inspect the scaling/loss implementation for decoding.
- Cumulative contact probability integrates nonnegative softplus hazard over
  known time intervals; monotonicity is not probability calibration.
- All arms train direct outcomes and representation variance/covariance terms
  (weights .1/.01). Supervised rollout adds recursive outcome loss; JEPA also
  predicts EMA target embeddings at genuinely observed future times. Future
  observations remain target-only, absent from inference. Every arm maintains
  the EMA, but only JEPA consumes it in the latent prediction objective.
- AdamW LR .001, weight decay 0, gradient clipping 1, EMA .99; CPU deterministic
  fits with 1,200 updates, batch six, three fixed seeds
  `2026091101`, `2026091102`, `2026091103`. Final checkpoints only, no resume.
- Variants: full, no_rgb, latest_packet_only, no_candidate_command.
  Latest-packet-only repeats the final packet, including its internal histories;
  it is NOT memoryless. No-candidate retains time/validity and past controls;
  it is NOT action-free. No-RGB removes RGB from both online and future targets.
- This learned predictor's interface is RGB/body/control. Ideal RGB-D tracking
  elsewhere in the stack must not be mistaken for learned depth input here.

### 5.2 Tracking and sensing

- `lewm/multi_reference_rgbd_pose_development.py`: retained-reference RGB-D
  correspondence tracking, with `MultiReferenceRGBDPose` and
  `MultiReferenceVisualLedMotion`.
- `lewm/temporal_anchor_continuity_development.py`: `TemporalAnchorRGBDPose` /
  `TemporalAnchorVisualLedMotion` use measured previous-frame increments when
  retained anchors are missing. At most 10 bridge frames, exact 100 ms spacing,
  .02 m / .10 rad anchor-increment disagreement checks. Failure is terminal;
  no resetting/resuming an observer to improve availability. Bridge-only poses
  do not become retained anchors. These thresholds are not calibrated error bounds.
- Both paths inherit gyro-fixed rotation. Their rotational agreement is not
  independent visual heading evidence: in exact arithmetic both compositions
  follow the same integrated gyro history. No gyro-bias correction is established.
- `lewm/joint_rgbd_rigid_pose_development.py` already has a joint RGB-D rigid
  rotation fit. Do not build another equivalent solver. Earlier nominal tests
  sometimes had worse position errors; joint mode still gates against gyro.
  If testing joint-plus-continuity, mode must reach BOTH retained and incremental
  branches and the new variant should retain incremental rotation witnesses.
- Read `docs/go2_independent_heading_followup_plan_2026-09-07.md` and
  `docs/go2_tracking_continuity_and_gyro_dependency_2026-09-07.md` before that work.
- Native challenge assumptions: ideal hidden-robot 640×480 RGB-D, body sensors
  at 50 Hz, gyro at 500 Hz, physics 500 Hz, observations/commands 10 Hz. Native
  rendering near plane 5 mm versus public valid depth 0.2–5 m. These are not
  calibrated hardware sensing, physical occlusion or real-time execution.

### 5.3 Continuous execution and memory: reuse these

| Component | Existing source / role |
| --- | --- |
| Sensor-bound local target | `lewm/sensor_anchored_goal_development.py`: bind episode/frame/time/RGB/depth/pose, transform target once into uninterrupted visual frame |
| Parameterized local pulse controller | `lewm/anchored_pulse_servo_development.py`: signed turns, braking, quiet hold and bounded local goals |
| Continuous mission accounting | `lewm/continuous_pulse_execution_development.py`: pose history, failure latch, frame chronology and mission/leg budgets across legs |
| Memory/executor bridge | `lewm/pulse_route_bridge_development.py`: observed branch dispatch and actual departure/arrival/fault events |
| Physical command interface | `scripts/pulse_mission_session_development.py`: distinct .20 m/s / .45 rad/s boundary; not a fresh-scene launcher by itself |
| Whole-task navigation | `lewm/whole_task_navigation_development.py`: `WholeTaskNavigation` |
| Episodic route hypotheses | `lewm/memory/episodic_route_hypotheses_development.py`: `EpisodicRouteHypotheses` |
| Observed exploration state | `lewm/memory/observed_exploration_development.py` |
| Empirical pulse/action model | `lewm/coupled_pulse_rollout_development.py`; inspect related coupled-controller plans before changing execution |

The continuous integration report documents default 360 s / 140 pulses / 36 legs
and per-leg 100 s / 35 pulse limits. Historical local target acceptance was
6 cm / .05 rad, displacement bound .4 m and yaw bound pi. These are implementation
facts, not permission to change current experiments or proof of safe maze travel.
Local arrival is not place recognition. A .4 m branch lookahead is not a detected
corridor endpoint. An emptied route stack is not home. Never reset pose per leg,
infer travel from commanded time, or substitute evaluator topology for sensing.

Useful map: `docs/go2_continuous_pulse_memory_integration_result_2026-09-06.md`.
Its historical "next" wording is superseded where subsequent physical results
exist. Latest audited room-return result remains 0/3, not its earlier local
three-nominal-success component test.

## 6. Completed learning science and its decisive limitation

Read these existing reports, not just favorable metrics:

- `docs/go2_independent_pulse_parallel_result_and_next_steps_2026-09-07.md`
- `docs/go2_independent_pulse_parallel_scientific_readout_2026-09-07.json`
- `docs/go2_independent_pulse_parallel_scientific_details_2026-09-07.json`
- `docs/go2_training_interaction_design_result_2026-09-07.md`
- `docs/go2_independent_pulse_training_interaction_diagnostic_2026-09-07.json`

Collection: 12 layouts / 1,440 eligible trials; 1,380 schedule-complete and 60
physical-terminal recordings, no other failure population. 256 positive-contact
horizon labels, not 256 independent missions. All positive-contact horizons
lack future image and motion targets, while their contact labels remain valid.
Never invent post-stop futures or discard these contacts to obtain complete pairs.

Study: 3 seeds × 4 input treatments × 3 objectives = 36 fits, 43,200 updates,
259,200 draws. Six training / three selection / three development-evaluation
layouts. Seeds repeat the same layouts: evaluation has three layout units, not
nine independent environments. Four CPU workers, one CPU thread each, roughly
1.7 GiB peak RSS per worker. About 98.3 min excluding preflight / 107 min total.
Reader authenticated 416 artifacts and ledgers, then aggregated saved scores;
it did not independently rerun training or raw prediction scoring.

Development macro means (lower is better):

| Primary comparison | Position mm | Yaw mrad | Contact Brier |
| --- | ---: | ---: | ---: |
| Action/time |18.223|9.052|.031722|
| Full direct |21.625|17.557|.032511|
| Full supervised rollout |25.834|57.793|.031539|
| Full JEPA |22.828|55.488|.032165|
| No-RGB JEPA |18.620|47.220|.031644|

JEPA improves position by 3.007 mm against matched supervised rollout (8/9 paired
cells), but is worse than action/time; removing RGB improves its position by
4.207 mm. This is no overall RGB-JEPA advantage. Do not select the best exposed
seed or ablation as a confirmed winner. The full report contains all models,
27 contrasts, missingness and secondary metrics.

The stronger problem is task informativeness:

- All 60 development action groups have the same action/time minimum-risk set:
  actions 2–5 (turn pulses). Only 9 groups have differing contact labels, all
  near-wall. No goal progress was scored and selections were not physically
  executed by the fitted model. Turning or stopping can avoid risk indefinitely.
- The six-layout training-only diagnostic confirms the shortcut in all 120
  matched six-action groups: 102 all-zero contact vectors, six `[0,1,0,0,0,0]`,
  twelve `[1,1,0,0,0,0]`. No strict action-pair contact-order reversal within a
  history/support stratum. Adding reversals alone would also be insufficient if
  another constant action still solves every progressing task.
- Under matched action/context/history/support, observed cross-layout endpoint
  RMS variation averages .261 mm (115 observed cells; five entirely censored).
  This is descriptive label dispersion, not prediction error or isolated RGB effect.
- Largest measured turn in that collection is about 14.3°, not maze-scale
  90°/180° execution. Observed forward pulses move roughly 9–38 mm or 31–108 mm.

Do NOT run another large unchanged-task fit sweep. The successor needs actual
geometry-dependent, goal-progressing decisions and informative pre-contact
targets, not only more layout IDs or optimizer steps.

## 7. Tracking V1: exact terminal state and artifacts

Define the fixed artifact base:

```text
/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/navigation_development_artifacts_v1
```

All following root names are children of that base:

| Root | State |
| --- | --- |
| `go2_independent_rgb_body_remaining_stages_v1_attempt_001` | Completed twelve-layout collection |
| `go2_independent_pulse_parallel_study_v1_attempt_001` | Completed learning study; do not retrain |
| `go2_independent_tracking_challenge_v1_attempt_001` | Failed during full native audit after complete recording/replay |
| `go2_independent_tracking_supervision_v1_attempt_001` | Completed outside failure record, return code 1 |

Use `scripts/navigation_artifact_root_development.py`:
`validate_root`, `artifact_path`, `verify_artifacts`. These require exact owned
nonsymlink roots and ordinary explicit artifact paths. No root discovery or
recursive runtime scan is needed. Hash specified artifacts, do not search arrays.

Principal SHA-256 identities:

```text
Collection terminal:
5e39ff8b3b456578c13b878bdd12a030ffac62bd16cd257ca98f72e2e86e446d
Learning result:
588f24def6ec8810ae5a3411277576b0d965c77bf6ffdb8e18cfd80dce7b8122
Learning definition:
8d8c3456054a284aa83031ea417d8c433beddbcc04a8b47d3164f120bc0ae5d8
Learning launch:
7af8abf6be59776cee4ea5970611d28fa9daf457b98d66ceb456a7dbfa40479e
Tracking definition:
223056ac7ddcb47b9d1a4b1b188028761fb15f56443f02fda28ecfa668326744
Tracking launch.json:
ac4f532d0b8cf6e3cab4e984f7a5845ebc0ac3f8d225ed6c696a9bdc5f79c4f2
Tracking collection_complete.json:
eb94e4db44120ab5eb90c500fd4d832d5fa8bc216e83399aff56af45dbabac7d
Tracking sensor_phase_complete.json:
5c9d451f35fe64e87219234878e7445bb0285fbb5bf72ded8be9b86ff5c30348
Tracking stress_sensor_phase_complete.json:
671aa02549a748ca34af9258142fc698b920974baf529e245cd1dc691547e163
Tracking failure.json:
d0902842b1446882430093c4969ed78127a50256e0b66195207a40ce631a7486
Outside request.json:
4693ef9ccf3dbea5e11346f1a5c5bceb243108e97fba22a5e8e1b65ba557bfa1
Outside terminal.json:
0cb3f84351b721ee4eb50db1ee837655c2a1acfc376410009893871dccd0c37d
Outside unit.log:
359c8e27ceed84d6012e3935b7d12a22173674c115c86fa23a935a577e05d8a1
```

Frozen source/config document:
`docs/go2_independent_tracking_challenge_v1_frozen_definition_2026-09-07.json`,
SHA `3ba130e506f237be7c5e8bb250338e2a5d576f6f3f1bd18261df3396206bb9ce`.
It binds 825 execution-source paths plus eight additional reader/helper/test
identities. Historical "not launched" text in that document records freeze time,
not current runtime state.

Actual trial population, in order:

```text
offset_niche_nominal_left
offset_niche_nominal_right
offset_niche_lower_friction_left
offset_niche_lower_friction_right
unequal_baffles_nominal_left
unequal_baffles_nominal_right
unequal_baffles_lower_friction_left
unequal_baffles_lower_friction_right
```

Two scene clusters × friction 1.0/.15 × two turn directions, not eight independent
mazes. Starts `[-.83,-.47,.17]` and `[-.61,.39,-.23]` (world SE2 setup only).
Per tape: 750 settle samples, 442 command intervals / 443 RGB-D frames / 22,850
physics samples. All eight recording schedules completed with no reported
physical stop or infrastructure failure. Total raw bytes 5,331,919,572.

Fixed 100 ms command segments:

```text
20 hold
40 approach (forward .12 m/s)
20 brake
126 outward turn (yaw ±.25 rad/s)
20 brake
30 translated view (forward .12 m/s)
20 brake
126 reverse turn
40 final hold
```

The frozen high-level command tape does not use the observer, JEPA, friction or
native goal pose to choose actions. Native state is setup/stop/evaluation only.
The low-level gait checkpoint remains unchanged.

All eight base paired replays and 88 fixed fault replays completed. Eleven
scenarios per tape: nominal, retained-anchor absence 1/10/11 frames, current
RGB/depth/gyro unavailability, repeated RGB, depth drift, shared gyro bias and
anchor/increment conflict. Onset frame 84. Scheduled faults and actual exposure
before a terminal failure must be distinguished. These are mechanism tests, not
calibrated hardware noise or physical occlusion.

Failure:

```text
inner status: TERMINAL_TRACKING_COHORT_PHASE_FAILURE
stage: complete_stress_native_audit_or_admission
reason: ValueError('unit native quaternions required')
outside status: SCOPED_COMMAND_FAILED
systemd_run_returncode: 1
log_complete: true
log_omitted_bytes: 0
```

No `result.json` or `challenge_result.json` completing scoring was observed.
The complete predecessor comparison and original complete-result reader were
NOT completed. Do not run that reader on this failed attempt: it correctly
requires absent failure and exact successful outside/inside terminals.

Historical systemd unit:
`lewm-independent-tracking-challenge-20260907-v1.service`.
It is inactive/dead. Keeper session 99647 exited 1; original keeper/parent PIDs
2180142/2180321 are no longer live. Service runtime 1h21m18.7s, CPU 1h24m02.6s,
peak charged memory about 7.1 GiB, swap peak 0. Sampled memory-limit/OOM counters
were zero. This was an application audit exception, not an established OOM.

The historical envelope was 8 GiB cgroup memory, zero swap, 512 tasks, 48h,
5 GiB/episode, 52 GiB combined output allowance plus 40 GiB reserve (92 GiB
prelaunch free-space requirement). These are historical launch limits, not
permission to restart or a mandatory budget for every small read-only analysis.

## 8. Latest implementation and completed diagnostic results

Read `docs/go2_tracking_precision_failure_and_motion_readout_2026-09-07.md` first.
It supersedes earlier "accuracy pending because still running" messages.

### 8.1 Quaternion diagnostic — COMPLETE

- Pure function `summarize_quaternions` in
  `lewm/tracking_quaternion_precision_diagnostic_development.py` checks bounded
  nonempty XYZW arrays and complete 500 Hz clocks, compares NumPy norms with
  `math.hypot`, counts original gate failures, distinguishes settling/capture
  samples, checks binary32 representability and quantifies yaw sensitivity on
  private normalized copies. It does NOT repair data or score poses.
- Adapter `scripts/read_go2_independent_tracking_quaternion_diagnostic_v1.py`
  binds the exact terminal failure and all phase/launch/outside identities,
  authenticates 825 frozen source bindings, runs
  `stress.admit_complete_sensor_phase` for all 96 streams BEFORE native arrays,
  then reads only receipt-bound pose/time arrays for all eight tapes. It
  reauthenticates sources/raw artifacts/stream identities afterward.
- Source identity uses compact sorted JSON **plus terminal newline**. Initial
  new-adapter session 83980 failed before sensor/native admission because that
  newline was missing. Corrected only the new adapter and added a regression
  test. This was not a retried native experiment.
- Corrected session 54799 completed exit 0 in 363.04s. Do not repeat it to
  rediscover known numbers. Its durable result is
  `docs/go2_independent_tracking_quaternion_diagnostic_2026-09-07.json`.
- Across 182,800 native samples, 765 exceeded 1e-7 norm error, including 48
  settling samples and 13 camera samples. All 731,200 quaternion components
  are exactly representable as binary32 despite float64 storage.
- Largest norm error `1.1591634696550557e-7`. Independent norm implementations
  differ at most `2.220446049250313e-16`. Maximum raw-v-normalized yaw difference
  `2.1727786791991832e-7` rad (~0.00001245°).
- Recorder source `scripts/whole_task_physics_session_development.py` widens
  native quaternions and reorders WXYZ→XYZW without normalizing.
  `scripts/independent_tracking_snapshot_development.py` stacks recorded rows.
  `lewm/physical_execution_development.py:rotation_xyzw` accepts bounded norm
  errors up to 1e-5, then normalizes for sensor rotations. The coverage checker
  uses 1e-7 and raw quaternion yaw. This is a concrete interface inconsistency.
- Evidence supports a precision-interface mismatch, not grossly invalid
  orientations. It does not prove the exact native arithmetic cause, integration
  stability, full sensor validity, pose accuracy or mission qualification.

### 8.2 Separate sensor-convention coverage — COMPLETE

- `lewm/representation_aware_tracking_coverage_development.py` exports
  `coverage_with_sensor_rotation_convention`. It uses the EXISTING sensor
  rotation acceptance function; it does not invent a tolerance selected from
  these outcomes. It normalizes only a private numerical pose view, retains the
  original norm-gate outcome and invokes original coverage arithmetic plus a
  separately implemented coverage reconstruction. All motion/stop thresholds
  remain unchanged. Original sources and arrays stay untouched.
- `scripts/read_go2_tracking_sensor_convention_coverage_v1.py:admitted_inputs`
  requires the exact completed diagnostic JSON hash, reauthenticates phase,
  failure, launch, outside, all raw receipt bindings, 825 sources and all 96
  admitted sensor streams. It reuses exact-byte sensor-admission evidence instead
  of decoding the same transforms again. This reuse must be disclosed, not
  represented as independent fresh sensor reconstruction.
- `readout()` reads each complete native pose/twist/time array, computes the
  separate coverage and reauthenticates afterward. Session 69712 completed
  exit 0 in 9.98s. Result:
  `docs/go2_tracking_sensor_convention_coverage_2026-09-07.json`.

Actual descriptive post-hoc measurements:

| Scene / support / direction | Approach m | Second translation m | Outward ° | Return ° |
| --- | ---: | ---: | ---: | ---: |
| Offset / nominal / left |.2190|.1966|125.57|-162.69|
| Offset / nominal / right |.2190|.1462|-164.29|128.83|
| Offset / low friction / left |.2008|.1596|160.48|-180.56|
| Offset / low friction / right |.2008|.1388|-180.87|160.48|
| Baffles / nominal / left |.1960|.2086|126.89|-165.08|
| Baffles / nominal / right |.1960|.1372|-163.00|126.78|
| Baffles / low friction / left |.1970|.1033|164.43|-180.76|
| Baffles / low friction / right |.1970|.1002|-180.55|163.97|

Unchanged criteria: both translations ≥.20 m, both signed turns ≥150°, last
second speed ≤.02 m/s and angular speed ≤.05 rad/s. Outcome: **0/8 all intended
coverage**, 4/8 approach, 1/8 second translation, 4/8 paired turns, 8/8 stopping.
These remain descriptive coverage pending the full sensor/physics audit. They
are not tracker accuracy, independent-maze reliability, or original-attempt passes.
The clear planning consequence is that requested motion duration did not ensure
executed coverage. Fixing the norm gate alone cannot solve navigation.

### 8.3 Tests and verification boundaries

Latest combined focused suite: **23 passed in 2.57s** (terminal session 40409).
Sixteen diagnostic tests cover invalid inputs, clocks, nonmutation, population
counts, float32 reproduction, canonical identity and admission-before-native
ordering. Seven coverage tests cover independent arithmetic, unchanged no-motion
failure, stop rules, missing samples and rejection of grossly invalid quaternions.
The post-hoc reader has real-data hash/numerical checks but no separate full
independent-reader qualification. No full repository regression was run for this
latest patch. Earlier 2,420-test integration results apply to that historical
revision, not automatically to the current untracked files.

Focused verification command, safe if needed after changing these files:

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:lewm_genesis:lewm_worlds OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .generated/venvs/genesis_rocm_0_4_6_v1/bin/python -m pytest -q -p no:cacheprovider lewm/tests/test_tracking_quaternion_precision_diagnostic_development.py lewm/tests/test_tracking_quaternion_diagnostic_adapter_development.py lewm/tests/test_representation_aware_tracking_coverage_development.py
```

Do not rerun the two completed real-data diagnostics merely to produce another
progress update. Their JSON results and exact identities are already available.

## 9. Concrete NEXT implementation: complete the distinct post-hoc raw/accuracy analysis

This is the immediate unfinished task. It has NOT been implemented or launched.
Do not confuse the completed coverage reader with it.

1. Design one separately named post-hoc reader/analysis, preserving the failed
   original attempt and every scientific outcome. Reuse the exact admitted input
   evidence through `read_go2_tracking_sensor_convention_coverage_v1.admitted_inputs`
   after inspecting that function, validating its source identity and all bound
   original bytes. If any identity differs, stop rather than silently trusting
   old admission. No new training/native collection is needed for this analysis.
2. Complete raw sensor/contact/geometry/setup/command/stop/raster verification
   once per tape, NOT once per stress stream. Existing implementation map:
   - `scripts/independent_tracking_evaluation_development.py:_raw_audit`
   - `scripts/near_field_sensor_audit_development.py:audit_sensors`
   - `scripts/independent_tracking_command_audit_development.py:audit_commands`
   - `scripts/audit_go2_independent_pulse_context_pilot_v1.py:audit_setup,audit_stops`
   - `lewm/raster_footprint_visibility_development.py:evaluate_footprint`
   - `scripts/independent_tracking_native_contact_guard_development.py:validate_report`
   The original `_raw_audit` does substantial sensor/physics work BEFORE the
   rejecting coverage call. No saved completion proves all eight audits passed.
3. Introduce the justified sensor representation convention in the NEW analysis
   at the coverage boundary only. Keep raw arrays immutable and record the
   original gate failure and transformation provenance. Do not monkeypatch the
   old module's `measured_coverage` binding or overwrite original result files.
   Preserve full completeness, safety/contact, visibility and timing checks.
   If a narrowly copied adapter is required, document its exact differences
   against the frozen source; reuse existing subordinate algorithms.
4. Score every base and stress stream (8 × 12 = 96 paired streams), retaining
   failed/unavailable rows. Existing `_score_trial` in the evaluation module
   already uses SciPy normalized rotations for truth; inspect before reuse.
   New destinations must not target the old output root. Include absolute and
   incremental position/orientation, shared-support comparisons, each arm's
   separate availability and first failure, bridge/rejoin behavior, actual
   fault exposure, and observer timing with its acquisition/control exclusion.
5. Independent numerical checker:
   `lewm/independent_tracking_numerical_verification_development.py` and
   `lewm/independent_tracking_continuity_verification_development.py`.
   The numerical checker's `_native` also has a 1e-7 norm gate; do not pretend
   it checks original failing arrays unchanged. Use a distinct, explicitly
   documented representation-aware interface if needed. Two algorithms on the
   same normalized view are independent arithmetic, NOT independent native data.
6. Preserve the full original scope if describing a complete challenge readout:
   all 48 comparisons against the six explicitly exposed predecessor tapes,
   through `scripts/independent_tracking_predecessor_comparison_development.py`.
   Do not discover other legacy runtime roots. Nonidentical box/start/prefix
   evidence is not statistical independence or novel topology. If these are not
   performed, label the product as only a raw/accuracy sub-analysis.
7. Produce a compact machine-readable result and a scientific interpretation,
   with exact new source/input bindings, incomplete motion coverage and all
   negative outcomes. A distinct derived artifact root, if used, must obey the
   existing ownership/path guard and declared finite byte/memory envelope.
   Recheck hardware; benchmark independent per-tape concurrency before freezing
   this new workload. No unbounded parallel decoding and no full-tree tests
   that might collect protected material. No new audit bureaucracy as a substitute
   for actually running the complete defined analysis.

Acceptance: the new analysis either completes all of its stated population and
checks, or honestly records the precise failure. It must NEVER claim the old
frozen attempt passed, intended motion coverage was complete, or navigation was
achieved. A precision fix is allowed to expose a negative scientific result.

## 10. Ordered plan after that analysis

### A. Reliable motion from sensors, not command integrals

Use paired tracker errors/availability and measured motion to choose the next
bounded closed-loop change. Nominal signed-turn response is asymmetric and
translation undershoots the command integrals. Lower-friction response differs.
Use sensor feedback, braking and explicit support uncertainty; do not simply
lengthen every tape or choose a favorable tolerance. Preserve sensor-loss stops.
Use actual current/stored observations for targets, not native evaluator state.

Test fresh departure, substantial turns, translation and return continuously,
with both directions and stated support conditions. Define tolerances from
clearance/stopping needs and fix the population before evaluation. A nominal-only
development operating domain may be declared, but low-friction failures stay
visible and terrain-independent reliability cannot be claimed.

Latest audited continuous room return is **0/3**:
`docs/go2_inner_arrival_collection_result_2026-09-06.md`.
Nominal left/right failed tracking during TURN_BACK (ticks 1127/1003, 3/7 stages);
lower friction failed excursion at tick 565 (0/7) despite fairly accurate tracking.
Median observation/control 143–147 ms exceeded the 100 ms interval while physics
was paused. Tracking and dynamics are separate gaps; neither correct latent
loss nor candidate replay availability repairs these actual missions.

### B. A learning task that actually needs visual world prediction

This design/implementation can progress alongside execution when independent
resources permit. Build balanced branch/obstacle situations where different
actions make safe goal progress as geometry changes. Do not score permanent
stopping/turning as success. Audit fixed-action/action-time and appearance/start
shortcuts on training data before costly fitting. Cross geometry and appearance
independently; different IDs or one-pixel RGB differences are not sufficient.

Collect real approach, interaction, signed turning, braking and slip transitions
with known causal action prefixes and useful observed pre-contact futures.
Preserve censored motion/images and known terminal-event labels separately.
Keep deployment input modality choices explicit: learned RGB/body currently
differs from ideal RGB-D estimation; ablate RGB and depth separately if both are
present in a future learned input. Do not claim RGB usefulness when depth alone
or an action prior explains the result.

### C. Train and diagnose the world model

Reuse the existing encoder/temporal/partial-plan/cumulative-hazard interfaces
where suitable. Compare direct supervised prediction, supervised rollout,
JEPA, action/time and persistence using matched data/actions/observations and
training budgets, with actual compute reported. Check training curves, gradient
flow, representation variation/collapse and observation dependence before scaling.
Do not promote a winner selected retrospectively from the previous study.

Exit: meaningful prospectively specified independent-development predictive or
action-selection benefit, or a retained negative result that informs a clearly
declared successor. JEPA superiority is not assumed.

### D. Use predictions to choose and execute actions

Wire a bounded receding-horizon planner: current observed state/history →
candidate action sequences → learned predicted consequences → declared
progress/risk costs → bounded executed command → fresh observation.
Verify that model predictions actually influence commands. Compare online
rollout on/off using the SAME frozen predictor/cost/action interface. Avoid
attributing a changed controller or privileged observation to the learned model.

### E. Make memory useful during real maze missions

Reuse existing observation-grounded visit/attempt semantics. Include branch
choice, dead-end discovery, physical backtracking, target/marker observation and
home return, with repeated-looking places and UNKNOWN association allowed.
Compare memory on/off with matched sensing, gait, executor, action vocabulary,
observation opportunities and budgets. Score actual mission success, collisions,
false place/home claims, path/time cost and every incomplete mission.

### F. Establish generalization and deployment validity

Split geometry before fitting; use multiple training seeds but analyze independent
layouts as replication units. Choose layout counts from a declared meaningful
effect and development variability, not arbitrary confidence claims from three
layouts. Separately test JEPA training, online planning and memory contributions.
Keep final evaluation custodian-isolated and never inspect reserved material.

Test calibrated/realistic sensing, robot self-visibility, noise/dropout, temporal
synchronization and full-loop latency with physics continuing during computation.
Report deadline misses and safe failure behavior, not only mean latency. Physical
Go2 runs require actual access and appropriate supervision/stop procedures.
Simulation success is a milestone, not unqualified hardware success.

## 11. Stop repeating these mistakes

- Do not launch or monitor the original tracking unit again: it is terminal.
- Do not run the original successful-result reader on retained failure output.
- Do not rerun completed diagnostics or the 36 fits as a substitute for progress.
- Do not call 8 full command schedules 8 complete motion-coverage successes.
- Do not call partial tracker availability accurate tracking or safe control.
- Do not call a latent predictor a learned high-level navigation policy.
- Do not equate shared gyro agreement with independent heading/bias correction.
- Do not invent post-terminal RGB/motion targets, silently remove failed arms,
  accept only favorable surviving frames, or treat frame counts as independent N.
- Do not mutate frozen sources/attempts to simplify an analysis or force a pass.
- Do not let ever-more integrity replacements, manifests or plan files displace
  experiments that address sensor-based control and genuinely visual decisions.
- Do not turn a product goal-state limitation into a claimed scientific blocker
  when meaningful local work is available. Do not falsely complete a goal to
  make its bookkeeping convenient.

## 12. Exact new local files to preserve

These eleven files were untracked at handoff creation. Hashes below are their
actual current content identities, not Git commit IDs. They are NOT a source
export manifest or authority to copy any whole tree. The new handoff itself is
additional and intentionally not self-hashed in this section.

```text
fef3abaa997851c42da736af2428eecd31ff3f27e5e0ff7b571f883c138725f8  docs/go2_long_term_goal_execution_plan_2026-09-07.md
e1283fb8a6eb9f77082c96fa8d691ae716487fcd6f64b3ccd21357001a9b9f3c  docs/go2_tracking_precision_failure_and_motion_readout_2026-09-07.md
44f7d6d5307320d2acca00178747693dbfe109179de008bb694e8f723b3897ff  docs/go2_independent_tracking_quaternion_diagnostic_2026-09-07.json
a62f9f4d0e6998640c961a380f2bf9a5f55bd504a18ae259ddcbd32ec782e80d  docs/go2_tracking_sensor_convention_coverage_2026-09-07.json
b7912108c7827c0024a69fffb268646212557a7be64f93f1bad00d524530f923  lewm/tracking_quaternion_precision_diagnostic_development.py
2a04935198bbefa992db8e0adcdf049f68a091b7b7a63ab5915dfaaf1d8e7484  lewm/representation_aware_tracking_coverage_development.py
fb9addc99f0697e84ba37d5bbadf79779c586797af75f81479672e2d2f12bc0c  scripts/read_go2_independent_tracking_quaternion_diagnostic_v1.py
0c4681710b48ae6458551459d8f34cca2ac58bc3877f5db97f46c4c433748ae2  scripts/read_go2_tracking_sensor_convention_coverage_v1.py
65ea93a86bc94cd83f8139a7a7c191ef073e36648092da651c32b2d7a96a0b73  lewm/tests/test_tracking_quaternion_precision_diagnostic_development.py
d2a63fd24ec0c278ea8289e06e5520d254c96748bab92576cb2046182f5230f7  lewm/tests/test_tracking_quaternion_diagnostic_adapter_development.py
1e3137a1d329bc3e18fa4b000bc55e6c98e8f3c6746cd912505fc2cfbc6e17e7  lewm/tests/test_representation_aware_tracking_coverage_development.py
```

## 13. Completion/status reporting for the new thread

Report evidence at the correct level: source implemented → synthetic tests →
recording complete → sensor/raw verification → measured coverage/accuracy →
closed-loop mission outcome → matched contribution/generalization → deployment.
Do not skip levels in language even if several run in the same program.

Every substantial iteration should yield a diagnosed cause, tested implementation,
completed experiment or substantive interpretation that changes the next action.
The immediate handoff is ready for implementation; no user choice is needed to
begin section 9. Fresh hardware checks and custody compliance remain mandatory.
The user may need to supply physical access/supervision or authorize external
coordination later; do not infer that authority from this software goal.
