# Broader training views for the frozen-feature motion readout

The completed physical branch diagnostic showed that the original motion head
decoded a real left turn with the wrong sign even from actual future features.
A geometric decoder recovered actual-future yaw much better, but prediction
errors remained too large for navigation. This experiment tests whether broader
training views improve the learned readout's transfer, with extra optimization
controlled. The pretrained encoder and both action/blind predictors stay frozen.

## Completed collection

Eight recordings use only the original two training geometry clusters, both
openings and both appearance seeds. The physical geometries and source role
assignments are unchanged. The navigation renderer includes the robot; the old
training renderer disabled its visualization. A matched quiet-start comparison
of frames 0, 5 and 10 in all eight environments found small primary-camera
differences in all 24 pairs: 3,218–5,568 pixels differed per image, with mean
absolute channel differences of 0.032–0.074 on the 0–255 scale. These differences
do not establish robot visibility as their cause or explain the navigation
failure. Added heading coverage and renderer alignment remain a combined
intervention, not independently isolated. The measured comparison is retained
in `docs/go2_full_heading_quiet_start_rgb_comparison_2026-09-18.json`.

Each fixed 36.8-second tape contains six heading stages, quiet intervals, both
full-duration turns, and short forward/arc pulses. This is training excitation,
not closed-loop navigation. Two processes used CPU groups 4–7 and 8–11.

| Collection result | Value |
|---|---:|
| Complete recordings | 8 / 8 |
| Camera pairs per recording | 369 |
| Completed 500-ms training windows | 2,832 |
| Disallowed contacts / physical stops | 0 / 0 |
| Occupied 30° heading bins in each recording | 12 / 12 |
| Unwrapped heading span | 6.028 radians (about 345°) |

The physical trajectories repeat across the visual conditions; these are not
eight independent dynamics demonstrations. All completed pre-contact windows
with departure frame at least ten are included. RGB, physics, command histories
and depth hashes are retained; raw depth arrays are omitted.

Collection root:
`.generated/navigation_development_artifacts_v1/go2_full_heading_training_v1_attempt_001/`.
Its `plan.json`, `result.json` and `samples.json` identify the complete population.
The collector is `scripts/collect_go2_full_heading_training_development.py`.

## Completed matched fit

| Treatment | Shared half-batch | Other half-batch | Updates |
|---|---|---|---:|
| Old-data continuation | 32 original examples | 32 original examples | 440 |
| Mixed-data continuation | Same 32 original examples | 32 new examples | 440 |

Both arms start from the same original motion-head checkpoint, retain its target
normalization, and use the same AdamW settings and batch size 64. Every original
example appears in both arms; every new example appears in the mixed arm.
Only fixed final checkpoints are retained. No exposed maze data enters fitting.
The old pool has 3,518 examples. Encoder feature extraction covers 6,492 unique
image files; the pooled FP16 cache occupies about 2.78 GB in RAM and is not saved.
The collection left about 2.4 GiB workspace space free; final heads are small.

Fit and evaluation scripts:
`scripts/train_go2_full_heading_readout_development.py` and
`scripts/evaluate_go2_full_heading_readout_development.py`.

Fit root:
`.generated/navigation_development_artifacts_v1/go2_full_heading_readout_v1_attempt_001/`.
Session 81869 completed the fit and fixed stall-branch evaluation with exit 0. The fit
started as PID 102310, creation time 1789700961.07. Check the live handle/process
and terminal artifacts; do not restart merely because a polling call times out.

The predeclared transfer readout compares original, old-data and mixed-data
heads on actual/action-predicted/blind-predicted features at 300/500/700/800 ms,
with 500 ms the supervised endpoint and 300–700 ms the planning interval.
This remains one exposed diagnostic context. The four prospective mazes are
untouched. Both heads completed 440 updates; total fitting job time was
1,371.94 seconds, of which 1,359.9 seconds encoded images. Evaluation took
7.22 seconds and reproduced the original outputs within 2e-5.

| Head | Actual-future XY / yaw at 500 ms | Predicted-future XY / yaw at 500 ms | Predicted-future XY / yaw at 700 ms |
|---|---:|---:|---:|
| Original | 29.09 mm / 3.36° | 30.83 mm / 2.91° | 33.05 mm / 7.56° |
| Old-data continuation | 30.11 mm / 2.58° | 30.82 mm / 2.62° | 34.35 mm / 6.78° |
| Mixed-data continuation | 14.21 mm / 0.98° | 13.43 mm / 2.79° | 14.87 mm / 6.01° |

These are RMSE across hold/left/right branches at one context. Broader data
improves position decoding substantially beyond extra old-data optimization;
predicted-feature yaw at the supervised endpoint does not beat old-data
continuation. Over the 300–700-ms execution interval, actual left/right yaw is
+9.17°/−9.24°. Action-predicted features decode to −0.76°/−1.10° with the original
head, +0.45°/−1.38° with old-data continuation and +3.43°/−1.06° with mixed data.
The mixed head fixes the left sign but still severely underestimates turns.
Its actual-future interval estimates are +4.25°/−0.05°, so readout transfer
remains a problem even with perfect future features. At 500 ms the mixed
predicted-feature right turn also has the wrong sign (+0.69°, actual −3.92°).

The next experiment uses the fixed mixed-data head on the same exposed full
maze, with all other controller/model settings unchanged and RGB-only depth
retention. The position improvement and corrected left interval justify
testing whether the earlier hold stall changes; success is not assumed.
This is development selection and does not isolate JEPA training. No
navigation benefit is established by these branch results.

Checkpoint SHA-256:
- Old-data: `81a09f36e90485132264214bf3cb02106b515fc0260990628fe149ca19ff1ecc`
- Mixed-data: `bbbb05fd2e2984ac4d818abc986ec0401e24e9d91f53ee4e5b49443b832bdf85`

The existing navigation runner now accepts
`--readout-arm {original,old_data,mixed_data}`. It defaults to the original head;
continued heads use distinct recording roots and their checkpoint identities
are recorded in the launch and physical readout. Loading a continued head
requires its completed fit result. Syntax and CLI import checks passed; no
continued-head native inference or navigation has run yet at this entry. The
completed transfer comparison above selects the mixed head for the next run.

The mixed-head full mission has now launched in session 54924, owner PID 105759,
creation time 1789702413.3. Its root is
`.generated/navigation_development_artifacts_v1/go2_dense_horizon_untimed_action_mixed_data_readout_exposed_maze_full_v1_attempt_001/`.
The launch records the mixed checkpoint above and unchanged action predictor
`5d39753fbdc7714c60bf8c24c1757fd0ffbbdfe9485d776d65b5c3caa9fbb7ca`.
The same session runs the physical reader after successful owner completion.
Mission outcome remains pending; verify the live handle or process and terminal
artifacts before attempting any restart.

Live mission evidence subsequently recorded an observed outbound arrival at
frame 2680 (269,500,000,000 ns), position [−0.0047142, 1.3117751] m relative to
the initial body frame, distance 0.0126837 m from the goal, after ten quiet
intervals. The mission switched to RETURN. This is explicitly not yet a native
physics-verified arrival or completed round trip; the owner and physical reader
are still pending. It must not be substituted for the terminal physical result.

While that mission runs, session 13313 evaluates all 35 existing exposed pilot
windows on CPU cores 4–7, separate from native navigation's CPU/GPU allocation.
`scripts/evaluate_go2_full_heading_readout_pilot_development.py` reuses the
completed correlation evaluator with the original, old-data and mixed-data
heads. It encodes 144 retained images, compares actual/action/blind future
features at 300/500/700 ms and the 300–700-ms interval, and checks reproduction
of the original online 500-ms forecasts. Results will be under
`go2_full_heading_readout_v1_attempt_001/pilot_evaluation/`. This is an additional
fixed diagnostic across the earlier pilot trajectory, not independent maze
evaluation, new training or a change to the ongoing navigation assignment.

### Completed pilot comparison

Session 13313 completed with exit 0 in 284.9 seconds on CPU. All 35 common
executed windows were evaluated, and the original online 500-ms forecasts were
reproduced within 2e-5. The windows overlap and mostly cover turning/holding in
one previously exposed maze; they are not 35 independent generalisation trials.

| Head | Actual-future XY / yaw at 500 ms | Action-predicted XY / yaw at 500 ms | Action-predicted XY / yaw over 300–700 ms |
|---|---:|---:|---:|
| Original | 44.30 mm / 8.32° | 37.16 mm / 7.23° | 14.62 mm / 6.46° |
| Old-data continuation | 44.10 mm / 8.31° | 37.72 mm / 7.21° | 12.63 mm / 6.74° |
| Mixed-data continuation | 21.15 mm / 5.45° | 18.83 mm / 5.56° | 10.07 mm / 6.67° |

The position improvement extends beyond the single stalled-state diagnostic
and exceeds the effect of extra old-data optimization. Endpoint yaw improves,
but interval yaw does not improve against the original head. At 500 ms, the
mixed head with the action-blind predictor has 17.86-mm / 6.95° errors: action
conditioning helps yaw here but does not reduce XY error. Zero-motion XY error
is only 6.76 mm on these turn-dominated windows, underscoring remaining spurious
translation estimates. Actual-future errors also remain substantial. These
results support broader-data readout transfer without establishing an adequate
physical world model, a JEPA-objective benefit or navigation success.

## Completed mixed-head mission

This section supersedes the pending language above, which was written at 05:10
while the mission was still in RETURN. Session 54924 completed at 05:30 and the
physical reader ran to completion. The mission terminal is
`OBSERVED_ROUND_TRIP_CANDIDATE`. No text above was altered.

| Mission result | Value |
|---|---:|
| Camera pairs / registered poses | 4,097 / 4,097 |
| Arrivals passing all checks | 2 |
| Outbound arrival | frame 2680, 12.7 mm observed |
| Return arrival | frame 4096, 9.6 mm observed |
| Disallowed contact samples | 0 |
| Median / maximum pose registration error | 7.18 mm / 12.19 mm |
| Neural calls / selected plans | 1,015 / 1,015 |
| Median model inference | 2,458 ms |

Both arrivals satisfied the physical distance, dwell and measured-motion-quiet
rules with all requested intervals zero. Native final distances were 18.8 mm
outbound and 1.3 mm on return. No dispatch veto of any kind fired: the recorded
reasons are 20,145 `CURRENT_NOMINAL_OBSTACLE_TEST_PASSED`, 200 `NO_ON_TIME_PLAN`
and 136 `MISSION_SETTLING_OR_TERMINAL`. Selected actions were 202 left turns,
192 right turns, 162 left arcs, 113 right arcs, 121 forward and 225 hold.

The broader-coverage readout head converted the earlier hold stall on this same
exposed layout into a completed round trip with the encoder, both predictors and
every controller setting unchanged. Only the motion readout differs. This is a
readout repair result.

It is not a demonstrated world-model benefit, for three reasons recorded here
rather than left to inference. First, the `reactive_feedback` control completed
the same round trip on the same layout at 03:44 with no forecast selecting any
command, so a completed mission on this layout does not separate forecast
quality from controller competence. Second, the forecast remains worse than
trivial references on the 1,014 matched 700-ms executed windows: 49.6 mm XY
RMSE against 10.6 mm for command history, 16.5 mm for the nominal model and
9.3 mm for the pose command. Third, 225 hold selections remain against the
reactive arm's 1.

Per action group, dense XY RMSE was 71.0 mm on 387 translation windows, 32.3 mm
on 393 turn windows, 24.3 mm on 225 hold windows and 6.0 mm on 9 translation
pulses. Command history beats the dense forecast in all groups except the
nine-window pulse group, which is too small to carry a claim. The windows
overlap and are not independent.

The physical record marks `raw_sensor_audit_complete`, `host_real_time_qualified`
and `real_sensor_uncertainty_calibrated` all false, and the run is
`native_state_evaluator_only`. This is one run on the exposed layout, which has
now been used for both diagnosis and selection and therefore cannot carry a
generalisation claim. The four prospective mazes remain untouched.
