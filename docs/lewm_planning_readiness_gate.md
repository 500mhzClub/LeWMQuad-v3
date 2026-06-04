# LeWM Planning-Readiness Gate

## Purpose

Training loss is not enough to authorize a longer LeWM sweep or downstream
training. The training objective is teacher-forced, while local planning uses
autoregressive predicted latents. A checkpoint must therefore receive an
explicit planning-readiness review before compute moves from `seq_len=4` to
`seq_len=8`, or from `seq_len=8` to `seq_len=16`.

This is an experiment gate, not a claim that a short-context model should
produce pixel-perfect long rollouts. The immediate question is whether the
learned rollout remains more useful than simple baselines as horizon increases.

## Sequence Semantics

For a training sample with `seq_len=4`, the loader returns four frames and four
action blocks:

```text
frames:   z0  z1  z2  z3
actions:  a0  a1  a2  a3
targets:      z1  z2  z3
```

The teacher-forced loss covers three transitions. The final loaded action has no
next-frame target inside that sample. The in-training `eval_rollout_pred` metric
also covers only the first three autoregressive transitions.

The predictor can still roll farther than three steps. Its inference helper
keeps the most recent `max_seq_len` predicted latent tokens as a sliding context
window. With the current `stride=5`, one macro step is approximately `0.5 s`;
a 20-step probe covers approximately `10 s`.

## Interpreting In-Training Metrics

The epoch-end metrics from `scripts/train_lewm.py` are useful health checks, but
they are not all planning-readiness metrics.

- `eval_pred` is the held-out, teacher-forced prediction MSE. This is the most
  direct validation analogue of the trained one-step objective.
- `eval_loss` is `eval_pred + sigreg_lambda * eval_sig`. It can be dominated by
  the SIGReg term, so use it mainly as a stability check rather than as a
  rollout-quality score.
- `eval_std` is a collapse check for the projected latent distribution. Very
  low or rapidly falling values are more concerning than small loss wiggles.
- `eval_action_zero_delta` and `eval_action_shuffle_delta` are action-sensitivity
  diagnostics. They check whether predicted latents materially change when the
  command sequence is zeroed or shuffled, but they are local probes, not paper
  metrics.
- `eval_rollout_pred` is a short open-loop latent MSE over the transitions
  available inside the training sample. For `seq_len=4`, it covers only three
  autoregressive transitions.

Do not reject a checkpoint solely because `eval_rollout_pred` wobbles while
`eval_pred` is improving and `eval_std` remains non-collapsed. Short open-loop
latent MSE is sensitive to compounding error, and the raw value is hard to
interpret without a baseline. A rollout can also have imperfect latent MSE while
still ranking actions usefully for local planning.

The relevant question for planning is comparative: does the learned,
recorded-action rollout beat simple alternatives such as persistence,
zero-action rollout, and shuffled-action rollout over useful horizons? The
bounded diagnostic below answers that question explicitly. Treat
`eval_rollout_pred` as an early warning signal if it explodes or degrades
consistently, not as the approval criterion by itself.

This also differs from the LeWM paper's headline reporting. The paper's decoded
open-loop rollouts are qualitative visualizations, while quantitative results
are reported through downstream planning/probing outcomes. The scalar latent
MSE reported here is a local engineering diagnostic for this implementation,
not a directly comparable paper number.

## Diagnostic

Run the bounded long-horizon diagnostic against a checkpoint:

```bash
.generated/venvs/genesis_render_vulkan/bin/python scripts/probe_lewm_rollout_horizons.py \
  --checkpoint models/checkpoints_textured_v03_full_20260531/sweep_seq4/lewm_seq4_e9.pt \
  --data-root .generated/datagen_full \
  --render-root .generated/datagen_full/render_textured_v03 \
  --horizons 1,2,3,5,8,10,16,20 \
  --output models/checkpoints_textured_v03_full_20260531/sweep_seq4/planning_gate_lewm_seq4_e9.json
```

The report uses a deterministic random sample from the scene holdout and
records point and cumulative MSE for:

- Learned rollout with the recorded action sequence.
- Persistence: repeat the initial projected latent at every horizon.
- Zero-action rollout.
- Shuffled-action rollout.
- Actual target displacement from the initial latent.
- Actual per-step latent change.

The report also records:

- `point_rollout_over_persistence`: lower than `1.0` means the learned rollout
  beats persistence at that horizon.
- `point_shuffled_minus_rollout`: positive means recorded actions outperform
  shuffled actions.
- `point_zero_minus_rollout`: positive means recorded actions outperform a
  zero-action rollout.

## Closed-Loop MPC Analogue

The planning-relevant metric for this task is closed-loop goal progress, not raw
latent MSE. Use the MPC benchmark as the closest current analogue to the LeWM
paper's downstream planning results. The first gate should be local, not a maze
solver: the base LeWM is a short-horizon local dynamics and goal-image cost
backbone. Long-horizon topological routing belongs to the H-JEPA memory,
reachability, and sub-goal stack.

### Local Visible-Beacon Gate

This is the appropriate first closed-loop benchmark. It samples held-out
`open_obstacle_field` scenes, constructs a local approach to a directly visible
beacon, and asks whether LeWM's latent cost can drive from roughly `1.2 m` away
to the beacon standoff pose under receding-horizon replanning:

```bash
.generated/venvs/genesis_render_vulkan/bin/python scripts/benchmark_lewm_closed_loop_mpc.py \
  --checkpoint models/checkpoints_textured_v03_full_20260531/sweep_seq4/lewm_seq4_e9_b050000.pt \
  --scene-corpus .generated/scene_corpus/minimum_20260520T080420Z \
  --split test_id \
  --family open_obstacle_field \
  --scene-limit 10 \
  --trials-per-scene 1 \
  --task visible-beacon \
  --mode kinematic \
  --backend cpu \
  --model-device cpu \
  --horizon 2 \
  --max-blocks 12 \
  --goal-radius-m 0.35 \
  --goal-standoff-m 0.85 \
  --beacon-approach-distance-m 1.2 \
  --beacon-start-yaw-jitter-rad 0.7 \
  --policies lewm,bearing,hold,random \
  --primitive-names hold,forward_slow,forward_medium,forward_fast,arc_left,arc_right,yaw_left,yaw_right,backward \
  --output models/checkpoints_textured_v03_full_20260531/sweep_seq4/closed_loop_mpc_visible_beacon_e9_b050000_testid_open_obstacle_field.json
```

`--mode kinematic` is a planning-signal benchmark. It renders the current camera
view, scores candidate primitive rollouts with the LeWM latent planner, then
updates the base pose with the command registry's kinematic displacement. It does
not prove physical Go2 control. `--mode physical` is reserved for the full
Genesis low-level-policy rollout path, which requires the locomotion training
environment with `rsl-rl-lib` and `tensordict` installed.

Existing zero-jitter result for `lewm_seq4_e9_b050000.pt` on
`test_id/open_obstacle_field`, `--task visible-beacon`:

```text
output: models/checkpoints_textured_v03_full_20260531/sweep_seq4/closed_loop_mpc_visible_beacon_e9_b050000_testid_open_obstacle_field.json
usable scenes: 9
skipped scenes: 1 (EGL_BAD_DISPLAY while rendering open_obstacle_field_cbc86eeaefd9)

policy   success  mean_progress_m  mean_final_distance_m  mean_path_efficiency
bearing  9/9      0.875            0.325                  1.000
lewm     7/9      0.781            0.419                  0.951
random   0/9      0.547            0.653                  0.643
hold     0/9      0.000            1.200                  0.000
```

This result is useful as a sanity check but is not yet the discriminating
benchmark: the bearing oracle is 9/9 and the initial heading is aligned to the
beacon. Re-run this gate with yaw jitter (the command above uses `0.7 rad`) so
the model must correct heading rather than only walk forward. The e9 checkpoint
does not solve global navigation, but it can often chase a nearby visible beacon
in an open local field. Its remaining local failure mode is early commitment to
`hold` after some initial progress:

```text
lewm primitive counts: forward_fast 27, hold 19, arc_left 12, arc_right 9,
forward_slow 8, forward_medium 4, yaw_right 3
```

Treat this as the first local-MPC readiness gate. A usable checkpoint should
show positive goal progress, beat hold and random baselines, and approach the
simple bearing controller on this local task. Passing this gate does not imply
maze readiness; it only says the local goal-image cost is not obviously
degenerate.

### Stress Gate

The earlier `test_id/visual_sensor_stress` benchmark is a stress test, not the
first approval criterion. It mixes longer initial distances, distractors, and
harder visual conditions, so it probes whether local LeWM can cope without the
future hierarchy. Current `lewm_seq4_e9_b050000.pt` result:

```text
output: models/checkpoints_textured_v03_full_20260531/sweep_seq4/closed_loop_mpc_kinematic_e9_b050000_testid_visual_sensor_stress.json
usable scenes: 6
skipped scenes: 1 (EGL_BAD_DISPLAY while rendering visual_sensor_stress_e4ab31a1a678)

policy   success  mean_progress_m  mean_final_distance_m  mean_path_efficiency
bearing  0/6      0.394            3.966                  0.673
random   0/6      0.112            4.247                  0.142
lewm     0/6      0.037            4.323                  0.228
hold     0/6      0.000            4.359                  0.000
```

This should **not** be used as a model-quality discriminator in its current
configuration: the privileged bearing oracle also scores 0/6, so the benchmark
is under-budgeted or otherwise invalid for the chosen starts. Before reporting a
stress result, either raise the budget to roughly `--max-blocks 40` for these
distances or shorten the starts so the bearing oracle has headroom. Only then
interpret the LeWM-vs-oracle gap as evidence about the local cost.

## Review Gate

The gate is initially review-based because task-calibrated hard thresholds have
not yet been established. Do not approve a checkpoint merely because the script
completed.

Before approval:

- Inspect all reported horizons, not only horizon `1`.
- Require the learned rollout to remain meaningfully better than persistence.
- Require a positive recorded-action advantage over shuffled and zero-action
  rollouts over useful planning horizons.
- Investigate sharp degradation before horizon `5`; local MPC depends on
  short-horizon ranking quality even when a 20-step rollout is imperfect.
- Run a receding-horizon local MPC smoke test before downstream training. The
  offline report is necessary but not sufficient.

Approve a reviewed report with:

```bash
bash scripts/approve_lewm_planning_gate.sh \
  models/checkpoints_textured_v03_full_20260531/sweep_seq4/planning_gate_lewm_seq4_e9.json
```

The approval marker is checkpoint-specific. A newer checkpoint requires a new
report and a new approval.

## Sweep Behavior

`scripts/train_lewm_sweep.sh` now runs the diagnostic automatically after
`seq_len=4` and `seq_len=8`. It stops before the next sweep phase unless the
checkpoint-specific approval marker exists. After review and approval, rerun
the same sweep command. Completed epochs are loaded from their checkpoint
directory and are not repeated.

`--skip-rollout-gates` exists for intentionally ungated experiments. Do not use
it for the main v3 training run.

Shell processes started before this gate was added are not retroactively
protected. Confirm that an already-running sweep stops after its current phase,
or restart the wrapper from the existing checkpoint directory before it can
advance.

## Current Main Run

The textured v03 main run is in a phase-only `seq_len=4` tmux session:

```text
tmux session: lewm_seq4_gate_20260602
checkpoint directory: models/checkpoints_textured_v03_full_20260531/sweep_seq4
log: .generated/train_logs/lewm_seq4_gate_20260602.log
```

This session resumes `seq_len=4` checkpoints and exits after epoch `9`. It does
not invoke the sweep wrapper. Run the long-horizon diagnostic manually against
`lewm_seq4_e9.pt`, review it, and approve the generated report before restarting
`scripts/train_lewm_sweep.sh` to consider `seq_len=8`.
