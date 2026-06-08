# Task-Aligned Data Readiness Review

Date: 2026-06-08

## Decision

The required task-aligned navigation data already exists. Do not collect new
rollouts before mining and training a baseline from the current corpus.

Artifact: `.generated/audits/task_aligned_data_readiness.json`

Executed bounded indexes and controls:

- `.generated/task_aligned_decisions/train32.jsonl`
- `.generated/task_aligned_decisions/val32.jsonl`
- `.generated/task_aligned_decisions/train32_scored.jsonl`
- `.generated/task_aligned_decisions/val32_scored.jsonl`
- `.generated/task_aligned_decisions/baselines_train32_val32.json`

## Exact Inventory

| item | count |
|---|---:|
| scenes | 1,450 |
| train / val / test_id / test_hard scenes | 1,000 / 150 / 150 / 150 |
| rendered RGB frames | 69,600,000 |
| privileged label rows | 69,600,000 |
| requested command blocks | 13,920,000 |
| executed command blocks | 13,920,000 |
| missing command joins | 0 |
| missing episode joins | 0 |

The corpus covers all eight scene families. All 1,450 scenes have synchronized
raw messages, derived labels, replay plans, and completed textured renders.

## Existing Decision Coverage

| existing signal | count |
|---|---:|
| T-junction + crossroad label rows | 38,893,502 |
| branch-state macro equivalent | 7,778,700 |
| route-teacher blocks | 4,038,877 |
| frontier blocks | 2,915,647 |
| loop-revisit blocks | 1,482,974 |
| recovery blocks | 1,316,626 |
| backward blocks | 921,662 |
| route completions | 12,748 |
| goal-target changes | 165,569 |

A 116-scene stratified label audit sampled 5,568,000 rows. Projected to the
full corpus, it indicates approximately 34.0M low-clearance rows and 10.7M
stuck rows. These are estimates; the exact branch and command-source counts
above come from all scene summaries.

A separate 29-scene message-schema audit covered 278,400 command blocks:

- 192,708 carried a valid `route_target_id`;
- 89,776 carried a valid `next_waypoint_id`;
- all 278,400 had matching executed command blocks;
- 89,578 executions were clipped.

## What Can Be Mined

### Branch-choice decisions

Each replay frame already contains:

- aligned RGB `frame_index`;
- pose and episode metadata;
- active `command_context`, including source, primitive, target, and waypoint.

Each aligned derived-label row contains `cell_id`, `local_graph_type`,
clearance, stuck status, landmark observations, and integrated motion.

At a command-context change, use the preceding rendered frame as the decision
observation, the new command as the teacher action, and the following boundary
as the realized outcome. For goal-directed commands, derive the oracle next
cell from `SceneGraph.next_waypoint(current_cell, route_target_id)`.

### Recovery and collision-safety decisions

Mine rows where any of these are true:

- command source is `recovery`;
- `stuck_label` is true;
- clearance is below the registered threshold;
- an executed command was clipped.

The existing manifest and planning grid can score all registered primitives
counterfactually from the recorded pose, as already demonstrated by
`build_first_action_dataset.py`. New simulation is not required for the first
counterfactual safety/ranking dataset.

### Goal conditioning

`route_target_id`, landmark BFS labels, and same-scene frames at target cells
are already available. A mining pass can attach a representative target-cell
RGB frame to each goal-directed decision without collecting goal images.

## Corrected Next Steps

1. Build a compact branch/recovery decision index from existing train and val
   scenes.
2. Audit unique states, target-cell coverage, class balance, and target-frame
   availability after deduplication.
3. Train a direct task-aligned baseline using pixels/history/actions as inputs
   and privileged oracle next-cell/safety labels only as targets.
4. Gate on held-out branch-choice accuracy, first-action regret, and collision
   rate.
5. Collect additional rollouts only if the mined index reveals a specific,
   quantified coverage gap that cannot be repaired by resampling or offline
   counterfactual scoring.

## Collection Trigger

New collection is justified only if the mined dataset shows one of:

- insufficient unique branch states after temporal deduplication;
- missing target-cell observations for goal conditioning;
- a severe class/family imbalance that resampling cannot fix;
- insufficient recovery outcomes for a specific failure mode;
- a deployment-domain shift not represented by the current simulator corpus.

None of those gaps is established by the current audit.

## Executed Mining Result

`scripts/mine_task_aligned_decisions.py` was run on 32 balanced train scenes
and 32 balanced validation scenes, with 512 decisions retained per scene.
Each row now includes the start/end RGB frames, target-cell RGB frame when
available, recorded start pose, scene manifest, requested and executed 15-D
action blocks, clipping outcome, realized motion, and privileged task labels.

| bounded index metric | train | validation |
|---|---:|---:|
| rows | 16,384 | 16,384 |
| branch rows | 9,625 | 9,097 |
| recovery rows | 11,665 | 12,468 |
| distinct scene/cell/target/type states | 4,612 | 4,066 |
| clipped executed rows | 5,354 | 5,466 |
| rows with target frame | 12,071 | 11,430 |
| rows with oracle next cell | 11,159 | 10,343 |

All nine trainable velocity primitives occur in both indexes. The miner
strictly verifies frame/label row counts, timestamp/env alignment, rendered
frame existence, and the executed-command join. The completed run found no
alignment or command-join failures.

This is enough data for the next bounded model cell. It does not justify new
rollout collection.

## Executed Counterfactual Screen

`scripts/score_task_aligned_counterfactuals.py` evaluated all nine registered
velocity primitives from every bounded-index start pose using the scene
manifest and inflated occupancy grid. This is offline privileged labeling,
not new simulation or rollout collection.

| metric | train | validation |
|---|---:|---:|
| scored rows | 16,384 | 16,384 |
| rows with a task target | 12,663 | 12,286 |
| logged action optimal rate | 28.6% | 30.4% |
| mean logged action regret | 0.423 | 0.442 |
| candidate collision rate | 25.4% | 29.7% |

The logged collector actions leave substantial measurable regret, so the
dataset supports a meaningful policy-learning test rather than only imitation.

The control in `scripts/evaluate_task_aligned_counterfactual_baselines.py`
also identified an important objective-design issue:

| validation control | mean regret | selected collision | target progress |
|---|---:|---:|---:|
| random action | 0.433 | 29.7% | +0.017 m |
| logged collector action | 0.442 | 31.6% | +0.039 m |
| global action-only prior (`yaw_left`) | **0.115** | **14.2%** | **0.000 m** |

A scalar safety-heavy cost can therefore be gamed by always turning in place.
Do not train or promote a model on scalar regret alone.

## Registered Next Experiment

Stop IDM, retrieval, and new-collection work. Train one compact candidate
action scorer on the mined/scored train index:

- deployed inputs: current RGB, short RGB/action history, optional target RGB,
  and one candidate 15-D action block;
- privileged training targets only: candidate collision, target progress,
  heading error, and post-action clearance;
- output separate task heads rather than only the current scalar cost;
- evaluation: scene-disjoint `val32_scored.jsonl`, three seeds.

Minimum validity gate:

- mean regret below the action-only prior (`< 0.115`);
- selected collision rate below the action-only prior (`< 14.2%`);
- positive mean target progress (`> 0.000 m`);
- each result must hold on all three seeds.

Promotion gate before closed-loop evaluation, in addition to the minimum
validity gate:

- random-regret ratio `<= 0.5`, equivalent here to mean regret `<= 0.216`;
- selected collision rate `<= 5%`;
- target progress above both random and action-only controls;
- no privileged fields consumed by the model at inference.

If this cell fails, inspect errors by scene family and decision type. Collect
new rollouts only when that analysis identifies a specific missing state or
outcome class. Do not respond to a model failure with undirected collection.

## Executed Frozen-Base Result

The registered candidate-scorer cell was executed using pooled raw features,
2x2 pooled spatial patch features, and four-frame history descriptors. All
three substrates failed the minimum gate in all three seeds. The best
three-seed mean was the spatial substrate at regret `0.139`, collision
`15.76%`, and progress `+0.0035 m`; the action-only `yaw_left` control remains
better at regret `0.116` and collision `14.04%`.

Stop frozen-head and inference-search variants. The next bounded escalation is
task-aligned adaptation of only the final two vision-encoder blocks. Full
results and gates are in
`docs/lewm_task_aligned_frozen_base_result_2026-06-08.md`.

The follow-up selection-rule grids also produced zero passing settings. A
first full final-two-block adapter screen improved the best frozen result
slightly but still failed at regret `0.134`, collision `15.25%`, and progress
`+0.0019 m`. Before larger retraining, review the target/goal-conditioning
contract that makes low-motion safe turns dominate held-out selection.

## Commands

```bash
.generated/venvs/genesis_render_vulkan/bin/python scripts/mine_task_aligned_decisions.py \
  --split train --family all --scene-limit 32 --max-rows-per-scene 512 \
  --output .generated/task_aligned_decisions/train32.jsonl

.generated/venvs/genesis_render_vulkan/bin/python scripts/score_task_aligned_counterfactuals.py \
  --input .generated/task_aligned_decisions/train32.jsonl \
  --output .generated/task_aligned_decisions/train32_scored.jsonl

.generated/venvs/genesis_render_vulkan/bin/python \
  scripts/evaluate_task_aligned_counterfactual_baselines.py \
  --train .generated/task_aligned_decisions/train32_scored.jsonl \
  --eval .generated/task_aligned_decisions/val32_scored.jsonl \
  --output .generated/task_aligned_decisions/baselines_train32_val32.json
```
