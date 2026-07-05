# Phase 2Z Occupancy Bridge And DINO/LeWM Paper Review

Date registered: 2026-06-15

Status: Phase 2Z failed bounded ROCm GPU smokes; train and validation data
only; no `test_id` or `test_hard` metric use.

Supersession note, 2026-06-15: this document originally registered Phase 2AA as
a DINO-WM-style next step. That branch is retained as an optional diagnostic
ceiling/control, not the mainline. The new primary direction is the foundational
JEPA plan in `docs/lewm_foundational_jepa_research_plan_2026-06-15.md`.

Status update, 2026-06-15: the foundational path now has Phase 3A
positive-control passes on the medium 2D navigation split without DINO,
including a stricter hidden-goal alias memory split where the memory-conditioned
model passes and the paired no-memory ablation fails. The DINO cache preflight
below remains useful only as a ceiling/control artifact; it is not the next
mainline implementation target.

## Plain-English Summary

The latest question was whether the model needs a map-like intermediate state
instead of raw RGB or a tiny ray vector.

Phase 2Z gave the learner a small local map around the robot. For each possible
first action it saw:

- which nearby cells are blocked or outside the scene;
- how much clearance each local cell has;
- where the goal lies in the local view;
- which cells the candidate first action would sweep through.

That is much closer to "what free space can I drive through?" than a single RGB
image vector. It still failed the action-choice gate. It could avoid total
primitive-collapse, but it did not choose actions with lower utility regret than
the action-only prior.

The literature review changes the next step: do not keep making small
first-action classifiers. The DINO/DINO-WM line suggests testing frozen
pretrained patch features plus action-conditioned latent rollout and
sampling-based planning. The LeWM paper comparison suggests adding a
paper-faithful 2D navigation replication as a positive-control task, so we can
separate implementation failure from the harder quadruped POMDP setting.

## Phase 2Z Implementation

Added implementation artifacts:

```text
lewm/benchmarks/phase2z_occupancy_affordance.py
lewm/models/primitive_affordance.py
scripts/train_jepa_phase2z_occupancy_affordance.py
lewm/tests/test_phase2z_occupancy_affordance.py
```

The feature schema is:

```text
phase2z_action_conditioned_local_occupancy_grid_v0
```

Per source state and candidate first primitive, the trainer constructs a
`4 x H x W` local ego-grid:

```text
occupied_or_out_of_bounds
clearance_norm
goal_heat
candidate_first_primitive_swept_path
```

It also supplies a compact vector:

```text
primitive one-hot
first command summary
source clearance
goal-relative distance/forward/left/bearing
```

The model is a small CNN over the per-primitive grid plus an MLP head over the
grid latent and vector features. It predicts the same six Phase 2O factorized
primitive-affordance targets used by earlier gates.

This is still a privileged diagnostic bridge: the grid is built from manifest
geometry and source pose. It is not a deployable RGB/depth policy.

## Phase 2Z Results

All runs used:

```text
train: .generated/jepa_counterfactual/phase2d_min_sources/train_spatial_future_v1.jsonl
validation: .generated/jepa_counterfactual/phase2d_min_sources/validation_spatial_future_v1.jsonl
device: ROCm CUDA-compatible torch runtime
grid_size: 32
half_extent_m: 4.0
optimization_steps: 512
```

### Ranking Weight 0.10

```text
primitive_match_rate: 0.3125
mean_target_utility_regret: 0.0747487065
selected_max_primitive_fraction: 0.32421875
oracle_max_primitive_fraction: 0.3515625
```

Primitive action-only prior:

```text
primitive_match_rate: 0.1640625
mean_target_utility_regret: 0.0585990126
selected_max_primitive_fraction: 1.0
```

Gate:

```text
passed: false
failure_reasons:
  - primitive_match_rate_below_threshold
  - regret_not_below_action_only_baseline
```

### Ranking Weight 1.00

```text
primitive_match_rate: 0.36328125
mean_target_utility_regret: 0.0950508479
selected_max_primitive_fraction: 0.3515625
oracle_max_primitive_fraction: 0.3515625
```

Gate:

```text
passed: false
failure_reasons:
  - primitive_match_rate_below_threshold
  - regret_not_below_action_only_baseline
```

Increasing the ranking term reduced distribution collapse and improved exact
match, but regret worsened. That is not a promotable tradeoff because the
navigation objective is utility-sensitive, not just top-1 label-sensitive.

## Interpretation

Phase 2Z rules out a narrow idea:

> A single-frame local occupancy grid with a first-action swept-path overlay is
> enough to recover the validation primitive decision under the current
> factorized affordance gate.

It does not rule out:

- recurrent local occupancy or belief-state memory;
- depth-derived occupancy with temporal accumulation;
- richer action-conditioned swept geometry;
- pretrained spatial patch features;
- sampling-based latent planning;
- JEPA navigation on the simpler LeWM-paper 2D task.

Together with Phase 2Y, the result says the missing state is not merely "more
local geometry." Phase 2Y had sparse rays; Phase 2Z had a dense local map. Both
improved over random but failed the utility-regret gate. The likely missing
piece is either temporal belief, a better visual/spatial substrate, or a
planning procedure that scores imagined futures rather than directly predicting
one primitive label.

## Paper Review

### DINO

DINO trains a student/teacher self-distillation system on image crops. The
important practical lesson for this project is that ViT patch tokens and
self-distilled attention can carry object/part-like spatial structure without
manual labels. A pooled token is useful, but the spatial tokens are the part
most relevant to navigation.

Source:

```text
Emerging Properties in Self-Supervised Vision Transformers
https://arxiv.org/abs/2104.14294
```

### DINOv2

DINOv2 scales this idea with stronger curation and combines image-level and
patch-level self-supervision. It is explicitly positioned as a general-purpose
visual feature model whose frozen features transfer to dense prediction tasks.

Lesson for this repo: before another end-to-end small RGB encoder, test a frozen
DINOv2 patch substrate as a controlled upper-bound visual representation.

Source:

```text
DINOv2: Learning Robust Visual Features without Supervision
https://arxiv.org/abs/2304.07193
```

### DINO-WM

DINO-WM is closest to the missing experiment here. It trains a world model on
frozen DINOv2 features and plans by optimizing action sequences toward a
goal-feature target. The planning lesson is important: it does not ask a small
head to output one primitive; it rolls out candidate futures and uses
sampling-based MPC/CEM over latent goal distance.

Lessons for this repo:

- use frozen spatial patch features as the target/state substrate;
- predict future patch features under action sequences;
- evaluate imagined futures against goal features;
- use sampling-based action-sequence selection, not only top-1 primitive
  classification;
- keep persistence, zero-action, and shuffled-action controls.

Source:

```text
DINO-WM: World Models on Pre-trained Visual Features enable Zero-shot Planning
https://arxiv.org/abs/2411.04983
```

### LeWM Paper And 2D Navigation

The LeWM paper basis matters because it provides a simpler 2D navigation setting
than this repo's current single-camera quadruped POMDP. If a paper-faithful
2D-style task passes while this quadruped task fails, the gap is likely task
difficulty, partial observability, and deployment geometry. If the 2D-style task
also fails here, the implementation or training objective is misaligned with the
paper basis.

Lesson for this repo: add a positive-control replication before claiming the
approach is blocked. The control should be much closer to the paper task than
the current quadruped camera benchmark.

Source:

```text
LeWorldModel / LeWM
https://le-wm.github.io/
https://arxiv.org/abs/2603.19312
```

## Historical Research Decision

Stop Phase 2Z. Do not launch a full training sweep from:

- Phase 2Y sparse local rays;
- Phase 2Z dense single-frame local occupancy;
- current single-frame RGB-to-swept-state bridges;
- current C2 image-aligned patch-token JEPA integration.

This section originally proposed Phase 2AA as the next bounded experiment:

1. Build a DINOv2 patch-feature cache for Phase 2D train/validation frames, or
   fail preflight if pretrained weights are unavailable locally and no download
   is approved.
2. Train an action-conditioned predictor on frozen DINOv2 spatial patch
   features, using normalized patch targets and the same split-overlap guards.
3. Evaluate real-action prediction against persistence, zero-action, and
   shuffled-action controls.
4. Add a DINO-WM-style sampling planner over candidate two-block action
   sequences: rollout each candidate, score latent distance to the goal-feature
   target, and compare primitive choice and regret against the existing
   primitive gate.
5. In parallel or immediately after, add a LeWM-paper-faithful 2D navigation
   positive-control task.

That recommendation is now superseded for mainline work. The 2D positive
control was implemented as Phase 3A and has a first-primitive receding-horizon
pass under `jepa_phase3a_positive_control_gate_v1`. Continue from the
foundational Phase 3A/3B/3C plan unless a DINO ceiling/control is explicitly
needed to bound the visual-representation gap.

## Phase 2AA Promotion Gate

A Phase 2AA result is not eligible for full training unless all of the following
hold on validation only:

- finite patch-token predictions and targets;
- real-action prediction beats persistence at the first and second action block;
- real action beats zero and shuffled actions by at least 10% of actual target
  change;
- sampling-based latent planner reaches primitive match `>= 0.50`;
- utility regret is below the primitive action-only prior;
- selected primitive distribution is no more than `0.20` above oracle max
  primitive fraction;
- no `test_id` or `test_hard` metrics are accessed.

If Phase 2AA fails, the next credible step is recurrent belief or motion/object
slots. If the LeWM-paper-faithful 2D control fails, fix implementation alignment
before any quadruped full sweep.

## Phase 2AA.0 Cache Preflight

Implemented:

```text
lewm/benchmarks/phase2aa_dinov2_cache.py
scripts/cache_jepa_phase2aa_dinov2_features.py
lewm/tests/test_phase2aa_dinov2_cache.py
```

The cache builder:

- loads frozen DINOv2 from the local torch hub cache;
- caches deterministic unique start and valid-future frames;
- defaults to `mode=complete` so smoke runs actually include future targets;
- records frame-role coverage and feature shapes;
- fails hard if frames or DINOv2 weights are missing.

The local machine already has:

```text
~/.cache/torch/hub/facebookresearch_dinov2_main
~/.cache/torch/hub/checkpoints/dinov2_vits14_pretrain.pth
```

ROCm cache smoke:

```text
data: train_spatial_future_v1.jsonl
mode: complete
max_rows: 4
feature_kind: patch_mean
unique_frames: 9
role_counts:
  start: 1
  future_step_0: 4
  future_step_1: 4
feature_shapes:
  cls: [9, 384]
  patch_mean: [9, 384]
```

This is runtime readiness evidence only. It is not a Phase 2AA research result
because it does not train or evaluate a DINOv2 dynamics model.

## Artifact Hashes

```text
e7f5d1ba61e7b2a43e07e31ac3fdf47d25b215df439134d4b4ba7d5d16300a80  phase2z_occupancy_affordance_rank010_smoke.json
3d923fdf5f573f098025c93505ea874f9eda9a49112d806c0b66e220b3cfee29  phase2z_occupancy_affordance_rank010_smoke_gate.json
29fbf456603686b1efe229caaa105c9b91e7865bd76883b45696d9865580a667  phase2z_occupancy_affordance_rank100_smoke.json
73866fbf64bb2ddd53f521efc818ecf99a495228238be8128bd4eb5bf2050e19  phase2z_occupancy_affordance_rank100_smoke_gate.json
67e4e59a8dc983f388ee5017d15f36211010e4ca7d7909f6faf5d9834199548a  phase2aa_dinov2_train_complete_patchmean_max4_smoke.json
1697f0cf9133a17183933a502c370834b39464e21777cf1569a330ddd44001c3  phase2aa_dinov2_train_complete_patchmean_max4_smoke.pt
```
