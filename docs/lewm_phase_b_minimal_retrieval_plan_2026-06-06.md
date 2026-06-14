# Minimal Phase B Retrieval Gate

Date: 2026-06-06

## Decision

Do not continue tuning IDM, pose auxiliaries, pooled single-frame action
rankers, or a learned cross-scene reachability regressor.

The next bounded experiment is a learned place-retrieval head on the frozen raw
LeWM encoder representation. If it passes, build the smallest explicit
topological memory around it. If it fails, stop the pooled-feature Phase B path
and move to spatial patch features before considering a heavy history encoder.

## Why This Is Next

The completed evidence changes the original v3 ordering:

- IDM has no held-out transition-specific decodability gain over a state-only
  control.
- Both pooled projected and pooled raw first-action rankers fail the registered
  local-action gate. Raw is consistently better than projected.
- Frozen single-frame place recognition is already far above chance, while a
  learned cross-scene reachability bucket head is near its trivial baseline.
- Short history improved the existing aliasing diagnostic only slightly.

This means the cheapest unresolved question is whether a small metric head can
turn the existing raw place signal into a reliable loop-closure substrate.
Graph distance should be computed by BFS over an explicit visited-place graph,
not regressed from two isolated frames.

## Registered P2.0 Experiment

### Inputs and split

- Source checkpoint:
  `models/checkpoints_textured_v03_rollout_stage2_20260604/seq11_rollout_lam0p25_h10_warm2_sess8k_ep12/lewm_seq11_e3.pt`
- Frozen feature: raw pooled LeWM encoder output.
- Training scenes: 4 per family from `train` (target 32 scenes).
- Evaluation scenes: 4 per family from `test_id` (target 32 scenes).
- Scene banks, frame limits, and retrieval metric reuse
  `scripts/probe_lewm_reachability_a3.py`.

### Head and loss

- Small MLP from frozen raw features to an L2-normalized retrieval embedding.
- Same `(scene_id, cell_id)` observations are positives.
- Different cells from the same scene with BFS distance at least 2 are
  negatives.
- Adjacent cells are ignored because rollout labels near boundaries are not a
  reliable hard-negative target.
- Train three fixed seeds: `20260606,20260607,20260608`.
- No LeWM weights are updated.

### Primary gate

Promote P2.0 only if all are true:

1. At least 32 held-out scenes are evaluated.
2. Mean held-out same-place Recall@5 improves over the frozen raw baseline by
   at least 15 percentage points.
3. Every seed improves Recall@5 over the frozen raw baseline.
4. Mean held-out Recall@1 does not regress.

Report graph-distance rank correlation as a secondary diagnostic. It is not the
primary gate because explicit graph BFS, rather than embedding distance, is the
planned routing mechanism.

## Execution Order After P2.0

### If P2.0 passes

1. Add a calibrated same-place/loop-closure classifier and require high
   precision before node merging.
2. Add a minimal explicit graph with conservative node creation, merge, and
   transition updates.
3. Add GoalAdapter into the same retrieval space and run its offline Recall@5
   gate.
4. Route with BFS over the graph.
5. Resume local-action work using spatial patch and/or history features for
   subgoal pursuit.

### If P2.0 fails

1. Close pooled raw/projected retrieval-head tuning.
2. Run the same retrieval gate on spatial patch features.
3. Add short history only if it improves held-out retrieval or loop-closure
   precision by a registered margin.
4. Do not build online memory or run physics navigation until an offline
   retrieval substrate passes.

## Explicit Graph Contract

The first online graph implementation, if promoted, must:

- store a retrieval embedding and representative observation per node;
- add directed traversed edges from executed transitions;
- use BFS over observed edges for distance and subgoal selection;
- merge nodes only through a calibrated high-precision loop-closure decision;
- report false merges, fragmentation, node purity, and trajectory coherence;
- never infer unobserved connectivity from latent distance alone.

## Executed P2.0 Result

Artifacts:

- `.generated/phase_b_retrieval_v0/full_gate.json`
- `.generated/phase_b_retrieval_v0/full_gate_heads/`

The run used 32 train scenes, 32 held-out `test_id` scenes, and all three
registered seeds.

| metric | frozen raw | learned head mean | change |
|---|---:|---:|---:|
| Recall@1 | 0.3420 | 0.3213 | -0.0206 |
| Recall@5 | 0.5491 | 0.5379 | -0.0112 |
| graph-distance Spearman | 0.0557 | 0.0512 | -0.0045 |

Every learned-head seed regressed Recall@5 despite reducing its training loss
substantially. P2.0 therefore fails its primary gate.

**Decision:** close learned pooled raw/projected retrieval-head tuning. Do not
build the online graph, GoalAdapter, or loop-closure system on this substrate.
The next experiment is P3.1, a deterministic spatial patch-token retrieval
screen on the same checkpoint and held-out scene set.

## Registered P3.1 Patch-Token Screen

P3.1 compares the existing raw CLS representation against:

- mean-pooled final-layer patch tokens;
- a spatial pyramid that concatenates 1x1, 2x2, and 4x4 regional means.

It reuses the same A3 scene selection and same-cell retrieval metric. Promote
the current checkpoint's patch substrate only if mean Recall@5 improves over
raw CLS by at least 5 percentage points without reducing Recall@1.

This is only a screen of the current from-scratch ViT's patch tokens. Failure
does not falsify stronger frozen pretrained patch features such as DINO; it
means the existing checkpoint does not justify a larger patch-token world-model
or action-head implementation.

## Executed P3.1 Result

Artifact: `.generated/phase_b_retrieval_v0/patch_full_gate.json`

| descriptor | Recall@1 | Recall@5 | Recall@5 change vs raw CLS |
|---|---:|---:|---:|
| raw CLS | 0.3420 | 0.5491 | - |
| patch mean | 0.3387 | 0.5327 | -0.0163 |
| 1x1/2x2/4x4 spatial pyramid | 0.3366 | 0.5284 | -0.0207 |

P3.1 fails. Spatially aggregating final-layer patch tokens from the current
from-scratch ViT does not improve held-out place retrieval.

**Decision:** do not build a patch-token world model or patch action head from
the current LeWM checkpoint. The next bounded substrate test is P3.2: run the
same retrieval gate with strong frozen pretrained DINOv2 features. That probe
requires obtaining official pretrained weights; no compatible pretrained
backbone or weights are currently present locally.

## Registered P3.2 Pretrained Patch Screen

P3.2 compares the raw LeWM CLS baseline against frozen DINOv2:

- normalized CLS;
- mean patch token;
- 1x1/2x2/4x4 patch-token spatial pyramid.

Use the same 32 held-out `test_id` scenes and A3 retrieval metric. Promote a
DINOv2 substrate only if at least one descriptor improves mean Recall@5 over
raw LeWM CLS by at least 5 percentage points without reducing Recall@1.

If P3.2 passes, build the next retrieval/local-action proxy on the winning
frozen feature. If it fails, stop single-frame substrate work and register a
short-history retrieval/action screen.

## Executed P3.2 Result

Artifact: `.generated/phase_b_retrieval_v0/dinov2_full_gate.json`

| descriptor | Recall@1 | Recall@5 | Recall@5 change vs raw LeWM CLS |
|---|---:|---:|---:|
| raw LeWM CLS | 0.3420 | 0.5491 | - |
| DINOv2 CLS | 0.3525 | 0.5432 | -0.0059 |
| DINOv2 patch mean | 0.3536 | 0.5409 | -0.0082 |
| DINOv2 spatial pyramid | 0.3375 | 0.5146 | -0.0345 |

DINOv2 improves exact nearest-neighbour Recall@1 slightly but reduces Recall@5
for every descriptor. P3.2 fails the registered gate.

**Decision:** stop single-frame retrieval substrate work. A stronger frozen
visual backbone does not fix the retrieval limitation on this benchmark. Run
one final bounded P3.3 short-history retrieval screen before deciding whether
to stop the retrieval/topology branch entirely.

## Registered P3.3 Short-History Screen

For identical held-out terminal windows, compare:

- the raw LeWM latent of the terminal frame;
- mean-pooled raw LeWM history;
- ordered concatenated raw LeWM history.

Evaluate history lengths 4 and 8 on 32 `test_id` scenes, using at most 8
terminal windows per cell and 160 windows per scene. Promote history only if at
least one descriptor improves mean Recall@5 over its same-window terminal-frame
baseline by at least 5 percentage points without reducing Recall@1.

If P3.3 fails, close offline place-retrieval/topology work on the current data
and objective. The next program should target the navigation objective directly
with better state/action supervision or collect trajectories designed for
loop-closure and partial-observability learning.

## Executed P3.3 Result

Artifact: `.generated/phase_b_retrieval_v0/history_full_gate.json`

All descriptors used identical H8-eligible terminal windows from 32 held-out
scenes.

| descriptor | Recall@1 | Recall@5 | Recall@5 change vs terminal |
|---|---:|---:|---:|
| terminal raw | 0.3038 | 0.5047 | - |
| H4 mean | 0.3216 | 0.5148 | +0.0101 |
| H4 ordered concat | 0.3073 | 0.5049 | +0.0001 |
| H8 mean | 0.3235 | 0.5245 | +0.0198 |
| H8 ordered concat | 0.3097 | 0.5136 | +0.0089 |

History helps slightly, consistent with the earlier A3 history diagnostic, but
the best Recall@5 gain is less than half the registered margin. P3.3 fails.

## Final Decision

Close the current retrieval/topology branch.

Do not build:

- a heavier BeliefEncoder on the current frozen latents;
- GoalAdapter or loop-closure heads;
- an online topological memory;
- additional pooled/patch retrieval-head variants;
- physics navigation around any of the failed offline substrates.

The evidence is now consistent across learned pooled heads, current-LeWM patch
tokens, frozen DINOv2 features, and short history. None produces the robust
held-out same-place retrieval improvement required to justify a hierarchy.

## Active Next Program

The next work should target the navigation decision directly rather than
continue representation readouts:

1. **Define a task-aligned supervised target.** Use privileged simulator state
   only during training to label collision-safe first actions, local subgoals,
   and branch choices. Evaluation remains pixels/actions only.
2. **Mine the existing corpus first.** The 2026-06-08 readiness audit found
   69.6M aligned rendered/label rows, 38.9M branch-state rows, 1.32M recovery
   blocks, and 13.92M requested/executed command blocks with no missing joins.
   Do not collect new rollouts until a mined-index audit identifies a specific
   coverage gap.
3. **Use spatial and temporal inputs jointly.** Train a compact policy/cost
   model from image patch features plus short action/observation history,
   evaluated first on the existing first-action regret/collision gate.
4. **Require direct task improvement.** Promote only on held-out first-action
   regret, collision, and then closed-loop goal success. Do not promote on
   decodability, retrieval, or latent-distance improvements alone.
5. **Reconsider hierarchy only after a local policy passes.** A graph or
   high-level planner is useful only after the local controller can reliably
   execute a chosen subgoal.

The cheapest next executable cell is a branch/recovery index and policy
baseline mined from the existing rollouts and graph manifests. The full audit
and mining contract are in
`docs/lewm_task_aligned_data_readiness_2026-06-08.md`.

### Executed task-aligned handoff

The branch/recovery mining and offline counterfactual-label cells have now
executed on 32 balanced train scenes and 32 balanced validation scenes. Each
split produced 16,384 usable decisions with all nine velocity primitives.
The logged validation action is counterfactually optimal only 30.4% of the
time, confirming usable policy-learning headroom.

The action-only control also showed that scalar regret is insufficient:
always selecting `yaw_left` obtains low regret and 14.2% collision while
making zero target progress. The registered next model must predict separate
collision, progress, heading, and clearance targets from pixels/history/actions
and pass all corresponding gates. Do not resume retrieval, IDM, or new rollout
collection before that direct task-aligned cell is evaluated.

That direct frozen-base cell has now failed across pooled raw, coarse spatial,
and four-frame-history descriptors, with zero minimum-gate passes across nine
total runs. Close frozen-head/search variants. The next bounded cell adapts
only the final two vision-encoder blocks using the same direct task labels and
held-out gates. See
`docs/lewm_task_aligned_frozen_base_result_2026-06-08.md`.
