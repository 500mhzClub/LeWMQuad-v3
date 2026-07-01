# Foundational JEPA Navigation Research Plan

Date registered: 2026-06-15

Status: new mainline research plan. This supersedes the DINO-first Phase 2AA
direction as the primary next step. DINOv2 remains an optional diagnostic
ceiling/control, not the main research path.

## Decision

Do not pivot the project to a DINO-WM replication as the mainline.

The mainline objective is foundational JEPA research:

> Determine whether a JEPA-style model can learn, from robot pixels, history,
> and actions, a structured predictive belief state that supports safe
> navigation action selection without using pretrained visual features or
> privileged geometry at runtime.

The desired end goal is stronger than remembering a supplied goal breadcrumb.
The deployed system should receive only observations, actions, proprioceptive
state if available, and a visually observable goal/beacon/object. It should
build online memory itself, explore by valuing novelty/frontiers while the goal
is unseen, bind the visual beacon to that memory when first observed, and then
select actions that claim or revisit the beacon without privileged simulator
goal coordinates, topological distance, or hidden breadcrumb channels at
runtime.

The recent DINO/DINO-WM review remains useful because it clarifies a known
working recipe. However, that recipe freezes the visual representation problem
with a strong pretrained encoder. That is a valuable control but not the core
question here.

## Boundary Conditions

Allowed for training, probes, labels, and audits:

- privileged geometry;
- depth, occupancy, pose, route teachers, and swept-volume oracles;
- kinematic and Genesis physics rollouts;
- topological memory diagnostics;
- DINO/DINOv2 features as a ceiling or sanity check only.

Not allowed for the mainline research claim:

- pretrained DINO/DINOv2 features as the deployed planning state;
- privileged geometry as a required runtime input;
- hidden goal-direction, goal-distance, or route breadcrumbs as deployed
  runtime inputs;
- success claims based on route replay, color servoing, or geometric vetoes
  unless the learned JEPA state makes the action-selection decision being
  claimed.

The topological memory stack remains a localization/routing scaffold and a
source of belief-state lessons. It is not by itself evidence that the JEPA
world model learned action consequences.

## Current Evidence

The repo has established several useful negatives and one narrow positive.

1. Pooled and current patch-token JEPA objectives fail direct action-conditioned
   prediction gates. Persistence beats non-collapsed learned predictions at the
   first block, and action changes do not produce meaningful target changes.
2. Anti-collapse regularization is necessary but insufficient. Removing it
   collapses the representation; keeping it preserves variation without
   organizing that variation around action consequences.
3. The factorized affordance target and selector are coherent. The true-factor
   ceiling passes strongly, so the primitive gate is not the primary failure.
4. Privileged action-conditioned swept geometry is sufficient for a learned
   affordance diagnostic to pass. The missing variable is consequence geometry,
   especially swept clearance, progress, unsafe fraction, heading, and
   continuation aggregates.
5. Current deployable bridges from single-frame RGB, local rays, dense
   single-frame occupancy, and current JEPA predicted tokens do not recover the
   action-relevant state well enough.
6. Online topological memory shows that history and filtering can extract useful
   place-belief structure from frozen LeWM latents, but pairwise verification is
   weak and yaw/viewpoint remains a dominant factor. The lesson is to make
   history, actions, and belief first-class JEPA inputs.
7. Phase 3A now has a small learned online-memory smoke result for the
   no-beacon exploration/claiming target. A trainable egocentric spatial
   frontier/marker memory head, initialized only with simple RGB evidence
   priors and trained from pixels/history/actions, beats the explicit
   nonprivileged online frontier scaffold on the
   `explore_claim_v4_spatial_frontier_smoke` validation split. This is evidence
   for the mechanism, not yet a full navigation claim: the split is deliberately
   small, and the full v4 run still needs a faster training/data path.

## 2026-06-16 Learned Online-Memory Update

Implemented in Phase 3A:

- a multi-channel egocentric spatial memory head inside
  `lewm/models/phase3a_jepa.py`;
- nonprivileged rolling maps for observed space, free space, blocked space, and
  marker belief;
- a spatial frontier/marker action score usable by the candidate selector;
- auxiliary occupancy, marker, score-distillation, and utility-ranking losses;
- report/export plumbing for `spatial_frontier_memory_score`;
- an MP4 demo exporter that can visualize the learned memory score.

The latest smoke checkpoint is:

```text
models/checkpoints/phase3a_explore_claim/phase3a_v4_spatial_frontier_directinit_64.pt
```

The corresponding score summary is:

```text
.generated/jepa_phase3a/explore_claim_v4_spatial_frontier_smoke/phase3a_v4_spatial_frontier_directinit_64_score_summary.json
```

Validation summary on the small smoke split:

```text
online_frontier_marker:
  explore_unseen: prim 0.167, seq_regret 2.275, new_cells 6.667
  discover_visible_marker: prim 0.500, seq_regret 1.108, new_cells 12.000, marker_seen 0.333, claim 0.167
  claim_after_marker_seen: prim 0.250, seq_regret 2.000, marker_seen 1.000, claim 0.250

learned_spatial_frontier:
  explore_unseen: prim 0.667, seq_regret 1.458, new_cells 9.000
  discover_visible_marker: prim 0.667, seq_regret 0.875, new_cells 12.667, marker_seen 0.333, claim 0.167
  claim_after_marker_seen: prim 0.500, seq_regret 1.900, marker_seen 1.000, claim 0.250
```

Demo artifact:

```text
.generated/jepa_phase3a/explore_claim_v4_spatial_frontier_smoke/phase3a_v4_spatial_frontier_directinit_64_demo.mp4
```

Interpretation:

- This addresses the earlier failure mode where a raw goal-image embedding did
  not organize views by navigational usefulness. The new score is not trying to
  compare the current image to the goal image in a generic latent space.
  Instead, it builds an online egocentric memory, values unexplored frontiers
  before the marker is seen, binds marker evidence when it appears, and then
  scores action sequences by future marker claiming/revisiting.
- The result does not use simulator breadcrumbs or privileged goal coordinates
  at runtime. Privileged state is still used for dataset labels, audits, and
  oracle comparisons.
- The current positive result is intentionally small-scale. It proves that the
  repo can now express and train the intended memory mechanism in a bounded
  setting. It does not yet prove robustness on the full Phase 3A v4 split, Go2,
  or scene-scale closed-loop navigation.
- Long full-v4 ROCm launches exposed a CPU/input-pipeline bottleneck before
  useful checkpointing. The trainer now has a reusable
  `Phase3AMaterializedDataset` cache that precomputes per-row tensors and hard
  negatives once, then reuses them during train/eval. On the smoke train split
  of 8192 rows / 32 grouped batches, one uncached pass over all batches took
  2.411 s; cache construction took 2.375 s; subsequent cached batch
  materialization took 0.048 s, a 49.9x speedup after the first pass. An
  8-step ROCm-wrapper smoke completed on `device cuda` and wrote
  `models/checkpoints/phase3a_explore_claim/phase3a_v4_spatial_frontier_cache_smoke_8.pt`.
  Report scoring and MP4 export now reuse the same cache.
- A bounded full-v4 post-cache GPU probe also completed on `device cuda`:
  `models/checkpoints/phase3a_explore_claim/phase3a_v4_spatial_frontier_cache_probe_gpu_16.pt`.
  This was only 16 optimization steps, so it is a throughput/proof-of-plumbing
  run rather than a converged model. Full validation covered 108 source states.
  Aggregate primitive match was 0.583 with sequence regret 1.708.

Full-v4 phase summary for that 16-step GPU probe:

```text
online_frontier_marker:
  explore_unseen: prim 0.588, seq_regret 1.978, new_cells 8.765
  discover_visible_marker: prim 0.613, seq_regret 2.060, new_cells 8.258, marker_seen 0.323, claim 0.000
  claim_after_marker_seen: prim 0.767, seq_regret 1.756, marker_seen 0.907, claim 0.326

learned_spatial_frontier:
  explore_unseen: prim 0.559, seq_regret 1.916, new_cells 8.941
  discover_visible_marker: prim 0.581, seq_regret 1.653, new_cells 9.419, marker_seen 0.387, claim 0.000
  claim_after_marker_seen: prim 0.721, seq_regret 1.472, marker_seen 0.837, claim 0.326
```

Full-v4 artifacts:

```text
.generated/jepa_phase3a/explore_claim_v4_novelty_then_claim/phase3a_v4_spatial_frontier_cache_probe_gpu_16_score_summary.json
.generated/jepa_phase3a/explore_claim_v4_novelty_then_claim/phase3a_v4_spatial_frontier_cache_probe_gpu_16_demo.mp4
```

Interpretation of the full-v4 probe:

- The previous full-v4 checkpointing stall is cleared when the GPU command is
  run outside the sandbox with ROCm device access and the Phase 3A cache is
  used.
- After only 16 steps, the learned spatial frontier score is not strictly
  better than the explicit scaffold on primitive-match rate, but it has lower
  sequence regret in all three phases and the same post-sighting claim rate.
- The next evidence-producing run should be a longer GPU training run with
  intermediate checkpoints and the cached report/export path, followed by the
  same phase summary. Do not treat this 16-step probe as final convergence.

## 2026-06-16 Full-v4 512-step Sweep and Closed Loop

Longer run:

```text
models/checkpoints/phase3a_explore_claim/phase3a_v4_spatial_frontier_gpu_512_b4.pt
```

Run configuration:

- full v4 no-beacon `novelty_then_claim` train/validation split;
- ROCm wrapper outside the sandbox on `device cuda`;
- `source_states_per_batch=4`;
- 512 optimization steps;
- checkpoints at steps 128, 256, and 384.

Final validation aggregate from the checkpoint report:

```text
primitive_match_rate: 0.630
mean_selected_sequence_target_utility_regret: 1.654
```

Phase score summary for the deployed `spatial_frontier_memory_score` remained:

```text
explore_unseen: prim 0.559, seq_regret 1.916, new_cells 8.941
discover_visible_marker: prim 0.581, seq_regret 1.653, new_cells 9.419, marker_seen 0.387, claim 0.000
claim_after_marker_seen: prim 0.721, seq_regret 1.472, marker_seen 0.837, claim 0.326
```

Important negative: the phase metrics were identical at steps 128, 256, 384,
and 512. The currently useful deployed score is therefore dominated by the
direct RGB-initialized egocentric memory mechanism, not by demonstrable
improvement of the spatial frontier score during this 512-step run. The
candidate-score aggregate does train, but the deployed spatial frontier score
itself did not move enough to affect action selection.

Ablation report:

```text
.generated/jepa_phase3a/explore_claim_v4_novelty_then_claim/phase3a_v4_spatial_frontier_gpu_512_b4_ablation_report.json
```

Key falsification results:

```text
identity:
  claim_after_marker_seen: prim 0.721, regret 1.472, claim 0.326

no_history:
  claim_after_marker_seen: prim 0.395, regret 4.621, claim 0.023

shuffled_history_actions:
  explore_unseen: prim 0.324, regret 2.903
  discover_visible_marker: prim 0.323, regret 2.710
  claim_after_marker_seen: prim 0.628, regret 2.944, claim 0.186

marker_color_removed:
  claim_after_marker_seen: prim 0.395, regret 4.705, claim 0.000

candidate_actions_shuffled:
  explore_unseen: prim 0.382, regret 4.950
  discover_visible_marker: prim 0.387, regret 5.037
  claim_after_marker_seen: prim 0.674, regret 4.158, claim 0.023

spatial_memory_disabled:
  explore_unseen: prim 0.382, regret 4.869
  discover_visible_marker: prim 0.452, regret 4.821
  claim_after_marker_seen: prim 0.465, regret 4.474, claim 0.000
```

Interpretation:

- The action sequence matters. Shuffling candidate actions badly damages
  exploration/discovery regret and nearly removes claiming.
- The online memory matters. Removing history or disabling the spatial memory
  destroys most post-sighting claiming.
- The visual marker matters. Removing green marker evidence drops claim rate to
  zero in the claim phase.
- These ablations support the claim that the current mechanism uses
  observation/action memory and marker evidence, not privileged goal
  coordinates or hidden route breadcrumbs.

Closed-loop 2D demo:

```text
.generated/jepa_phase3a/explore_claim_v4_novelty_then_claim/phase3a_v4_spatial_frontier_gpu_512_b4_closed_loop_demo.mp4
.generated/jepa_phase3a/explore_claim_v4_novelty_then_claim/phase3a_v4_spatial_frontier_gpu_512_b4_closed_loop_report.json
```

The closed-loop exporter reconstructs validation scenes, repeatedly samples all
4-step candidate sequences from the current state, scores them with the learned
spatial frontier memory score, executes the selected first primitive, and
updates the online history. Current observations show the visual marker when it
is in view, but no beacon or privileged coordinate channel is provided.

Closed-loop result:

```text
attempted episodes: 24
claimed episodes: 13
selected demo: claimed in 21 steps
start: (4, 10, yaw 3)
goal: (6, 4)
final: (6, 4, yaw 2)
marker first visible in selected demo: step 12
collisions in selected demo: 0
```

Status after this run:

- The controlled 2D end-to-end requirement is now met at the proof-of-mechanism
  level: no runtime breadcrumb, novelty/frontier exploration, visual marker
  binding, and closed-loop claiming are demonstrated in a generated 2D world.
- This is still not a Go2 claim and not yet a clean foundational-learning claim.
  The direct RGB-initialized map prior is doing too much work, and longer
  training did not improve the deployed spatial frontier score. The next
  research step is to reduce that hand-designed prior while preserving the
  ablation-sensitive closed-loop behavior.

## Core Hypothesis

A foundational JEPA path remains credible if the next model changes the problem
from "predict the next image-aligned token grid" to:

```text
history of observations + executed actions + candidate future actions
  -> learned structured belief state
  -> action-conditioned predictive consequences
  -> compare imagined futures under a registered navigation gate
```

The learned state should be structured but not pretrained:

- egocentric spatial tokens;
- learned slots tied by supervision/probes to clearance, safety, goal progress,
  heading, and visibility;
- recurrent belief tokens over history and actions;
- separate appearance/place memory branch from local consequence branch.

## Phase 3A: JEPA Positive Control

Purpose: prove the local JEPA implementation can learn action-conditioned
navigation consequences in a simpler controlled setting before re-entering the
Go2 POMDP.

Task:

- create a small 2D navigation world with image observations generated from
  grid/maze state;
- expose only pixels plus actions to the model during training/evaluation;
- keep privileged state only for labels, probes, and oracle action selection;
- generate same-source counterfactual branches for all candidate actions;
- include collisions, dead ends, turns, and visually aliased corridors.

Model:

- learned pixel encoder, no DINO;
- JEPA target encoder with stop-gradient/EMA or other explicitly registered
  anti-collapse mechanism;
- action-conditioned predictor over learned spatial or slot tokens;
- optional recurrent belief over short history.

Gates:

- finite training and validation metrics;
- no representation collapse or low effective rank;
- real-action decision prediction beats persistence at the first horizon;
- later decision horizons either beat persistence or show strong real-action
  contrast against zero and shuffled actions;
- real action beats zero and hard-negative actions by at least 10 percent of
  actual target change;
- latent rollout action selection beats the action-only prior on primitive
  match and utility regret;
- exact same implementation fails the gate when actions are randomized, proving
  the task is not solved by visual priors alone.

If Phase 3A fails, do not run another quadruped JEPA sweep. Fix the objective,
target geometry, or optimization on the positive control first.

## Phase 3B: Counterfactual JEPA Objective

Purpose: remove the main weakness of logged-future prediction by training
directly on same-source action contrast.

Training data:

- source observation/history;
- a valid candidate action sequence;
- future target observations when renderable;
- terminal/contact/invalid-event labels when future images are absent;
- hard-negative actions from the same source state.

Objective:

- normalized latent prediction for valid future targets;
- hard-negative action hinge: correct action prediction must beat wrong actions
  by a margin scaled to actual target change;
- zero-action hinge excluding true hold cases;
- event prediction head for collision/contact/invalid future slots;
- multi-step rollout loss with scheduled sampling or latent-noise recovery.

Promotion gate:

- one-step and two-step rollout/persistence ratios below 1.0;
- hard-negative and zero-action advantages above the registered margin;
- event prediction improves over source/action priors;
- action-selection regret below the primitive action-only prior;
- no test-id or test-hard access.

## Phase 3C: Learned Structured Belief State

Purpose: make the model represent the variables that Phase 2S/2W proved are
necessary, while keeping runtime state learned from observations/history/actions.

State candidates:

- learned egocentric spatial token map;
- learned consequence slots such as clearance, unsafe fraction, goal progress,
  heading, visibility, and contact risk;
- recurrent belief token updated by observation, action, and proprioceptive
  summary;
- split latent branches for place/view memory versus local action consequences.

Supervision and probes:

- use Phase 2W sanitized swept geometry as teacher/probe, not runtime input;
- use factorized Phase 2O labels as auxiliary targets;
- preserve JEPA latent prediction and action-identifiability as the primary
  gate;
- report whether the learned state can reconstruct or rank swept consequence
  factors on validation.

Promotion gate:

- passes Phase 3B action/persistence gates;
- learned consequence state passes the primitive affordance gate without
  privileged geometry input;
- state-ablation shows that removing history/actions degrades decision quality;
- no selected primitive collapse beyond the existing oracle-excess bound.

## Phase 3D: Go2 Re-Entry

Only after Phase 3A passes and Phase 3B/3C pass on bounded train/validation
data should the project re-enter full Go2 JEPA training.

Initial Go2 scope:

- train/validation only;
- scene-disjoint source states;
- candidate action blocks from the existing primitive registry;
- no test-id or test-hard model selection;
- topological memory may provide goal/subgoal context, but the JEPA state must
  decide local action among candidates.

Go2 gate:

- learned rollout beats persistence at first and second block;
- learned action choice beats primitive action-only prior on match and regret;
- hard-negative and zero-action controls pass;
- selected primitive distribution is not collapsed;
- model remains stable across at least three seeds before any closed-loop demo.

## Optional DINO Control

DINOv2/DINO-WM work is retained only as a control branch.

Use it if one of these questions becomes blocking:

- is the dynamics/planning code correct when the visual substrate is strong?
- is the Go2 task only failing because the visual representation is too weak?
- how far is the foundational learned substrate from a known pretrained ceiling?

A DINO result must not be described as the main foundational JEPA result.

## Immediate Implementation Units

1. Add `phase3a` positive-control data generation:
   - deterministic small 2D maze/image generator;
   - same-source candidate action branches;
   - train/validation split manifest;
   - persistence, action-only, zero-action, and oracle baselines.

2. Add a minimal Phase 3A JEPA trainer:
   - learned image encoder;
   - structured spatial or slot target geometry;
   - EMA/stop-gradient target option;
   - hard-negative and zero-action action-identifiability losses;
   - multi-step rollout evaluation.

3. Add executable gates:
   - collapse/effective-rank gate;
   - rollout-vs-persistence gate;
   - real-vs-hard-negative action gate;
   - latent action-selection regret gate.

4. Only after Phase 3A passes, port the objective back into the current Phase 2D
   Go2 counterfactual data path.

Phase 3A.0 implementation note, 2026-06-15:

```text
lewm/benchmarks/phase3a_positive_control.py
scripts/build_jepa_phase3a_positive_control.py
lewm/tests/test_phase3a_positive_control.py
```

The scaffold now generates deterministic 2D pixel-observation navigation
datasets with all same-source two-action candidate branches, privileged
consequence labels, source-level oracle first primitives, and a train-derived
first-primitive action-only prior. It is a data-contract scaffold only; no JEPA
trainer or scientific result is registered yet.

Phase 3A.1 implementation note, 2026-06-15:

```text
lewm/models/phase3a_jepa.py
lewm/benchmarks/phase3a_training.py
scripts/train_jepa_phase3a_positive_control.py
scripts/check_jepa_phase3a_gate.py
lewm/tests/test_phase3a_training.py
```

The first minimal trainer and executable gate are implemented. A tiny 8-step CPU
smoke verified the end-to-end report/gate path and failed the scientific gate,
as expected. A 256-step bounded overfit attempt on the tiny smoke split also
failed the gate: persistence still beat rollout, action advantages stayed below
the 10 percent target-change margin, primitive match was `0.333`, regret did
not beat the action-only prior, and the target representation tripped the
collapse/effective-rank warnings. This is a valid negative scaffold result, not
a Phase 3A positive-control pass.

Next fix: adjust the foundational objective on the positive control before Go2
work. Priority candidates are stronger anti-collapse/variance pressure, a
non-collapsing target geometry, action-contrast loss on free-running rollouts,
and explicit event/utility heads whose gradients cannot be satisfied by a
near-static latent.

Phase 3A.2 implementation note, 2026-06-15:

The Phase 3A trainer now includes action-identifiability losses on free-running
rollouts, not only teacher-forced one-step predictions. The predictor rolls out
the true candidate sequence, same-source wrong action sequences, and the zero
action sequence from the same source token state, then applies the hard-negative
and zero-action margins to the full imagined future.

A 256-step bounded CPU run with the free-running contrast still failed the
registered gate. It improved hard-negative action separation relative to the
previous overfit attempt, reaching shuffled-action advantages of roughly
`0.078` at step 1 and `0.069` at step 2, but both remain below the required
`0.1` target-change margin. Rollout still lost to persistence at both horizons,
zero-action advantages were below threshold, primitive match stayed at `0.333`,
regret stayed worse than the action-only prior, and the representation still
tripped the collapse warning.

Interpretation: free-running action contrast is directionally useful but not
sufficient. The next objective change should address target geometry/collapse
directly, for example with stronger non-collapsing token structure or auxiliary
event/consequence heads whose targets cannot be satisfied by a near-static
latent. Do not move this objective back to Go2 until Phase 3A passes.

Phase 3A.3 implementation note, 2026-06-15:

The Phase 3A batch contract and trainer now include an auxiliary per-step
consequence target over:

```text
step_collision
cumulative_collision_fraction
target_progress_fraction
reached_goal
safe_recoverable
target_utility_scaled
```

The model predicts these factors from imagined rollout tokens with binary
losses for event flags and scalar losses for progress/utility factors. This is
privileged supervision used only as a training/probe scaffold; model inputs
remain pixels plus candidate actions.

A 256-step bounded run with the consequence head still failed the registered
gate. It did improve part of the failure mode: step-1 zero-action separation
passed the `0.1` target-change threshold (`0.132`), and step-1 hard-negative
separation rose to `0.089`, just below threshold. Step 2 remained below
threshold (`0.094` zero-action, `0.060` hard-negative), rollout still lost to
persistence at both horizons (`1.89` and `1.78` ratios), primitive match stayed
at `0.333`, regret did not beat the action-only prior, and the representation
still tripped collapse/effective-rank warnings.

Interpretation: consequence supervision helps action sensitivity, but the
rollout objective is still not learning a target geometry where action dynamics
beat persistence. The next fix should make the prediction target itself less
persistence-friendly, rather than only adding heads on top of the current
tokens. Priority options are delta/consequence-token targets, source-local
utility ranking, or a stronger covariance/effective-rank objective.

Phase 3A.4 implementation note, 2026-06-15:

The Phase 3A trainer now includes two additional objective components:

- rollout transition-delta loss, comparing predicted step-to-step token changes
  against target step-to-step token changes;
- source-local utility ranking, reusing the existing grouped
  `action_utility_losses` helper so candidate utilities are compared only
  within each same-source action set.

The first high-weight ranking run (`utility_ranking_loss_lambda=1.0`) was a
useful negative ablation: hard-negative action separation passed by a wide
margin, but rollout quality collapsed relative to persistence and zero-action
controls became worse than the true action. The default ranking weight is now
`0.1` so ranking does not dominate the world-model objective.

The balanced 256-step run still failed the gate. It improved the
free-running-vs-persistence ratios relative to the consequence-head run
(`1.35` and `1.34`, versus `1.89` and `1.78`), and primitive match rose to
`0.667`. However, zero-action and hard-negative margins stayed below threshold,
regret worsened to `1.667`, rollout still lost to persistence, and
collapse/effective-rank warnings remained. This is not a positive-control pass.

Interpretation: direct delta pressure makes the rollout less persistence-bad,
but it does not by itself preserve action identity. Source-local ranking changes
the selected primitive, but the current scalar utility head is not learning the
right ordering under this tiny validation split. The next fix should separate
the decision state from the image-token prediction state: add explicit learned
consequence/decision tokens whose targets and gates are action-consequence
centric, while continuing to audit image-token rollout against persistence.

Phase 3A.5 implementation note, 2026-06-15:

The Phase 3A model now has an explicit decision-token branch:

- decision tokens are initialized from the current learned image-token state;
- a separate action-conditioned predictor rolls those decision tokens forward;
- fixed, frozen projections of privileged consequence factors provide
  non-collapsing decision-token targets during training/audit;
- consequence and utility heads read from decision-token rollout, while image
  token rollout remains in the report as an audit surface;
- the gate now evaluates `decision_rollout_controls` when present, falling back
  to image `rollout_controls` only for old reports.

The first 256-step decision-token run is the strongest positive-control signal
so far, but still not a pass. It eliminated the decision-state collapse warning,
passed effective-rank diagnostics, beat persistence at step 1 (`0.465` ratio),
and passed zero-action plus hard-negative action margins at both steps. It still
failed because step 2 did not beat persistence (`8.56` ratio) and primitive
selection remained wrong (`0.0` match, `1.667` regret).

The next ablation changed primitive scoring to use the predicted
`target_utility_scaled` consequence factor instead of a separate pooled utility
head. That improved selector behavior somewhat (`0.333` primitive match and
`1.333` regret), but weakened the step-1 zero/hard-negative margins and still
failed step-2 persistence. The trainer now exposes this as `--utility-source`
with choices `consequence` and `head`; `consequence` is the default because it
matches the intended decision-state interpretation, while `head` remains a
useful ablation.

Interpretation: explicit decision tokens are the first change to satisfy the
non-collapse and action-identifiability parts of the foundational hypothesis on
the positive control. The remaining failure is multi-step consistency and
source-local utility ordering, not basic action sensitivity. Next priority:
improve the recurrent decision-token rollout, likely by adding scheduled
teacher forcing/recovery or by training utility as a deterministic function of
predicted consequence factors rather than a separately learned scalar.

Phase 3A.6 implementation note, 2026-06-15:

The decision-token objective now also trains teacher-forced one-step decision
dynamics. For each horizon step, the predictor is trained from the image-seeded
decision start at step 1 and from target decision tokens at later steps, with
separate prediction, delta, hard-negative, and zero-action losses. These are
reported independently from the free-running decision losses; the gate still
uses only free-running `decision_rollout_controls`.

A 256-step run with the previous consequence target showed better free-running
decision MSE and a stronger step-1 persistence ratio (`0.31`), but still failed
step-1 action-margin thresholds, step-2 persistence, and primitive selection.
The same setup with `--utility-source head` reached a step-1 zero-action margin
of `0.095`, just below threshold, but still failed hard-negative margin,
step-2 persistence, and utility selection.

The data contract was then updated to make the decision target more genuinely
step-specific. Generated Phase 3A rows now include per-step progress and goal
distance fields, and the consequence target now separates:

```text
step_progress_fraction
cumulative_progress_fraction
```

instead of repeating one final `target_progress_fraction` at every horizon.
The target remains backward-compatible for older JSONL rows by falling back to
final progress divided over the horizon.

After rebuilding the smoke split, a 256-step run still failed the gate, but the
step-2 persistence failure became less pathological: step-2
free-running-vs-persistence ratio improved to `2.33`, from roughly `5.81` under
the repeated-progress target. Step 2 still passed zero/hard-negative margins;
step 1 still failed both action-margin thresholds; primitive selection stayed
collapsed to `turn_left` with `0.0` match and `1.333` regret.

Interpretation: teacher-forced decision dynamics and step-specific consequence
targets improve multi-step target geometry, but the model still needs stronger
first-step action separation and a better source-local utility decision rule.
The next fix should target same-source first-action discrimination directly,
for example by adding a supervised first-primitive distribution/ranking probe
from predicted consequence factors and using it only as a Phase 3A scaffold
until the learned utility ordering becomes reliable.

Phase 3A.7 implementation note, 2026-06-15:

Phase 3A now has a narrow positive-control pass on the medium deterministic
2D navigation split:

```text
data: .generated/jepa_phase3a/positive_control_medium_v1
run: phase3a_jepa_first_primitive_control_weighted_1024
gate: phase3a_jepa_first_primitive_control_weighted_1024_gate_v1.json
schema: jepa_phase3a_positive_control_gate_v1
passed: true
```

The changes that made the positive control observable and learnable were:

- adding a source observation beacon that encodes ego-frame goal direction and
  topological distance, so visually aliased local patches are not asked to
  infer an unobserved goal;
- seeding decision tokens from pooled image tokens plus the beacon token and
  center token, instead of only the mean token state;
- correcting the action-only prior and model selector metric to first-primitive
  receding-horizon regret, while retaining exact fixed-sequence regret as a
  diagnostic;
- weighting decision-token prediction, delta, real-vs-zero, real-vs-shuffled,
  and candidate-score losses strongly enough to keep action consequences
  identifiable.

Registered validation metrics for the passing run:

```text
primitive_match_rate: 0.53125
action_only_prior_match_rate: 0.125
mean_target_utility_regret: 0.6875
action_only_prior_regret: 0.75
decision_step_1_free_running_vs_persistence: 0.2817686200
decision_step_1_zero_margin: 0.2413155437
decision_step_1_shuffled_margin: 0.1913672686
decision_step_2_free_running_vs_persistence: 1.4568203145
decision_step_2_zero_margin: 1.7902541215
decision_step_2_shuffled_margin: 1.3107767388
collapse_warning: false
```

Gate v1 records an important claim boundary. For receding-horizon navigation,
the first selected primitive is the deployed decision, so step 1 must beat
persistence. Later decision-token horizons are still audited, but when the
target step delta is tiny, absolute persistence can be less meaningful than
whether the latent remains action-identifiable. The passing run therefore
passes step 2 because both zero-action and shuffled-action advantages are far
above the registered 0.10 target-change margin, despite not beating
persistence there.

Negative ablations during this iteration:

- direct candidate scoring from source tokens did not improve selection;
- linear decision-target geometry did not fix step-2 persistence;
- residual recurrent updates did not fix step-2 persistence;
- soft cross-entropy ranking did not improve regret;
- DINO was not needed for this positive-control pass and remains only a
  ceiling/control branch.

Remaining limitation: exact fixed two-action sequence regret is still poor
relative to the train-derived full-sequence prior. Phase 3A should therefore be
claimed only as a first-primitive, receding-horizon positive control. The next
mainline step is to add learned online topological/belief memory to the JEPA
state, not to return to a pretrained DINO substrate.

Phase 3A.8 implementation note, 2026-06-15:

The online memory extension is implemented and has a separate positive-control
pass where the current source observation does not contain the goal/topology
beacon:

```text
data: .generated/jepa_phase3a/positive_control_memory_v1
history_steps: 4
current_goal_beacon: false
history_goal_beacon: true
run: phase3a_memory_context_head_zero3_1024
gate: phase3a_memory_context_head_zero3_1024_gate_v1.json
schema: jepa_phase3a_positive_control_gate_v1
passed: true
```

Implementation changes:

- Phase 3A rows now carry `history_observations_rgb`, `history_actions`,
  `history_primitive_sequence`, and beacon flags;
- `materialize_phase3a_batch` returns fixed-shape history tensors while keeping
  old zero-history rows valid;
- `Phase3AJepaModel` has an optional learned memory GRU over
  pooled/beacon/center history token features plus executed history actions;
- the memory vector seeds the decision tokens together with the current source
  image tokens;
- the trainer exposes `--use-memory-context`, and the dataset builder exposes
  `--history-steps`, `--no-current-goal-beacon`, and
  `--no-history-goal-beacon`.

Registered validation metrics for the passing memory run:

```text
primitive_match_rate: 0.5
action_only_prior_match_rate: 0.125
mean_target_utility_regret: 0.75
action_only_prior_regret: 0.84375
decision_step_1_free_running_vs_persistence: 0.3306960464
decision_step_1_zero_margin: 0.1470506787
decision_step_1_shuffled_margin: 0.1000471413
decision_step_2_free_running_vs_persistence: 1.6984821901
decision_step_2_zero_margin: 1.5726709092
decision_step_2_shuffled_margin: 1.1286033944
collapse_warning: false
```

Iteration notes:

- `phase3a_memory_context_control_weighted_1024` learned strong decision
  controls and beat action-only regret, but failed primitive match at `0.46875`;
- `phase3a_memory_context_head_1024` reached primitive match `0.5` and regret
  `0.75`, but missed step-1 zero-action margin at `0.083`;
- `phase3a_memory_context_head_zero3_1024` kept the head selector and raised
  decision zero-action weights to `3.0`, which cleared the full gate.

Claim boundary: this was evidence that the local learned JEPA scaffold can use
history when current goal beacons are removed, but it was not yet a clean
memory-dependence result. Follow-up no-memory and markerless ablations could
still pass related variants, so the task still leaked enough local prior or
visible target information for non-memory models to solve the registered gate.
It remains useful as an engineering milestone, not as the final memory claim.

Phase 3A.9 implementation note, 2026-06-15:

The memory positive control has been tightened into a hidden-goal alias task.
For each sampled source state, the data builder can now generate multiple
same-grid goal variants:

```text
builder flag: --goal-variants-per-source
row fields: base_scene_id, base_source_index, goal_variant_index
```

With current goal beacon and current/future goal markers disabled, all aliases
for a source have identical current observations but different hidden goals.
Only the history stream retains the goal/topology beacon. The data contract has
a regression test that verifies:

```text
same base source + different goal variants -> identical start_observation_rgb
same base source + different goal variants -> distinct history_observations_rgb
```

The model was also changed so the learned memory/belief vector is an explicit
input to both scalar scoring heads, not only an indirect seed for decision
tokens. This prevents the candidate score from silently washing out memory
after token normalization.

Registered alias split:

```text
data: .generated/jepa_phase3a/positive_control_memory_alias_v1
train rows: 2048
validation rows: 1024
history_steps: 4
goal_variants_per_source: 4
current_goal_beacon: false
current_goal_marker: false
history_goal_beacon: true
history_goal_marker: false
future_goal_marker: false
action_only_prior_match_rate: 0.15625
action_only_prior_regret: 0.8125
```

Passing memory run:

```text
run: phase3a_memory_alias_candidate_memscore_act6_zero3_1024
gate: phase3a_memory_alias_candidate_memscore_act6_zero3_1024_gate_v1.json
schema: jepa_phase3a_positive_control_gate_v1
passed: true
utility_source: candidate_score
use_memory_context: true
candidate_score_source_tokens: true
decision_action_contrast_lambda: 6.0
decision_teacher_forced_action_contrast_lambda: 6.0
decision_zero_contrast_lambda: 3.0
decision_teacher_forced_zero_contrast_lambda: 3.0
```

Registered validation metrics:

```text
primitive_match_rate: 0.578125
action_only_prior_match_rate: 0.15625
mean_target_utility_regret: 0.78125
action_only_prior_regret: 0.8125
decision_step_1_free_running_vs_persistence: 0.2838943303
decision_step_1_zero_margin: 0.2592265308
decision_step_1_shuffled_margin: 0.1036261022
decision_step_2_free_running_vs_persistence: 1.6151267882
decision_step_2_zero_margin: 2.4131704951
decision_step_2_shuffled_margin: 0.7485530481
collapse_warning: false
```

Paired no-memory ablation on the same split and candidate-score setup:

```text
run: phase3a_no_memory_alias_candidate_memscore_act3_zero3_1024
gate: phase3a_no_memory_alias_candidate_memscore_act3_zero3_1024_gate_v1.json
passed: false
failure_reasons:
  - step_1_hard_negative_advantage_below_threshold
  - regret_not_below_action_only_prior
primitive_match_rate: 0.5
mean_target_utility_regret: 1.0
decision_step_1_shuffled_margin: 0.0663377941
```

Interpretation: this is the first Phase 3A result where a memory-conditioned
JEPA decision state passes the registered gate on a task constructed to make
the current frame insufficient, while the paired no-memory ablation fails. It
is still a controlled 2D result, not a Go2 navigation claim. The next mainline
step should port the memory-conditioned decision-token objective into the
bounded Go2 counterfactual data path, preserving the same ablation discipline:
memory on must pass, memory off must fail or materially degrade.

Phase 3A.10 implementation note, 2026-06-15:

The next scaffold toward the desired end goal is now registered as
no-privileged-beacon exploration and visual beacon claiming. This addresses the
claim gap in Phase 3A.9: the hidden-goal alias pass proves that memory can carry
a supplied goal/topology cue, but it does not prove that the agent can build
breadcrumbs from ordinary observations and actions.

Implemented data-contract changes:

```text
builder flag: --history-policy {turning,explore}
builder flag: --utility-mode {goal_progress,explore_then_claim}
row fields:
  history_policy
  utility_mode
  history_goal_marker_seen
  current_goal_marker_seen
  observed_free_cells_before_candidate
future observation fields:
  goal_marker_visible
  newly_observed_free_cells
  cumulative_new_free_cells
  goal_claimed
label fields:
  goal_known_before_candidate
  future_goal_marker_seen
  target_new_free_cells
  target_goal_progress_utility
  target_exploration_utility
```

In `explore_then_claim` mode, the privileged goal/topology beacon can be
disabled in current and history observations. Before the visual marker has been
seen, the candidate utility rewards newly observed free cells and collision
avoidance. Once the marker is seen in history/current, the utility returns to
goal-progress/claim behavior, with a small novelty term. If a previously unseen
candidate future sees or reaches the marker, that candidate receives explicit
visual-discovery/claim reward. Privileged simulator geometry is still used to
label novelty and marker visibility, but it is not present in the model input.

Initial contract test:

```text
seed: 59
history_steps: 4
history_policy: explore
utility_mode: explore_then_claim
current_goal_beacon: false
history_goal_beacon: false
current_goal_marker: true
history_goal_marker: true
future_goal_marker: true
```

The test verifies that all rows have no current/history beacon, histories can
include forward exploration, marker sightings are ordinary visual events, and
unknown-goal rows use novelty utility rather than hidden goal-progress utility.

CLI smoke artifacts:

```text
data: .generated/jepa_phase3a/explore_claim_contract_v0
train rows: 128
validation rows: 64
history_policy: explore
utility_mode: explore_then_claim
current_goal_beacon: false
history_goal_beacon: false
train future_goal_marker_seen rows: 37 / 128
validation future_goal_marker_seen rows: 44 / 64
```

A 16-step CPU trainer smoke also completed from this dataset:

```text
run: phase3a_explore_claim_smoke_16
status: contract smoke only; not a scientific pass
primitive_match_rate: 0.0
mean_target_utility_regret: 1.9125
```

This failed result is expected at 16 steps and should not be interpreted as
evidence against the exploration objective. It only proves that the new
no-beacon utility mode is consumable by the existing Phase 3A training path.

GPU execution note:

```text
wrapper: scripts/phase3a_rocm_train.sh
python: /home/andrewknowles/TinyQuadJEPA/bin/python
device observed outside sandbox: AMD Radeon AI PRO R9700
```

The managed sandbox does not expose the ROCm GPU, so Phase 3A GPU training now
goes through the repo-owned wrapper above. That wrapper sets the ROCm path,
selects GPU 0 by default, unsets `HSA_OVERRIDE_GFX_VERSION`, and forwards
arguments to the Phase 3A trainer. The wrapper command prefix has been approved
once so future Phase 3A GPU sweeps can vary dataset and checkpoint arguments
without repeated permission prompts.

Larger no-beacon exploration/discovery result:

```text
data: .generated/jepa_phase3a/explore_claim_v1
train source states: 128
validation source states: 48
current_goal_beacon: false
history_goal_beacon: false
history_policy: explore
utility_mode: explore_then_claim

memory run:
  artifact: phase3a_explore_claim_memory_candidate_act6_zero3_2048
  aggregate primitive_match_rate: 0.5833333333
  aggregate mean_target_utility_regret: 0.6739583333

no-memory run:
  artifact: phase3a_explore_claim_no_memory_candidate_act6_zero3_2048
  aggregate primitive_match_rate: 0.5625
  aggregate mean_target_utility_regret: 0.8854166667

phase report:
  artifact: phase3a_explore_claim_comparison_report.json
  explore_unseen source states: 46
  discover_visible_marker source states: 2
  claim_after_marker_seen source states: 0
  memory discover marker selected rate: 1.0
  no-memory discover marker selected rate: 0.5
```

Interpretation: this is a small positive signal for no-beacon exploration and
visual discovery, but it is underpowered. The validation split contains only
two marker-discovery source groups and no post-sighting claim groups, so it is
not a pass on the full exploration-to-claim objective.

Strict history-claim contract:

```text
data: .generated/jepa_phase3a/explore_claim_history_claim_h4_v0
train source states: 40
validation source states: 30
horizon: 4
candidate rows per source: 256
current_goal_beacon: false
history_goal_beacon: false
current_goal_marker: false
history_goal_marker_seen rows: all
oracle claim_after_marker_seen goal-claimed rate: 0.6

memory run:
  artifact: phase3a_history_claim_h4_memory_1024
  claim selected_goal_claimed_rate: 0.0333333333
  claim primitive_match_rate: 0.5666666667
  claim mean_target_utility_regret: 1.5366666667
  selected first primitive: turn_left for all 30 validation sources

no-memory run:
  artifact: phase3a_history_claim_h4_no_memory_1024
  claim selected_goal_claimed_rate: 0.0333333333
  claim primitive_match_rate: 0.5666666667
  claim mean_target_utility_regret: 1.5366666667
  selected first primitive: turn_left for all 30 validation sources
```

Interpretation: this is a negative result for the stricter "saw beacon in
history, current view hides it, now claim it" test. The data contract is now
capable of testing exact claim behavior because the oracle claims in 60% of
validation source states, but the learned memory model does not beat the
no-memory ablation. Both learned models exploit the same first-action bias and
almost never select a sequence that actually claims the goal. The missing piece
is not GPU throughput; it is a stronger training/evaluation formulation for
candidate sequence scoring under large same-source candidate sets.

Promotion gate for this line, not yet passed:

- train on no-beacon `explore_then_claim` rows;
- memory-on model beats action-only and no-memory ablations on exploration
  novelty before marker sighting;
- memory-on model beats action-only and no-memory ablations on beacon claiming
  after first visual sighting;
- learned state records enough online memory that repeated observations of the
  same region become less valuable than frontier/novel observations;
- no current/history goal beacon, no privileged topological distance, and no
  simulator route breadcrumbs are present in model inputs;
- report a separate failure if goal-image embedding distance does not correlate
  with true progress, because the current positive result does not claim that
  latent goal-image geometry is solved.

## Phase 3A Online Marker-Memory Result

Follow-up on 2026-06-15: the stricter history-claim contract was narrowed to an
adjacent-goal alias split so every validation source has a visible marker in
history, a hidden current marker/beacon, and 256 candidate four-step action
sequences. This isolates the question "after seeing the beacon once, can the
agent use online memory to claim it without simulator breadcrumbs?"

```text
data: .generated/jepa_phase3a/explore_claim_history_claim_alias_h4_adjacent_tiny_v0
validation source states: 12
candidate sequences per source: 256
current_goal_beacon: false
history_goal_beacon: false
current_goal_marker: false
history marker: visible/seen
future marker: visible for scoring/reporting only
```

The new non-privileged egocentric marker-memory baseline detects the green
marker in history RGB, rolls that remembered relative marker location forward
through observed history actions, then scores candidate action sequences by the
terminal remembered marker delta. It uses no simulator coordinates, no hidden
goal id, no topological distance, and no privileged breadcrumb. On the claim
phase it reached:

```text
selected_goal_claimed_rate: 0.9166666667
mean_selected_sequence_target_utility_regret: 0.5
top5_claimed_rate: 0.9166666667
artifact: phase3a_egocentric_marker_memory_demo.mp4
```

This is the important first-principles control: the task is solvable from
pixels plus actions alone. We do not need privileged simulator breadcrumbs to
leave a path to the beacon once it has been visually observed.

The learned JEPA memory variants did not yet learn this operation. The best
shortlist behavior came from marker-memory auxiliary/ranking heads: correct
claim candidates often enter the top-5, but top-1 selection remains poor.

```text
best learned checkpoint for top-1 so far:
  phase3a_alias_h4_adjacent_tiny_spatial_memory_actionseq_marker_rank_1024_step000512.pt
  candidate_marker_memory_distance_score selected_goal_claimed_rate: 0.4166666667
  mean_selected_sequence_target_utility_regret: 4.2666666667

latest distance-mode checkpoint:
  phase3a_alias_h4_adjacent_tiny_spatial_memory_actionseq_marker_distance_rank_512.pt
  selected_goal_claimed_rate: 0.25
  mean_selected_sequence_target_utility_regret: 5.3583333333
  top5_claimed_rate: 0.8333333333
  artifact: phase3a_learned_marker_distance_demo.mp4
```

The promotion gate still fails for the learned model, while its new
`non_privileged_egocentric_marker_memory` block passes. This distinction should
be preserved: a hand-coded visual marker memory is a control and target, not a
claim that learned JEPA navigation has succeeded.

Next implementation target: replace the generic vector GRU/MLP candidate scorer
with a learned but structured egocentric belief state. The model should predict
or maintain a small egocentric marker/novelty occupancy field from RGB history,
apply action-conditioned transforms in that field, and score candidate terminal
poses by reading from the transformed belief. This keeps the approach
foundational and non-privileged while adding the spatial equivariance that the
current learned scorer is missing.

First structured-head implementation attempt:

```text
code path:
  structured_marker_memory_start_delta_prediction
  deterministic rollout_marker_delta(actions)
  structured_marker_memory_score = -abs(final_delta).sum

checkpoint:
  phase3a_alias_h4_adjacent_tiny_spatial_memory_actionseq_structured_marker_512_step000256.pt
  structured score selected_goal_claimed_rate: 0.3333333333
  structured score mean_selected_sequence_target_utility_regret: 4.5333333333
  structured score top5_claimed_rate: 0.4166666667

checkpoint:
  phase3a_alias_h4_adjacent_tiny_spatial_memory_actionseq_structured_marker_512.pt
  structured score selected_goal_claimed_rate: 0.0833333333
  structured score mean_selected_sequence_target_utility_regret: 6.2416666667
  structured score top5_claimed_rate: 0.1666666667
```

Interpretation: simply adding a continuous start-delta head and deterministic
candidate-action transform is not sufficient. It provides the right interface,
but the learned RGB/history encoder still underfits or averages the remembered
marker location enough that ranking collapses. The next version should not be a
single vector regression head; it should maintain a spatial belief/heatmap with
equivariant update/readout so uncertainty and aliases are represented explicitly.

2026-06-16 categorical and online-memory update:

```text
categorical learned memory:
  checkpoint: phase3a_alias_h4_adjacent_tiny_spatial_memory_saliency_categorical_marker_s1_256.pt
  report: phase3a_explore_claim_spatial_memory_online_marker_s1_256_report.json
  categorical selected_goal_claimed_rate: 0.3333333333
  categorical mean_selected_sequence_target_utility_regret: 4.75
  categorical top5_claimed_rate: 0.3333333333
  train remembered-cell source accuracy: 8/24
  validation remembered-cell source accuracy: 3/12

explicit online RGB+odometry memory:
  code path: online_marker_memory_score_prediction
  selected_goal_claimed_rate: 0.9166666667
  mean_selected_sequence_target_utility_regret: 0.5
  top5_claimed_rate: 0.9166666667
  gate block: model_online_marker_memory_score passed
  artifact: phase3a_online_marker_memory_demo.mp4
```

Interpretation: the categorical head confirmed the same failure mode as the
scalar head. Even with RGB marker saliency features and a discrete cell target,
a generic vector memory head collapses to a few common cells and cannot fit the
small training set. The model-side online score is therefore a successful
non-privileged scaffold, not a learned-JEPA success: it detects the visual
marker from RGB, rolls that memory through observed egocentric actions, and
scores candidate actions in that explicit memory frame. The learned JEPA gate
still fails, while both the standalone egocentric baseline and the model-side
online-memory diagnostic pass.

Next learned target: replace the vector GRU/MLP remembered-cell head with a
spatial belief map that is updated by RGB marker/novelty evidence and shifted or
rotated by actions. The first acceptance target is to match the explicit online
RGB+odometry score on this tiny alias split without simulator coordinates,
hidden beacons, goal ids, topological distance, or breadcrumb channels.

2026-06-16 learned spatial belief-map result:

```text
learned spatial marker memory:
  code path: spatial_marker_memory_start_belief
  action update: roll_spatial_marker_belief
  candidate score: spatial_marker_memory_score_prediction
  checkpoint: phase3a_alias_h4_adjacent_tiny_spatial_belief_marker_s1_256.pt
  report: phase3a_explore_claim_spatial_belief_marker_s1_256_report.json
  gate: phase3a_explore_claim_spatial_belief_marker_s1_256_gate.json
  demo: phase3a_spatial_belief_marker_demo.mp4

validation claim_after_marker_seen:
  selected_goal_claimed_rate: 0.9166666667
  mean_selected_sequence_target_utility_regret: 0.5666666667
  top5_claimed_rate: 0.9166666667
  claim-only gate: passed

belief probe:
  train remembered-cell argmax accuracy: 21/23 valid sources
  validation remembered-cell argmax accuracy: 11/12 valid sources
  validation mean belief mass: 0.9947
  validation mean max cell probability: 0.0245
```

Interpretation: this is the first learned-memory success for the tiny
history-claim alias split. The learned detector plus equivariant egocentric
belief map reaches the same claim rate as the explicit online RGB+odometry
control, with slightly higher sequence regret. This is a materially different
result from the failed GRU/vector, scalar-delta, and categorical-cell heads:
the model now has the spatial state structure needed to bind a seen marker to
future actions.

Remaining caveats: this is still a tiny supervised positive-control result, not
full exploration. The belief is diffuse even when its argmax is usually right,
and the task starts after history has already seen the marker. The next gate
should scale this spatial memory to the larger no-beacon explore-then-claim
setting: novelty/frontier seeking until marker discovery, then use the learned
belief map to claim or revisit the marker without privileged breadcrumbs.

2026-06-16 no-beacon novelty-to-claim update:

```text
new target mode:
  utility_mode: novelty_then_claim
  pre-marker target: visible novelty - collision cost
  post-marker target: goal progress / claim
  no current beacon, no history beacon, no current goal marker

dataset:
  .generated/jepa_phase3a/explore_claim_v4_novelty_then_claim
  train: 216 source groups / 55,296 candidate rows
  validation: 108 source groups / 27,648 candidate rows
  phases: explore_unseen=34, discover_visible_marker=31,
          claim_after_marker_seen=43

online frontier + marker memory:
  code path: egocentric_explore_claim_score
  model output: online_frontier_marker_score_prediction
  summary: phase3a_v4_online_frontier_marker_summary.json
  model summary: phase3a_v4_model_online_frontier_marker_summary.json
  demo: phase3a_v4_online_frontier_marker_demo.mp4
```

Validation phase metrics for the non-privileged online scaffold:

```text
explore_unseen:
  primitive_match_rate: 0.588
  sequence_regret: 1.978
  selected_new_free_cells: 8.765 / oracle 13.912

discover_visible_marker:
  primitive_match_rate: 0.613
  sequence_regret: 2.060
  selected_new_free_cells: 8.258 / oracle 13.774
  selected_future_goal_marker_seen_rate: 0.323

claim_after_marker_seen:
  primitive_match_rate: 0.767
  sequence_regret: 1.756
  selected_goal_claimed_rate: 0.326
  best_claim_available_rate: 0.326
```

Interpretation: the desired non-privileged loop is now represented as an
explicit online control. Before the marker is remembered, it maintains an
egocentric observed-cell set from RGB history/current views and rewards
frontier exposure. After the marker is remembered, it switches to RGB marker
memory and action odometry. It claims every validation source where a claim is
reachable in the candidate horizon (`14/14` available claim groups). The
single-horizon discovery rate is partial because the marker is genuinely
hidden; discovery should be judged in closed-loop exploration, not as if the
policy could know which unseen frontier contains the marker.

The same finite-crop online score is also exposed by `Phase3AJepaModel` as
`online_frontier_marker_score_prediction` and integrated into the reporting
script. The model-side output is prediction-allclose to the Python reference on
the v4 validation set (`max_abs_prediction_diff ~= 3.8e-7`); exact JSON summary
equality is false only because raw floating-point tie values differ at sub-micro
precision.

Learned follow-ups on the same v4 data:

```text
checkpoint:
  phase3a_v4_novelty_online_marker_memory_512.pt
  learned candidate + explicit online marker term
  aggregate primitive_match_rate: 0.574
  aggregate mean_target_utility_regret: 0.759
  claim_after_marker_seen selected_goal_claimed_rate: 0.326
  explore_unseen selected_new_free_cells: 6.471 / oracle 13.912

checkpoint:
  phase3a_v4_online_frontier_distill_512.pt
  pure learned distillation of online frontier-marker score
  aggregate primitive_match_rate: 0.676
  failure mode: selected forward for all 108 validation groups
```

Interpretation: the learned candidate scorer can use the explicit marker
memory term to claim after sighting, but it still under-learns online frontier
memory. Direct ranking distillation to the online policy collapsed to an
action prior. This is a negative result: the online memory algorithm proves the
information is available in RGB/actions, but the current generic candidate
scorer is not sufficient to internalize the topological memory update.

Next learned target: make the spatial belief memory multi-map. It should carry
both marker evidence and explored/known-free/known-blocked occupancy evidence,
with deterministic egocentric action transforms and learned RGB evidence
updates. The model should then score candidate paths from this memory, rather
than asking an MLP candidate head to infer frontier novelty implicitly.

ROCm note: `scripts/phase3a_rocm_train.sh` was used for a 1024-step structured
run, but this path did not reach the first 512-step checkpoint after several
minutes and `ps` showed one saturated CPU core. The process was stopped. Before
larger sweeps, optimize the Phase 3A training loop for GPU throughput: cache
materialized tensors, reduce Python per-source grouping overhead, or run a
single-process tmux job with explicit GPU telemetry.

## 2026-06-16 Randomized-Palette Caveat Update

The fixed-color/direct-init v4 result was useful, but it did not fully answer
the original goal-image/latent-organization concern. It relied on a direct RGB
marker prior: marker removal ablated performance, which means the successful
closed loop was partly a hand-coded color-memory controller rather than a
learned marker concept.

The v5 randomized-palette split was built to remove that shortcut:

```text
dataset:
  .generated/jepa_phase3a/explore_claim_v5_random_palette
  train: 64 source groups / 16,384 candidate rows
  validation: 32 source groups / 8,192 candidate rows
  palette: scene_random
  no current beacon, no history beacon, no current goal marker
```

Implemented changes:

- per-row randomized render palettes;
- palette-aware target extraction for training labels only;
- neutral/MLP frontier detector initialization, without fixed green weights;
- direct per-frame marker/observed/free/blocked observation-map supervision;
- configurable learned-score constants:
  `spatial_frontier_collision_penalty`,
  `spatial_frontier_novelty_reward`,
  `spatial_frontier_marker_gate_threshold`,
  `spatial_frontier_marker_gate_width`, and
  `spatial_marker_memory_score_temperature`;
- an optional `spatial_frontier_marker_source=spatial_marker` branch, tested
  as a negative result.

Best randomized-palette open-loop checkpoint variant so far:

```text
base weights:
  models/checkpoints/phase3a_explore_claim/phase3a_v5_random_palette_obsaux_mlp_maponly_512_threshold.pt

derived scorer:
  models/checkpoints/phase3a_explore_claim/phase3a_v5_random_palette_obsaux_mlp_maponly_512_threshold_p4_tau050.pt

variant:
  collision_penalty=4.0
  novelty_reward=0.35
  marker_gate_threshold=0.5
  marker_gate_width=0.25
  marker_score_temperature=0.5
```

Open-loop randomized-palette phase metrics for this best variant:

```text
explore_unseen:
  primitive_match_rate: 1.000
  sequence_regret: 1.517
  selected_future_goal_marker_seen_rate: 0.000

discover_visible_marker:
  primitive_match_rate: 0.846
  sequence_regret: 1.373
  selected_future_goal_marker_seen_rate: 0.385

claim_after_marker_seen:
  primitive_match_rate: 0.563
  sequence_regret: 2.613
  selected_goal_claimed_rate: 0.250
```

Closed-loop randomized-palette result:

```text
demo:
  .generated/jepa_phase3a/explore_claim_v5_random_palette/phase3a_v5_random_palette_obsaux_mlp_maponly_512_threshold_p4_tau050_block4_closed_loop_demo_v2.mp4

report:
  .generated/jepa_phase3a/explore_claim_v5_random_palette/phase3a_v5_random_palette_obsaux_mlp_maponly_512_threshold_p4_tau050_block4_closed_loop_report_v2.json

claimed_episodes: 1 / 16
marker_seen_ever: 11 / 16
mean_final_goal_distance: 6.5 cells
collision_steps: 11 / 439
```

This is the current main caveat. The randomized-palette model often discovers
the marker, but usually fails to claim it afterward. That means the earlier
"goal embedding cannot select paths that make the world look like the goal"
failure is only partially addressed. The representation is no longer a raw
goal-image similarity embedding and no longer depends on a fixed green runtime
detector, but the learned memory is still too local and too weak for robust
receding-horizon beacon claiming.

The dedicated marker-source branch was also tested:

```text
checkpoint:
  models/checkpoints/phase3a_explore_claim/phase3a_v5_random_palette_spatialmarker_frontier_256.pt

configuration:
  spatial_frontier_marker_source=spatial_marker
  spatial_marker_memory_loss_lambda=5.0
  spatial_marker_memory_ranking_loss_lambda=1.0

result:
  spatial_frontier_memory_score claim_after_marker_seen claim rate: 0.000
  spatial_marker_memory_score claim_after_marker_seen claim rate: 0.000
```

Interpretation: the separate spatial-marker head did not solve the issue. The
learned memory support was then decoupled from `view_size=7` by adding
`spatial_memory_size`. Observations are now inserted into a larger egocentric
memory support and the marker/frontier/free/blocked maps roll through candidate
actions on that larger support.

Latest larger-memory randomized-palette result:

```text
derived checkpoint:
  models/checkpoints/phase3a_explore_claim/phase3a_v5_random_palette_obsaux_mlp_maponly_512_threshold_p4_tau050_mem15.pt

configuration:
  view_size=7
  spatial_memory_size=15
  collision_penalty=4.0
  novelty_reward=0.35
  marker_gate_threshold=0.5
  marker_gate_width=0.25
  marker_score_temperature=0.5

closed-loop demo:
  .generated/jepa_phase3a/explore_claim_v5_random_palette/phase3a_v5_random_palette_obsaux_mlp_maponly_512_threshold_p4_tau050_mem15_block4_closed_loop_demo.mp4

closed-loop report:
  .generated/jepa_phase3a/explore_claim_v5_random_palette/phase3a_v5_random_palette_obsaux_mlp_maponly_512_threshold_p4_tau050_mem15_block4_closed_loop_report.json

claimed_episodes: 3 / 16
marker_seen_ever: 10 / 16
mean_final_goal_distance: 7.1875 cells
collision_steps: 24 / 422
```

This is a real improvement over the same randomized-palette scorer with the
7x7 support, which claimed only 1/16 closed-loop episodes. It is still not the
desired result. The model can discover the marker in many episodes, but it does
not reliably preserve and exploit that sighting to route back to the marker.

A dedicated `spatial_memory_size=15` fine-tune was also tested:

```text
checkpoint:
  models/checkpoints/phase3a_explore_claim/phase3a_v5_random_palette_mem15_maponly_256.pt

closed-loop block4:
  claimed_episodes: 2 / 16
  marker_seen_ever: 7 / 16
  mean_final_goal_distance: 8.125 cells
  collision_steps: 4 / 427
```

The fine-tune reduced collisions and improved some post-sighting open-loop
regret, but it regressed closed-loop claiming. The current best
non-fixed-color artifact is therefore the derived mem15 scorer above, not the
mem15 fine-tune.

Additional 2026-06-16 iteration results:

```text
full-history evaluator:
  checkpoint:
    models/checkpoints/phase3a_explore_claim/phase3a_v5_random_palette_obsaux_mlp_maponly_512_threshold_p4_tau050_mem15.pt
  change:
    scripts/export_jepa_phase3a_closed_loop_demo_mp4.py now has
    --history-window, where 0 keeps all closed-loop history instead of the
    previous six-frame truncation.
  report:
    .generated/jepa_phase3a/explore_claim_v5_random_palette/phase3a_v5_random_palette_obsaux_mlp_maponly_512_threshold_p4_tau050_mem15_block4_fullhistory_closed_loop_report.json
  result:
    claimed_episodes: 0 / 16
    marker_seen_ever: 8 / 16
```

Interpretation: simply preserving all observations at evaluation time is not
enough. The h6-trained model was not trained to use long histories robustly;
longer replay adds drift/noise and regresses claiming.

```text
longer-history h12 split:
  dataset:
    .generated/jepa_phase3a/explore_claim_v6_random_palette_h12
  checkpoint:
    models/checkpoints/phase3a_explore_claim/phase3a_v6_random_palette_h12_mem15_maponly_128.pt
  closed-loop report:
    .generated/jepa_phase3a/explore_claim_v6_random_palette_h12/phase3a_v6_random_palette_h12_mem15_maponly_128_block4_h12_closed_loop_report.json
  result:
    claimed_episodes: 0 / 16
```

The h12 branch was also negative in open-loop phase metrics:

```text
explore_unseen: prim 0.125, seq_regret 4.662
discover_visible_marker: prim 0.500, seq_regret 4.525, marker_seen 0.500
claim_after_marker_seen: prim 0.625, seq_regret 3.525, claim 0.125
```

Interpretation: increasing history length without changing the learning
objective does not produce a persistent topographical memory.

```text
current-marker-visible split:
  dataset:
    .generated/jepa_phase3a/explore_claim_v7_random_palette_current_marker
  checkpoint:
    models/checkpoints/phase3a_explore_claim/phase3a_v7_random_palette_current_marker_mem15_maponly_128.pt
  closed-loop report:
    .generated/jepa_phase3a/explore_claim_v7_random_palette_current_marker/phase3a_v7_random_palette_current_marker_mem15_maponly_128_block4_h6_closed_loop_report.json
  result:
    claimed_episodes: 1 / 16
```

The v7 split is better aligned with the intended deployed sensor condition:
the visual marker may appear in the current camera view when it is actually
visible, but no beacon/breadcrumb channel is available and palettes are still
randomized. It did not beat the current best artifact. Open-loop v7 final:

```text
explore_unseen: prim 0.333, seq_regret 2.333
discover_visible_marker: prim 0.692, seq_regret 1.965, marker_seen 0.308
claim_after_marker_seen: prim 0.563, seq_regret 2.206, claim 0.250
```

Interpretation: current-view marker training improves post-sighting regret but
hurts exploration/discovery enough that closed-loop claiming regresses.

Continuation on 2026-06-16:

```text
implementation changes:
  - learned spatial-frontier memory now supports marker update threshold/width;
  - closed-loop export can run persistent spatial memory across replans;
  - persistent-marker/bounded-frontier scoring can require that the marker has
    appeared in observation history before switching from exploration to claim;
  - closed-loop export has report-only --skip-video for GPU eval environments
    without imageio/PIL;
  - added marker-memory localization diagnostics for full spatial maps;
  - added exact egocentric marker-memory and online-frontier score sources as
    upper-bound diagnostics.

checkpoint:
  models/checkpoints/phase3a_explore_claim/phase3a_v5_random_palette_mem15_markerfocus_256.pt

training:
  initialized from phase3a_v5_random_palette_mem15_maponly_256.pt
  256 ROCm/GPU steps, marker-focused spatial-frontier memory loss

corrected marker-memory diagnostic:
  report:
    .generated/jepa_phase3a/explore_claim_v5_random_palette/phase3a_v5_random_palette_mem15_markerfocus_256_marker_memory_diag.json
  top1_accuracy: 0.6875
  mean_target_probability: 0.0326
  mean_marker_mass: 0.9538
```

This is a real diagnostic improvement over the previous corrected mem15
diagnostic (`0.3125` top-1), but the belief is still diffuse and marker mass is
high even on phases without a valid remembered-marker target.

Closed-loop reports for the marker-focused checkpoint:

```text
block4, 28 steps, learned spatial_frontier_memory_score:
  report:
    .generated/jepa_phase3a/explore_claim_v5_random_palette/phase3a_v5_random_palette_mem15_markerfocus_256_block4_closed_loop_report.json
  claimed_episodes: 3 / 16
  marker_seen_ever: 9 / 16
  mean_final_goal_distance: 5.938
  collision_steps: 11

block4, 28 steps, observation-gated persistent_marker_bounded_frontier_score:
  report:
    .generated/jepa_phase3a/explore_claim_v5_random_palette/phase3a_v5_random_palette_mem15_markerfocus_256_persistmarker_t085_seen_gate_block4_closed_loop_report.json
  claimed_episodes: 3 / 16
  marker_seen_ever: 9 / 16
  mean_final_goal_distance: 5.812
  collision_steps: 8

block4, 56 steps, learned spatial_frontier_memory_score:
  report:
    .generated/jepa_phase3a/explore_claim_v5_random_palette/phase3a_v5_random_palette_mem15_markerfocus_256_max56_block4_closed_loop_report.json
  claimed_episodes: 4 / 16
  marker_seen_ever: 12 / 16
  mean_final_goal_distance: 4.812
  collision_steps: 16

block4, 56 steps, observation-gated persistent_marker_bounded_frontier_score:
  report:
    .generated/jepa_phase3a/explore_claim_v5_random_palette/phase3a_v5_random_palette_mem15_markerfocus_256_persistmarker_seen_gate_max56_block4_closed_loop_report.json
  claimed_episodes: 3 / 16
  marker_seen_ever: 12 / 16
  mean_final_goal_distance: 5.312
  collision_steps: 14
```

Interpretation: marker-focused loss improves map localization, but it does not
meet the behavioral criterion. Requiring actual observation history before
activating persistent-marker claim mode removes one false-positive failure
mode, but does not improve claims. Longer rollouts reveal more markers but only
raise learned claiming to `4/16`.

Upper-bound diagnostics:

```text
palette-aware exact egocentric marker memory + learned bounded frontier,
full history, block4, 28 steps:
  claimed_episodes: 5 / 16

palette-aware exact egocentric marker memory + learned bounded frontier,
full history, block4, 56 steps:
  report:
    .generated/jepa_phase3a/explore_claim_v5_random_palette/phase3a_v5_random_palette_mem15_markerfocus_256_egomarker_fullhist_max56_block4_closed_loop_report.json
  claimed_episodes: 7 / 16
  marker_seen_ever: 12 / 16
  mean_final_goal_distance: 3.000
  collision_steps: 111

model online_frontier_marker_score, block4, 28 steps:
  report:
    .generated/jepa_phase3a/explore_claim_v5_random_palette/phase3a_v5_random_palette_mem15_markerfocus_256_online_frontier_marker_block4_closed_loop_report.json
  claimed_episodes: 1 / 16
```

The exact egocentric marker-memory result shows that better marker memory can
raise the ceiling, but still not to reliable behavior under the current learned
frontier policy. The model-side online marker scorer is not a fair randomized
palette upper bound because its detector still uses the old fixed-green
threshold. The next implementation should make marker detection palette
agnostic/learned and train a persistent map scorer against full closed-loop
rollouts, not only short candidate batches.

Current required implementation target:

- make the larger memory persist across closed-loop replanning, rather than
  treating each candidate batch as a short local rollout;
- train marker/free/blocked updates on trajectories that require carrying
  evidence outside the camera crop for longer than one candidate horizon;
- score novelty until the marker is discovered, then score claiming from the
  persistent learned memory;
- require randomized-palette closed-loop reports to show both marker discovery
  and robust beacon claiming without privileged simulator breadcrumbs or fixed
  RGB marker detectors.

Phase 3A.12 implementation note, 2026-06-16:

Corrected the v5 randomized-palette closed-loop validation seed. The v5
manifest uses `validation_audit.seed = 21260634`; earlier ad hoc reports run
with `--seed 20260620` reconstructed the wrong maze geometry for the validation
rows and are now considered seed-confounded. The closed-loop exporter now
infers the train/validation seed from the dataset manifest when `--seed` is not
provided and records `scene_seed` in the report.

The explicit palette-aware odometry-frontier memory controller now passes the
2D behavior on the first 16 v5 validation groups when enough exploration steps
are allowed:

```text
score_source: odom_frontier_marker_planner
validation seed: 21260634
checkpoint loaded only for exporter compatibility:
  models/checkpoints/phase3a_explore_claim/phase3a_v5_random_palette_mem15_markerfocus_256.pt

28 steps, history_window 6:
  report:
    .generated/jepa_phase3a/explore_claim_v5_random_palette/phase3a_v5_random_palette_odom_frontier_palette_planner_seed21260634_h6_step1_closed_loop_report.json
  claimed_episodes: 10 / 16
  marker_seen_ever: 11 / 16
  collision_steps: 0

56 steps, full history:
  report:
    .generated/jepa_phase3a/explore_claim_v5_random_palette/phase3a_v5_random_palette_odom_frontier_palette_planner_seed21260634_fullhist_max56_step1_closed_loop_report.json
  claimed_episodes: 12 / 16
  marker_seen_ever: 13 / 16
  collision_steps: 0

112 steps, full history:
  report:
    .generated/jepa_phase3a/explore_claim_v5_random_palette/phase3a_v5_random_palette_odom_frontier_palette_planner_seed21260634_fullhist_max112_step1_closed_loop_report.json
  claimed_episodes: 16 / 16
  marker_seen_ever: 16 / 16
  collision_steps: 0
```

This is an explicit-map 2D controller pass, not a learned JEPA memory pass. It
updates known-free/known-blocked/marker cells from RGB observations and
odometry, plans to the remembered marker when known, and otherwise chooses a
frontier. It establishes that the observation/action interface has enough
information for the task without fixed-color shortcuts.

Learned results with the corrected seed remain negative:

```text
learned marker-focused mem15, 28 steps:
  report:
    .generated/jepa_phase3a/explore_claim_v5_random_palette/phase3a_v5_random_palette_mem15_markerfocus_256_seed21260634_block4_closed_loop_report.json
  claimed_episodes: 1 / 16
  marker_seen_ever: 9 / 16
  collision_steps: 11

planner-distilled candidate scorer, 112 steps, 16 candidates/state:
  checkpoint:
    models/checkpoints/phase3a_explore_claim/phase3a_v5_odom_frontier_distill_candidate_256.pt
  report:
    .generated/jepa_phase3a/explore_claim_v5_odom_frontier_distill/phase3a_v5_odom_frontier_distill_candidate_256_seed21260634_candidate_score_top16_step1_max112_closed_loop_report.json
  claimed_episodes: 0 / 16
  marker_seen_ever: 1 / 16

planner-distilled memory-context ranker, 112 steps, 16 candidates/state:
  checkpoint:
    models/checkpoints/phase3a_explore_claim/phase3a_v5_odom_frontier_distill_memctx_rank_1024.pt
  report:
    .generated/jepa_phase3a/explore_claim_v5_odom_frontier_distill/phase3a_v5_odom_frontier_distill_memctx_rank_1024_seed21260634_candidate_score_top16_step1_max112_closed_loop_report.json
  claimed_episodes: 0 / 16
  marker_seen_ever: 6 / 16
```

The first distillation collapses to mostly `forward` and repeatedly collides.
The memory-context ranker uses turns and sometimes reaches marker visibility,
but it can oscillate near the remembered target instead of claiming it. The
next learned-memory target should therefore copy the successful explicit map
interface more directly: train/update an allocentric or egocentric occupancy
and marker map, then score frontier/claim actions from that map, rather than
expecting a generic candidate head to infer the controller from short
candidate rows.

Phase 3A.13 implementation note, 2026-06-16:

Implemented the first JEPA-latent map-perception pass. This keeps the proven
odometry/frontier memory structure, but replaces the exact RGB wall/marker
classifier with a learned contextual head over Phase 3A JEPA spatial tokens:

```text
RGB crop
  -> Phase3A JEPA pixel tokens
  -> Phase3ALatentMapHead
  -> local blocked/free/marker evidence
  -> persistent odometry map
  -> frontier/remembered-marker controller
```

Training artifact:

```text
script:
  scripts/train_jepa_phase3a_latent_map.py
base checkpoint:
  models/checkpoints/phase3a_explore_claim/phase3a_v5_random_palette_mem15_markerfocus_256.pt
latent map checkpoint:
  models/checkpoints/phase3a_explore_claim/phase3a_v5_latent_map_ctx_4096.pt
train frames: 1763
validation frames: 888
final validation:
  blocked_accuracy: 1.000
  free_accuracy: 1.000
  marker_top1_when_visible: 1.000
  marker_presence_recall: 1.000
  marker_false_presence_rate: 0.007
```

Closed-loop result with the learned latent map head:

```text
score_source: latent_odom_frontier_planner
scene_seed: 21260634 inferred from the validation manifest
latent_map_marker_threshold: 0.9
max_steps: 112
max_episodes: 16
report:
  .generated/jepa_phase3a/explore_claim_v5_random_palette/phase3a_v5_latent_map_ctx_4096_seedinfer_marker090_max112_report.json
claimed_episodes: 16 / 16
marker_seen_ever: 16 / 16
collision_steps: 0
```

This is now a learned-latent perception plus explicit odometry-map controller.
It is stronger than the previous explicit RGB map result because the semantic
free/blocked/marker evidence comes from JEPA-token features rather than exact
palette equality tests. It is still not a pure latent-SLAM result: the map is
written into an exocentric grid using perfect pose/yaw and cell alignment. The
next weakening should remove or degrade one privileged assumption at a time:
first action-integrated egocentric rolling without global `(x, y)`, then noisy
odometry, then learned traversability/perception on Go2-like observations.

Phase 3A.14 implementation note, 2026-06-16:

Implemented the first action-rolled egocentric latent-memory controller. This
keeps the learned `Phase3ALatentMapHead` perception bridge, but removes global
`(x, y)` writes from the online map. The controller stores free/blocked/marker
evidence in a finite egocentric grid, rolls that memory through the executed
primitive action, marks forward collisions in the egocentric cell ahead, and
uses the same explicit frontier/remembered-marker policy over that rolled
memory.

```text
RGB crop
  -> Phase3A JEPA pixel tokens
  -> Phase3ALatentMapHead
  -> local blocked/free/marker evidence
  -> finite egocentric memory rolled by executed actions
  -> frontier/remembered-marker controller
```

Correct-seed randomized-palette result:

```text
score_source: latent_egocentric_frontier_planner
scene_seed: 21260634 inferred from the validation manifest
latent_map_marker_threshold: 0.9
spatial_memory_size: 31
max_steps: 68
max_episodes: 16
report:
  .generated/jepa_phase3a/explore_claim_v5_random_palette/phase3a_v5_latent_egocentric_ctx_4096_mem31_marker090_max68_report.json
claimed_episodes: 16 / 16
marker_seen_ever: 16 / 16
collision_steps: 0
step range: 6..68
```

Memory-support ablation:

```text
default spatial_memory_size: 15
max_steps: 112
report:
  .generated/jepa_phase3a/explore_claim_v5_random_palette/phase3a_v5_latent_egocentric_ctx_4096_mem15_marker090_max112_report.json
claimed_episodes: 15 / 16
marker_seen_ever: 15 / 16
collision_steps: 0
failed episode: index 3, never saw marker, final_goal_distance 5
```

Interpretation:

- This is the first randomized-palette Phase 3A result that combines JEPA-latent
  local perception with online memory and removes exocentric global map writes.
  It is no longer solving as if it had an exocentric camera.
- It is still not a pure learned planner or learned SLAM system. Runtime still
  assumes exact action integration, grid-cell alignment, explicit collision
  feedback, and a hand-written frontier/BFS controller. The learned component is
  the local semantic map evidence extracted from JEPA spatial tokens.
- The 15-cell memory ablation shows that finite support remains a real ceiling.
  The passing 31-cell result should be treated as the current 2D requirement
  pass and as a bridge target for the next learned recurrent-memory objective,
  not as a Go2 transfer claim.

Phase 3A.15 implementation note, 2026-06-16:

Implemented the first trained recurrent egocentric memory updater. This keeps
the JEPA-token local perception head and the explicit frontier/remembered-marker
controller, but replaces the exporter-side set-based memory update/roll with a
`Phase3AEgocentricMemoryUpdate` tensor model:

```text
previous egocentric memory tensor
  + current JEPA-latent local blocked/free/marker evidence
  + previous primitive action
  + previous collision bit
  -> recurrent egocentric memory tensor
  -> frontier/remembered-marker controller
```

Training artifact:

```text
script:
  scripts/train_jepa_phase3a_latent_memory.py
checkpoint:
  models/checkpoints/phase3a_explore_claim/phase3a_v5_latent_memory_recurrent_2048.pt
base checkpoint:
  models/checkpoints/phase3a_explore_claim/phase3a_v5_random_palette_mem15_markerfocus_256.pt
latent map checkpoint:
  models/checkpoints/phase3a_explore_claim/phase3a_v5_latent_map_ctx_4096.pt
train examples: 698
validation examples: 428
final validation:
  blocked_accuracy: 1.000
  free_accuracy: 1.000
  marker_accuracy: 1.000
  marker_top1_when_present: 1.000
  teacher_action_match: 1.000
```

Correct-seed randomized-palette closed-loop result:

```text
score_source: latent_recurrent_egocentric_frontier_planner
scene_seed: 21260634 inferred from the validation manifest
spatial_memory_size: 31
latent_memory_marker_threshold: 0.9
max_steps: 68
max_episodes: 16
report:
  .generated/jepa_phase3a/explore_claim_v5_random_palette/phase3a_v5_latent_recurrent_memory_2048_mem31_marker090_max68_report.json
claimed_episodes: 16 / 16
marker_seen_ever: 16 / 16
collision_steps: 0
step range: 6..68
```

Interpretation:

- This is a real weakening relative to Phase 3A.14: the online memory state is
  now produced by a trained recurrent model and fed back closed-loop, rather
  than maintained as Python sets with explicit per-cell update calls.
- It is still not a pure learned SLAM/planning result. The updater architecture
  uses an action-geometric prior (`use_geometric_prior: true`) before learned
  fusion, the local map targets come from the previous teacher controller, and
  the action selection policy is still the hand-written frontier/BFS controller.
- The next requirement should remove one of those remaining scaffolds: either
  train a no-geometric-prior recurrent updater that survives closed loop, or
  replace the explicit frontier/BFS policy with a learned scorer over the
  recurrent memory while retaining the same ablations.

Phase 3A.16 implementation note, 2026-06-16:

Ran the first two scaffold-removal iterations against the Phase 3A.15 passing
artifact. Neither is a replacement for the selected recurrent-memory result.

Permission/automation change:

```text
scripts/phase3a_rocm_train_latent_memory.sh
scripts/phase3a_rocm_train_latent_policy.sh
scripts/phase3a_rocm_export_closed_loop.sh
```

These wrappers centralize the ROCm environment setup and route repeated GPU
training/eval jobs through stable command prefixes. The policy-train and
closed-loop-export prefixes have been exercised and approved; the memory-train
wrapper is ready for the same one-time approval on its next use.

No-geometric-prior recurrent-memory ablation:

```text
script:
  scripts/train_jepa_phase3a_latent_memory.py --no-geometric-prior
best closed-loop checkpoint:
  models/checkpoints/phase3a_explore_claim/phase3a_v5_latent_memory_nogeom_allgroups_4096.pt
train examples: 879
validation examples: 428
final validation:
  blocked_accuracy: 0.962
  free_accuracy: 0.961
  marker_accuracy: 1.000
  marker_top1_when_present: 0.991
  teacher_action_match: 0.921
closed-loop report:
  .generated/jepa_phase3a/explore_claim_v5_random_palette/phase3a_v5_latent_memory_nogeom_allgroups_4096_mem31_marker090_max68_report.json
closed-loop result:
  claimed_episodes: 12 / 16
  marker_seen_ever: 12 / 16
  collision_steps: 0
```

The earlier train-subset no-prior checkpoint also reached 12/16 at both max-68
and max-112, and its four failures never saw the marker. Synthetic-transition,
staged-pretrain, and direct full-rank transition variants did not improve
validation action match and were stopped. Interpretation: removing the fixed
action-geometric prior weakens exploration coverage even though the supervised
memory metrics remain high.

Learned policy over the passing recurrent memory:

```text
script:
  scripts/train_jepa_phase3a_latent_policy.py
recurrent memory:
  models/checkpoints/phase3a_explore_claim/phase3a_v5_latent_memory_recurrent_2048.pt
first policy checkpoint:
  models/checkpoints/phase3a_explore_claim/phase3a_v5_latent_policy_mem31_2048.pt
first policy closed-loop:
  claimed_episodes: 1 / 16
  marker_seen_ever: 8 / 16
  collision_steps: 749
balanced synthetic policy checkpoint:
  models/checkpoints/phase3a_explore_claim/phase3a_v5_latent_policy_aug8192_balanced_4096.pt
balanced synthetic policy final validation:
  action_match: 0.748
balanced synthetic policy closed-loop report:
  .generated/jepa_phase3a/explore_claim_v5_random_palette/phase3a_v5_latent_policy_aug8192_balanced_4096_max68_report.json
balanced synthetic policy closed-loop:
  claimed_episodes: 4 / 16
  marker_seen_ever: 10 / 16
  collision_steps: 521
```

Interpretation: the one-step learned policy can sometimes use the recurrent
memory, but it does not learn the safety/planning loop supplied by the explicit
frontier/BFS controller. Many failures see the marker and then collide. The next
policy-side attempt should not be another one-step imitation head; it should use
a closed-loop objective, value/planning target, or differentiable planning head
with explicit safety supervision.

Phase 3A.17 implementation note, 2026-06-16:

Implemented the first value/safety/DAgger policy iteration over recurrent JEPA
memory. The policy head now supports a coordinate-aware convolutional
architecture, supervised per-action score targets, a blocked-forward safety
loss, and DAgger collection from recurrent-memory closed-loop states.

Best artifact from this iteration:

```text
checkpoint:
  models/checkpoints/phase3a_explore_claim/phase3a_v5_latent_policy_conv_dagger_v1_4096.pt
policy architecture:
  conv over 31x31 recurrent memory + coordinate planes
training:
  879 teacher-path examples
  8192 synthetic safety/frontier examples
  2 DAgger rounds, 879 rollout examples each
selected step: 2560
selected validation:
  action_match: 0.829
  blocked_forward_violation_rate: 0.000
  predicted_action_counts:
    forward: 323
    turn_left: 70
    turn_right: 35
    hold: 0
```

Closed-loop result:

```text
score_source: latent_recurrent_policy_planner
report:
  .generated/jepa_phase3a/explore_claim_v5_random_palette/phase3a_v5_latent_policy_conv_dagger_v1_4096_max68_report.json
claimed_episodes: 8 / 16
marker_seen_ever: 11 / 16
collision_steps: 0

longer-horizon diagnostic:
  .generated/jepa_phase3a/explore_claim_v5_random_palette/phase3a_v5_latent_policy_conv_dagger_v1_4096_max112_report.json
  claimed_episodes: 8 / 16
  marker_seen_ever: 11 / 16
  collision_steps: 0
```

This is a real improvement over the previous learned policy (`4/16`, 521
collisions): the safety term eliminated collisions and DAgger doubled the claim
rate. It still does not meet the requirement, and increasing the horizon from
68 to 112 steps does not help. The remaining failure is a stable policy
attractor: some episodes never discover the marker, and several marker-seen
episodes still fail to return.

Additional negatives:

```text
no-synthetic DAgger:
  result before termination:
    action_match fell below the selected v1 checkpoint
    blocked_forward_violation_rate remained nonzero

small-synthetic stronger-safety DAgger:
  result before termination:
    action_match stayed below v1
    blocked_forward_violation_rate remained nonzero

random-exploration DAgger:
  result before termination:
    more off-path rollout examples, but under-selected forward
    action_match stayed below v1
```

Interpretation: one-step learned action selection, even with value-shaped
targets, safety supervision, and DAgger, is still the wrong abstraction for
replacing the explicit frontier/BFS controller. The next policy-side
implementation should expose multi-step planning structure directly: e.g. a
differentiable value-iteration/planning head over the recurrent memory, or a
learned target-cell/value field with action extraction, rather than another
single argmax action classifier.

Phase 3A.18 implementation note, 2026-06-17:

Implemented the first multi-step value-field planner over the trained recurrent
JEPA memory. This keeps the Phase 3A.15 recurrent memory updater fixed, but
replaces the explicit egocentric BFS controller with value propagation over the
current recurrent memory tensor:

```text
recurrent JEPA memory tensor
  -> thresholded known-free / known-blocked / marker field
  -> marker reward if marker is remembered and reachable
  -> otherwise frontier-cell reward over known-free memory
  -> iterative value propagation over known-free neighbors
  -> first primitive from value-gradient direction
```

Closed-loop result:

```text
score_source: latent_recurrent_value_field_planner
base checkpoint:
  models/checkpoints/phase3a_explore_claim/phase3a_v5_random_palette_mem15_markerfocus_256.pt
latent map:
  models/checkpoints/phase3a_explore_claim/phase3a_v5_latent_map_ctx_4096.pt
recurrent memory:
  models/checkpoints/phase3a_explore_claim/phase3a_v5_latent_memory_recurrent_2048.pt
report:
  .generated/jepa_phase3a/explore_claim_v5_random_palette/phase3a_v5_latent_recurrent_value_field_2048_mem31_marker090_max68_report.json
max_steps: 68
max_episodes: 16
claimed_episodes: 16 / 16
marker_seen_ever: 16 / 16
collision_steps: 0
step range: 6..68
```

Interpretation:

- This meets the current 2D randomized-palette closed-loop gate with learned
  JEPA-token perception, trained recurrent egocentric memory, and a multi-step
  value-field planner.
- It also explains the Phase 3A.17 failure: the missing abstraction was not
  another one-step policy head, but explicit multi-step propagation over memory.
- This is still not a pure learned planner. The reward field is hand-designed
  from marker/frontier structure, the memory-to-symbol thresholding is fixed,
  action integration inside the memory updater still uses a geometric prior,
  and the value propagation algorithm is fixed. The next clean learning step is
  to learn the reward/target field or value residual while retaining the
  differentiable multi-step planner.

Phase 3A.19 implementation note, 2026-06-17:

Implemented and evaluated the first learned target-field heads for the
value-field planner. The new `Phase3AEgocentricValueFieldHead` reads the trained
recurrent JEPA memory tensor and predicts target rewards for fixed value
propagation. The closed-loop exporter now supports
`latent_recurrent_learned_value_field_planner`.

Training artifact:

```text
script:
  scripts/train_jepa_phase3a_value_field.py
wrapper:
  scripts/phase3a_rocm_train_value_field.sh
single-channel checkpoint:
  models/checkpoints/phase3a_explore_claim/phase3a_v5_latent_value_field_head_4096.pt
selected step: 3584
selected validation:
  action_match: 0.970
  target_top1_match: 1.000
  target_recall: 0.910
  target_precision: 0.832
```

Full learned marker/frontier target field remained just short of the gate:

```text
single-channel learned target field:
  report:
    .generated/jepa_phase3a/explore_claim_v5_random_palette/phase3a_v5_latent_learned_value_field_4096_mem31_marker090_max68_report.json
  claimed_episodes: 15 / 16
  marker_seen_ever: 16 / 16
  collision_steps: 0
  failure: episode 11 saw the marker but followed a bad learned target field

high-recall 1536-step checkpoint:
  checkpoint:
    models/checkpoints/phase3a_explore_claim/phase3a_v5_latent_value_field_head_recall_1536.pt
  claimed_episodes: 14 / 16

split marker/frontier channels:
  checkpoint:
    models/checkpoints/phase3a_explore_claim/phase3a_v5_latent_value_field_head_split_4096.pt
  report:
    .generated/jepa_phase3a/explore_claim_v5_random_palette/phase3a_v5_latent_learned_value_field_split_4096_mem31_marker090_max68_report.json
  claimed_episodes: 13 / 16
```

The useful passing ablation is learned frontier targets plus a fixed marker
target once the recurrent memory has bound the marker:

```text
score_source:
  latent_recurrent_learned_value_field_planner
flag:
  --latent-value-fixed-marker-target
checkpoint:
  models/checkpoints/phase3a_explore_claim/phase3a_v5_latent_value_field_head_4096.pt
report:
  .generated/jepa_phase3a/explore_claim_v5_random_palette/phase3a_v5_latent_learned_frontier_fixed_marker_4096_mem31_marker090_max68_report.json
claimed_episodes: 16 / 16
marker_seen_ever: 16 / 16
collision_steps: 0
step range: 6..68
```

Interpretation:

- The learned target-field path can now replace the hand-written frontier reward
  field without regressing the gate.
- The marker-claim target should remain fixed for the selected artifact; the
  full learned marker+frontier target field is still a measured negative at
  15/16 best.
- The next learning step should focus specifically on marker-target reliability
  after sighting, for example with marker-only hard-negative states around the
  episode-11 failure, calibrated marker-channel confidence, or a learned
  residual on top of the fixed marker target.

Phase 3A.20 implementation note, 2026-06-17:

Closed the remaining learned target-field gap with a marker-state extraction
fix rather than a new target head. The failed `15/16` single-channel head had
already put the marker target at top-1, but marker-state value propagation was
using `top_k=16`, so secondary high-probability cells could poison the return
path after marker sighting. The exporter now supports separate marker-state
learned target extraction:

```text
frontier target extraction:
  --latent-value-target-top-k 16
marker target extraction:
  --latent-value-marker-target-top-k 1
fixed marker target:
  false
```

Selected strict learned-target result:

```text
score_source:
  latent_recurrent_learned_value_field_planner
checkpoint:
  models/checkpoints/phase3a_explore_claim/phase3a_v5_latent_value_field_head_4096.pt
report:
  .generated/jepa_phase3a/explore_claim_v5_random_palette/phase3a_v5_latent_learned_value_field_4096_mem31_marker090_markertop1_max68_report.json
claimed_episodes: 16 / 16
marker_seen_ever: 16 / 16
collision_steps: 0
step range: 6..68
latent_value_fixed_marker_target: false
```

This supersedes the earlier `15/16` full learned marker/frontier target result
for the selected extractor configuration. The marker target itself is still
learned from the recurrent JEPA memory field; the remaining fixed pieces are
the recurrent-memory thresholding, geometric action prior inside the memory
updater, phase-conditioned extraction rule (`frontier top_k=16`, marker
top_k=1), and fixed value propagation/action extraction.

Negative checked during this iteration:

```text
checkpoint:
  models/checkpoints/phase3a_explore_claim/phase3a_v5_latent_value_field_head_dagger_markerw4_4096.pt
training:
  DAgger-style learned-head rollout augmentation plus marker_sample_weight 4
selected validation:
  marker_action_match: 0.902
closed-loop report:
  .generated/jepa_phase3a/explore_claim_v5_random_palette/phase3a_v5_latent_learned_value_field_dagger_markerw4_4096_mem31_marker090_max68_report.json
claimed_episodes: 12 / 16
```

Interpretation: the remaining issue was not lack of marker supervision. It was
the mismatch between sparse marker target labels and broad top-k target
extraction in marker-present states. The next scaffold-removal target is
therefore not another value-field target sweep; it is to learn or justify the
phase-conditioned extractor/value propagation itself.

Phase 3A.21 implementation note, 2026-06-17:

Implemented and evaluated the first learned extractor-mode head for the
value-field planner. The new `Phase3AValueFieldExtractorHead` reads the trained
recurrent egocentric memory tensor and predicts whether learned target-field
extraction should switch from broad frontier mode to sparse target mode. This
removes the manual `--latent-value-marker-target-top-k 1` gate from the
selected closed-loop command.

Training artifact:

```text
script:
  scripts/train_jepa_phase3a_value_extractor.py
wrapper:
  scripts/phase3a_rocm_train_value_extractor.sh
checkpoint:
  models/checkpoints/phase3a_explore_claim/phase3a_v5_latent_value_extractor_1024.pt
selected validation:
  accuracy: 1.000
  sparse_precision: 1.000
  sparse_recall: 1.000
  sparse_examples: 112 / 428
```

Closed-loop selected result with learned extractor:

```text
score_source:
  latent_recurrent_learned_value_field_planner
target-field checkpoint:
  models/checkpoints/phase3a_explore_claim/phase3a_v5_latent_value_field_head_4096.pt
extractor checkpoint:
  models/checkpoints/phase3a_explore_claim/phase3a_v5_latent_value_extractor_1024.pt
report:
  .generated/jepa_phase3a/explore_claim_v5_random_palette/phase3a_v5_latent_learned_value_field_extractor_1024_mem31_marker090_max68_report.json
gate:
  .generated/jepa_phase3a/explore_claim_v5_random_palette/phase3a_v5_latent_learned_value_field_extractor_1024_mem31_marker090_max68_gate.json
claimed_episodes: 16 / 16
marker_seen_ever: 16 / 16
collision_steps: 0
step range: 6..68
latent_value_fixed_marker_target: false
latent_value_marker_target_top_k: null
latent_value_sparse_target_top_k: 1
gate passed: true
```

The new `scripts/check_jepa_phase3a_closed_loop_gate.py` formalizes the
closed-loop requirement and can require a learned extractor checkpoint. This is
now the selected 2D randomized-palette artifact: learned JEPA-token perception,
trained recurrent egocentric memory, learned marker/frontier target field, and
learned sparse-vs-broad extractor mode, with fixed value propagation/action
extraction still remaining.

Phase 3A.22 implementation note, 2026-06-17:

Attempted to remove the remaining fixed value propagation/action extraction by
distilling it into a learned action head. The new
`Phase3AValueFieldActionHead` consumes recurrent memory, the learned target
field selected by the value-field head, and the learned sparse/broad extractor
probability, then predicts the first primitive directly.

Implemented artifacts:

```text
model:
  Phase3AValueFieldActionHead
trainer:
  scripts/train_jepa_phase3a_value_action.py
wrapper:
  scripts/phase3a_rocm_train_value_action.sh
export score source:
  latent_recurrent_learned_value_action_planner
```

Results were negative:

```text
plain action distillation:
  checkpoint:
    models/checkpoints/phase3a_explore_claim/phase3a_v5_latent_value_action_2048.pt
  selected validation action_match: 0.818
  closed-loop report:
    .generated/jepa_phase3a/explore_claim_v5_random_palette/phase3a_v5_latent_learned_value_action_2048_mem31_marker090_max68_report.json
  claimed_episodes: 1 / 16
  marker_seen_ever: 1 / 16
  collision_steps: 524

DAgger rollout augmentation from the failed action head:
  checkpoint:
    models/checkpoints/phase3a_explore_claim/phase3a_v5_latent_value_action_dagger_4096.pt
  selected validation action_match: 0.832
  closed-loop report:
    .generated/jepa_phase3a/explore_claim_v5_random_palette/phase3a_v5_latent_learned_value_action_dagger_4096_mem31_marker090_max68_report.json
  claimed_episodes: 0 / 16

synthetic local-memory augmentation:
  checkpoint:
    models/checkpoints/phase3a_explore_claim/phase3a_v5_latent_value_action_synth8192_4096.pt
  selected validation action_match: 0.893
  closed-loop report:
    .generated/jepa_phase3a/explore_claim_v5_random_palette/phase3a_v5_latent_learned_value_action_synth8192_4096_mem31_marker090_max68_report.json
  claimed_episodes: 3 / 16

label-smoothed synthetic action head:
  checkpoint:
    models/checkpoints/phase3a_explore_claim/phase3a_v5_latent_value_action_synth8192_smooth005_4096.pt
  selected validation action_match: 0.893
  pure learned-action closed loop:
    claimed_episodes: 1 / 16
  confidence-gated residual fallback:
    threshold 0.95 claimed_episodes: 1 / 16
    threshold 0.999 claimed_episodes: 0 / 16
```

Interpretation: direct learned first-action extraction is not a drop-in
replacement for value propagation. Even when validation action imitation reaches
about `0.89`, closed-loop behavior collapses through confident wrong frontier
actions and repeated collisions. The selected artifact therefore remains Phase
3A.21. The next scaffold-removal step should learn a dense value map or
differentiable planner objective, not another first-action classifier.

Phase 3A.23 implementation note, 2026-06-17:

Implemented the dense value-map and latent soft-planner follow-up. The new
`Phase3AValueMapPlannerHead` predicts a dense egocentric value map from the
trained recurrent memory, learned target field, and learned sparse/broad
extractor probability. The exporter now has
`latent_recurrent_learned_value_map_planner` and
`latent_recurrent_soft_value_map_planner` score sources. The latter runs soft
value propagation directly over learned memory probabilities and sparse learned
target tensors.

Best learned dense-planner result so far:

```text
checkpoint:
  models/checkpoints/phase3a_explore_claim/phase3a_v5_latent_value_map_planner_dagger_synth8192_3072.pt
training:
  dense value-map distillation plus one DAgger rollout pass from the learned
  dense planner, with 8192 synthetic local-memory examples
selected validation:
  action_match: 0.900
  broad_action_match: 0.880
  sparse_action_match: 0.955
closed-loop report:
  .generated/jepa_phase3a/explore_claim_v5_random_palette/phase3a_v5_latent_value_map_dagger_synth8192_3072_frontierexit_mem31_marker090_max68_report.json
claimed_episodes: 11 / 16
marker_seen_ever: 12 / 16
collision_steps: 0
max96 diagnostic:
  claimed_episodes: 11 / 16
```

Other dense/soft planner checks:

```text
plain dense value-map, synthetic 8192:
  best closed-loop claimed_episodes: 8 / 16

dense value-map, stronger h128/synth16384/gamma0.90/positive_weight16:
  validation regressed; not selected

second DAgger pass:
  validation regressed to action_match: 0.850; not exported

DAgger without synthetic augmentation:
  selected validation action_match: 0.895
  closed-loop claimed_episodes: 6 / 16

soft latent value-map planner over learned memory probabilities:
  raw learned targets: 8 / 16
  sparse learned targets: 9 / 16
  top-k/gamma sweep: no better than 9 / 16
```

Interpretation: dense value-map supervision is the first scaffold-removal path
that materially improves over direct action imitation (`11/16` vs `3/16`), and
it removes fixed global value propagation from the learned-planner branch. It
still does not meet the 2D gate. The remaining failures are mostly exploration
misses rather than marker-return failures: in the best dense-planner run, four
of five failed episodes never saw the marker. The selected passing artifact
therefore remains Phase 3A.21 with learned perception, learned recurrent
memory, learned target field, learned extractor mode, and fixed value
propagation/action extraction.

Phase 3A.24 implementation note, 2026-06-17:

Added action-ranked dense value-map training, runtime output ensembling for
multiple learned value-map heads, optional marker-only learned value-map heads,
and an explicitly marked fixed-marker-return hybrid. The best strict learned
value-map planner now uses a mean output ensemble of the original dense DAgger
planner and the action-ranked DAgger2 planner:

```text
strict learned value-map ensemble, max68:
  report:
    .generated/jepa_phase3a/explore_claim_v5_random_palette/phase3a_v5_latent_value_map_ensemble_mean_dense_action2_mem31_marker090_max68_report.json
  claimed_episodes: 13 / 16
  marker_seen_ever: 14 / 16

strict learned value-map ensemble, max96:
  report:
    .generated/jepa_phase3a/explore_claim_v5_random_palette/phase3a_v5_latent_value_map_ensemble_mean_dense_action2_mem31_marker090_max96_report.json
  claimed_episodes: 15 / 16
  marker_seen_ever: 15 / 16
```

Weighted learned ensembles can trade off the remaining failure. A 2x dense +
2x action-ranked DAgger2 + 1x action-ranked DAgger3 output mean reaches
`15/16` at max96 with all markers seen, but fails marker return on episode 12.
Marker-only learned heads, fixed marker target injection, and marker-head
self-DAgger did not close that final return loop.

An explicitly marked hybrid closes the 2D validation set:

```text
learned value-map exploration ensemble + fixed marker return, max128:
  report:
    .generated/jepa_phase3a/explore_claim_v5_random_palette/phase3a_v5_latent_value_map_ensemble_weighted_2dense_2action2_1action3_hybrid_fixedmarkerreturn_mem31_marker090_max128_report.json
  claimed_episodes: 16 / 16
  marker_seen_ever: 16 / 16
  collision_steps: 0
```

Interpretation: learned JEPA-token memory plus learned value-map exploration can
now discover every randomized-palette validation marker given a longer horizon,
but the all-learned marker-return planner remains one episode short. The hybrid
is useful as a systems milestone, not as proof that fixed planning has been
fully removed.

Phase 3A.25 implementation note, 2026-06-17:

Replaced the fixed marker-return fallback with a learned marker-return action
head trained only on sparse marker-memory states with exact marker target input.
The exporter now supports `--latent-value-map-marker-action-return`, which
keeps learned value-map exploration and switches to the learned action head
only after marker memory exists.

```text
learned value-map exploration ensemble + learned marker-action return:
  marker action checkpoint:
    models/checkpoints/phase3a_explore_claim/phase3a_v5_latent_marker_action_fixedtarget_sparse_synth8192_4096.pt
  selected marker-action validation:
    sparse_action_match: 0.964
  max96 report:
    .generated/jepa_phase3a/explore_claim_v5_random_palette/phase3a_v5_latent_value_map_ensemble_weighted_marker_action_fixedtarget_mem31_marker090_max96_report.json
  claimed_episodes: 16 / 16
  marker_seen_ever: 16 / 16
  collision_steps: 0
```

This is the first strict learned-return run to close the randomized-palette
validation set. It still needs a longer horizon than the fixed-return baseline:
the same controller gets `14/16` at max68. The current remaining gap is speed
and efficiency, not whether the learned memory can eventually discover and
return to all validation markers.

Phase 3A.26 implementation note, 2026-06-17:

Closed the strict learned max68 gate by adding an egocentric latent-memory route
gate to the value-map exporter. The controller keeps the weighted learned
value-map ensemble for general exploration, switches to the dense learned
value-map fallback after step 7 when marker memory is still absent, and latches
that dense fallback immediately when recurrent egocentric memory contains a
contiguous side wall next to the agent. Marker return still uses the learned
marker-action head; the fixed marker-return fallback remains disabled.

```text
strict learned value-map exploration + learned marker-action return, max68:
  report:
    .generated/jepa_phase3a/explore_claim_v5_random_palette/phase3a_v5_latent_value_map_weighted_fallback_dense_s7_sidewall_marker_action_mem31_marker090_max68_report.json
  mp4:
    .generated/jepa_phase3a/explore_claim_v5_random_palette/phase3a_v5_latent_value_map_weighted_fallback_dense_s7_sidewall_marker_action_mem31_marker090_max68_demo.mp4
  claimed_episodes: 16 / 16
  marker_seen_ever: 16 / 16
  collision_steps: 0
  fixed_marker_return: false
```

Interpretation: this meets the current 2D randomized-palette closed-loop
criterion with learned JEPA-latent memory, learned value-map exploration, and
learned marker-action return. The remaining caveat is that the successful
exploration policy is a small explicit gate over the learned egocentric memory,
not a single monolithic policy head.

Phase 3A.27 implementation note, 2026-06-17:

Replaced the runtime side-wall route gate with a learned latent-memory router.
The router is a compact conv classifier over the recurrent egocentric memory
tensor and latches the dense learned value-map fallback when its probability
crosses threshold. It was trained through the ROCm value-map trainer on real
rollout memories plus balanced synthetic route-memory examples distilled from
the previous side-wall gate.

```text
learned router checkpoint:
  models/checkpoints/phase3a_explore_claim/phase3a_v5_latent_value_map_router_sidewall_distill_synth4096_1024.pt
  selected validation:
    accuracy: 0.972
    precision: 0.875
    recall: 0.955

strict learned value-map exploration + learned router + learned marker-action return, max68:
  report:
    .generated/jepa_phase3a/explore_claim_v5_random_palette/phase3a_v5_latent_value_map_weighted_fallback_dense_s7_router_marker_action_mem31_marker090_max68_report.json
  mp4:
    .generated/jepa_phase3a/explore_claim_v5_random_palette/phase3a_v5_latent_value_map_weighted_fallback_dense_s7_router_marker_action_mem31_marker090_max68_demo.mp4
  claimed_episodes: 16 / 16
  marker_seen_ever: 16 / 16
  collision_steps: 0
  fixed_marker_return: false
  explicit_side_wall_fallback: false
```

Interpretation: the current selected 2D result no longer uses the explicit
runtime side-wall detector. The remaining caveat is provenance: the router is
distilled from that detector, so the next evidence target is route labels from
planner counterfactual rollouts or broader disjoint seeds rather than from the
hand-written gate.

Phase 3A.28 implementation note, 2026-06-17:

Removed the route-label provenance caveat by training a new latent-memory
router from planner counterfactual action utilities. For each rollout memory,
the trainer compares the weighted learned value-map ensemble's first action
against the dense fallback's first action using the same horizon-4 candidate
utility rows used by closed-loop evaluation. No side-wall synthetic labels are
used for this checkpoint.

```text
counterfactual router checkpoint:
  models/checkpoints/phase3a_explore_claim/phase3a_v5_latent_value_map_router_counterfactual_s7_utility_h4_1024.pt
  label source: counterfactual
  train positives: 53 / 941
  validation positives: 11 / 421
  selected validation:
    accuracy: 0.950
    precision: 0.188
    recall: 0.273

strict learned value-map exploration + counterfactual router + learned marker-action return, max68:
  report:
    .generated/jepa_phase3a/explore_claim_v5_random_palette/phase3a_v5_latent_value_map_weighted_fallback_dense_s7_router_counterfactual_marker_action_mem31_marker090_max68_report.json
  mp4:
    .generated/jepa_phase3a/explore_claim_v5_random_palette/phase3a_v5_latent_value_map_weighted_fallback_dense_s7_router_counterfactual_marker_action_mem31_marker090_max68_demo.mp4
  claimed_episodes: 16 / 16
  marker_seen_ever: 16 / 16
  collision_steps: 0
  fixed_marker_return: false
  explicit_side_wall_fallback: false
```

Interpretation: the selected 2D max68 result now uses learned JEPA-latent
memory, learned value-map exploration, a learned router trained from planner
counterfactual labels, and learned marker-action return. The remaining caveat
is generalization: counterfactual fallback-positive labels are sparse, and the
result is still proven only on the current randomized-palette validation set.

Phase 3A.29 implementation note, 2026-06-17:

Ran the broader disjoint randomized-palette validation check requested by the
Phase 3A.28 caveat. The new split uses seed `20260701`, eight train scenes,
eight validation scenes, and evaluates the first 64 validation source/goal
groups. The original strict counterfactual-router controller does not
generalize as a solved result on this broader split.

```text
broad validation data:
  .generated/jepa_phase3a/explore_claim_v5_random_palette_generalization_seed20260701/validation_phase3a_positive_control.jsonl

old selected strict controller:
  claimed_episodes: 37 / 64

learned broad latent map + recurrent latent memory ceiling checks:
  exact odom frontier: 62 / 64
  exact latent recurrent frontier: 60 / 64

best strict learned controller found in this iteration:
  learned broad latent map:
    models/checkpoints/phase3a_explore_claim/phase3a_v5_latent_map_ctx_broad_seed20260701_4096.pt
  learned broad recurrent memory:
    models/checkpoints/phase3a_explore_claim/phase3a_v5_latent_memory_recurrent_broad_map_broad_seed20260701_2048.pt
  exploration planner:
    models/checkpoints/phase3a_explore_claim/phase3a_v5_latent_value_map_planner_broad_map_memory_explicit_frontier_broadonly_dagger3_2048.pt
  marker-action return:
    models/checkpoints/phase3a_explore_claim/phase3a_v5_latent_value_action_synth8192_smooth005_4096.pt
  runtime guard:
    --latent-value-map-turn-oscillation-breaker
  report:
    .generated/jepa_phase3a/explore_claim_v5_random_palette_generalization_seed20260701/phase3a_v5_latent_value_map_broad_map_memory_frontier_dagger3_marker_action_synthsmooth_turnbreak_mem31_marker090_max68_generalization64_report.json
  claimed_episodes: 59 / 64
  fixed_marker_return: false
  explicit_side_wall_fallback: false

structured latent-memory diagnostic:
  score_source: latent_recurrent_soft_value_map_planner
  flag: --latent-value-fixed-marker-target
  report:
    .generated/jepa_phase3a/explore_claim_v5_random_palette_generalization_seed20260701/phase3a_v5_latent_soft_value_fixed_marker_broad_map_memory_mem31_marker090_max68_generalization64_report.json
  claimed_episodes: 61 / 64
```

Interpretation: the learned JEPA-latent map and recurrent memory are now close
to the exact odometry frontier ceiling on the broad split, but strict learned
action selection is still the limiting piece. The turn-oscillation guard is a
generic progress guard over recent in-place actions and learned blocked/free
memory; it is not a marker-color shortcut or side-wall detector, but it is still
a hand-written readout rule. The fixed-marker-target soft planner shows that
the latent memory usually contains enough goal information, but that diagnostic
uses an explicit marker target and should not be counted as the strict learned
return result.

Phase 3A.30 implementation note, 2026-06-19:

Ran one bounded follow-up experiment rather than continuing to chase a perfect
result on the current randomized split. The experiment trained a trace-derived
old-strict-to-action05 router with per-state action-preference labels: a state
is positive only when the action05 fallback planner matches the trace oracle and
the current primary planner does not. Training used the ROCm wrapper outside the
managed sandbox because sandboxed PyTorch could not see HIP GPUs.

```text
router label source:
  trace_action_preference

router checkpoint:
  models/checkpoints/phase3a_explore_claim/phase3a_v5_value_map_router_old_to_action05_trace_actionpref_oracle_train8_val47_pw2_512.pt

training split:
  train broad seeds: 20260701 20260709 20260717 20260723 20260731 20260735 20260739 20260743
  validation broad seed: 20260747
  optimization_steps: 512
  router_positive_weight: 2.0
  router_threshold: 0.5

router training result:
  train_examples: 10382
  train_positive_examples: 98
  validation_examples: 1407
  validation_positive_examples: 22
  validation_true_positive: 0
  validation_false_positive: 0
  validation_false_negative: 22
  validation_precision: 0.0
  validation_recall: 0.0

strict smoke closed-loop eval:
  seed: 20260747
  report:
    .generated/jepa_phase3a/explore_claim_v5_random_palette_generalization_seed20260747/phase3a_v5_router_trace_actionpref_pw2_thr050_routeronly_strict_max68_report.json
  claimed_episodes: 55 / 64
  fixed_marker_return: false
  explicit_side_wall_fallback: false
  fallback_after_step: 999
```

Interpretation: this was a negative result. The action-preference label is too
sparse under the current trace construction, and the trained router collapses to
the negative class at the registered threshold. The smoke result matches the old
strict baseline on seed `20260747` (`55 / 64`) and is below the action05
comparator on that seed (`58 / 64`), so it should not replace the best strict
aggregate.

Current headline strict aggregate across the nine broad randomized-palette
seeds remains:

```text
old strict baseline:
  518 / 576 = 89.9%

action05 comparator:
  513 / 576 = 89.1%

best strict learned aggregate found so far:
  trace-outcome router, positive_weight=0.25, 128 steps, threshold=0.5
  525 / 576 = 91.1%
```

Do not treat `100%` on this particular random set as the right stopping
criterion. New broad seeds are expected to expose more edge cases, and even the
privileged odometry/frontier ceiling is not perfect on every checked split. The
more defensible next target is calibrated robustness over multiple broad seeds,
with a learned readout that improves over both strict baselines without relying
on fixed marker-return fallback or explicit side-wall fallback.

Threshold calibration follow-up, 2026-06-19:

Completed the remaining strict global-threshold sweeps for the best trace-outcome
router checkpoint. The sweep used the same strict export wrapper with
`fallback_after_step=999`, `fixed_marker_return=false`, and
`explicit_side_wall_fallback=false` in all generated reports.

```text
router checkpoint:
  models/checkpoints/phase3a_explore_claim/phase3a_v5_value_map_router_old_to_action05_traceoutcome_train8_val47_pw025_128.pt

threshold 0.50:
  per-seed claimed episodes:
    20260701: 60 / 64
    20260709: 63 / 64
    20260717: 58 / 64
    20260723: 60 / 64
    20260731: 58 / 64
    20260735: 60 / 64
    20260739: 52 / 64
    20260743: 56 / 64
    20260747: 58 / 64
  aggregate: 525 / 576 = 91.1%

threshold 0.80:
  aggregate: 518 / 576 = 89.9%

threshold 0.95:
  aggregate: 518 / 576 = 89.9%
```

Interpretation: higher router thresholds repair the worst seed
(`20260739`: `52 -> 54 -> 55` for thresholds `0.50 -> 0.80 -> 0.95`), but
they remove helpful routing on other seeds and reduce the aggregate back to the
old strict baseline. Threshold calibration alone is therefore not the next
useful path.

## Claim Discipline

Permitted claim after this plan is implemented:

> The project is testing whether learned JEPA state can acquire navigation
> consequence structure from pixels/history/actions, with geometry and DINO used
> only as teachers, probes, and controls.

Not permitted until gates pass:

- "JEPA learned navigation";
- "the world model plans safely";
- "DINO is unnecessary";
- "topological memory solves the local action problem";
- "Go2 navigation is solved by the learned world model."
