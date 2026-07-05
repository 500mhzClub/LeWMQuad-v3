# JEPA Navigation: Experiment Failure and Next Steps

Date: 2026-06-14

## Supersession Note

This document records the broad JEPA-first research direction. The current
evidence-qualified repository audit and authoritative next experiment are:

- `docs/lewm_jepa_repository_research_audit_2026-06-14.md`;
- `docs/lewm_jepa_phase2d_preregistered_research_plan_2026-06-14.md`.
- `docs/lewm_jepa_phase2d_stage2_corrected_model_2026-06-14.md`.
- `docs/lewm_jepa_phase2d_stage3_trainer_statistics_2026-06-14.md`.
- `docs/lewm_jepa_phase2d_stage4_source_state_table_2026-06-14.md`.
- `docs/lewm_jepa_phase2d_stage5_split_run_readiness_2026-06-14.md`.
- `docs/lewm_jepa_phase2d_stage6_generation_contract_2026-06-14.md`.
- `docs/lewm_jepa_phase2d_stage7_training_start_gate_2026-06-14.md`.
- `docs/lewm_jepa_phase2d_stage8_source_selection_render_readiness_2026-06-14.md`.
- `docs/lewm_jepa_phase2d_stage9_training_ready_2026-06-14.md`.
- `docs/lewm_jepa_phase2d_stage10_full_training_launch_2026-06-14.md`.
- `docs/lewm_jepa_phase2h_action_utility_audit_2026-06-15.md`.
- `docs/lewm_jepa_phase2i_source_action_utility_plan_2026-06-15.md`.
- `docs/lewm_jepa_phase2j_film_utility_plan_2026-06-15.md`.
- `docs/lewm_jepa_phase2k_interaction_only_utility_plan_2026-06-15.md`.
- `docs/lewm_jepa_phase2l_soft_utility_objective_plan_2026-06-15.md`.
- `docs/lewm_foundational_jepa_research_plan_2026-06-15.md`.

The preregistered reproducibility, data, control, split, render-readiness, and
training-start gates passed for the registered-minimum Phase 2D primary
C0/C1/C2 training matrix. The first full-training launch exposed a
validation-diagnostic OOM; that implementation failure is fixed and documented
in Stage 10. A CPU-only v2 relaunch was stopped as a runtime-selection error.
The corrected ROCm GPU launch completed C0/C1 but failed to produce a valid C2
checkpoint. Follow-on bounded pilots through Phase 2L also failed before test
access. The current branch is not ready for a full JEPA training matrix. The
next registered direction should replace the current RGB CLS utility ranker with
a structured source-local affordance/dynamics state that has explicit
primitive-balanced supervision and geometry-derived calibration before
re-entering JEPA latent prediction. Do not access test-ID or test-hard results
until a new selected checkpoint candidate and staged access gates exist.

Update on 2026-06-15: the DINO/DINO-WM Phase 2AA branch is no longer the primary
mainline. It remains an optional diagnostic ceiling/control. The mainline is now
the foundational JEPA program in
`docs/lewm_foundational_jepa_research_plan_2026-06-15.md`: first build a
paper-faithful learned positive-control navigation task, then add
counterfactual action-identifiability and learned structured belief-state
objectives before returning to Go2.

Further update on 2026-06-15: Phase 3A now has a narrow positive-control pass
on the medium 2D navigation split under
`jepa_phase3a_positive_control_gate_v1`. The result validates a learned
pixel-to-decision-token JEPA scaffold for first-primitive receding-horizon
selection, with DINO unused. The claim remains bounded: exact fixed two-action
sequence regret is still weak, so the next research step is learned online
topological/belief memory and structured consequence state before Go2 re-entry.

Memory update on 2026-06-15: the first current-beacon-free Phase 3A memory pass
was useful but too weak as a memory-dependence claim, because later no-memory
ablations could pass related variants. The stronger result is now the hidden
goal-alias split: `phase3a_memory_alias_candidate_memscore_act6_zero3_1024`
passes `jepa_phase3a_positive_control_gate_v1`, while the paired no-memory
candidate-score ablation fails on regret and step-1 hard-negative advantage.
This still remains a controlled 2D positive-control result, not a Go2
navigation claim.

Exploration target update on 2026-06-15: the desired end goal is explicitly not
runtime breadcrumbs from simulator geometry. The target is online memory built
from observations/actions, novelty/frontier-seeking before a visual beacon is
seen, and beacon claiming or revisiting after that visual sighting. The Phase
3A data contract now has an `explore_then_claim` mode to start testing this
without current/history goal beacons.

Progress update on 2026-06-15: GPU Phase 3A training now runs through
`scripts/phase3a_rocm_train.sh`, a repo-owned ROCm wrapper whose prefix can be
approved once for future sweeps. The no-beacon `explore_claim_v1` split showed
a small memory advantage over no-memory on sparse marker-discovery cases, but
it had only two discovery source groups and zero post-sighting claim groups.
A stricter `explore_claim_history_claim_h4_v0` contract then hid the current
marker, exposed the marker in history, and made oracle exact claiming possible
in 60% of validation source states. That stricter contract failed: memory and
no-memory models behaved identically, selected `turn_left` for every source,
and claimed the goal in only 3.3% of source states. The next technical target
is therefore candidate-sequence scoring and/or candidate sampling for the
history-claim contract, not another claim of navigation success.

Learned-memory update on 2026-06-16: Phase 3A now has a small but concrete
no-beacon online-memory smoke result. The implemented path is no longer a raw
goal-image embedding or a privileged breadcrumb scorer. It is a trainable
egocentric spatial frontier/marker memory head that rolls observation-derived
maps forward through candidate actions, rewards novelty/frontiers while the
marker is unseen, binds marker evidence when it appears, and then scores action
sequences by marker claiming/revisiting. On
`explore_claim_v4_spatial_frontier_smoke`, the checkpoint
`phase3a_v4_spatial_frontier_directinit_64.pt` improves over the explicit
online frontier scaffold: primitive match is 0.667 vs 0.167 for exploration,
0.667 vs 0.500 for marker discovery, and 0.500 vs 0.250 after marker sighting.
Sequence regret is also lower in all three phases. This is still a bounded
smoke result, not a full Phase 3A v4 or Go2 navigation claim; the full-v4 ROCm
runs exposed a CPU/input-pipeline bottleneck before useful checkpointing. The
trainer now reuses a `Phase3AMaterializedDataset` tensor cache; on the smoke
train split this made repeated batch materialization 49.9x faster after the
initial cache build, report/export now use the same cache, and GPU access is
confirmed outside the sandbox on the AMD Radeon AI PRO R9700. A bounded
full-v4 16-step ROCm probe completed on `device cuda` and wrote
`phase3a_v4_spatial_frontier_cache_probe_gpu_16.pt`. On the full-v4 validation
split, the learned spatial frontier score had lower sequence regret than the
explicit scaffold in all three phases, while primitive-match rate was slightly
lower and post-sighting claim rate was tied at 0.326. This is a
throughput/plumbing pass, not a converged model. Demos:
`.generated/jepa_phase3a/explore_claim_v4_spatial_frontier_smoke/phase3a_v4_spatial_frontier_directinit_64_demo.mp4`
and
`.generated/jepa_phase3a/explore_claim_v4_novelty_then_claim/phase3a_v4_spatial_frontier_cache_probe_gpu_16_demo.mp4`.

Closed-loop update on 2026-06-16: a longer full-v4 GPU run completed:
`models/checkpoints/phase3a_explore_claim/phase3a_v4_spatial_frontier_gpu_512_b4.pt`.
The final checkpoint reports aggregate primitive match 0.630 and sequence
regret 1.654 on full validation. Phase-level deployed spatial-frontier metrics
did not change from steps 128 through 512, which is a useful negative: the
current behavior is dominated by the direct RGB-initialized egocentric
frontier/marker memory mechanism rather than learned improvement of that score
during the 512-step run. The ablations are nevertheless strong: no-history
drops post-sighting claim rate from 0.326 to 0.023, marker-color removal drops
claiming to 0.000, candidate-action shuffling drops claiming to 0.023, and
disabling spatial memory drops claiming to 0.000. A closed-loop 2D rollout
export now reconstructs validation scenes, scores candidate sequences from the
current state, executes the selected first action, and updates online history.
It claimed the marker in 13/24 attempted validation episodes; the selected demo
starts with no marker visible, first sees the marker at step 12, reaches the
goal in 21 steps, and has zero collisions. Demo:
`.generated/jepa_phase3a/explore_claim_v4_novelty_then_claim/phase3a_v4_spatial_frontier_gpu_512_b4_closed_loop_demo.mp4`.

Randomized-palette caveat update on 2026-06-16: the fixed-color v4 closed-loop
result is not enough for the intended research claim, because it relied on a
direct RGB marker prior. The new randomized-palette v5 split removes that
shortcut:
`.generated/jepa_phase3a/explore_claim_v5_random_palette`. The best current
non-fixed-color open-loop variant is
`models/checkpoints/phase3a_explore_claim/phase3a_v5_random_palette_obsaux_mlp_maponly_512_threshold_p4_tau050.pt`,
with randomized-palette phase metrics:

```text
explore_unseen: prim 1.000, seq_regret 1.517
discover_visible_marker: prim 0.846, seq_regret 1.373, marker_seen 0.385
claim_after_marker_seen: prim 0.563, seq_regret 2.613, claim 0.250
```

But closed-loop remains a negative result:

```text
demo:
  .generated/jepa_phase3a/explore_claim_v5_random_palette/phase3a_v5_random_palette_obsaux_mlp_maponly_512_threshold_p4_tau050_block4_closed_loop_demo_v2.mp4
report:
  .generated/jepa_phase3a/explore_claim_v5_random_palette/phase3a_v5_random_palette_obsaux_mlp_maponly_512_threshold_p4_tau050_block4_closed_loop_report_v2.json
claimed_episodes: 1 / 16
marker_seen_ever: 11 / 16
```

The model often discovers the marker but fails to claim it. A separate
spatial-marker-source branch was also tested
(`phase3a_v5_random_palette_spatialmarker_frontier_256.pt`) and regressed to
0% randomized-palette claim rate. The next implementation target was therefore
larger learned memory support decoupled from the camera crop.

That decoupling is now implemented through `spatial_memory_size`. The best
current non-fixed-color derived artifact is:

```text
models/checkpoints/phase3a_explore_claim/phase3a_v5_random_palette_obsaux_mlp_maponly_512_threshold_p4_tau050_mem15.pt
```

It uses `view_size=7` with `spatial_memory_size=15`. Closed-loop block4
improved from 1/16 to 3/16 claimed episodes:

```text
demo:
  .generated/jepa_phase3a/explore_claim_v5_random_palette/phase3a_v5_random_palette_obsaux_mlp_maponly_512_threshold_p4_tau050_mem15_block4_closed_loop_demo.mp4
report:
  .generated/jepa_phase3a/explore_claim_v5_random_palette/phase3a_v5_random_palette_obsaux_mlp_maponly_512_threshold_p4_tau050_mem15_block4_closed_loop_report.json
claimed_episodes: 3 / 16
marker_seen_ever: 10 / 16
```

A direct mem15 fine-tune
(`phase3a_v5_random_palette_mem15_maponly_256.pt`) reduced collisions but
regressed claiming to 2/16, so it is not the selected artifact.

Further iterations on 2026-06-16 did not beat the v5 mem15 artifact:

```text
full-history eval:
  checkpoint: phase3a_v5_random_palette_obsaux_mlp_maponly_512_threshold_p4_tau050_mem15.pt
  change: --history-window 0 keeps all closed-loop history
  result: 0 / 16 claimed, marker_seen_ever 8 / 16

h12 training:
  checkpoint: phase3a_v6_random_palette_h12_mem15_maponly_128.pt
  result: 0 / 16 claimed

current-marker-visible training:
  checkpoint: phase3a_v7_random_palette_current_marker_mem15_maponly_128.pt
  result: 1 / 16 claimed
```

These are useful negatives. More history by itself introduces drift/noise; h12
training does not teach persistent topographical memory; and training current
marker visibility improves post-sighting open-loop regret but hurts
exploration/discovery enough to regress closed-loop claiming. The desired end
goal remains unmet: novelty/frontier reward until beacon discovery, then robust
beacon claiming from learned persistent memory, without privileged simulator
breadcrumbs or fixed RGB goal detectors at runtime. The next target is a real
persistent belief-state update objective, not another small scorer/fine-tune
sweep.

## Research Objective

The purpose of this project is not merely to produce a successful navigation
controller. It is to determine how far a JEPA-style predictive world model can
be pushed toward reliable embodied navigation.

The current experiment should therefore be treated as a falsification of one
particular formulation:

> A globally pooled, short-horizon, action-conditioned JEPA representation,
> trained primarily for latent next-observation prediction with uniform
> anti-collapse regularisation, is not sufficient for reliable safe-path
> selection in this task.

It does not establish that JEPA-style world models are unsuitable for
navigation. The next phase will redesign the world model around the information
and predictive capabilities that navigation demonstrably requires.

## Research Constraints

The next phase remains JEPA-first:

- the learned world model must make the decisions that determine navigation;
- deployed planning should operate directly on learned latent state and latent
  predictions;
- no pixel reconstruction is required;
- explicit depth, occupancy, pose, and oracle plans may be used for training,
  probing, counterfactual labels, and evaluation;
- privileged geometry must not become a required runtime input to the final
  system;
- conventional controllers and geometric planners remain baselines and oracles,
  not the final research contribution.

This distinction is essential. Privileged geometry can tell us precisely what
the latent knows and where it fails without replacing the latent as the planning
substrate.

## Why The Current Experiment Failed

### 1. Global pooling discarded action-relevant spatial structure

The current LeWM compresses an observation into a single pooled vector. This is
effective for coarse view and place recognition, but local navigation requires
structured facts such as:

- an obstacle occupies the left side of the view;
- the centre corridor is traversable;
- a safe opening lies behind a near obstacle;
- two similar surfaces occupy different positions;
- a small heading change alters clearance.

The experiments show that the pooled latent retains appearance and yaw more
strongly than metric position, free space, or traversability. A small learned
head cannot reliably recover information that the encoder discarded.

### 2. The representation was an observation encoding, not a belief state

Navigation is partially observable. Safe action selection depends on recently
observed space, executed actions, motion uncertainty, and facts outside the
current field of view.

The current system does not maintain a sufficiently capable recurrent latent
belief. This forces repeated observation, re-alignment, and artificial spins.
The final demo reaches the goal partly by repeatedly reacquiring route bearings
rather than by maintaining a stable predictive state.

### 3. Observational prediction did not teach counterfactual action consequences

Training primarily asks the model to predict the future associated with the
action that occurred in the dataset. Planning requires comparing futures under
actions that were not executed:

- forward collides;
- a left arc makes safe progress;
- a right arc enters a dead end;
- turning reveals useful information.

A model can predict recorded next observations accurately while remaining unable
to rank alternative actions. The repeated failures of first-action, nav-cost,
IDM, and task-aligned heads are consistent with this gap.

### 4. The predictive horizon was too weak for planning

Short-horizon latent prediction is dominated by visual persistence because
adjacent observations change little. Current rollouts lose to a persistence
baseline beyond the earliest horizons.

This means the planner cannot trust predicted consequences far enough ahead to
choose paths around obstacles or through junctions. Increasing planning logic on
top of an unreliable rollout model cannot solve this.

### 5. One latent was asked to satisfy conflicting requirements

The current design asks one representation to be:

- invariant enough for place recognition;
- viewpoint-sensitive enough for control;
- stable enough for long-term memory;
- action-sensitive enough for dynamics;
- locally precise enough for clearance and collision decisions.

These properties conflict. Place recognition should often ignore heading and
small obstacle-layout changes, while safe control must respond strongly to them.
The resulting pooled latent is useful for recognition but unreliable for local
action selection.

### 6. Uniform SIGReg did not create navigation geometry

SIGReg successfully resists trivial collapse and appears to support recognition
and representation dispersion. However, navigational state lies on structured,
non-isotropic manifolds containing cyclic heading, local metric relationships,
topological branches, and collision boundaries.

The scaled ablation showed that simply lowering SIGReg did not recover that
structure. Lower weights degraded retrieval and dispersion and reduced action
sensitivity while projected geometry remained poor.

The failure is therefore not merely an incorrect regularisation coefficient. It
is a mismatch between the representation roles, predictive objective, and the
uniform regularisation applied to them.

### 7. Learned heads optimized proxies rather than deployed decisions

Several learned heads succeeded on their own objective or diagnostic while
failing closed-loop navigation gates. Examples include strong energy-head
ranking that produced worse navigation and improved pose decodability that did
not improve first-action selection.

These results demonstrate that:

- offline decodability is not sufficient;
- proxy ranking objectives can exploit dataset shortcuts;
- heads on frozen pooled latents cannot manufacture missing spatial state;
- evaluation must use counterfactual, scene-disjoint, closed-loop decisions.

### 8. The successful demo did not exercise learned world-model planning

The final ego-depth demo is a valuable integration and diagnostic artifact, but
it does not validate the JEPA planner:

- the graph route comes from a privileged scene-graph DFS tour;
- mapping-time bearings and reactive heading logic dominate execution;
- simulator-rendered depth and ground-truth pose support local obstacle handling;
- the final object approach uses a pixel-space colour servo;
- the successful artifact reports `fallback_blocks = 0`, so the LeWM rollout
  planner made no fallback decisions.

The trajectory also required 215 yaw primitives, 127 alignment blocks, and 64
veto escapes, with a path/direct ratio of 3.21x. It succeeded through engineered
recovery and persistence, not reliable learned safe-path selection.

## Revised Research Hypothesis

The next phase tests:

> A JEPA-style world model can support reliable navigation if its latent state
> preserves spatial organization, integrates observations and actions through
> time, predicts counterfactual multi-step futures, separates conflicting
> representation roles, and is trained and evaluated on deployed action
> consequences.

The following directions are parts of one coherent redesign, not independent
patches to the current controller.

## Direction 1: Spatially Structured JEPA Tokens

### Hypothesis

Spatially organized and rotation-aware latent tokens preserve local affordances
that global pooling removes.

### Proposed architecture

Retain a grid or set of visual tokens rather than reducing the complete
observation to one vector. Each token should retain correspondence to a region,
surface, object, or learned spatial element.

The model should support predictable transformations under robot rotation.
Turning should transform, rotate, or re-associate spatial state rather than
producing an unrelated global representation.

A pooled appearance token may still be retained for retrieval, but it must not
be the sole planning state.

### Initial experiments

Train spatial and pooled variants on identical data and capacity budgets. At
held-out observations, rank candidate actions for:

- collision versus non-collision;
- available forward clearance;
- left versus right openings;
- recoverable versus trapped futures;
- safe progress toward a local target.

Use privileged geometry only to label and evaluate these decisions.

### Promotion gate

The spatial representation must materially outperform the pooled LeWM and
action-only baselines on held-out safe-action ranking without requiring decoded
occupancy at runtime.

## Direction 2: Recurrent Action-Conditioned Latent Belief

### Hypothesis

A persistent latent belief can integrate partial observations and reduce the
need for repeated spins and route-bearing reacquisition.

### Proposed architecture

Update a persistent latent state from:

- previous belief tokens;
- current spatial observation tokens;
- the previously executed action;
- onboard proprioception;
- optionally onboard odometry as a controlled experimental input.

The belief should distinguish observed evidence from unseen or uncertain state.
Possible implementations include recurrent transformers, persistent spatial
memory tokens, learned egocentric latent maps, or object/surface slots.

### Initial experiments

Construct controlled partial-observability tests:

1. observe an obstacle or opening;
2. rotate until it leaves the field of view;
3. execute intervening motion;
4. ask the belief to rank safe actions without re-observing the feature.

Run ablations with vision only, proprioception, onboard odometry, and privileged
pose. Privileged pose remains an oracle diagnostic.

### Promotion gate

The recurrent belief must improve hidden-obstacle and remembered-opening action
ranking, reduce active re-observation in closed loop, and express higher
uncertainty when evidence is absent or contradictory.

## Direction 3: Counterfactual Multi-Step JEPA Prediction

### Hypothesis

Training on branched action futures and free-running rollout is necessary for a
JEPA to learn the consequences required by planning.

### Data design

From the same initial simulator state, execute several candidate action
sequences. Preserve branches containing:

- successful progress;
- collision and near-collision;
- dead ends;
- recoveries;
- information-gathering rotations;
- visually similar outcomes with different safety.

This directly removes the observational-data ambiguity between action
correlation and action consequence.

### Training objective

Predict target latent trajectories for every candidate sequence. During
multi-step training, recursively consume predicted state so the model is exposed
to its own rollout errors.

Auxiliary event targets may include contact, progress, recoverability, and
uncertainty. These guide representation learning but must not replace latent
trajectory prediction as the central objective.

### Promotion gate

On scene-disjoint counterfactual branches, latent rollouts must:

- beat persistence and action-only baselines across the planning horizon;
- rank the safest-progressing sequence correctly;
- identify collision and unrecoverable futures;
- remain calibrated enough that uncertainty predicts rollout failure.

## Direction 4: Factorized Representation Roles

### Hypothesis

Separating conflicting invariances during representation learning will produce a
more navigation-sufficient world model than adding heads to a universal pooled
latent.

### Proposed factorization

Use a shared visual stem with learned branches for:

- **appearance/place:** stable retrieval and goal-image matching;
- **spatial affordance:** local traversability and action opportunities;
- **dynamics:** action-conditioned state transformation;
- **belief:** temporal integration and uncertainty.

The branches may exchange information, but each receives objectives and
regularisation appropriate to its role.

This differs from the previous frozen-head approach because specialization
changes the encoder representation itself.

### Initial experiments

Measure whether each branch exhibits its intended properties and whether removing
it causes the expected failure. Control for total capacity and compare against a
single-branch model.

### Promotion gate

Factorization must improve direct action selection or multi-step prediction, not
only probe accuracy. Redundant branches that do not affect deployed decisions
should be removed.

## Direction 5: Branch-Specific Anti-Collapse Regularisation

### Hypothesis

Anti-collapse can preserve useful variance without erasing structured navigation
geometry when applied according to branch semantics.

### Proposed experiments

Compare regularisation strategies rather than only changing `sigreg_lambda`:

- current uniform global SIGReg;
- strong SIGReg on appearance features only;
- token-local or group-wise variance constraints;
- effective-rank constraints without global isotropy;
- temporal/action-sensitivity regularisation;
- teacher-target prediction with stop-gradient;
- combinations selected before evaluation.

Track retrieval, effective rank, spatial correspondence, action sensitivity,
rollout accuracy, safe-action ranking, and closed-loop behavior.

### Promotion gate

A strategy must resist collapse while improving navigation decisions. Better
dispersion or retrieval alone is not sufficient.

## Direction 6: Direct Latent-Space Planning

### Hypothesis

A sufficiently structured JEPA belief can support model-predictive control
without pixel reconstruction or explicit runtime geometry.

### Planner design

At each decision:

1. encode history into the current latent belief;
2. generate candidate action sequences;
3. predict their latent trajectories;
4. score predicted goal progress, collision risk, uncertainty, and
   recoverability;
5. execute the first action from the selected sequence;
6. observe and replan.

The planner must actually control the robot. It cannot remain an unused fallback
behind a conventional controller.

### Model-exploitation controls

Because planners exploit model errors, use:

- uncertainty penalties;
- ensemble disagreement or equivalent epistemic estimates;
- short receding horizons initially;
- conservative scoring outside the training distribution;
- targeted data collection for exploited failure regions.

### Promotion gate

Direct latent planning must outperform persistence, reactive control, and the
current pooled LeWM on held-out safe progress and closed-loop efficiency. It must
approach a privileged oracle while using only permitted runtime observations.

## Experimental Roadmap

### Phase 0: Preserve the current result

- Freeze the current branch as the completed pooled-LeWM/topological-navigation
  experiment.
- Preserve its artifacts, gates, and negative results.
- Treat the final demo as a baseline, not evidence of learned planning.

### Phase 1: Build the counterfactual decision benchmark

- Generate scene-disjoint branched action futures from matched initial states.
- Include collision, clearance, progress, dead-end, and recovery labels.
- Register pooled-LeWM, persistence, action-only, reactive, and privileged-oracle
  baselines.

This benchmark is required before training a new architecture because it tests
the deployed decision directly and prevents another proxy-metric detour.

### Phase 2: Test spatial tokens before recurrence

- Replace pooled planning state with spatial tokens.
- Test single-observation safe-action ranking.
- Test rotation equivariance and spatial correspondence.
- Stop or redesign immediately if the representation does not beat pooled
  baselines.

### Phase 3: Add recurrent belief

- Integrate observation history, actions, and proprioception.
- Evaluate memory under occlusion and rotation.
- Measure uncertainty and reduction in re-observation behavior.

### Phase 4: Train counterfactual free-running prediction

- Train branched multi-action latent futures.
- Increase rollout horizon only when the model beats persistence.
- Use rollout failure to drive targeted data collection.

### Phase 5: Factorize and regularize

- Separate appearance, affordance, dynamics, and belief roles.
- Run branch and regularisation factorials.
- Select using counterfactual and closed-loop gates.

### Phase 6: Deploy latent MPC

- Make latent rollouts the primary action-selection mechanism.
- Compare against the current demo controller and privileged oracle.
- Remove privileged runtime sources before claiming perception navigation.

### Phase 7: Restore long-range image-goal navigation

- Reintroduce topological memory only after local latent planning is reliable.
- Test autonomous memory construction rather than privileged route replay.
- Use the appearance branch for retrieval and the latent belief/planner for safe
  execution.

## Stop Conditions

The next phase should remain ambitious but falsifiable. Reconsider the broader
JEPA navigation hypothesis if, after controlled architecture and data changes:

- spatial tokens do not improve held-out safe-action ranking;
- recurrent belief does not retain action-relevant hidden state;
- counterfactual training does not beat action-only and persistence baselines;
- free-running latent prediction remains unreliable at the minimum useful
  planning horizon;
- latent MPC repeatedly exploits errors that uncertainty cannot identify;
- learned latent planning provides no benefit over reactive perception under
  matched runtime inputs.

These outcomes would be research results about the practical limits of latent
predictive objectives, not merely engineering failures.

## Immediate Next Work

The first implementation target is the counterfactual decision benchmark, not a
new end-to-end navigation controller.

Update on 2026-06-15: Phase 2D reached the GPU training gate but did not
produce a valid action-conditioned spatial JEPA checkpoint. C0/C1 failed the
persistence gate, original C2 failed numerically, and a detached-control C2
pilot stayed finite but collapsed and lost badly to persistence. The current
image-aligned spatial-token objective should not receive another full training
launch until a bounded smoke passes the explicit validation gate:

- no collapse, effective-rank, or near-static-target warning;
- real action beats zero action by at least `0.10` of target change;
- real action beats shuffled/hard-negative actions by at least `0.10` of
  target change;
- one-step real prediction beats persistence with ratio `< 1.0`.

The next implementation target is therefore an architecture-level redesign of
the target/state geometry, not another hyperparameter-only C2 run. The preferred
bounded direction is a factorized affordance/dynamics state or
motion-equivariant slots where action consequences are represented in a
geometry that is not dominated by image-aligned appearance persistence.

Follow-up on 2026-06-15: the first target-geometry pilot tested learned slot
state/target pooling under the same Phase 2D controls. It completed on the ROCm
GPU runtime and avoided the original C2 non-finite-gradient failure, but it
still collapsed, failed effective-rank diagnostics, ranked hard-negative actions
above the real action, and lost to one-step persistence by `24.96x`. This pilot
is documented in:

```text
docs/lewm_jepa_phase2e_target_geometry_plan_2026-06-15.md
```

Do not launch a full slot-geometry training matrix. The next bounded experiment
should replace generic image/slot target prediction with a factorized
affordance/dynamics/event state and must keep the same explicit gate:

- no collapse, effective-rank, or near-static-target warning;
- real action beats zero action by at least `0.10` of target change;
- real action beats shuffled/hard-negative actions by at least `0.10` of
  target change;
- one-step real prediction beats persistence with ratio `< 1.0`.

Follow-up on 2026-06-15: Phase 2F added a masked sequence-level consequence
head using the privileged generator labels already present in the Phase 2D
JSONL rows. C2 plus the consequence head was numerically unstable in two
bounded smokes, failing before validation with infinite pre-clip gradient norm.
A C1 consequence-isolation smoke completed, but still collapsed, ranked
hard-negative actions above the real action, and lost to persistence by
`229.69x`. This pilot is documented in:

```text
docs/lewm_jepa_phase2f_factorized_consequence_plan_2026-06-15.md
```

Do not launch a full consequence-head matrix. The next candidate must put
navigation consequence structure into the state itself, for example dedicated
affordance/dynamics tokens or direct 81-way per-source action-utility
supervision, rather than a mean-pooled auxiliary head attached to the same
image-aligned latent prediction target.

Follow-up on 2026-06-15: Phase 2G added a source-local direct action-utility
head over the predicted future spatial tokens. The bounded C1 smoke stayed
finite but did not pass the utility-selection or latent gates. It selected the
oracle utility row in `1/5` validation source states, matched the oracle first
primitive in `1/5`, collapsed to low-rank latent targets, ranked hard-negative
actions above the real action on average, and lost to one-step persistence by
`117.98x`. This pilot is documented in:

```text
docs/lewm_jepa_phase2g_action_utility_plan_2026-06-15.md
```

Do not launch a full action-utility-head matrix attached to the current
image-aligned latent future. The next immediate diagnostic is a Phase 2H
utility-label and action-only baseline audit. It must determine whether the
81-way utility target is genuinely source-conditioned or mostly explained by
source-independent action-sequence bias. Only if action-only baselines fail to
explain validation utility should the next model add dedicated
source-conditioned affordance/utility state.

Follow-up on 2026-06-15: Phase 2H audited the utility target and train-derived
action-only baselines. Utility labels had complete train/validation coverage,
non-trivial within-source spread, and rare top ties. The full-sequence
action-only baseline was below uniform random top-1 on validation
(`0.0078` vs `0.0123` expected), and the first-primitive baseline improved
coarse primitive choice but not exact source-local action selection
(`0.0117` top-1). This audit is documented in:

```text
docs/lewm_jepa_phase2h_action_utility_audit_2026-06-15.md
```

The next bounded implementation target is Phase 2I: a source-conditioned
affordance/utility pilot that predicts utility from the current observation and
candidate action sequence directly. This is a diagnostic prerequisite for a new
JEPA full run: it must beat the Phase 2H action-only baselines before the
research should invest in integrating the utility state into a latent
world-model objective.

Follow-up on 2026-06-15: Phase 2I trained a source-conditioned utility ranker
on the ROCm GPU runtime. It completed with finite metrics and improved exact
top-1 utility selection over the action-only priors (`0.2148` vs `0.0117`),
but failed the promotion gate. It selected `backward` as the first primitive for
all 256 validation source states, matched the first-primitive baseline exactly
(`0.3516`), and had worse regret than both action-only baselines. This pilot is
documented in:

```text
docs/lewm_jepa_phase2i_source_action_utility_plan_2026-06-15.md
```

Do not integrate this Phase 2I ranker into JEPA. The next bounded fix should
force source-action interaction explicitly, for example a FiLM or bilinear
source-conditioned action scorer with per-source score centering and a
trainable action-only control under the same architecture.

The matched trainable action-only control reached the same validation summary
as the source-conditioned ranker. Therefore, the source-conditioned
concatenation model did not measurably use the source observation.

It should answer three questions before substantial architecture investment:

1. Does the current pooled LeWM contain enough information to distinguish safe
   counterfactual actions when evaluated correctly?
2. Do spatial tokens materially improve that decision?
3. How far into free-running prediction must a model remain accurate to improve
   the deployed action?

This establishes a direct, falsifiable bridge between JEPA representation
learning and navigation behavior, and provides the evaluation contract for all
subsequent world-model designs.

Follow-up on 2026-06-15: Phases 2J, 2K, and 2L tested stronger utility-ranker
controls after Phase 2I. FiLM conditioning, interaction-only fusion, and a soft
utility-distribution objective all failed the executable utility gate. The
models continued to choose a global backward-first pattern rather than a
source-local action. These pilots are documented in:

```text
docs/lewm_jepa_phase2j_film_utility_plan_2026-06-15.md
docs/lewm_jepa_phase2k_interaction_only_utility_plan_2026-06-15.md
docs/lewm_jepa_phase2l_soft_utility_objective_plan_2026-06-15.md
```

Do not launch a full JEPA training run from the RGB CLS source-action ranker
family. The active next gate is Phase 2M: collapse the two-block utility target
to source-local first-primitive affordances, train a source-image-only
primitive utility vector, and require it to beat a source-independent primitive
prior before any JEPA integration pilot. This bounded plan is documented in:

```text
docs/lewm_jepa_phase2m_structured_affordance_state_plan_2026-06-15.md
```

Follow-up on 2026-06-15: Phase 2M completed as a bounded ROCm GPU smoke and
failed the executable gate. It improved primitive match over the
source-independent primitive prior (`0.3125` vs `0.1641`), but had worse regret
(`0.1084` vs `0.0586`) and a more collapsed selected primitive distribution
than the oracle (`0.6875` max selected fraction vs `0.3516` oracle).

Do not integrate the Phase 2M scalar primitive-affordance head into JEPA. The
next bounded fix is Phase 2N: keep the source-only primitive model but replace
the soft scalar-utility objective with class-balanced hard-oracle primitive
supervision. This isolates whether the remaining collapse is primarily caused
by oracle class imbalance/objective geometry. If Phase 2N cannot reduce
collapse and regret, stop RGB CLS-only affordance variants and move to
geometry-derived local affordance targets before any JEPA full run.

Follow-up on 2026-06-15: Phase 2N completed as a bounded ROCm GPU smoke and
failed the same executable gate. Class balancing marginally improved primitive
match (`0.3320` vs Phase 2M `0.3125`) and slightly reduced selected primitive
collapse (`0.5977` vs `0.6875`), but utility regret became much worse
(`0.2525` vs `0.1084`) and remained worse than the primitive action-only prior
(`0.0586`). This pilot is documented in:

```text
docs/lewm_jepa_phase2n_class_balanced_affordance_plan_2026-06-15.md
```

Stop RGB CLS-only scalar primitive-affordance objective variants. The active
next step is Phase 2O: build factorized geometry-derived primitive affordance
targets separating safety/recoverability, swept clearance, task progress, and
heading tie-breakers. The model should be judged by a safety-first selection
rule before any JEPA latent integration.

Follow-up on 2026-06-15: Phase 2O implemented and audited the factorized
primitive-affordance target contract. Core factors are complete on train and
validation for every source/primitive target: safety/recoverability,
task-gain, p05 clearance, minimum clearance, and unsafe fraction.
`heading_alignment` is partial and should remain an optional tie-breaker. This
audit is documented in:

```text
docs/lewm_jepa_phase2o_factorized_affordance_targets_2026-06-15.md
```

The next bounded implementation is a factorized primitive affordance head with
safety-first selection. It must still pass the primitive gate before any JEPA
latent integration: primitive match at least `0.50`, regret below the
source-independent primitive prior, and selected primitive distribution no more
collapsed than oracle by more than `0.20`.

Follow-up on 2026-06-15: Phase 2P implemented the factorized primitive
affordance head and safety-first selector. The ROCm GPU smoke stayed finite but
failed the executable gate. It scored below the primitive action-only prior on
validation primitive match (`0.1367` vs `0.1641`), had worse regret (`0.1248`
vs `0.0586`), and still concentrated selection too strongly (`0.6094` max
selected fraction vs `0.3516` oracle). This pilot is documented in:

```text
docs/lewm_jepa_phase2p_factorized_affordance_model_2026-06-15.md
```

Do not integrate the Phase 2P source-only factorized head into JEPA. The next
bounded fix should change the state substrate, not only the loss. The active
next step is a geometry-exposed affordance state pilot: local ray/clearance
tokens or factorized affordance slots trained from the Phase 2O labels, with
the same primitive gate retained before any full JEPA training run.

Follow-up on 2026-06-15: Phase 2Q audited the true-factor ceiling for the
Phase 2O labels plus Phase 2P safety-first selector. With true factor targets,
the selector passed the primitive gate on validation: primitive match `0.8672`,
mean utility regret `0.00142`, and selected max primitive fraction `0.3398`
against oracle `0.3516`. This audit is documented in:

```text
docs/lewm_jepa_phase2q_factorized_affordance_ceiling_2026-06-15.md
```

This means the target/selector contract is coherent enough to keep. The next
bounded implementation should not tune the Phase 2P RGB source-only head; it
should implement Phase 2R, a geometry-exposed learned affordance state, and
hold it to the same primitive gate before any JEPA full training launch.

Follow-up on 2026-06-15: Phase 2R implemented a privileged geometry-feature
affordance diagnostic. It joined source pose, optional goal pose, local
obstacle rays, source clearance, and scene bounds from the referenced
train/validation render metadata and scene manifests, then trained an MLP to
predict the Phase 2O factors. The bounded ROCm GPU smoke completed with finite
metrics but failed the primitive gate. Validation primitive match improved to
`0.3867` versus the primitive action-only prior at `0.1641`, and selected
primitive distribution was not more collapsed than oracle (`0.3242` selected
max fraction vs `0.3516` oracle). However, match remained below the `0.50`
threshold and regret was worse than the primitive prior (`0.0961` vs
`0.0586`). This pilot is documented in:

```text
docs/lewm_jepa_phase2r_geometry_affordance_state_2026-06-15.md
```

Do not launch a full JEPA training run from Phase 2R. Source-local geometry is
not enough. The next bounded implementation should expose action-conditioned
swept geometry or factorized affordance slots before learning, then hold that
state to the unchanged primitive gate before any JEPA integration.

Follow-up on 2026-06-15: Phase 2S implemented an action-conditioned
swept-geometry affordance diagnostic. For each source state and first
primitive, it built per-primitive features from the first command block,
kinematic swept clearance, unsafe fraction, goal progress, heading alignment,
and two-block continuation aggregates. The bounded ROCm GPU smoke passed the
primitive gate: validation primitive match `0.53125`, mean utility regret
`0.04997` versus the primitive action-only prior at `0.05860`, and selected max
primitive fraction `0.4141` versus oracle `0.3516`, within the registered
`+0.20` collapse bound. This pilot is documented in:

```text
docs/lewm_jepa_phase2s_swept_geometry_affordance_2026-06-15.md
```

This is the first passed learned affordance-state diagnostic in the Phase 2
redesign chain. It does not validate RGB perception or JEPA latent prediction.
The next bounded implementation should be a JEPA integration smoke that predicts
or exposes the Phase 2S-style swept-affordance state while retaining the
primitive gate, action-identifiability gate, zero/shuffled-action controls, and
one-step persistence gate.

Follow-up on 2026-06-15: Phase 2T implemented that first JEPA integration
smoke by keeping the C2 normalized spatial JEPA objective and adding a
factorized consequence head on predicted future spatial tokens. The bounded
ROCm GPU smoke completed but failed both gates. The primitive gate scored
`0.1094` primitive match and `0.2833` regret, worse than the primitive
action-only prior (`0.1641` match, `0.0586` regret). The spatial JEPA gate also
failed stability, hard-negative action advantage, zero-action advantage, and
persistence (`1842.87x`). This smoke is documented in:

```text
docs/lewm_jepa_phase2t_factorized_jepa_affordance_integration_2026-06-15.md
```

Do not launch a full JEPA training matrix from Phase 2T. The next fix should
change the target state itself: direct swept-affordance state prediction,
RGB-to-Phase-2S state distillation, or factorized affordance slots tied to
action-conditioned consequence geometry.

Follow-up on 2026-06-15: Phase 2U tested a narrower source/action bridge:
source RGB plus candidate two-block action sequence directly predicted the six
factorized consequence labels. The ROCm GPU smoke completed with finite metrics
but failed the primitive gate. It improved primitive match over the
source-independent prior (`0.2148` vs `0.1641`) but stayed below the `0.50`
threshold and had much worse regret (`0.1903` vs `0.0586`). This bridge is
documented in:

```text
docs/lewm_jepa_phase2u_source_action_factorized_affordance_2026-06-15.md
```

Do not launch a full JEPA training run from Phase 2U. The current small
source/action RGB encoder is not enough. The next bounded fix must add stronger
state supervision: RGB-to-swept-geometry distillation, factorized affordance
slots, or another compact non-image-aligned state target tied directly to
swept clearance, safety, progress, and heading.

Follow-up on 2026-06-15: Phase 2V implemented RGB-to-Phase-2S swept-state
distillation. Both the plain full-target bridge and the anchored/ranked
full-target bridge completed on the ROCm GPU runtime but failed the primitive
gate. The worst reconstructed full-target feature was consistently in the
`best_progress_second_onehot_*` family, so Phase 2W audited whether that
explicit continuation-choice identity was necessary.

Phase 2W found a cleaner privileged teacher target: remove
`best_progress_second_onehot_*` and add a light primitive-ranking term
(`0.10`). This sanitized privileged target passed the primitive gate with
validation primitive match `0.5117`, mean regret `0.0365`, and selected max
primitive fraction `0.4102` against oracle `0.3516`. A first-block-only target
failed, showing that continuation aggregates remain useful; only the explicit
second-action identity should be removed.

Phase 2X then repeated the RGB bridge using the passed Phase 2W sanitized
target. It reconstructed the sanitized state within the registered feature
error thresholds but still failed primitive choice: validation primitive match
`0.125` and regret `0.1302`, both worse than the primitive action-only prior.
This result is documented in:

```text
docs/lewm_jepa_phase2v_phase2w_phase2x_swept_state_bridge_2026-06-15.md
```

Do not launch a full JEPA training sweep from Phase 2X. The current single-frame
RGB bridge is still not sufficient. The next bounded fix should introduce a
stronger deployable state bridge: local metric/depth observations, temporal
memory, occupancy-like intermediate state, or factorized slots supervised by
the Phase 2W sanitized target. A candidate must pass the primitive gate before
JEPA integration, and any JEPA integration must still pass action
identifiability, zero/shuffled-action, stability, and persistence gates.

Follow-up on 2026-06-15: Phase 2Y added the same light primitive-ranking
objective to the Phase 2R source-local geometry bridge. This tests whether a
small local metric/ray state plus source and goal pose is enough when the loss
is aligned with the primitive gate. The bounded ROCm GPU smoke completed but
failed: validation primitive match `0.3906`, regret `0.0980`, and selected max
primitive fraction `0.3281`. It is documented in:

```text
docs/lewm_jepa_phase2y_ranked_geometry_bridge_2026-06-15.md
```

Do not launch a full JEPA sweep from Phase 2Y. Simple local rays are not enough
either. The next bounded bridge should accumulate or construct a richer
deployable state, for example depth-derived swept occupancy, temporal memory,
or factorized affordance slots supervised by the Phase 2W sanitized target.

Follow-up on 2026-06-15: Phase 2Z implemented an action-conditioned local
occupancy bridge: per-candidate local occupancy/clearance/goal/path grids plus
compact command and goal-relative vector features. The bounded ROCm GPU smokes
completed but failed the primitive gate. With ranking weight `0.10`, validation
primitive match was `0.3125` and regret was `0.0747`; with ranking weight
`1.00`, match improved to `0.3633` but regret worsened to `0.0951`. Both failed
because match stayed below `0.50` and regret did not beat the primitive
action-only prior (`0.0586`). This is documented in:

```text
docs/lewm_jepa_phase2z_occupancy_and_dino_review_2026-06-15.md
```

Do not launch a full JEPA sweep from Phase 2Z. A dense single-frame local
occupancy bridge is not enough under the current action-choice gate.

The DINO/DINOv2/DINO-WM and LeWM-paper review changes the next registered step.
The missing control is not another small affordance classifier; it is a
DINO-WM-style frozen spatial-patch substrate and action-conditioned rollout
screen, plus a LeWM-paper-faithful 2D navigation positive control. Register
Phase 2AA as:

1. cache frozen DINOv2 patch features for Phase 2D train/validation frames;
2. train an action-conditioned predictor over normalized patch features;
3. gate real-action rollout against persistence, zero-action, and shuffled-action
   controls;
4. evaluate candidate two-block action sequences by DINO-WM-style latent
   goal-feature distance and the existing primitive match/regret/collapse gate;
5. run a simpler LeWM-paper-style 2D navigation positive-control task before any
   full quadruped sweep.

No `test_id` or `test_hard` metrics should be accessed for Phase 2AA model
selection. If DINOv2 weights are unavailable locally, Phase 2AA must fail
preflight or explicitly request approval for the one-time download instead of
silently falling back to a weaker encoder.

Phase 2AA.0 implementation note: the DINOv2 cache preflight is now implemented
and smoke-tested on the ROCm runtime. Local DINOv2 weights are present in the
torch hub cache. A `mode=complete`, `max_rows=4`, `patch_mean` smoke cached 9
unique frames with both future steps represented. The next implementation unit
is the action-conditioned DINOv2 feature predictor and its persistence /
zero-action / shuffled-action gates.

Phase 3A marker-memory follow-up on 2026-06-15: the no-beacon
history-claim alias split now has a non-privileged egocentric marker-memory
control that claims the hidden current beacon in `11/12` validation sources
(`0.9167` claim rate, `0.5` sequence regret). The latest learned
marker-distance checkpoint still claims only `3/12` (`0.25`) despite putting
claim candidates in top-5 on `10/12`. This means the blocker is no longer
whether pixels/actions contain enough information; they do. The next learned
implementation should add an equivariant egocentric belief/occupancy memory
with action-conditioned transforms, not another broad sweep of the generic
GRU/MLP candidate scorer.

Phase 3A structured-marker follow-up: a first structured scalar-delta head was
implemented (`structured_marker_memory_start_delta_prediction` plus
deterministic `rollout_marker_delta`). It improved the best learned structured
checkpoint only to `4/12` claims at 256 steps and collapsed to `1/12` by 512
steps. This keeps the conclusion unchanged but sharper: the next model should
use a spatial belief/heatmap memory with equivariant update/readout. A scalar
remembered-delta regression head is still too brittle under aliasing.

Phase 3A online-memory follow-up on 2026-06-16: a categorical remembered-cell
head with RGB marker saliency features still failed (`4/12` validation claims,
`4.75` sequence regret, `3/12` validation remembered-cell accuracy). The
explicit model-side `online_marker_memory_score_prediction`, however, matches
the non-privileged egocentric control exactly: `11/12` validation claims,
`0.5` sequence regret, and `11/12` top-5 claim coverage. This is recorded in
`phase3a_explore_claim_spatial_memory_online_marker_s1_256_report.json`,
`phase3a_explore_claim_spatial_memory_online_marker_s1_256_gate.json`, and
`phase3a_online_marker_memory_demo.mp4`.

Treat that as the scaffold and target, not as learned JEPA success. The next
implementation should learn a spatial RGB/action belief map that reproduces the
online-memory score without simulator coordinates, hidden beacons, topological
distance, goal ids, or breadcrumb channels.

Phase 3A learned spatial-memory update on 2026-06-16: the first belief-map
implementation passed the tiny no-beacon history-claim alias gate. The learned
`spatial_marker_memory_score_prediction` reached `11/12` validation claims,
`0.5667` sequence regret, and `11/12` top-5 claim coverage in
`phase3a_explore_claim_spatial_belief_marker_s1_256_report.json`; the
claim-only gate passes in
`phase3a_explore_claim_spatial_belief_marker_s1_256_gate.json`. The MP4 demo is
`phase3a_spatial_belief_marker_demo.mp4`.

This is a learned-memory positive control, not full navigation. The belief-map
argmax is correct on `11/12` validation sources, but max cell probability is
still low, and the split starts after the marker has already been seen in
history. The next registered step is to scale the same spatial belief memory to
the larger no-beacon explore-then-claim task: learned novelty/frontier seeking
before discovery, learned marker binding at first sighting, then learned
claim/revisit behavior.

Phase 3A no-beacon novelty-to-claim update on 2026-06-16: the larger task was
made better aligned with the desired end goal by adding `novelty_then_claim`.
Unlike `explore_then_claim`, it does not reward privileged hidden future-marker
visibility before the marker is known. Pre-marker utility is novelty/frontier
exposure minus collision cost; post-marker utility is claim/progress.

The v4 split is:

```text
.generated/jepa_phase3a/explore_claim_v4_novelty_then_claim
train: 216 source groups / 55,296 candidate rows
validation: 108 source groups / 27,648 candidate rows
validation phases:
  explore_unseen: 34
  discover_visible_marker: 31
  claim_after_marker_seen: 43
```

The explicit online frontier+marker memory scaffold now passes the intended
information test without beacons, goal ids, simulator coordinates, or
breadcrumbs:

```text
summary: phase3a_v4_online_frontier_marker_summary.json
model summary: phase3a_v4_model_online_frontier_marker_summary.json
demo: phase3a_v4_online_frontier_marker_demo.mp4

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

This means the online memory can build itself from RGB/action history, explore
for novelty, and claim every validation source where a claim is reachable in
the candidate horizon (`14/14` available claim groups). Single-horizon marker
discovery remains partial because the marker is genuinely hidden until explored.
`Phase3AJepaModel` now exposes the same finite-crop score as
`online_frontier_marker_score_prediction`; the model output matches the Python
reference on this split within `max_abs_prediction_diff ~= 3.8e-7`.

Learned v4 result: `phase3a_v4_novelty_online_marker_memory_512.pt` learns a
candidate scorer plus an explicit online marker-memory term. It claims after
sighting (`0.326` selected claim rate, matching `0.326` availability) but still
under-learns frontier novelty (`6.471` selected new free cells vs `13.912`
oracle on `explore_unseen`). Direct distillation in
`phase3a_v4_online_frontier_distill_512.pt` collapsed to always selecting
`forward`, so do not treat it as learned topological memory.

Next implementation step: convert the spatial belief-map marker memory into a
multi-channel egocentric map for marker evidence, explored cells, known-free
cells, and known-blocked cells. Candidate scoring should read from this map
after deterministic action transforms, instead of relying on a generic MLP
candidate head to rediscover online frontier memory.

Phase 3A randomized-palette closed-loop continuation on 2026-06-16:

Implemented the first persistent spatial-frontier memory loop for closed-loop
evaluation: marker/free/blocked memory maps can now be rolled between replans,
marker writes can be thresholded, and the persistent-marker branch can be gated
on actual marker observation history. A report-only exporter mode
(`--skip-video`) was added so ROCm evals can run in the GPU Python environment
without optional video dependencies. A corrected marker-memory diagnostic now
measures full-grid target localization rather than the old 5x5 categorical
support.

The marker-focused mem15 fine-tune improved the corrected marker-memory top-1
diagnostic from `0.3125` to `0.6875`, but closed-loop behavior still does not
meet the target:

```text
checkpoint:
  models/checkpoints/phase3a_explore_claim/phase3a_v5_random_palette_mem15_markerfocus_256.pt

28-step learned spatial-frontier closed-loop:
  report:
    .generated/jepa_phase3a/explore_claim_v5_random_palette/phase3a_v5_random_palette_mem15_markerfocus_256_block4_closed_loop_report.json
  claimed_episodes: 3 / 16
  marker_seen_ever: 9 / 16

56-step learned spatial-frontier closed-loop:
  report:
    .generated/jepa_phase3a/explore_claim_v5_random_palette/phase3a_v5_random_palette_mem15_markerfocus_256_max56_block4_closed_loop_report.json
  claimed_episodes: 4 / 16
  marker_seen_ever: 12 / 16

56-step observation-gated persistent-marker closed-loop:
  report:
    .generated/jepa_phase3a/explore_claim_v5_random_palette/phase3a_v5_random_palette_mem15_markerfocus_256_persistmarker_seen_gate_max56_block4_closed_loop_report.json
  claimed_episodes: 3 / 16
  marker_seen_ever: 12 / 16
```

Upper-bound diagnostics are also still below the desired behavior. A
palette-aware exact egocentric marker-memory branch with the learned bounded
frontier scorer reaches `7/16` claims at 56 steps, while the model-side
`online_frontier_marker_score` falls to `1/16` on randomized palettes because
its detector is still the old fixed-green threshold.

Updated next implementation step: keep the persistent map machinery, but train
it with full closed-loop rollouts and an explicit palette-agnostic marker
observation objective. The learned map must reduce false marker mass before
sighting, keep a sharper target distribution after sighting, and improve the
frontier policy enough that exact marker memory is no longer capped at `7/16`.

Phase 3A randomized-palette seed correction and explicit-map result on
2026-06-16:

The v5 validation closed-loop seed is `21260634`, from
`.generated/jepa_phase3a/explore_claim_v5_random_palette/phase3a_positive_control_manifest.json`
`validation_audit.seed`. Earlier ad hoc reports using `--seed 20260620`
reconstructed the wrong maze geometry for those validation rows and should be
treated as seed-confounded. The closed-loop exporter now infers the split seed
from the manifest when `--seed` is omitted and writes `scene_seed` into the
report.

The palette-aware explicit odometry-frontier memory controller passes the 2D
randomized-palette validation behavior when allowed a longer exploration
budget:

```text
score_source: odom_frontier_marker_planner
validation seed: 21260634

28 steps, history_window 6:
  claimed_episodes: 10 / 16
  marker_seen_ever: 11 / 16
  collision_steps: 0
  report:
    .generated/jepa_phase3a/explore_claim_v5_random_palette/phase3a_v5_random_palette_odom_frontier_palette_planner_seed21260634_h6_step1_closed_loop_report.json

56 steps, full history:
  claimed_episodes: 12 / 16
  marker_seen_ever: 13 / 16
  collision_steps: 0
  report:
    .generated/jepa_phase3a/explore_claim_v5_random_palette/phase3a_v5_random_palette_odom_frontier_palette_planner_seed21260634_fullhist_max56_step1_closed_loop_report.json

112 steps, full history:
  claimed_episodes: 16 / 16
  marker_seen_ever: 16 / 16
  collision_steps: 0
  report:
    .generated/jepa_phase3a/explore_claim_v5_random_palette/phase3a_v5_random_palette_odom_frontier_palette_planner_seed21260634_fullhist_max112_step1_closed_loop_report.json
```

This is the first clean 2D "explore, build a map, find the marker, remember it,
and return" pass in the randomized-palette setting. It is not yet a learned
JEPA memory pass: the controller uses explicit RGB/odometry map updates for
known-free, known-blocked, and marker cells.

Correct-seed learned checks remain negative:

```text
marker-focused mem15 learned spatial-frontier scorer:
  claimed_episodes: 1 / 16
  marker_seen_ever: 9 / 16
  report:
    .generated/jepa_phase3a/explore_claim_v5_random_palette/phase3a_v5_random_palette_mem15_markerfocus_256_seed21260634_block4_closed_loop_report.json

planner-distilled candidate scorer:
  checkpoint:
    models/checkpoints/phase3a_explore_claim/phase3a_v5_odom_frontier_distill_candidate_256.pt
  claimed_episodes: 0 / 16
  marker_seen_ever: 1 / 16
  failure: selected `forward` at every step in the selected episode and hit
    repeated wall collisions
  report:
    .generated/jepa_phase3a/explore_claim_v5_odom_frontier_distill/phase3a_v5_odom_frontier_distill_candidate_256_seed21260634_candidate_score_top16_step1_max112_closed_loop_report.json

planner-distilled memory-context ranker:
  checkpoint:
    models/checkpoints/phase3a_explore_claim/phase3a_v5_odom_frontier_distill_memctx_rank_1024.pt
  claimed_episodes: 0 / 16
  marker_seen_ever: 6 / 16
  failure: can reach marker visibility but often oscillates near distance 2
    instead of claiming
  report:
    .generated/jepa_phase3a/explore_claim_v5_odom_frontier_distill/phase3a_v5_odom_frontier_distill_memctx_rank_1024_seed21260634_candidate_score_top16_step1_max112_closed_loop_report.json
```

Next step: preserve the explicit controller as the 2D reference implementation
and move the learned work one level closer to it. Train a map-update module
against the explicit known-free/known-blocked/marker state, then train a
frontier/claim scorer over that map. Direct candidate-head distillation from
short rows is currently a negative path.

Phase 3A JEPA-latent map result on 2026-06-16:

Implemented the first version of that map-update bridge. The new
`Phase3ALatentMapHead` reads Phase 3A JEPA spatial tokens and predicts local
blocked/free/marker evidence. The online controller still writes those learned
local predictions into a persistent odometry map and uses the same
frontier/remembered-marker planner.

```text
training script:
  scripts/train_jepa_phase3a_latent_map.py
base checkpoint:
  models/checkpoints/phase3a_explore_claim/phase3a_v5_random_palette_mem15_markerfocus_256.pt
latent map checkpoint:
  models/checkpoints/phase3a_explore_claim/phase3a_v5_latent_map_ctx_4096.pt

validation frames: 888
blocked_accuracy: 1.000
free_accuracy: 1.000
marker_top1_when_visible: 1.000
marker_presence_recall: 1.000
marker_false_presence_rate: 0.007
```

Correct-seed closed-loop result:

```text
score_source: latent_odom_frontier_planner
scene_seed: 21260634 inferred from manifest
latent_map_marker_threshold: 0.9
max_steps: 112
max_episodes: 16
claimed_episodes: 16 / 16
marker_seen_ever: 16 / 16
collision_steps: 0
report:
  .generated/jepa_phase3a/explore_claim_v5_random_palette/phase3a_v5_latent_map_ctx_4096_seedinfer_marker090_max112_report.json
```

This is the current best 2D result: learned JEPA-latent perception feeding an
explicit odometry map. The remaining privileged assumptions are perfect
pose/yaw, cell alignment, and the explicit frontier/BFS controller. Next
iteration should weaken those assumptions in order:

- first, replace global `(x, y)` writes with action-integrated egocentric map
  rolling;
- then add odometry noise/dropout and require graceful degradation;
- then replace grid-cell labels with Go2-style traversability and marker
  perception targets.

Phase 3A egocentric latent-memory result on 2026-06-16:

The first weakening step is now implemented. The new
`latent_egocentric_frontier_planner` score source keeps the learned
`Phase3ALatentMapHead` local perception bridge, but no longer writes local
evidence into global `(x, y)` cells. It maintains a finite egocentric memory of
free, blocked, and marker evidence, rolls that memory by the executed primitive
action, adds forward collision feedback in the cell ahead, and runs the
frontier/remembered-marker controller in that egocentric memory.

Correct-seed randomized-palette closed loop:

```text
score_source: latent_egocentric_frontier_planner
scene_seed: 21260634 inferred from manifest
latent_map_marker_threshold: 0.9
spatial_memory_size: 31
max_steps: 68
max_episodes: 16
claimed_episodes: 16 / 16
marker_seen_ever: 16 / 16
collision_steps: 0
report:
  .generated/jepa_phase3a/explore_claim_v5_random_palette/phase3a_v5_latent_egocentric_ctx_4096_mem31_marker090_max68_report.json
```

The selected trajectory explores under `latent_egocentric_frontier`, first sees
the marker at step 13, switches to `latent_egocentric_marker`, and reaches the
goal after 19 steps with zero collisions.

Memory-support ablation:

```text
default spatial_memory_size: 15
max_steps: 112
claimed_episodes: 15 / 16
marker_seen_ever: 15 / 16
collision_steps: 0
failed episode: index 3, never saw marker, final_goal_distance 5
report:
  .generated/jepa_phase3a/explore_claim_v5_random_palette/phase3a_v5_latent_egocentric_ctx_4096_mem15_marker090_max112_report.json
```

This is now the current best 2D randomized-palette result. It answers the
exocentric-camera concern for this toy setting: the passing controller uses
JEPA-token local perception and action-rolled egocentric memory rather than
global coordinate writes. It still relies on exact primitive odometry,
grid-cell alignment, explicit collision feedback, and a hand-written
frontier/BFS policy, so the next step is not another candidate-score sweep. The
next useful experiment is to replace the hand-written memory roll/controller
with a trained recurrent latent-memory update while keeping the same
egocentric-map targets and ablations.

Phase 3A recurrent latent-memory update on 2026-06-16:

That first recurrent-memory step is now implemented. The new
`Phase3AEgocentricMemoryUpdate` model consumes the previous egocentric memory
tensor, current JEPA-latent local evidence, previous primitive action, and
previous collision bit, then emits the current egocentric memory tensor used by
the same frontier/remembered-marker controller.

Training artifact:

```text
script:
  scripts/train_jepa_phase3a_latent_memory.py
checkpoint:
  models/checkpoints/phase3a_explore_claim/phase3a_v5_latent_memory_recurrent_2048.pt
train examples: 698
validation examples: 428
final validation:
  blocked_accuracy: 1.000
  free_accuracy: 1.000
  marker_accuracy: 1.000
  marker_top1_when_present: 1.000
  teacher_action_match: 1.000
```

Correct-seed randomized-palette closed loop:

```text
score_source: latent_recurrent_egocentric_frontier_planner
scene_seed: 21260634 inferred from manifest
spatial_memory_size: 31
latent_memory_marker_threshold: 0.9
max_steps: 68
max_episodes: 16
claimed_episodes: 16 / 16
marker_seen_ever: 16 / 16
collision_steps: 0
step range: 6..68
report:
  .generated/jepa_phase3a/explore_claim_v5_random_palette/phase3a_v5_latent_recurrent_memory_2048_mem31_marker090_max68_report.json
```

This is the new selected 2D randomized-palette artifact. Compared with the
previous egocentric pass, the online memory state is now generated by a trained
recurrent model and fed back closed-loop. The caveat is still material:
`use_geometric_prior` is true inside the updater, so action integration is a
fixed architectural prior before learned fusion, and the action-selection
policy remains an explicit frontier/BFS controller. The next clean ablation is
therefore either a no-geometric-prior recurrent updater, or a learned policy
head over the recurrent memory that replaces the explicit controller.

Phase 3A scaffold-removal follow-up on 2026-06-16:

Both first clean ablations have now been run, and neither replaces the selected
artifact.

No-geometric-prior memory:

```text
checkpoint:
  models/checkpoints/phase3a_explore_claim/phase3a_v5_latent_memory_nogeom_allgroups_4096.pt
final validation teacher_action_match: 0.921
closed-loop report:
  .generated/jepa_phase3a/explore_claim_v5_random_palette/phase3a_v5_latent_memory_nogeom_allgroups_4096_mem31_marker090_max68_report.json
claimed_episodes: 12 / 16
marker_seen_ever: 12 / 16
collision_steps: 0
```

Learned policy over the passing recurrent memory:

```text
first policy checkpoint:
  models/checkpoints/phase3a_explore_claim/phase3a_v5_latent_policy_mem31_2048.pt
claimed_episodes: 1 / 16
marker_seen_ever: 8 / 16
collision_steps: 749

balanced synthetic policy checkpoint:
  models/checkpoints/phase3a_explore_claim/phase3a_v5_latent_policy_aug8192_balanced_4096.pt
final validation action_match: 0.748
closed-loop report:
  .generated/jepa_phase3a/explore_claim_v5_random_palette/phase3a_v5_latent_policy_aug8192_balanced_4096_max68_report.json
claimed_episodes: 4 / 16
marker_seen_ever: 10 / 16
collision_steps: 521
```

Current interpretation: the recurrent-memory pass is real but still scaffolded.
Removing the action-geometric prior weakens exploration coverage, and replacing
the frontier/BFS controller with a one-step imitation head causes repeated
post-marker collisions. The next policy-side step should be a closed-loop
objective, value/planning target, or differentiable planning head with explicit
safety supervision rather than another single-step action classifier.

To reduce manual permission prompts, repeated GPU jobs should use the stable
wrappers:

```text
scripts/phase3a_rocm_train_latent_memory.sh
scripts/phase3a_rocm_train_latent_policy.sh
scripts/phase3a_rocm_export_closed_loop.sh
```

Phase 3A learned-policy iteration on 2026-06-16:

The first value/safety/DAgger policy iteration improved the learned policy but
still did not replace the explicit frontier/BFS controller.

```text
checkpoint:
  models/checkpoints/phase3a_explore_claim/phase3a_v5_latent_policy_conv_dagger_v1_4096.pt
selected validation:
  action_match: 0.829
  blocked_forward_violation_rate: 0.000
closed-loop report:
  .generated/jepa_phase3a/explore_claim_v5_random_palette/phase3a_v5_latent_policy_conv_dagger_v1_4096_max68_report.json
claimed_episodes: 8 / 16
marker_seen_ever: 11 / 16
collision_steps: 0
longer-horizon diagnostic:
  claimed_episodes: 8 / 16 at max_steps 112
```

This is better than the previous learned policy (`4/16`, 521 collisions), but
it is still below the requirement and below the explicit-planner reference. The
new failure mode is no longer wall collision; it is a stable under-exploration
or failed-return attractor. No-synthetic DAgger, smaller-synthetic stronger
safety, and random-exploration DAgger were all dominated before completion.

Updated next implementation step: stop training one-step action classifiers.
The learned replacement for the explicit controller needs a multi-step
structure, such as a differentiable value-iteration/planning head over the
recurrent memory or a learned target-cell/value field with deterministic action
extraction.

Phase 3A value-field planner result on 2026-06-17:

The multi-step planner hypothesis passed. A new
`latent_recurrent_value_field_planner` score source keeps the trained recurrent
JEPA memory updater but replaces the explicit BFS controller with value
propagation over the recurrent memory tensor. It builds a marker reward field
when a marker is remembered; otherwise it builds a frontier reward field and
propagates values through known-free cells before extracting the first primitive.

```text
score_source: latent_recurrent_value_field_planner
recurrent memory:
  models/checkpoints/phase3a_explore_claim/phase3a_v5_latent_memory_recurrent_2048.pt
report:
  .generated/jepa_phase3a/explore_claim_v5_random_palette/phase3a_v5_latent_recurrent_value_field_2048_mem31_marker090_max68_report.json
max_steps: 68
claimed_episodes: 16 / 16
marker_seen_ever: 16 / 16
collision_steps: 0
step range: 6..68
```

This is now the selected 2D randomized-palette result for the current gate:
learned JEPA-token perception, trained recurrent egocentric memory, and
multi-step value-field action selection. The caveat remains important: the
marker/frontier reward field, memory thresholding, geometric action prior, and
value propagation algorithm are still fixed. The next clean learning step is to
learn the reward/target field or value residual on top of this planner, not to
return to one-step policy classifiers.

Phase 3A learned target-field follow-up on 2026-06-17:

That target-field step is now implemented. The new
`Phase3AEgocentricValueFieldHead` predicts target rewards from the trained
recurrent JEPA memory tensor, and
`latent_recurrent_learned_value_field_planner` uses those learned targets with
the same fixed value propagation and first-action extraction.

```text
training script:
  scripts/train_jepa_phase3a_value_field.py
wrapper:
  scripts/phase3a_rocm_train_value_field.sh
single-channel checkpoint:
  models/checkpoints/phase3a_explore_claim/phase3a_v5_latent_value_field_head_4096.pt
selected validation:
  action_match: 0.970
  target_top1_match: 1.000
  target_recall: 0.910
  target_precision: 0.832
```

Full learned marker/frontier target selection remains just below the gate:

```text
report:
  .generated/jepa_phase3a/explore_claim_v5_random_palette/phase3a_v5_latent_learned_value_field_4096_mem31_marker090_max68_report.json
claimed_episodes: 15 / 16
marker_seen_ever: 16 / 16
collision_steps: 0
main failure: episode 11 saw the marker but followed a bad learned target field
```

Two direct variants did not fix it: a high-recall 1536-step checkpoint reached
`14/16`, and a split marker/frontier two-channel checkpoint reached `13/16`.
The useful passing ablation is learned frontier targets plus fixed marker
targets once the recurrent memory has bound the marker:

```text
score_source: latent_recurrent_learned_value_field_planner
flag: --latent-value-fixed-marker-target
checkpoint:
  models/checkpoints/phase3a_explore_claim/phase3a_v5_latent_value_field_head_4096.pt
report:
  .generated/jepa_phase3a/explore_claim_v5_random_palette/phase3a_v5_latent_learned_frontier_fixed_marker_4096_mem31_marker090_max68_report.json
claimed_episodes: 16 / 16
marker_seen_ever: 16 / 16
collision_steps: 0
step range: 6..68
```

Initial interpretation before the extractor follow-up: the hand-written
frontier reward field could be replaced by a learned JEPA-memory target field
without regressing the 2D gate, but the marker-claim target still looked brittle
enough to keep fixed for the selected artifact.

Follow-up in the same iteration: the strict learned marker/frontier target path
now clears the 2D gate when marker-present states use learned top-1 extraction
while frontier states keep broad top-k extraction. This does not use the fixed
marker target branch.

```text
score_source: latent_recurrent_learned_value_field_planner
checkpoint:
  models/checkpoints/phase3a_explore_claim/phase3a_v5_latent_value_field_head_4096.pt
frontier extraction: --latent-value-target-top-k 16
marker extraction: --latent-value-marker-target-top-k 1
fixed marker target: false
report:
  .generated/jepa_phase3a/explore_claim_v5_random_palette/phase3a_v5_latent_learned_value_field_4096_mem31_marker090_markertop1_max68_report.json
claimed_episodes: 16 / 16
marker_seen_ever: 16 / 16
collision_steps: 0
step range: 6..68
```

The earlier marker-weighted DAgger attempt is a negative:
`phase3a_v5_latent_value_field_head_dagger_markerw4_4096.pt` regressed to
`12/16` closed-loop claims. The useful conclusion is sharper: the single-channel
head already learned the marker as the top target, but marker-state value
propagation should not keep the same broad `top_k=16` used for frontier
exploration. The selected 2D result is now learned JEPA-token perception,
trained recurrent egocentric memory, learned marker/frontier target field, and
fixed value propagation with a phase-conditioned learned-target extractor.

Next scaffold-removal follow-up: the phase-conditioned extractor is now learned
as a separate mode head. `Phase3AValueFieldExtractorHead` predicts sparse-vs-
broad extraction from the recurrent memory tensor, so the selected command no
longer passes `--latent-value-marker-target-top-k 1`.

```text
extractor checkpoint:
  models/checkpoints/phase3a_explore_claim/phase3a_v5_latent_value_extractor_1024.pt
extractor validation:
  accuracy: 1.000
  sparse_precision: 1.000
  sparse_recall: 1.000

closed-loop report:
  .generated/jepa_phase3a/explore_claim_v5_random_palette/phase3a_v5_latent_learned_value_field_extractor_1024_mem31_marker090_max68_report.json
gate:
  .generated/jepa_phase3a/explore_claim_v5_random_palette/phase3a_v5_latent_learned_value_field_extractor_1024_mem31_marker090_max68_gate.json
claimed_episodes: 16 / 16
marker_seen_ever: 16 / 16
collision_steps: 0
latent_value_fixed_marker_target: false
latent_value_marker_target_top_k: null
latent_value_extractor_head: phase3a_v5_latent_value_extractor_1024.pt
gate passed: true
```

The remaining fixed piece is value propagation/action extraction itself. The
next meaningful 2D experiment should learn a value residual or differentiable
planning head over the learned target field, while keeping this new closed-loop
gate as the regression check.

Follow-up: the direct learned-action replacement for value propagation is a
negative. `Phase3AValueFieldActionHead` was added and trained to consume the
recurrent memory, learned target field, and learned sparse/broad extractor
probability, then predict the first primitive directly.

```text
plain action head:
  checkpoint: phase3a_v5_latent_value_action_2048.pt
  selected validation action_match: 0.818
  closed-loop claimed_episodes: 1 / 16
  marker_seen_ever: 1 / 16
  collision_steps: 524

DAgger action head:
  checkpoint: phase3a_v5_latent_value_action_dagger_4096.pt
  selected validation action_match: 0.832
  closed-loop claimed_episodes: 0 / 16

synthetic local-memory action head:
  checkpoint: phase3a_v5_latent_value_action_synth8192_4096.pt
  selected validation action_match: 0.893
  closed-loop claimed_episodes: 3 / 16

label-smoothed synthetic action head:
  checkpoint: phase3a_v5_latent_value_action_synth8192_smooth005_4096.pt
  selected validation action_match: 0.893
  pure closed-loop claimed_episodes: 1 / 16
  residual fallback threshold 0.95: 1 / 16
  residual fallback threshold 0.999: 0 / 16
```

Conclusion: fixed value propagation/action extraction is still the selected
scaffold. First-action imitation over the learned memory/target stack is not
stable enough closed-loop, even with DAgger states, synthetic local-memory
safety examples, and label smoothing. The next attempt should supervise a dense
value map or differentiable planning objective rather than another primitive
classifier.

Follow-up: dense value-map and soft latent planning were implemented and tested.
`Phase3AValueMapPlannerHead` predicts a dense egocentric value map from the
trained recurrent memory, learned target field, and learned sparse/broad
extractor probability. The exporter now supports
`latent_recurrent_learned_value_map_planner`. It also supports
`latent_recurrent_soft_value_map_planner`, which runs soft value propagation
over learned memory probabilities and sparse learned target tensors.

```text
plain dense value-map planner:
  checkpoint: phase3a_v5_latent_value_map_planner_synth8192_2560.pt
  selected validation action_match: 0.900
  closed-loop claimed_episodes: 8 / 16

dense value-map planner with one DAgger pass:
  checkpoint: phase3a_v5_latent_value_map_planner_dagger_synth8192_3072.pt
  selected validation action_match: 0.900
  sparse_action_match: 0.955
  closed-loop report:
    .generated/jepa_phase3a/explore_claim_v5_random_palette/phase3a_v5_latent_value_map_dagger_synth8192_3072_frontierexit_mem31_marker090_max68_report.json
  claimed_episodes: 11 / 16
  marker_seen_ever: 12 / 16
  collision_steps: 0
  max96 diagnostic: 11 / 16

soft latent value-map planner:
  raw learned targets: 8 / 16
  sparse learned targets: 9 / 16
  top-k/gamma sweep: no better than 9 / 16

negative dense variants:
  h128/synth16384/gamma0.90/positive_weight16 regressed offline
  second DAgger pass regressed offline
  no-synthetic DAgger closed-loop: 6 / 16
```

Conclusion: dense value-map supervision is a better scaffold-removal direction
than first-action imitation, but it has not met the closed-loop gate. The best
learned-planner failures are now mostly exploration misses: the policy does not
reliably drive the learned memory to marker discovery in all starts. Keep the
Phase 3A.21 learned-target/extractor plus fixed value propagation artifact as
the selected passing baseline while the next iteration targets learned
frontier/exploration stability.

Follow-up, 2026-06-17: action-ranked dense value-map training plus runtime
output ensembling improved the strict learned planner but did not fully close
the gate.

```text
best strict learned value-map ensemble:
  heads:
    phase3a_v5_latent_value_map_planner_dagger_synth8192_3072.pt
    phase3a_v5_latent_value_map_planner_actionloss01_dagger2_synth8192_3072.pt
  max68 report:
    .generated/jepa_phase3a/explore_claim_v5_random_palette/phase3a_v5_latent_value_map_ensemble_mean_dense_action2_mem31_marker090_max68_report.json
  max68 claimed_episodes: 13 / 16
  max96 claimed_episodes: 15 / 16

weighted strict learned ensemble:
  2x dense + 2x actionloss DAgger2 + 1x actionloss DAgger3
  max96 claimed_episodes: 15 / 16
  marker_seen_ever: 16 / 16
  remaining failure: marker-return loop after marker was seen

hybrid systems milestone:
  learned value-map exploration ensemble + fixed marker return
  report:
    .generated/jepa_phase3a/explore_claim_v5_random_palette/phase3a_v5_latent_value_map_ensemble_weighted_2dense_2action2_1action3_hybrid_fixedmarkerreturn_mem31_marker090_max128_report.json
  claimed_episodes: 16 / 16
  marker_seen_ever: 16 / 16
  collision_steps: 0
```

Next technical target: remove the fixed marker-return fallback. The remaining
all-learned failure is not perception or marker discovery; it is a learned
return/value-map loop after marker memory exists. Marker-only sparse heads,
fixed marker target injection, and marker-head self-DAgger did not fix it.

Follow-up, 2026-06-17: the fixed marker-return fallback was removed by adding a
learned marker-return action head trained on sparse marker-memory states with
exact marker targets.

```text
learned value-map exploration ensemble + learned marker-action return:
  marker action checkpoint:
    phase3a_v5_latent_marker_action_fixedtarget_sparse_synth8192_4096.pt
  selected marker-action sparse_action_match: 0.964
  max96 report:
    .generated/jepa_phase3a/explore_claim_v5_random_palette/phase3a_v5_latent_value_map_ensemble_weighted_marker_action_fixedtarget_mem31_marker090_max96_report.json
  claimed_episodes: 16 / 16
  marker_seen_ever: 16 / 16
  collision_steps: 0

same controller at max68:
  claimed_episodes: 14 / 16
```

Next target: improve efficiency rather than basic feasibility. The strict
learned controller now solves the 16 validation starts by max96, but it is still
slower than the fixed-return baseline. Focus on faster exploration/value-map
selection for the two max68 failures.

Follow-up, 2026-06-17: the strict learned controller now also closes max68 by
adding a small egocentric latent-memory route gate. The gate latches the dense
learned value-map fallback when recurrent memory contains a contiguous side wall
next to the agent, otherwise it uses the weighted learned value-map ensemble
before falling back to dense after step 7 if marker memory is still absent.
Marker return remains the learned marker-action head; the fixed return fallback
is off.

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

Next target: replace the explicit side-wall gate with a trained latent-memory
router or absorb it into a single planner while preserving the `16/16` max68
result on disjoint starts.

Follow-up, 2026-06-17: replaced the runtime side-wall gate with a learned
latent-memory router distilled from that gate. The router is a conv classifier
over recurrent egocentric memory and is loaded with
`--latent-value-map-router-head`; the explicit
`--latent-value-map-side-wall-fallback` flag is off in the selected run.

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

Next target: remove the remaining label-provenance caveat by training route
labels from planner counterfactual rollouts or validating the router across
broader disjoint seeds.

Follow-up, 2026-06-17: trained a counterfactual-labeled latent-memory router.
The trainer labels each rollout memory by comparing the weighted learned
value-map ensemble's first-action utility against the dense fallback's
first-action utility using the same horizon-4 candidate rows used by closed-loop
evaluation. This checkpoint uses no side-wall synthetic labels.

```text
counterfactual router checkpoint:
  models/checkpoints/phase3a_explore_claim/phase3a_v5_latent_value_map_router_counterfactual_s7_utility_h4_1024.pt
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

Next target: generalization. The current counterfactual-positive route labels
are sparse but sufficient on this validation set, so evaluate the selected
controller across broader disjoint validation seeds and larger validation sets
before treating the 2D result as robust.

Follow-up, 2026-06-17: the broader disjoint randomized-palette validation did
not preserve the original 16/16 claim. The new split uses seed `20260701`, eight
train scenes, eight validation scenes, and the first 64 validation source/goal
groups.

```text
broad validation data:
  .generated/jepa_phase3a/explore_claim_v5_random_palette_generalization_seed20260701/validation_phase3a_positive_control.jsonl

old selected strict controller:
  claimed_episodes: 37 / 64

ceiling checks on refreshed learned latent memory:
  exact odom frontier: 62 / 64
  exact latent recurrent frontier: 60 / 64

best strict learned result from this iteration:
  report:
    .generated/jepa_phase3a/explore_claim_v5_random_palette_generalization_seed20260701/phase3a_v5_latent_value_map_broad_map_memory_frontier_dagger3_marker_action_synthsmooth_turnbreak_mem31_marker090_max68_generalization64_report.json
  claimed_episodes: 59 / 64
  fixed_marker_return: false
  explicit_side_wall_fallback: false

structured latent-memory diagnostic:
  report:
    .generated/jepa_phase3a/explore_claim_v5_random_palette_generalization_seed20260701/phase3a_v5_latent_soft_value_fixed_marker_broad_map_memory_mem31_marker090_max68_generalization64_report.json
  claimed_episodes: 61 / 64
  note: uses --latent-value-fixed-marker-target, so it is not the strict learned-return result
```

Next target: close the remaining strict learned action-selection gap. The broad
latent map and recurrent latent memory are near the exact odometry frontier
ceiling, but learned value-map/action readouts still lose episodes through
wrong exploration turns and imperfect marker return. Retraining stale target
heads improved offline target metrics but did not improve closed-loop score, so
the next useful work is on closed-loop-aware planner training or learned
readouts that directly optimize the failed trajectory states.

Follow-up, 2026-06-19: ran one bounded trace-action-preference router
experiment and stopped rather than continuing to optimize the same random set.
The router labels a state positive only when the action05 fallback planner
matches the trace oracle and the old strict primary planner does not. Training
used the ROCm wrapper outside the managed sandbox because sandboxed PyTorch
could not see HIP GPUs.

```text
router checkpoint:
  models/checkpoints/phase3a_explore_claim/phase3a_v5_value_map_router_old_to_action05_trace_actionpref_oracle_train8_val47_pw2_512.pt

training setup:
  label_source: trace_action_preference
  train broad seeds: 20260701 20260709 20260717 20260723 20260731 20260735 20260739 20260743
  validation broad seed: 20260747
  optimization_steps: 512
  positive_weight: 2.0
  threshold: 0.5

router validation:
  train positives: 98 / 10382
  validation positives: 22 / 1407
  validation true positives: 0
  validation false positives: 0
  validation recall: 0.0

strict smoke eval:
  report:
    .generated/jepa_phase3a/explore_claim_v5_random_palette_generalization_seed20260747/phase3a_v5_router_trace_actionpref_pw2_thr050_routeronly_strict_max68_report.json
  claimed_episodes: 55 / 64
  fixed_marker_return: false
  explicit_side_wall_fallback: false
  fallback_after_step: 999
```

Interpretation: negative result. The action-preference labels are too sparse
under this construction, so the router collapses to the negative class at the
registered threshold and does not improve closed-loop behavior. Current
confirmed headline aggregate remains `525 / 576 = 91.1%` for the best strict
learned broad result, versus `518 / 576 = 89.9%` for the old strict baseline and
`513 / 576 = 89.1%` for the action05 comparator. A perfect score on this
particular random set should not be treated as the target; more broad seeds will
almost certainly expose additional edge cases.

Threshold-calibration follow-up, 2026-06-19: completed strict global threshold
sweeps for the best trace-outcome router checkpoint
`phase3a_v5_value_map_router_old_to_action05_traceoutcome_train8_val47_pw025_128.pt`.
All reports preserved the strict contract (`fixed_marker_return=false`,
`explicit_side_wall_fallback=false`, `fallback_after_step=999`).

```text
threshold 0.50:
  aggregate: 525 / 576 = 91.1%

threshold 0.80:
  aggregate: 518 / 576 = 89.9%

threshold 0.95:
  aggregate: 518 / 576 = 89.9%
```

Higher thresholds improve the worst seed (`20260739`, from `52/64` at `0.50`
to `54/64` at `0.80` and `55/64` at `0.95`) but lose enough episodes on other
seeds to erase the aggregate gain. Threshold calibration alone is not a useful
next path.
