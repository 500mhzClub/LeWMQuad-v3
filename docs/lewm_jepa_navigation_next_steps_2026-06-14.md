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
