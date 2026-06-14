# JEPA Navigation: Experiment Failure and Next Steps

Date: 2026-06-14

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

It should answer three questions before substantial architecture investment:

1. Does the current pooled LeWM contain enough information to distinguish safe
   counterfactual actions when evaluated correctly?
2. Do spatial tokens materially improve that decision?
3. How far into free-running prediction must a model remain accurate to improve
   the deployed action?

This establishes a direct, falsifiable bridge between JEPA representation
learning and navigation behavior, and provides the evaluation contract for all
subsequent world-model designs.
