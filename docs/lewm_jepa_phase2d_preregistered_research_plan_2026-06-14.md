# JEPA Navigation Phase 2D Preregistered Research Plan

Date registered: 2026-06-14

Branch at registration: `jepa-spatial-world-model-nav`

Code commit at registration: `5c06bba76955c37b7223e9255756dffa59338680`

Status: design preregistration implemented through Stage 10; GPU primary
training attempted; original C2 failed; detached-control C2 stabilization pilot
failed the explicit validation gate

Retrospective evidence audit:
`docs/lewm_jepa_repository_research_audit_2026-06-14.md`

Implementation status:
`docs/lewm_jepa_phase2d_stage0_stage1_implementation_2026-06-14.md`

Corrected model status:
`docs/lewm_jepa_phase2d_stage2_corrected_model_2026-06-14.md`

Trainer/statistics status:
`docs/lewm_jepa_phase2d_stage3_trainer_statistics_2026-06-14.md`

Source-state table status:
`docs/lewm_jepa_phase2d_stage4_source_state_table_2026-06-14.md`

Split/run readiness status:
`docs/lewm_jepa_phase2d_stage5_split_run_readiness_2026-06-14.md`

Generation-contract status:
`docs/lewm_jepa_phase2d_stage6_generation_contract_2026-06-14.md`

Training-start gate status:
`docs/lewm_jepa_phase2d_stage7_training_start_gate_2026-06-14.md`

Source-selection and render-readiness status:
`docs/lewm_jepa_phase2d_stage8_source_selection_render_readiness_2026-06-14.md`

Training-ready status:
`docs/lewm_jepa_phase2d_stage9_training_ready_2026-06-14.md`

Full-training launch status:
`docs/lewm_jepa_phase2d_stage10_full_training_launch_2026-06-14.md`

Stage 0 and Stage 1 reusable foundations are partially implemented, and the
Stage 2 corrected-model implementation gate passes. The Phase 2D pilot trainer,
diagnostic controls, strict data gate, and statistical utilities are
implemented for smoke/pilot use. The trainer now emits the required
source-state prediction/control table. Split and run-readiness guards now block
held-out access without verified lineage and frozen C0/C1/C2 manifests. The
counterfactual generation path now records the full two-block sequence grid and
propagates topology/visual lineage into final spatial rows. Confirmatory
training now has a separate training-start preflight and the trainer enforces
it before optimization. Registered-minimum source rows, all-candidate render
plans, complete render accounting, spatial-future datasets, a verified split
manifest, the C2 training-start preflight, and frozen C0/C1/C2 primary run
manifests now exist. The registered primary full-training matrix on train and
validation data exposed a validation-diagnostic OOM in the first launch attempt;
that implementation failure was documented, fixed, and smoke-tested on the full
validation path. A CPU-only v2 relaunch was stopped as a runtime-selection
error. The corrected GPU launch completed C0 and C1 seeds, but all completed
C0/C1 runs failed the registered persistence gate, and final C1 stability also
failed. Original C2 failed through a combination of concurrent GPU OOM and
non-finite objective dynamics. A bounded detached-control C2 stabilization
pilot remained finite but collapsed, failed action-identifiability, and lost to
persistence by `247.69x`. The trainer now records an explicit
`final_validation_gate` and permits checkpoint selection only when stability,
zero-action advantage, hard-negative action advantage, and persistence criteria
all pass. No test-ID or test-hard result has been opened.

## Purpose

Phase 2D is a bounded causal diagnostic. It tests whether corrected target
normalization and a valid action-identifiability objective make the current
spatial JEPA formulation learn action-conditioned one-step transitions.

It is not:

- a final navigation model;
- a physical-safety claim;
- a test of recurrent belief;
- a full-scale generalization claim;
- a license to tune on the final test set.

## Primary Research Question

> On approximately Markov-visible, scene-disjoint counterfactual transitions,
> does a normalized EMA spatial JEPA trained with valid hard action negatives
> predict the correct one-step future better than persistence, zero action, and
> non-identical wrong actions?

## Hypotheses

### Primary Hypothesis H1

Adding a masked action-identifiability loss to normalized EMA targets improves
one-step prediction under the real action relative to matched non-identical
wrong actions.

Primary estimand:

`wrong_action_mse - real_action_mse`, normalized by target-change MSE, on the
eligible non-hold hard-negative subset.

Primary success threshold:

- lower bound of the paired `95%` hierarchical bootstrap confidence interval is
  greater than `0`;
- point estimate is at least `0.10`;
- the individual-seed point estimate is positive and at least `0.10` for at
  least `2` of `3` optimization seeds.

### Co-Primary Hypothesis H2

The primary C2 Phase 2D cell beats one-step persistence on validation and
untouched test-ID scenes.

Primary estimand:

`real_action_rollout_mse / persistence_mse`.

Primary success threshold:

- upper bound of the paired `95%` hierarchical bootstrap confidence interval is
  below `1.0`;
- point estimate is at most `0.90`;
- the individual-seed point estimate is below `1.0` and at most `0.90` for at
  least `2` of `3` optimization seeds.

The `0.90` point threshold prevents promotion on a negligible numerical win.

### Secondary Hypothesis H3

Normalized target geometry prevents the Phase 2C scale-growth failure without
causing low-variance collapse.

Success requires:

- no registered collapse warning;
- no registered near-static-target warning;
- pre-normalization target feature scale remains within the fixed interval
  defined below;
- final prediction loss does not exceed its minimum epoch value by more than
  `50%`.

### Exploratory Hypothesis H4

Corrected action-identifiable dynamics improve conditional counterfactual
consequence ranking. This is exploratory because labels remain kinematic and
future-observation validity remains outcome-dependent.

No Phase 2D cell may pass solely on H4.

## Required Corrections Before Training

Phase 2D must not run until all corrections below have automated tests and a
machine-readable data/control audit.

### 1. Per-Slot Observation Masks

Use every planned candidate sequence. For each future slot:

- compute token prediction loss only when the observation is valid;
- retain the row, action, validity, and consequence-event metadata when the RGB
  target is invalid;
- never infer collision directly from renderer invalidity;
- report valid-slot coverage by action, family, consequence bucket, and horizon.

Complete-sequence filtering is prohibited for the primary analysis.
Privileged consequence labels remain evaluation and audit labels in C0-C2;
they are not training targets in Phase 2D.

### 2. Valid Hard Action Negatives

For each eligible source state and step:

- the negative must come from the same source state;
- the negative primitive at the evaluated step must differ from the real
  primitive;
- true-hold positives are excluded from zero-action and hard-negative primary
  estimands;
- identical action vectors are excluded even if primitive names differ;
- negative eligibility and exclusion reasons are recorded.

Batch rolling without semantic checks is prohibited.

### 3. Source-Grouped Balanced Sampling

Training batches must preserve the matched-state counterfactual structure.

Required audit targets:

- every source state contributes multiple distinct first actions when available;
- first-action counts are reported before and after validity masking;
- no action receives less than `5%` of eligible first-action training examples,
  or inverse-frequency weighting is preregistered before training;
- source states and scenes, not candidate rows, define statistical groups.

The confirmatory dataset must include the full `81` two-block primitive
sequences for every selected source state. Candidate subsampling is permitted
only for smoke tests and pilots.

Source states must be selected without using future candidate outcomes.
Selection may use split, scene family, valid current observation, reset
integrity, and a preregistered current-state stratification. The source-state
selection manifest must be frozen before counterfactual outcomes are analyzed.

### 4. Separate Prediction And Target Projection Paths

The spatial predictor output and target encoder output must not share one
BatchNorm projector.

Registered implementation choice:

- L2-normalize each spatial token before prediction loss and diagnostics;
- use a separate predictor projection head;
- use a separate EMA target projection head;
- do not use BatchNorm in either spatial projection head;
- retain EMA stop-gradient targets;
- continue using the appearance branch only as a separately reported
  anti-collapse control.

LayerNorm or a linear head is acceptable only if selected before seeing Phase
2D validation results and documented as a protocol amendment.

### 5. Split Correction

The existing Phase 2B evaluation data is redesignated `validation`.

Required splits:

| Split | Purpose | Access rule |
| --- | --- | --- |
| train | parameter fitting | unrestricted during training |
| validation | checkpoint selection and protocol debugging | may be evaluated each epoch |
| test-ID | final same-distribution estimate | opened once after cell and checkpoint rule are frozen |
| test-hard | final stress estimate | opened once after test-ID report is generated |

No scene, topology seed, visual seed, or source state may cross splits.

The test split generation seed, scene IDs, and dataset checksums must be
committed before training begins. Aggregate test labels must not be inspected
while selecting the model.

Minimum confirmatory split size:

| Split | Minimum scenes | Minimum source states per scene |
| --- | ---: | ---: |
| train | 32 | 16 |
| validation | 16 | 16 |
| test-ID | 16 | 16 |
| test-hard | 16 | 16 |

Scenes must remain balanced across the eight registered families wherever the
split definition permits. A smaller run is a smoke test or pilot and may not be
reported as the confirmatory Phase 2D result.

### 6. Nominal And Executed Action Contract

Phase 2D kinematic futures use deterministic nominal primitive actions. The
report must state this explicitly.

For later physics-trained models, the dataset must store both nominal and
executed blocks. The planner-facing model will condition on nominal action and
predict execution-conditioned uncertainty. This later change is out of scope
for the primary Phase 2D gate.

## Experimental Cells

All cells use identical corrected data, splits, source-grouped sampling,
architecture capacity, optimizer family, optimizer-step budget, checkpoint
rule, and three optimization seeds.

| Cell | Target | Action-identifiability loss | Purpose |
| --- | --- | --- | --- |
| C0: corrected online spatial control | normalized online target | no | Measures effect of corrected data, negatives, and projector alone |
| C1: normalized EMA | normalized EMA target | no | Isolates target stabilization |
| C2: normalized EMA plus action ID | normalized EMA target | yes | Tests H1 and H2 |

The original Phase 2B `spatial_var` result remains a historical reference, not
a capacity-matched cell in the corrected factorial.

### Fixed Design Constants

- latent dimension: `48`;
- encoder depth: `2`;
- encoder heads: `3`;
- encoder MLP ratio: `2`;
- predictor layers: `2`;
- predictor heads: `4`;
- predictor head dimension: `12`;
- predictor MLP dimension: `96`;
- optimizer: AdamW;
- learning rate: `3e-4`;
- weight decay: `1e-4`;
- EMA momentum for C1 and C2: `0.99`;
- appearance SIGReg weight: `0.09`;
- normalized spatial variance-floor weight: `1.0`;
- optimization seeds: `20260614`, `20260615`, and `20260616`.

The optimizer-step budget, source-group batch construction, and evaluation
interval must be fixed in the immutable run manifests before any confirmatory
training or validation result is inspected. They must be identical across C0,
C1, and C2.

### Optional Ablation

One action-loss-weight ablation may be added only if its two fixed weights are
registered before any training. Continuous hyperparameter search is out of
scope for this bounded gate. This ablation is exploratory and may not replace
the fixed C2 cell in the confirmatory decision.

## Loss Contract

For valid target slot `t`, let:

- `z_t` be the normalized target;
- `p_real` be the prediction under the real action;
- `p_wrong` be the prediction under a valid non-identical wrong action;
- `p_zero` be the zero-action prediction when the real action is non-hold.

For C1 and C2, `z_t` is produced by the stop-gradient EMA target path. C0 uses
the normalized online target path as a historical control.

Base prediction loss:

`L_pred = mean(mask_valid * d(p_real, z_t))`

Action-identifiability loss:

`L_action = mean(mask_hard * max(0, margin_i + d(p_real, z_t) - d(p_wrong, z_t)))`

Zero-action contrast:

`L_zero = mean(mask_non_hold * max(0, margin_i + d(p_real, z_t) - d(p_zero, z_t)))`

`d` is mean squared error between per-token L2-normalized spatial features.
For each valid transition:

`margin_i = 0.10 * max(target_change_mse_i, 1e-4)`

For training, average `L_action` over every eligible same-source non-identical
wrong action present in the source-grouped batch. Randomly choosing one
convenient negative is prohibited for the confirmatory cells.

Apply the spatial variance floor to online per-token L2-normalized features
with target coordinate standard deviation `1 / sqrt(spatial_feature_dim)`.
Retain appearance SIGReg on the separate appearance branch.

The complete registered objectives are:

- C0 and C1:
  `L_total = L_pred + 0.09 * L_appearance_sigreg + 1.0 * L_spatial_var`;
- C2:
  `L_total = L_pred + 1.0 * L_action + 1.0 * L_zero + 0.09 * L_appearance_sigreg + 1.0 * L_spatial_var`.

The primary action metric remains a held-out measured advantage; passing the
training margin alone is not success.

## State And Eligibility Scope

The primary Phase 2D analysis is restricted to approximately Markov-visible
transitions:

- valid current RGB observation;
- valid future RGB observation for the evaluated slot;
- no reset between state and future;
- no known physical fall;
- sufficient candidate action diversity at the source state.

The following subsets must be reported separately:

- non-hold first actions;
- turns/arcs;
- translations;
- visible-obstacle or opening transitions where labels are available;
- invalid or terminal future-observation events;
- history-required/reveal transitions where identifiable.

History-required subsets are diagnostic only in Phase 2D. Their purpose is to
prevent single-frame failure from being misinterpreted as a general target
failure.

## Metrics

### Primary Metrics

1. One-step rollout/persistence MSE ratio.
2. Hard-negative action advantage normalized by target-change MSE.
3. Zero-action advantage normalized by target-change MSE on non-hold examples.

All primary metrics are paired at source-state level and reported with
hierarchical confidence intervals.

### Primary Estimand Definitions

For every eligible non-hold real transition `i`, evaluate all unique
same-source action vectors whose action at the evaluated step differs from the
real action. Deduplicate negatives by action vector at one step and by action
history at longer horizons. Define:

`A_i = (mean_wrong_mse_i - real_mse_i) / max(target_change_mse_i, 1e-4)`

The `1e-4` denominator floor is fixed before training to avoid numerically
unstable ratios on nearly static targets. Report the fraction of examples for
which the floor is active.

For H1:

1. average `A_i` across eligible transitions within each source state;
2. average source states within each scene;
3. average scenes with equal weight.

For H2:

1. average real and persistence MSE separately within each source state;
2. average each quantity within scene and then across scenes with equal weight;
3. compute the ratio of the resulting real and persistence means.

Bootstrap replicates must recompute these complete estimands. Do not average
candidate-level ratios or let scenes with more valid candidates receive more
weight.

### Required Secondary Metrics

- two-step free-running/persistence ratio;
- action advantage by primitive and scene family;
- train versus validation persistence ratio;
- valid-slot and negative-eligibility coverage;
- target feature standard deviation;
- target effective rank;
- target-change/feature-variance ratio;
- target norm distribution;
- train/eval-mode prediction stability;
- selected consequence labels under the registered candidate cost;
- invalid/terminal event coverage and consequence-label audits.

### Required Baselines And Controls

- persistence;
- state-only predictor;
- action-only predictor;
- zero action on eligible non-hold examples;
- same-source non-identical wrong action;
- corrected online-target spatial control;
- privileged consequence oracle;
- random and fixed action priors for candidate selection.

The state-only and action-only controls test whether apparent performance comes
from state persistence or action-frequency shortcuts.

Train both diagnostic controls on the same splits and three seeds. The
state-only control receives the current state with action inputs fixed to zero.
The action-only control receives the action and a learned constant state token.
Match predictor capacity as closely as practical and report parameter counts.
Neither control participates in checkpoint selection for C0-C2.

## Collapse And Stability Criteria

The current fixed `mean_feature_std < 0.05` collapse warning is retained for
comparability but is not sufficient.

Phase 2D must also report:

- covariance effective rank;
- mean and percentile pre-normalization token norms;
- target-change distribution;
- pairwise state discrimination within and across source states;
- feature scale by epoch.

Registered failure conditions:

- mean feature standard deviation below `0.05`;
- target-change/feature-variance ratio below `0.01`;
- effective rank below `10%` of spatial feature dimension;
- non-finite loss or feature norms;
- median pre-normalization target token norm changes by more than `2x` from
  epoch one to final;
- final prediction loss exceeds minimum epoch prediction loss by more than
  `50%`.

Any failure blocks promotion regardless of primary metric.

## Statistical Analysis Plan

### Experimental Unit

The primary experimental unit is the source state. Candidate actions from one
source state are paired observations, not independent samples. Source states
are clustered within scenes.

### Replication

- optimization seeds: exactly `3` for the bounded gate;
- identical data split and sampling protocol across cells;
- report each seed and the across-seed aggregate;
- no seed may be silently excluded.

Failed runs caused by implementation or infrastructure errors may be rerun only
with the reason recorded. Statistically unfavorable completed runs may not be
rerun or removed.

### Precision And Power Check

Before opening test-ID, estimate scene-level and source-state-level variance
using train and validation results only. Run a simulation or cluster-aware
power analysis for the registered H1 effect of `0.10` and H2 ratio of `0.90`.

Required condition:

- estimated power is at least `80%` for both co-primary thresholds; and
- the validation confidence interval procedure has at least `16` independent
  scene clusters.

If the condition fails, expand the still-unopened test manifests before any
test metric is computed. Do not weaken the effect thresholds after inspecting
validation or test results.

### Confidence Intervals

Use a paired hierarchical bootstrap:

1. sample scenes with replacement;
2. within each sampled scene, sample source states with replacement;
3. retain all paired candidate predictions required for the estimand;
4. compute the paired cell difference or ratio;
5. repeat at least `10,000` times.

Report percentile `95%` confidence intervals and the complete bootstrap
configuration. Do not bootstrap candidate rows independently.

Compute each seed's result separately. For the across-seed confirmatory
estimate, preserve matched optimization seeds across cells, bootstrap scenes
and source states within each seed, compute the estimand per seed, and average
the three seed-level estimands with equal weight. Do not pool all candidate
rows or source states across seeds as if they were independent.

### Multiple Comparisons

H1 and H2 are co-primary and both must pass. H3 is required as a stability
gate. All other analyses are secondary or exploratory and must be labeled as
such. No correction is required for the gate because promotion requires all
co-primary and stability conditions; exploratory p-values, if any, must not be
used for promotion.

### Checkpoint Selection

Select one checkpoint per seed using validation data only.

Registered lexicographic rule:

1. reject checkpoints that trigger a collapse or stability failure;
2. maximize validation hard-negative action advantage;
3. among ties within `0.01`, minimize validation one-step
   rollout/persistence ratio;
4. among remaining ties, choose the earlier epoch.

The rule must be implemented before training and applied identically to all
cells.

## Promotion And Stop Rules

### Promote Current Spatial Formulation

Promote to a larger, physics-grounded, recurrent experiment only if C2:

- passes H1 and H2 on validation and untouched test-ID;
- passes all collapse and stability criteria;
- improves over C1 on H1 with a paired confidence interval excluding zero;
- does not materially regress two-step prediction relative to C1;
- does not increase newly unsafe conditional selections relative to C1,
  acknowledging that this remains exploratory;
- reproduces the direction of H1 and H2 in at least `2` of `3` seeds.

### Stop And Redesign Target Geometry

Stop direct image-aligned patch dynamics as the primary state if corrected C2:

- fails H1 or H2 on validation across all three seeds; or
- passes validation but fails untouched test-ID; or
- passes action identifiability only by triggering collapse/stability failure.

The next registered alternatives should be tested in this order:

1. egomotion-aligned target features with visibility/occlusion masks;
2. motion-equivariant learned slots;
3. factorized affordance, dynamics, event, and uncertainty state;
4. recurrent belief for history-required subsets.

### Inconclusive Outcome

The result is inconclusive, not negative, if:

- eligible hard-negative coverage is below `70%` of non-hold source-step pairs;
- any action has less than `5%` eligible coverage without registered weighting;
- test split integrity fails;
- artifact lineage or dataset checksums cannot be verified;
- infrastructure failures prevent three completed seeds;
- fewer than `16` test scenes or fewer than `12` eligible source states per
  test scene remain after masks;
- the registered precision and power check fails and the unopened test set
  cannot be expanded.

An inconclusive result requires data or infrastructure correction. It must not
be reported as an architecture failure.

## Execution Sequence

### Stage 0: Reproducibility And Audit Infrastructure

1. Add a tracked experiment manifest schema.
2. Record code commit, exact command, environment fingerprint, seeds, data
   hashes, split IDs, model config, checkpoint rule, and artifact hashes.
3. Add a reusable Phase 2 audit script that emits data balance, negative
   contamination, split integrity, and artifact lineage.
4. Add automated train-set and validation-set persistence/action diagnostics.
5. Add a tracked claims registry mapping every reported claim to artifacts and
   analysis code.

Gate: one command regenerates the Phase 2B/2C audit tables from available
artifacts.

### Stage 1: Correct Data And Controls

1. Implement per-slot masks.
2. Implement source-grouped balanced sampling.
3. Implement valid same-source hard negatives.
4. Add state-only and action-only controls.
5. Generate and freeze validation, test-ID, and test-hard manifests.

Gate: automated audit reports zero identical hard negatives, zero split
overlap, and acceptable eligible action coverage.

### Stage 2: Correct Model And Diagnostics

1. Separate predictor and EMA target projection heads.
2. Remove BatchNorm from spatial projection paths.
3. Add per-token normalization.
4. Add effective-rank, norm, scale, and train/eval stability diagnostics.
5. Add the registered masked action-identifiability loss.

Gate: unit tests pass and a smoke run confirms every metric and mask is emitted.

### Stage 3: Run Bounded Factorial

1. Freeze run manifests for C0, C1, and C2. Completed in Stage 9.
2. Run three seeds per cell.
3. Select checkpoints using the registered validation-only rule.
4. Produce validation report without accessing test labels.
5. If validation gate passes, open test-ID once and evaluate the frozen
   selected checkpoints from all three preregistered cells.
6. Generate test-hard report only after test-ID reporting is complete.

### Stage 4: Decide

1. Promote only under the registered promotion rule.
2. Otherwise classify the result as negative or inconclusive using the stop
   rules.
3. Record protocol deviations, including deviations that appear harmless.
4. Register the next experiment before implementing it.

## Required Artifact Set

Every completed Phase 2D run must produce:

- immutable run manifest;
- environment fingerprint;
- train/validation/test split manifest and hashes;
- per-slot data and eligibility audit;
- per-epoch training and validation metrics;
- selected checkpoint hash;
- per-source-state prediction/control table;
- hierarchical bootstrap samples or deterministic bootstrap seed and code;
- final validation report;
- final test-ID report if opened;
- final test-hard report if opened;
- limitations and protocol deviations;
- machine-readable gate decision.

Publication tables must be generated from the per-source-state results, not
manually transcribed.

## Protocol Amendment Rule

Any change after registration must be appended to this document before the
affected result is inspected. The amendment must state:

- date and code commit;
- exact change;
- reason;
- whether any validation or test result had already been observed;
- which hypotheses or gates are affected.

Changes made after viewing test-ID or test-hard results define a new experiment.
They may not be reported as part of the original Phase 2D confirmatory result.

## Registered Decision

Proceed with the Stage 9 registered primary full-training matrix only. Do not
start Phase 2D training on the old complete-sequence-filtered data or old
batch-rolled action control. Do not access test-ID or test-hard results until
the selected-checkpoint and staged-access gates pass.

## Contingent Program-Level Roadmap

The following roadmap is not part of the Phase 2D confirmatory gate. Each phase
requires a separate preregistration after the previous gate is resolved.

### Capability Ladder

| Level | Required evidence | Claim permitted |
| --- | --- | --- |
| 0: contract integrity | immutable splits, valid controls, reproducible artifacts | dataset and analysis are auditable |
| 1: representation validity | non-collapse, stable scale, state discrimination | latent is non-trivial |
| 2: action-conditioned prediction | beats persistence and valid wrong actions | latent transition uses action information |
| 3: consequence ranking | ranks safety, progress, recoverability, and uncertainty | latent supports offline counterfactual decisions |
| 4: physical grounding | physics-calibrated predictions and event labels | latent models embodied consequences |
| 5: learned local MPC | learned rollouts are the primary closed-loop selector | world model powers local navigation |
| 6: recurrent belief | improves registered history-required tasks | world model handles partial observability |
| 7: long-range and transfer | topology, aliasing, sensor, simulator, and embodiment tests | broader navigation generalization |

No result may claim a higher level solely from evidence at a lower level.

### Phase 2E: Target Geometry Redesign

Trigger: corrected Phase 2D fails H1 or H2 without an inconclusive data result.

Compare egomotion-aligned features, motion-equivariant slots, and a factorized
affordance/dynamics/event state. Use the corrected Phase 2D data, controls,
splits, and statistics. Test one target family at a time against the corrected
C2 reference.

### Phase 3: Recurrent Belief

Trigger: a single-frame model passes Level 2 on Markov-visible transitions.

Create explicit history-required tests for occluded obstacles, remembered
openings, reveal actions, motion uncertainty, and missing observations. Compare
vision-only, proprioceptive, onboard-odometry, and privileged-pose oracle
beliefs. Promote only for gains on history-required subsets.

### Phase 4: Physics-Grounded Counterfactual Dynamics

Trigger: the learned state passes Levels 2 and 3 on kinematic targets.

Expand Genesis counterfactual replay while preserving source physical state,
nominal and executed commands, contact, fall, slip, and uncertainty. Calibrate
kinematic labels rather than treating them as truth.

### Phase 5: Learned Local MPC

Trigger: physics-grounded consequence ranking passes.

Use learned latent rollouts as the primary local action selector. Geometric and
reactive controllers remain baselines and emergency intervention oracles.
Report interventions separately; a run dominated by interventions does not
validate learned planning.

### Phase 6: Long-Range Image-Goal Navigation And Transfer

Trigger: learned local MPC passes on disjoint scenes.

Combine learned local planning with non-privileged topological memory and goal
retrieval. Evaluate alias stress, larger mazes, sensor changes, simulator
transfer, and embodiment changes. Keep local planning and long-range routing
claims separate.
