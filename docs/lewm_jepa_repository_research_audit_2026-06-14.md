# JEPA Navigation Repository Research Audit

Date: 2026-06-14

Audited branch: `jepa-spatial-world-model-nav`

Audited commit: `5c06bba76955c37b7223e9255756dffa59338680`

Audit status: retrospective repository and artifact audit

Companion preregistration:
`docs/lewm_jepa_phase2d_preregistered_research_plan_2026-06-14.md`

Foundation implementation:
`docs/lewm_jepa_phase2d_stage0_stage1_implementation_2026-06-14.md`

## Research Objective

The project's scientific objective is:

> Determine which learned latent target geometry, action-conditioning
> objective, and belief-state structure allow a JEPA-style world model to
> predict and compare navigation consequences well enough to select safe,
> useful actions from permitted runtime observations.

The objective is not merely to build a navigation system that succeeds using
any available engineering method. Geometric planners, privileged simulator
state, depth, occupancy, and hand-written recovery logic are valid teachers,
oracles, probes, and baselines. They do not constitute evidence that the
learned world model performs navigation.

## Executive Conclusion

The repository has a strong scientific foundation:

- it distinguishes engineered task success from learned-model competence;
- it uses matched-state counterfactual branches and scene-disjoint evaluation;
- it includes persistence, action, collapse, and privileged-oracle controls;
- it records negative results and stops failed approaches before scaling;
- it retains privileged geometry as labels and diagnostics rather than silently
  making it a runtime input.

The strongest current result is a bounded negative result:

> Under the Phase 2B/2C data, architecture, optimization budget, and evaluation
> procedure, every non-collapsed learned predictor loses to latent persistence
> at the first action block. The unregularized spatial representation collapses.
> Adding the tested EMA teacher does not pass the registered gate.

This result is robust enough to stop scaling the current formulation. It does
not establish that JEPA, spatial tokens, image-aligned tokens, or recurrence are
generally unsuitable for navigation.

The current action-identifiability and conditional action-selection
measurements contain important confounds. These do not invalidate the
persistence failure, but they prevent strong causal claims about why the models
failed. Phase 2D must remove these confounds before testing the next objective.

## Evidence Classification

This audit uses the following evidence classes.

| Class | Definition | Permitted claim |
| --- | --- | --- |
| A: registered direct evidence | Metric and gate defined before the run; result available in a machine-readable artifact | May support the registered bounded conclusion |
| B: post-hoc diagnostic evidence | Computed after seeing the result to diagnose threats or mechanisms | May identify confounds or motivate a new hypothesis; may not retroactively pass a gate |
| C: interpretation | Mechanistic explanation consistent with results but not isolated experimentally | Must be described as a hypothesis |
| D: untested proposal | Future design or expected mechanism | No empirical claim |

## Scope And Provenance

The audit covered the tracked repository, relevant generated Phase 2B/2C
artifacts, and the current documentation lineage.

Repository state at audit:

- tracked files: `354`;
- tracked Python source: approximately `51,381` lines;
- tracked documentation: `60` Markdown files;
- tracked `lewm/tests` Python files: `27`;
- worktree: clean before this documentation change;
- generated experiment artifacts and most model checkpoints: ignored by Git.

Primary generated artifact hashes:

| Artifact | SHA-256 |
| --- | --- |
| Phase 2B analysis | `39d5b7f497e9517e9410c6b7554d5900884302fd648c96cfc12100fde053141b` |
| Phase 2C analysis | `0450317eb175c6de033260f28c589707a36833a84b37460a9e7d85e896a0fb02` |
| Phase 2B train spatial dataset | `8a51b3fd1a940b037ec7645d8fc6a56cb5d9c934a0c65dbdf30d0798cc167294` |
| Phase 2B evaluation spatial dataset | `23b0e408246bf4720effef4c0cdcd7bc87e39abaa4fc22b140cf95d662aa8ade` |

The generated artifacts are local evidence, not durable publication artifacts.
Their summaries, checksums, configurations, and data lineage must be promoted
into a tracked experiment registry before publication.

Key implementation references:

- `scripts/train_jepa_spatial_lewm.py::_evaluate`: persistence, zero-action,
  batch-rolled action, and selection evaluation;
- `scripts/train_jepa_spatial_predictor.py::_load_rows`: complete-valid
  filtering;
- `lewm/models/spatial_lewm.py::SpatialLeWorldModel`: online/EMA targets,
  shared spatial projection, and spatial variance loss;
- `lewm/benchmarks/rollout_diagnostics.py::summarize_rollout_controls`:
  prediction, action, and collapse metrics;
- `scripts/build_jepa_spatial_future_dataset.py`: per-slot validity metadata;
- `scripts/build_jepa_counterfactual_render_plans.py`: kinematic future
  endpoint generation;
- `lewm/planning/primitive_bank.py`: nominal planning action bank;
- `scripts/train_lewm.py`: executed-command training action ingestion;
- `scripts/replay_jepa_physics_calibration.py`: bounded physical calibration.

## Claims Matrix

### Established Within The Registered Bounded Experiment

1. **The tested pooled, regularized spatial, and EMA spatial predictors fail
   the one-step persistence gate.**

   Phase 2B final one-step rollout/persistence ratios are `2.07x` pooled and
   `2.69x` regularized spatial. Phase 2C EMA finishes at `8.37x`. Lower is
   better and passing requires below `1.0x`.

2. **The tested unregularized spatial representation collapses.**

   Its final mean feature standard deviation is `0.027`, its target-change MSE
   is near zero, and its collapse warning triggers. The resulting low raw
   prediction error and apparently strong selection metrics are not valid
   evidence of useful dynamics.

3. **The tested variance floor is necessary but insufficient.**

   Removing it collapses the representation. Retaining it preserves variation
   but does not pass persistence or action gates.

4. **The tested EMA teacher is insufficient.**

   It prevents the observed low-variance collapse but does not pass the
   prediction or action gates and exhibits increasing feature scale and
   prediction loss.

5. **The current pooled LeWM is not justified as the primary planning state.**

   It fails direct counterfactual decision and prediction gates. It remains a
   valid appearance/retrieval baseline and may remain useful as an auxiliary
   branch.

### Supported But Limited

1. **The tested models exhibit weak measured action sensitivity.**

   Real actions produce only small improvements over zero actions and do not
   reliably beat shuffled actions. However, the current shuffled and zero
   controls contain false negatives, so the exact magnitude is not a clean
   estimate of action use.

2. **The current spatial target geometry is likely mismatched to navigation.**

   Average image-aligned patch MSE gives substantial weight to appearance and
   egomotion-induced patch displacement. This is a plausible explanation, not
   an isolated causal result.

3. **The Phase 2C instability is consistent with target-scale drift.**

   Feature standard deviation and prediction loss both increase. The shared
   BatchNorm projection path is an additional architectural confound, so the
   audit cannot attribute the failure solely to unnormalized EMA targets.

4. **Conditional action-selection metrics are exploratory.**

   They are computed only on candidate groups with valid rendered future
   observations, use a noisy kinematic consequence proxy, and rank direct
   goal-latent MSE. They are not safety estimates.

### Not Established

The repository does not currently establish that:

- JEPA-style world models are unsuitable for navigation;
- spatial tokens are unsuitable for navigation;
- image-aligned patch targets can never work;
- recurrence or belief state cannot improve one-step prediction;
- the current latent action-selection metric measures physical safety;
- the learned model can perform reliable closed-loop navigation;
- EMA is generally unsuitable;
- a larger model would fail under a corrected objective and data contract.

## Detailed Methodological Findings

### 1. Persistence Failure Is The Cleanest Current Result

Persistence compares the learned future with the representation of the current
state. It does not depend on shuffled-action construction or privileged safety
labels. Every valid non-collapsed Phase 2B/2C cell loses at the first action
block.

A post-hoc training-set diagnostic also found that final Phase 2B checkpoints
lose to persistence on their own complete-valid training rows:

| Cell | Train step-one rollout/persistence | Train step-two rollout/persistence |
| --- | ---: | ---: |
| pooled | `2.28x` | `4.51x` |
| regularized spatial | `3.14x` | `3.72x` |

Evidence class: B. This diagnostic was not preregistered and is not yet emitted
as a registered gate. It is now reproducibly emitted by
`scripts/evaluate_jepa_phase2_checkpoint.py` and recorded in
`docs/lewm_jepa_phase2d_stage0_stage1_implementation_2026-06-14.md`. It
strengthens the optimization-failure hypothesis but cannot retroactively pass a
gate.

### 2. Shuffled-Action Negatives Are Contaminated

The evaluator constructs shuffled actions by rolling each evaluation batch by
one row. It does not require the negative action to differ from the real action.
Because dataset rows are ordered by source state and candidate, the shuffled
action usually comes from the same source state.

Reproducing the evaluator's batch-size-eight rolling procedure gives:

| Split | Same source | Same full sequence | Same step-one action | Same step-two action |
| --- | ---: | ---: | ---: | ---: |
| train | `79.70%` | `1.27%` | `42.13%` | `6.35%` |
| evaluation | `79.15%` | `0.00%` | `40.52%` | `4.98%` |

At step one, approximately `40%` of shuffled negatives are therefore not
negative actions at all. This biases shuffled-action advantage toward zero.

Evidence class: B. The audit reproduces the implemented evaluator behavior from
the Phase 2B datasets. The result does not rescue the models because they still
lose to persistence. It does mean that claims about the exact degree of action
ignorance must be qualified.

### 3. The Zero-Action Control Includes True Holds

Among complete-valid rows, the real first action is `hold` for `32.23%` of
training rows and `32.94%` of evaluation rows. For these rows, replacing the
real action with zero is not an intervention.

Action-identifiability metrics must exclude true-hold positives when comparing
against zero action. Hold remains useful as a legitimate candidate and a
persistence-related baseline.

Evidence class: B.

### 4. Complete-Valid Filtering Changes The Action And Outcome Distribution

The dataset records per-slot observation validity, but current Phase 2A/2B/2C
trainers retain only sequences whose complete two-step future is valid.

Of `576` planned candidate sequences per split:

- training retains `394`;
- evaluation retains `422`.

The retained first-action counts are:

| Action | Train | Evaluation |
| --- | ---: | ---: |
| hold | 127 | 139 |
| backward | 85 | 91 |
| arc right | 97 | 84 |
| forward fast | 33 | 34 |
| arc left | 20 | 34 |
| yaw right | 13 | 21 |
| yaw left | 13 | 14 |
| forward slow | 5 | 4 |
| forward medium | 1 | 1 |

The retained family counts also vary materially despite the planned
one-scene-per-family design. Renderer validity is already known to be
outcome-dependent. Training only complete sequences therefore changes the
effective action, family, and likely consequence distribution.

This filtering is defensible for a first token-prediction diagnostic, but it is
not an adequate final navigation-learning contract. Phase 2D must use per-slot
masks and separately model terminal, invalid, contact, or observation-absence
events.

Evidence class: A for retained counts and the outcome-dependent invalidity
finding; B for the resulting selection-bias interpretation.

### 5. The Evaluation Split Has Been Used As Validation

The Phase 2B/2C trainer evaluates the same disjoint split after every epoch.
That split has also informed successive design decisions across phases. It is
therefore a validation split, not an untouched test split.

Scene disjointness prevents direct scene leakage but does not prevent adaptive
overfitting through repeated model and objective selection.

Future experiments require separate validation, test-ID, and test-hard splits.
Test splits must remain unopened until the design and checkpoint-selection rule
are frozen.

Evidence class: B, based on the implemented training loop and experiment
history.

### 6. The Shared Spatial Projector Is An Architectural Confound

The spatial model applies one `TokenProjector` to both:

- raw encoder targets; and
- raw predictor outputs.

The projection path contains BatchNorm. The pooled LeWM instead has separate
encoder and predictor projectors. This difference means the Phase 2B spatial
cell is not only a test of spatial state: its prediction and target
distributions also share normalization parameters and running statistics.

For EMA, target projector buffers are copied from the online projector every
update. This further complicates interpretation of target-scale drift.

The current result remains a valid failure of the implemented cell. It is not a
clean test of spatial tokens versus pooled state. Phase 2D must remove this
confound before attributing failure to target geometry.

Evidence class: B.

### 7. A Single RGB Observation Is Not Generally A Markov Navigation State

The bounded spatial experiment intentionally uses one RGB observation to
isolate spatial representation. This is scientifically useful, but it imposes a
construct limit.

Collision risk and future views can depend on hidden variables including:

- recently observed but now occluded geometry;
- robot velocity and low-level controller state;
- gait phase, contact, slip, and recovery state;
- action latency and execution error;
- surfaces outside the current camera field of view.

Failure on single-frame prediction does not show that recurrent belief is
premature in all settings. Future tests should distinguish approximately
Markov-visible transitions from history-required and reveal/occlusion
transitions.

Evidence class: C, grounded in the task's observability contract.

### 8. Training And Planning Use Different Action Variables

The base LeWM training contract uses executed command blocks. The planner
enumerates nominal primitive command blocks. Executed commands are useful
prediction inputs when explaining recorded transitions, but nominal commands
are the variable available to a planner before execution.

For embodied planning, the learned model should estimate:

`p(future state, executed action | current belief, nominal action)`.

A deterministic model trained only on executed commands is queried with a
different variable at deployment and cannot represent execution uncertainty.

Evidence class: B for the contract mismatch; D for the proposed probabilistic
factorization.

### 9. Kinematic Future Observations Are A Debugging Target, Not Physical Dynamics

Phase 2 counterfactual future images are rendered at kinematically integrated
endpoint poses. They are explicitly marked `physics_validated = false`.

The bounded 32-case Genesis calibration reports:

- mean endpoint error: `0.094 m`;
- mean yaw error: `0.093 rad`;
- newly-unsafe label agreement: `62.5%`;
- ends-unsafe label agreement: `75.0%`.

The calibration also resets the robot to a standard stance and clears prior
executed-action state, so it does not reproduce every hidden physical state at
the source observation.

The kinematic corpus remains valuable for testing visual target geometry and
action identifiability. It is insufficient for physical-safety or embodied
dynamics claims.

Evidence class: A for registered calibration measurements; B for the hidden
state limitation.

### 10. Direct Goal-Latent MSE Is Not A Validated Planning Cost

Conditional candidate selection currently chooses the final predicted spatial
tokens closest to a goal image under direct position-aligned token MSE.

This cost may prefer appearance similarity without representing:

- collision or contact;
- traversability;
- target progress;
- recoverability;
- uncertainty;
- information gain;
- whether image patches correspond after egomotion.

Selection results are useful diagnostics but cannot override persistence and
action-identifiability failure. A planning-ready model requires separately
reported consequence predictions and a registered decision rule.

Evidence class: B.

### 11. Statistical Evidence Is Currently Insufficient For General Claims

Phase 2B/2C use one seed, eight training scenes, eight evaluation scenes, and
one reduced-capacity architecture. Candidate sequences from the same source
state are correlated. Treating candidates as independent samples would
overstate confidence.

The correct primary statistical unit is the source state, clustered within
scene. Generalization claims require multiple optimization seeds and confidence
intervals resampled by scene and source state.

The bounded experiments remain valid diagnostics and stop gates. They are not
adequately powered architecture comparisons.

Evidence class: A for experiment scope; B for statistical interpretation.

### 12. Reproducibility And Claim Traceability Need Consolidation

Strengths:

- seeds, data paths, and limitations are commonly written into result JSON;
- core experiment code and tests are tracked;
- many decisions and negative findings are documented;
- platform policy artifacts include hashes.

Risks:

- `.generated/` and most `models/` artifacts are ignored;
- no tracked central experiment registry currently links claims to artifact
  hashes, data lineage, commands, and code commit;
- no tracked top-level environment lock or CI workflow was found in this audit;
- many date-stamped documents can diverge without an explicit supersession
  graph;
- several important audit diagnostics are not emitted by reusable scripts.

Before publication, every reported table should be regenerable from immutable
or content-addressed inputs with one documented command.

Evidence class: B.

## Threats To Validity

### Internal Validity

- contaminated shuffled and zero-action controls;
- shared BatchNorm projector between target and prediction paths;
- repeated use of the same evaluation split;
- outcome-dependent complete-sequence filtering;
- single-seed optimization noise;
- source-order-dependent negative construction.

### Construct Validity

- image-aligned patch MSE may not measure navigation-sufficient prediction;
- direct goal-latent MSE may not measure safe useful action selection;
- single-frame state may omit required hidden variables;
- kinematic endpoint renders do not measure embodied dynamics;
- mean feature standard deviation is only a coarse collapse diagnostic.

### External Validity

- eight train and eight evaluation scenes are a bounded learnability screen;
- reduced CPU architecture may not represent full-capacity behavior;
- fixed camera, simulator, robot, and primitive bank limit transfer claims;
- conditional valid-render subsets omit difficult physical outcomes.

### Statistical Conclusion Validity

- no multi-seed estimate for Phase 2B/2C;
- no confidence intervals;
- hierarchical dependence among candidates, source states, and scenes;
- successive hypotheses were informed by the same evaluation split.

## Revised Publication-Safe Claims

Use:

> In a bounded, scene-disjoint, single-seed diagnostic, the tested pooled,
> online spatial, unregularized spatial, and EMA spatial JEPA formulations all
> failed to beat latent persistence at the first action block. The
> unregularized spatial cell collapsed. These findings justify redesigning the
> target and action-identifiability objective before scaling.

Do not use:

> Spatial JEPA cannot learn navigation dynamics.

Use:

> The tested models showed little measured advantage from real actions over
> zero or batch-rolled actions. Because many negative controls were identical
> to the positive action, the exact magnitude of action insensitivity remains
> unresolved.

Do not use:

> The model ignores actions.

Use:

> The tested EMA configuration exhibited increasing feature scale and
> prediction loss and failed the registered gate.

Do not use:

> EMA causes feature-scale drift or is unsuitable for this task.

## Research Decision

1. Retain the current Phase 2B/2C outcome as a bounded negative result.
2. Do not scale the current formulation or compensate with planner heuristics.
3. Correct the data, control, split, projector, and statistical confounds before
   testing the next objective.
4. Treat Phase 2D as an objective-and-controls sanity experiment, not as a
   likely final architecture.
5. If corrected Phase 2D still fails, move from direct image-aligned patch
   dynamics toward egomotion-aligned features, motion-equivariant slots, or a
   factorized affordance/dynamics belief state.

The exact next experiment and stop conditions are preregistered in
`docs/lewm_jepa_phase2d_preregistered_research_plan_2026-06-14.md`.
