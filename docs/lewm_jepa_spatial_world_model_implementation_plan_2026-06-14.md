# JEPA Spatial World Model Implementation Plan

Date: 2026-06-14

Branch: `jepa-spatial-world-model-nav`

## Objective

Build and evaluate a JEPA-style world model whose learned latent state can
directly select safe, useful navigation actions under partial observability.

The first implementation milestone is not a new model. It is a model-agnostic
counterfactual decision benchmark that directly measures the capabilities the
new model is intended to learn. This prevents another cycle in which proxy
prediction or retrieval metrics improve without improving deployed decisions.

## Research Contract

The final learned system must:

- maintain a learned latent state from permitted runtime observations;
- predict action-conditioned latent futures;
- use those predictions as the primary action-selection mechanism;
- operate without privileged runtime pose, depth, occupancy, or route geometry;
- beat persistence, action-only, reactive, and current pooled-LeWM baselines;
- report safety, progress, clearance, recoverability, and uncertainty
  separately.

Privileged geometry and simulator state are allowed as teachers, labels,
diagnostic probes, and oracle upper bounds.

## Phase 1: Counterfactual Decision Benchmark

### Purpose

Create matched-state branches that answer:

> From the same observation history and start state, can a model distinguish
> which candidate action sequence remains safe, preserves recoverability, and
> makes useful progress?

The existing task-aligned v2 indexes provide scene-disjoint start observations,
history frames, poses, and local targets. They are retained as seed states.

### v0 benchmark contract

For each seed state, branch every configured primitive sequence for a short
multi-block horizon. Record:

- complete primitive and active-block sequence;
- endpoint pose and path length;
- whether the start is already inside the inflated grid;
- whether the candidate newly enters unsafe space;
- whether it ends in unsafe space;
- minimum and fifth-percentile swept configuration clearance;
- unsafe sample fraction and clearance gain;
- target progress and final heading error;
- whether the target remains reachable from the endpoint.

The privileged kinematic labels are explicitly marked
`physics_validated = false`. They establish the benchmark shape and cheap
screening contract. A bounded physics replay later calibrates their validity.

### Why labels remain separate

A single scalar cost hides whether a model succeeds by becoming stationary,
accepting collisions, or exploiting weighting choices. The benchmark therefore
stores each consequence separately.

A transparent lexicographic oracle is included only to define one reference
decision:

1. avoid newly entering unsafe space;
2. end outside unsafe space;
3. preserve target recoverability;
4. maximize target progress, or clearance gain for targetless recovery;
5. maximize low-percentile swept clearance;
6. minimize heading error;
7. minimize path length.

### v0 promotion gate

Before training a new encoder, the benchmark must demonstrate:

- scene-disjoint train and evaluation indexes;
- non-trivial candidate diversity in safety and progress;
- target alignment for all goal-conditioned rows;
- explicit coverage of starts already inside the inflated grid;
- useful separation between random, action-only, logged-action, and oracle
  controls;
- deterministic reproduction from an index and primitive registry.

The initial registered validity checks are:

- zero train/evaluation scene overlap;
- every logged velocity primitive is represented in the candidate bank;
- random expected new-unsafe rate is at least 5%;
- random expected safe-positive-progress rate is at least 5%;
- the oracle selects zero newly unsafe trajectories;
- oracle safe-positive-progress rate exceeds the action-only prior by at least
  10 percentage points.

### v1 physics calibration

Replay a bounded stratified sample of identical start states and candidate
sequences in Genesis. Add:

- physical contact;
- fall;
- realized displacement and yaw;
- realized minimum body clearance where available;
- primitive execution deviation;
- kinematic-label agreement.

Do not train a safety-critical model against the kinematic proxy until this
agreement is quantified.

## Phase 2: Spatial-Token Baseline

### Architecture

Retain spatial encoder tokens and train an action-conditioned predictor over
them. Preserve a separate pooled token only for the current recognition
baseline.

Start with single-observation state so the effect of spatial structure is
isolated from memory.

### Required comparisons

- action-only prior;
- current pooled raw and projected LeWM features;
- current patch-token diagnostic;
- new spatial-token predictor;
- privileged oracle.

### Gate

Spatial tokens must improve scene-disjoint safe-sequence ranking and progress
without increasing unsafe selections. Better latent loss or probe accuracy
alone does not pass.

## Phase 3: Recurrent Latent Belief

Add persistent belief tokens updated from observation tokens, actions, and
proprioception. Run controlled vision-only and onboard-odometry ablations.

Evaluate hidden-obstacle memory, remembered openings, uncertainty under missing
evidence, and the reduction in active re-observation behavior.

The recurrent model must improve decisions that genuinely require history; it
must not be promoted for gains confined to static single-frame cases.

## Phase 4: Counterfactual Free-Running Prediction

Train on branched action futures from matched states. Recursively consume
predicted state during training and evaluation.

Increase rollout horizon only when the model beats persistence at the current
horizon. Track consequence ranking as well as latent prediction error.

Required events include safe progress, collision/contact, dead ends,
recoverability, and information-gathering actions.

## Phase 5: Factorized Objectives And Anti-Collapse

Separate appearance/place, spatial-affordance, dynamics, and recurrent-belief
roles. Apply anti-collapse regularisation according to each branch's semantics
rather than uniformly across one pooled latent.

Run controlled capacity-matched factorials. Select variants on counterfactual
and closed-loop action gates, not representation dispersion alone.

## Phase 6: Direct Latent Model-Predictive Control

Use latent rollouts as the primary controller:

1. encode the current belief;
2. generate candidate action sequences;
3. predict their latent futures;
4. estimate progress, safety, recoverability, and uncertainty;
5. select and execute the first action;
6. observe and replan.

Use uncertainty penalties and targeted data collection to address planner
exploitation. Conventional reactive and geometric controllers remain baselines
and intervention oracles.

## Phase 7: Long-Range Image-Goal Navigation

Reintroduce topological memory only after local learned planning is reliable.
Use the appearance branch for retrieval and the latent belief/planner for route
execution. Build memory through autonomous or fixed non-privileged exploration,
not a privileged scene-graph tour.

## Initial Implementation Started

The first v0 benchmark components are implemented:

- continuous configuration-clearance queries in
  `lewm_worlds.planning_grid.InflatedOccupancyGrid`;
- reusable multi-block swept-trajectory labeling in
  `lewm.benchmarks.counterfactual`;
- `scripts/build_jepa_counterfactual_benchmark.py` to upgrade existing
  task-aligned decision indexes;
- focused tests for continuous clearance, start-in-inflation semantics, swept
  unsafe detection, and safety-first oracle ordering.

### Initial smoke result

Artifacts:

- `.generated/jepa_counterfactual/v0_train_smoke.jsonl`
- `.generated/jepa_counterfactual/v0_val_smoke.jsonl`
- `.generated/jepa_counterfactual/v0_smoke_baselines.json`

The smoke uses 16 rows from one train scene and 16 rows from one disjoint
validation scene. Each state branches all 81 two-block sequences over the nine
trainable velocity primitives.

The smoke contract gate passes:

- train/evaluation scene overlap: `0`;
- logged-action candidate fallbacks: `0`;
- random expected newly-unsafe rate: `26.93%`;
- random expected safe-positive-progress rate: `31.48%`;
- oracle newly-unsafe rate: `0%`;
- oracle safe-positive-progress rate: `62.5%`;
- action-only safe-positive-progress rate: `0%`.

The action-only prior selects `yaw_left, yaw_left`, reproducing the conservative
low-motion failure mode that prior task-aligned experiments exposed. The oracle
instead reaches `+0.190 m` mean target progress while preserving zero newly
unsafe selections and full target recoverability on the smoke subset.

This establishes that the v0 contract is discriminative on a bounded sample. It
does not yet establish full-distribution coverage or physical validity.

## Phase 1 Full-Distribution Result

Phase 1 is complete. The full generated artifacts are:

- `.generated/jepa_counterfactual/v0_train_full.jsonl`
- `.generated/jepa_counterfactual/v0_val_full.jsonl`
- `.generated/jepa_counterfactual/v0_full_coverage_audit.json`
- `.generated/jepa_counterfactual/v0_full_baselines.json`
- `.generated/jepa_counterfactual/v0_val_full_seq4_rollout.json`
- `.generated/jepa_counterfactual/physics_sample_full.jsonl`
- `.generated/jepa_counterfactual/physics_replay_full.jsonl`

The full benchmark contains 14,890 train rows from 32 scenes and 14,381
evaluation rows from 32 disjoint scenes. Every row branches all 81 two-block
primitive sequences. The coverage audit passes every registered contract
check:

- train/evaluation scene overlap: `0`;
- goal-conditioned rows with a matched local-target frame: `100%`;
- held-out candidate newly-unsafe rate: `29.08%`;
- held-out candidate safe-positive-progress rate: `35.65%`;
- held-out oracle newly-unsafe rate: `0%`;
- held-out oracle safe-positive-progress rate: `85.21%`.

The target-conditioned held-out control results are:

| Selector | Newly unsafe | Ends unsafe | Mean progress | Safe positive progress |
| --- | ---: | ---: | ---: | ---: |
| action-only prior | 0.00% | 15.65% | 0.000 m | 0.00% |
| random expected | 25.34% | 34.21% | 0.053 m | 35.65% |
| logged then hold | 20.20% | 34.31% | 0.059 m | 36.74% |
| current pooled LeWM rollout | 14.26% | 24.10% | 0.027 m | 23.13% |
| transparent oracle | 0.00% | 2.69% | 0.138 m | 85.21% |

The current pooled LeWM rollout planner is therefore not a competitive
decision representation on this task. It reduces unsafe selections relative
to random, but does so while making roughly half the random selector's mean
progress and producing substantially less safe positive progress than either
random or logged-then-hold. Its exact oracle-sequence match rate is `2.50%`.
This is consistent with the architecture-reset diagnosis: the pooled latent
supports appearance recognition better than spatially grounded action ranking.

## Physics-Calibration Result

The bounded Genesis calibration replays 32 cases, balanced across all eight
scene families and four kinematic safety/progress buckets.

- fall rate: `0%`;
- mean two-block endpoint error versus the kinematic proxy: `0.094 m`;
- mean yaw error: `0.093 rad`;
- newly-unsafe label agreement: `62.5%`;
- ends-unsafe label agreement: `75.0%`.

Agreement is asymmetric. Safe cases mostly remain safe, but the kinematic
proxy substantially over-predicts unsafe outcomes: only one of ten candidates
labelled as newly unsafe entered unsafe space physically. It also misses three
physical newly-unsafe cases. The proxy remains useful for cheap exhaustive
candidate generation and benchmark shaping, but it is not acceptable as the
sole safety target or as evidence of physical safety.

Physical contact-force reads remain unavailable on the current AMD Genesis
path, and physical clearance is sampled once per command tick. Those
limitations must remain explicit.

## Phase 1 Decision

Proceed to the spatial-token JEPA experiment, with two constraints:

1. Use the full counterfactual benchmark to test whether retaining spatial
   structure improves action ranking over the pooled LeWM.
2. Treat kinematic safety as a noisy teacher. Promote safety claims only on
   physics-calibrated labels, and expand physical replay if a model appears to
   exploit proxy errors.

Do not add further heads or navigation heuristics to the current pooled latent.
The next implementation target is a capacity-matched single-observation
spatial-token predictor and consequence-ranking evaluation. Persistent belief,
longer free-running rollouts, and factorized anti-collapse objectives remain
later controlled experiments, not simultaneous changes.

## Phase 2 Started: Counterfactual Future Observations

A spatial-token JEPA predictor requires matched future observations for each
branched action. Privileged endpoint labels alone are not sufficient: training
a scorer directly from current tokens and geometry labels would repeat the
frozen-feature head strategy that the reset rejected.

`scripts/build_jepa_counterfactual_render_plans.py` now converts benchmark rows
into deterministic replay-compatible plans with one egocentric future frame at
every action-block endpoint. It supports bounded candidate subsets for smoke
tests and all-candidate output for training data.

The first end-to-end smoke used one held-out state, nine deterministic
candidate sequences including the oracle, and both block endpoints:

- planned future frames: `18`;
- valid rendered future frames: `18/18`;
- unique rendered images: `17/18` (`hold, hold` correctly repeats a pose);
- maximum reconstructed final-endpoint error: `< 5e-9 m`.

These are kinematic endpoint observations and inherit the physics-calibration
warning above. The next implementation unit is the capacity-matched
spatial-token predictor trained on these matched current/action/future tuples,
with kinematic versus physically replayed subsets reported separately.
