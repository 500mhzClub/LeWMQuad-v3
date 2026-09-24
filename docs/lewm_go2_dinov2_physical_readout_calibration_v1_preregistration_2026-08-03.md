# Go2 DINOv2 physical-readout calibration V1

**Frozen:** 2026-08-03, before any evaluation-role RGB leaf is opened for
this calibration.

## Purpose and claim boundary

This is a development-only evaluator calibration.  It asks whether frozen
DINOv2 true-future spatial tokens contain enough scene-disjoint information
for one fixed, train-only physical-ranking readout to beat the strongest
simple controls.  It does not train a dynamics model and cannot establish
world-model capacity, navigation usefulness, rollout, memory, G2--G8
qualification, promotion, or deployment.

The previous bounded-branch evaluator called a model target-encoder readout a
`true_future_ceiling`.  That readout achieved mean normalized physical-rank
regret `0.36349`, versus `0.40186` for the current-state control and `0.17441`
for the task/action-only control.  Its paired interval against current crossed
zero.  It was therefore not a privileged oracle and did not establish usable
evaluator headroom.  This calibration does not alter that completed result.

The decision is deliberately prior to the proposed backbone-trainable DINOv2
JEPA.  If true frozen-DINO future features cannot beat the task/action-only and
current-state controls through the exact relational readout below, predictions
in that feature space cannot be qualified by this planning interface and the
new model run stops before training.  Failure does not reject every possible
nonlinear or differently conditioned DINO planning interface.

## Fixed development inputs

Use only the reviewed split-root bounded-branch development bundle at
`.generated/dev/lewm-go2-wm-bounded-branch-posthoc-join-admission-v1`:

- manifest: 11,964 bytes, SHA-256
  `87448995c905107453814a5e7e4cd9968d31cbc0e308513d17bc038c6585f15e`;
- terminal: 1,250 bytes, SHA-256
  `a1590fffc673f7676016bb70d4b4f5530f24b9a49bf05e84dcec6bc1756fbe56`;
- independent terminal review: 2,844 bytes, SHA-256
  `bfd0250357d0f681c674db6c54ea4a8c4d5e617230332383beda3db3e0f38669`.

Each role contains 128 states from 16 scenes, balanced across eight families,
with three context observations and all nine executed one-step successors per
state.  Train and evaluation scenes and RGB identities are disjoint.  The
`train` role fits every readout.  The `eval` role supplies every reported
comparison.  No row from `eval` may affect a fitted parameter, feature
standardizer, descriptor choice, regularization coefficient, or threshold.

Reuse the exact existing frozen-DINO train cache:

- cache: 302,107,682 bytes, SHA-256
  `164f1fef8c859976c93f7fc978e938c6c8f7f9963cf92bb154f51b23d158b34b`;
- receipt: 1,770 bytes, SHA-256
  `e94ec5d188811c44d4cc870e76d1888aa6f30ee6d423557ee9f3e2918a700994`.

The cache contains the exact ordered 1,536 train artifacts as float16
`[1536,256,384]` per-token-L2-normalized features.  The calibration may open
each of the 1,536 evaluation-role RGB artifacts exactly once through the
reviewed bound reader and create one equivalently ordered evaluation cache.
No other RGB role is eligible.

The role order is executable and identical for train and evaluation: sort
groups by `(group_index,state_id)`; within each group append the three context
artifacts in chronological order, followed by the nine successor artifacts in
requested action-ID order 0 through 8.  The cache binds this complete artifact
list and its canonical compact-JSON SHA-256.

## Frozen DINOv2 encoder

- source: official local DINOv2 repository commit
  `7764ea0f912e53c92e82eb78a2a1631e92725fc8`;
- model: `dinov2_vits14` without register tokens;
- checkpoint: 88,283,115 bytes, SHA-256
  `b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9`;
- input: exact existing 224-pixel ImageNet-normalized preprocessing;
- output: final 16 by 16 grid of 384-channel patch tokens, with per-token L2
  normalization and float16 cache storage.

The encoder is frozen, in evaluation mode, and run without gradients.  The
cache receipt binds repository, checkpoint, preprocessing, artifact order,
shape, dtype, file identity, decoded-pixel verification counts, and access
counts.  The previously reviewed train-cache receipt is trusted at its exact
binding; this attempt does not reopen any train RGB leaf.

## Fixed descriptor and readout

Promote cached tokens to float64, reshape every tensor to `[16,16,384]`, and
split it into four nonoverlapping
8 by 8 quadrants in row-major order: top-left, top-right, bottom-left,
bottom-right.  For each quadrant concatenate its channel-wise mean and
population standard deviation (`ddof=0`).  Concatenate the four results to one
fixed 3,072-dimensional descriptor.

For each group let `c` be the last-context descriptor and `s_a` the successor
descriptor for candidate action `a`.  The primary relational latent feature is
exactly `[c,s_a,s_a-c]`.  Construct its task-conditioned feature using the
repository's existing `task_conditioned_feature_v1` contract:

`[relational, target_x, target_y, 1, relational*target_x,
relational*target_y]`.

Fit three separate sets of nine deterministic ridge heads, one head per
requested action, on the 128 train states.  The target in every set is that
branch's dense physical rank divided by the largest dense rank in its state.
The sets receive respectively (a) the relational current-plus-true-successor
feature plus task, (b) the last-context descriptor plus task, and (c) only
`[target_x,target_y,1]`.  Thus each control is fitted in its own feature space;
the current-state arm is not weakened by applying a future-feature head out of
distribution.  Use the existing standardized dual ridge implementation with
fixed lambda `1e-3`.  Actions never share an intercept.  Tied physical ranks
remain equivalent and receive the same target.  No hyperparameter search,
descriptor alternative, seed search, nonlinear readout, or result-dependent
refit is allowed.

## Fixed evaluation arms

Evaluate the following scores on all 128 evaluation states:

1. `privileged_physical_oracle`: the bound dense physical ranks themselves;
2. `dinov2_true_future`: each action's current-plus-true-successor relational
   feature through its train-only relational action head;
3. `dinov2_current_state`: the last context DINO descriptor reused across all
   candidate actions and scored by the separately fitted current-state action
   heads;
4. `task_action_only`: target coordinates through the separate action-only
   heads;
5. `relational_persistence`: the primary relational heads with successor set
   equal to current, i.e. `[c,c,0]` for every candidate action;
6. `random_expected`: the exact mean normalized rank within each state; and
7. `hold_constant`: action ID 6 for every state.

For deterministic arms, choose the minimum predicted score with action ID as
the fixed tie-break.  Report normalized dense-rank regret, oracle-equivalent
selection rate, target progress, chosen-action histogram, per-family results,
and per-scene rows.  Falls and tips are reported as corpus-support facts only:
the bundle contains none, so safety is `NOT_TESTABLE_ZERO_EVENT_SUPPORT`, never
a passed safety claim.  `random_expected` has no selected action, target
progress, path, or chosen-action histogram; those fields are `NOT_APPLICABLE`.

The strict bound loader must recompute physical dense ranks and equivalence
from the bound branch labels under the frozen one-centimetre calibration
tolerance.  A declared rank field is never trusted without that recomputation.

## Uncertainty and exact gates

Use paired scene-cluster bootstrap intervals with equal weights for the eight
fixed families, 10,000 resamples, seed `2026080302`, and percentile 95%
intervals.  The scene, not a branch row, is the resampling unit.

The calibration passes only if all of the following hold:

1. all source, input, role, hash, decoded-pixel, finiteness, determinism, and
   no-protected-access checks pass;
2. the privileged physical oracle has exactly zero normalized rank regret and
   selects an oracle-equivalent action in every evaluation state;
3. the upper 95% endpoint of paired
   `dinov2_true_future - task_action_only` normalized rank regret is strictly
   below zero;
4. the upper 95% endpoint of paired
   `dinov2_true_future - dinov2_current_state` normalized rank regret is
   strictly below zero;
5. the upper 95% endpoint of paired
   `dinov2_true_future - relational_persistence` normalized rank regret is
   strictly below zero;
6. the DINO true-future point estimate beats random expected regret, with all
   per-family comparisons reported as diagnostics; and
7. an independent deterministic replay reproduces the cache identity,
   readout identities, selected actions, summaries, and intervals exactly.

These gates use measured controls and zero-effect boundaries rather than a
round absolute performance cutoff.  A pass freezes the observed
task/action-only-to-true-future headroom for a later, separately preregistered
model gate.  It does not authorize model training automatically.  A failure
means the proposed DINO prediction experiment cannot be qualified by this
readout and must stop or adopt a materially different, prospectively reviewed
planning interface.

Replay reuses and rehashes the immutable evaluation cache and recomputes the
readouts, selections, summaries, and intervals.  It must not reopen an
evaluation RGB leaf.

The terminal statuses are exactly:

- `PASS_DINO_TRUE_FUTURE_PHYSICAL_READOUT_HEADROOM_ESTABLISHED`; or
- `STOP_DINO_TRUE_FUTURE_PHYSICAL_READOUT_HEADROOM_NOT_ESTABLISHED`; or
- `FAIL_INFRASTRUCTURE_NO_SCIENTIFIC_DECISION`.

## Output and custody

One separately reviewed authority may create only
`.generated/dev/go2_dinov2_physical_readout_calibration_v1/attempt_v1` with an
exclusive reservation, evaluation feature cache and receipt, `result.json`,
and `terminal.json`.  Existing artifacts are immutable.  No retry, resume,
overwrite, model training, feature fine-tuning, collection, generation,
closed-loop run, held-out access, sealed access, promotion, or deployment is
authorized by this preregistration.

After this attempt, both bounded-branch roles remain development/calibration
data.  In particular, this evaluation role cannot later be described as a
fresh confirmatory set merely because candidate training did not consume its
RGB leaves.
