# Go2 DINOv2 dense shared spatial-readout calibration V1

**Frozen:** 2026-08-03, before implementation or execution of this mechanism
and before its readout loads either bound token cache.

## Purpose and claim boundary

This is one final development-only oracle-future interface calibration.  It
asks whether a low-capacity readout that preserves the full DINOv2 patch grid,
shares parameters across actions, and conditions on the goal and requested
command can extract scene-disjoint physical action-ranking information from
the *actual* successor tokens beyond the same strong controls used previously.

It is not a dynamics-model experiment.  It cannot establish prediction,
rollout, memory, closed-loop planning, navigation usefulness, safety, G2--G8
qualification, promotion, or deployment.  The evaluation role has already
been used by the completed quadrant-readout calibration and its aggregate and
family results have been inspected.  This is therefore prospective only with
respect to the materially new mechanism; it is not fresh confirmatory data.

The completed quadrant/separate-ridge result and corrected terminal review
remain immutable.  Their STOP is not retried or relabelled.  This experiment
tests the one explicit scope left open by that result: a genuinely dense,
shared, low-capacity, differently conditioned planning interface.

There is exactly one mechanism attempt.  No architecture, seed, epoch,
regularization, family routing, threshold, or control may change after its
evaluation result is known.  Failure closes this frozen-DINO oracle-future
planning-interface route rather than authorizing another readout variation.

## Fixed development inputs

Use only the reviewed train/evaluation roles of the same split-root bounded
matched-branch development bundle.  Each role contains exactly 128 states,
16 scenes, two scenes in each of eight families, and all nine executed
one-step successors per state.  Train/evaluation scenes, states, and artifacts
must remain disjoint.

No RGB or encoder execution is permitted.  The direct runtime input closure is
exactly these 16 files:

- train token cache:
  `.generated/dev/go2_matched_branch_successor_screen_v1/attempt_v1/features/dinov2.pt`,
  302,107,682 bytes, SHA-256
  `164f1fef8c859976c93f7fc978e938c6c8f7f9963cf92bb154f51b23d158b34b`;
- train receipt at the sibling `dinov2.json`, 1,770 bytes, SHA-256
  `e94ec5d188811c44d4cc870e76d1888aa6f30ee6d423557ee9f3e2918a700994`;
- evaluation token cache:
  `.generated/dev/go2_dinov2_physical_readout_calibration_v1/attempt_v2_integrity_replacement_v1/dinov2_eval.pt`,
  302,106,281 bytes, SHA-256
  `00a2e197d98effcd192392f50170648622a7210f954075002dc8b43110c636f8`;
- evaluation receipt at the sibling `dinov2_eval.json`, 1,770 bytes,
  SHA-256
  `d3e928cc563beb4dd850f34ca41915b8e5974c6d0b1b182602f3e3f20828421c`;
- prior calibration result at
  `.generated/dev/go2_dinov2_physical_readout_calibration_v1/attempt_v2_integrity_replacement_v1/result.json`,
  581,557 bytes, SHA-256
  `d87eed0cb8a4912be8fcf0bb2dd582a8394c363ad39cfd9cced8a4f0507a53ee`;
- prior terminal at the sibling `terminal.json`, 575 bytes, SHA-256
  `5bb8409a085917caee78b404534f5f3bf5537a928f8165793c34ce54a180f0a0`;
- corrected prior terminal review at
  `docs/lewm_go2_dinov2_physical_readout_calibration_integrity_replacement_v1_terminal_review_2026-08-03.json`,
  14,382 bytes, SHA-256
  `7074779bdc506548d903c0319b74243f2b2934a1888325f813ee52f5a115c679`;
- prior compatibility receipt at
  `.generated/dev/go2_dinov2_physical_readout_calibration_v1/attempt_v2_integrity_replacement_v1/compatibility_receipt.json`,
  3,017 bytes, SHA-256
  `3bd0f06e2970966a9471f352a76cd6859580336d86a69dec945a989c971e0710`;
- posthoc manifest at
  `.generated/dev/lewm-go2-wm-bounded-branch-posthoc-join-admission-v1/manifest.json`,
  11,964 bytes, SHA-256
  `87448995c905107453814a5e7e4cd9968d31cbc0e308513d17bc038c6585f15e`;
- posthoc terminal at the sibling `terminal.json`, 1,250 bytes, SHA-256
  `a1590fffc673f7676016bb70d4b4f5530f24b9a49bf05e84dcec6bc1756fbe56`;
- posthoc terminal review
  `docs/lewm_go2_world_model_bounded_branch_posthoc_join_admission_v1_terminal_review_2026-08-02.json`,
  2,844 bytes, SHA-256
  `bfd0250357d0f681c674db6c54ea4a8c4d5e617230332383beda3db3e0f38669`;
- posthoc RGB manifest at
  `.generated/dev/lewm-go2-wm-bounded-branch-posthoc-join-admission-v1/rgb_manifest.json`,
  1,880,307 bytes, SHA-256
  `5e03afa7665ffef54a1cab5e37135a18d42761bc844ecefacaa433f75a1b1f7e`;
- posthoc train rows at the sibling `train.jsonl`, 30,432,624 bytes,
  SHA-256
  `edc6f88bb105c39575477fbfbb0224bf0312cf5ee3e90551f86a9c11c2ebb447`;
- posthoc evaluation rows at the sibling `eval.jsonl`, 30,411,588 bytes,
  SHA-256
  `531debbc431f2f8afc83a491b491b8822134c831b16ca4d283fe1e7f4ba07768`;
- stored task-relevance result
  `docs/lewm_go2_world_model_visual_domain_parity_task_relevant_input_adequacy_result_v1_2026-08-02.json`,
  94,165 bytes, SHA-256
  `5094104ac29b4652cd577015c5fbf23b42f0768c78a205cbf07a77d992339ca7`;
- stored task-relevance review
  `docs/lewm_go2_world_model_visual_domain_parity_task_relevant_input_adequacy_independent_review_v1_2026-08-02.json`,
  2,080 bytes, SHA-256
  `29eb00a486604824effb56502194855553f87c81a9691d4075a5810273c92ca9`.

Both token caches must have shape `[1536,256,384]`, float16 storage,
per-token L2 normalization with maximum absolute norm error at most `2e-3`,
and the exact artifact orders in their receipts.
The strict posthoc bundle loader and the already-preregistered singleton SSIM
compatibility admission remain mandatory.  The only admissible live/stored
task-relevance difference remains
`measurements.pixels.minimum_reference_candidate_rgb_ssim`, finite and within
absolute tolerance `1e-12`, with both values at or above `0.99`; all other
fields are canonical-exact.

## Permitted inputs and prohibited leakage

Readout inputs are limited to:

- the last context DINO patch-token grid;
- the candidate's actual successor DINO patch-token grid for the oracle-future
  arm, or the last context grid in the registered control substitutions;
- the relative target `(x,y)` in body coordinates;
- the requested canonical action command `(vx,wz)`.

The requested command catalog is fixed to:

| action | name | `vx` | `wz` |
|---:|---|---:|---:|
| 0 | arc_left | 0.20 | 0.45 |
| 1 | arc_right | 0.20 | -0.45 |
| 2 | backward | -0.20 | 0.00 |
| 3 | forward_fast | 0.30 | 0.00 |
| 4 | forward_medium | 0.25 | 0.00 |
| 5 | forward_slow | 0.20 | 0.00 |
| 6 | hold | 0.00 | 0.00 |
| 7 | yaw_left | 0.00 | 0.45 |
| 8 | yaw_right | 0.00 | -0.45 |

Inputs must not include executed command tapes, clipping, endpoint pose or
state, path length, target progress, falls, tips, contacts, physical ranks,
family/scene/state identity, artifact identity/path/hash, evaluation-derived
normalization, or any protected material.  Physical dense ranks are train
targets and evaluation metrics only.  Nine branches and 256 patches are never
treated as independent uncertainty units.

## Train-only PCA

Fit one PCA basis using only the train role's 128 last-context grids and all
1,152 train successor grids: 1,280 grids or 327,680 patch-token rows.  The row
order is all 128 last-context grids in role-plan order followed by all 1,152
successors in state-major/action-major order; patches within each grid are
row-major.  Promote the float16 tokens to float64.  Compute population
covariance exactly as `(X-mu)^T(X-mu)/327680`, use `numpy.linalg.eigh`, order
eigenpairs by descending eigenvalue and then original ascending eigenvector
index on an exact eigenvalue tie, and retain exactly `K=8` components.  Fix
each component's sign so its largest-absolute loading (smallest channel index
on a tie) is positive.  Whiten by `sqrt(max(eigenvalue, 1e-12))`.  No clipping
is permitted.

Apply that frozen mean, basis, and whitening scale to both roles.  The PCA
identity binds the float64 mean, ordered eigenvalues, signed basis, epsilon,
source artifact order, and implementation source.

## Fixed dense shared scorer

For each co-located patch `i`, let `z_c` and `z_s` be its eight-dimensional
whitened current and successor vectors and define
`r_i = [z_c, z_s, z_s - z_c]` in 24 dimensions.  For cache patch `(row,col)`
in row-major order, define
`u=2*(col+0.5)/16-1`, `v=2*(row+0.5)/16-1`, and `p_i=(u,v)`.  Define the
four-dimensional condition

`q = [goal_x / 10, goal_y / 10, requested_vx / 0.30, requested_wz / 0.45]`.

The scorer is exactly:

```
h_i     = tanh(W_r r_i + W_p p_i + W_q q + b_h)       # H=4
alpha_i = softmax_i(w_alpha^T h_i)                     # over 256 patches
v_i     = W_v r_i                                      # V=4, no bias
z       = sum_i alpha_i v_i
score   = w_z^T z + z^T B q + b_score
```

Shapes are `W_r:[4,24]`, `W_p:[4,2]`, `W_q:[4,4]`,
`b_h:[4]`, `w_alpha:[4]`, `W_v:[4,24]`, `w_z:[4]`,
`B:[4,4]`, and scalar `b_score`: exactly 245 supervised parameters per member,
735 for the primary three-member true-future ensemble, another matched 735 for
the current ensemble, and 1,470 dense supervised parameters across the six
networks stored in the checkpoint.  The fixed task/action base has 27 fitted
coefficients and is reported separately.  Every patch can affect attention and
value.  Parameters are shared across all nine actions and all 1,152 train
branches.  There is no q-only value or score shortcut, convolution, token
masking, dropout, batch normalization, scheduler, or hidden family/scene
embedding.

## Fixed fitting protocol

First fit the exact frozen task/action-only ridge control.  For every train
branch, define `base_score` as that control's fitted score and fit the dense
scorer only to the residual
`dense_rank / max_state_dense_rank - base_score`.  At evaluation, every dense
arm score is `task_action_only_score + dense_residual_score`.  This nesting
prevents the dense mechanism from losing merely because a small neural model
must relearn the already-strong action/goal prior; its incremental patch-token
contribution must still earn every gate.

Use state-balanced residual mean squared error.  Each minibatch contains 16
complete states and all nine actions.  Use exactly:

- three seeds: `2026080303`, `2026080304`, `2026080305`;
- float32 scorer inputs and parameters in the bound ROCm runtime;
- Xavier-uniform initialization with gain `1.0` for `W_r`, `W_p`, `W_q`,
  `W_v`, and `B`, and for `w_alpha`/`w_z` viewed as `[1,4]` before flattening;
  all biases are zero.  For each member, initialize on CPU using a dedicated
  `torch.Generator(device="cpu").manual_seed(member_seed)`, drawing in the
  exact order `W_r`, `W_p`, `W_q`, `W_v`, `B`, `w_alpha[1,4]`, `w_z[1,4]`,
  then clone that state for TRUE/CURRENT and move both models to ROCm;
- AdamW with learning rate `1e-3`, weight decay `1e-2` applied to all 245
  parameters, betas `(0.9,0.999)`, epsilon `1e-8`, `amsgrad=False`,
  `maximize=False`, `foreach=False`, and `fused=False`;
- L2 gradient-norm clipping at `1.0`;
- 256 epochs, eight batches per epoch, exactly 2,048 optimizer steps;
- a separate CPU `torch.Generator` initialized from the same member seed for
  data ordering, with one consecutive `torch.randperm(128)` draw per epoch,
  split into eight groups of 16 states;
- `torch.use_deterministic_algorithms(True)` and float32 matmul precision
  `highest`;
- no early stopping, validation monitoring, checkpoint selection, or retry.

For each seed, instantiate one scorer, clone its exact initial state into the
true-future and current-control models, and fit them independently with the
same state order.  The primary score for an arm is the arithmetic mean of the
three seed scores before action selection.  Individual seed results are
diagnostics only; no seed may be selected or dropped.

## Fixed arms

1. `privileged_physical_oracle`: exact dense physical ranks.
2. `dense_shared_true_future`: the exact task/action-only score plus the
   ensemble true-future residual from the actual action-matched successor grid.
3. `dense_shared_current_state`: the ensemble of capacity-matched current
   residual scorers, trained and evaluated with `r=[z_c,z_c,0]`, added to the
   same task/action-only score.
4. `task_action_only`: the exact previous nine-head target-only ridge control,
   using three target features, ridge `1e-3`, and 128 train rows per action.
   Its identity must equal
   `69895316b19bc179e35fdd76905aadbd50b6ad3e22e965b662ba59672c52886a`
   and its evaluation regret must equal `0.17441406250000002`.
5. `dense_relational_persistence`: the fitted true-future residual ensemble
   evaluated with `r=[z_c,z_c,0]`, without refitting, and added to the same
   task/action-only score.
6. `hold_constant`: fixed action 6.
7. `random_expected`: exact uniform expectation, not a realized sample.

The completed quadrant `dinov2_true_future` result may be copied as a
report-only historical comparator.  It is not a gate and cannot be used to
select this mechanism.

## Metrics, uncertainty, and unchanged gates

Select the minimum predicted normalized rank, breaking ties by action ID.
Report normalized physical rank regret, oracle-equivalent selection, physical
target progress, path length, action histograms, all per-family and per-scene
summaries, attention entropy, seed dispersion, train loss, and finiteness.
Safety remains `NOT_TESTABLE_ZERO_EVENT_SUPPORT` when the fixed labels contain
no falls or tips; zero events are not a safety pass.

Use the unchanged paired scene-cluster bootstrap: equal weight for the eight
families, scene as the resampling unit, two scenes per family, 10,000
resamples, seed `2026080302`, percentile 95% interval.

The calibration passes only if all hold:

1. all authority, source, input, role, cache-rehash, artifact-order,
   compatibility, finiteness, no-RGB, and no-protected-access checks pass;
2. the privileged physical oracle has exactly zero normalized regret and
   selects an oracle-equivalent action in every evaluation state;
3. the upper 95% endpoint of
   `dense_shared_true_future - task_action_only` regret is strictly below zero;
4. the upper 95% endpoint of
   `dense_shared_true_future - dense_shared_current_state` regret is strictly
   below zero;
5. the upper 95% endpoint of
   `dense_shared_true_future - dense_relational_persistence` regret is strictly
   below zero;
6. the dense true-future point regret beats random expected regret, with all
   family comparisons reported; and
7. a fresh-process cache-only deterministic replay rehashes both caches,
   rebuilds the PCA, reinitializes and retrains all six networks for the exact
   2,048 steps, and reproduces PCA and state-dict identities, per-seed and
   ensemble scores, selected actions, summaries, intervals, and verdict
   exactly.  A separate read-only terminal audit must confirm this gate before
   the result is accepted.

These are the same zero-effect superiority boundaries and physical controls as
the previous calibration, not a new absolute threshold.  A favorable single
seed, training loss, attention map, family, or point estimate cannot override
a failed gate.

## Terminal route

The exact scientific statuses are:

- `PASS_DENSE_SHARED_DINO_PHYSICAL_READOUT_HEADROOM_ESTABLISHED`;
- `STOP_FROZEN_DINO_VISUAL_PLANNING_INTERFACE_NOT_ESTABLISHED`; or
- `FAIL_INFRASTRUCTURE_NO_SCIENTIFIC_DECISION`.

A PASS permits a separately preregistered dense action-conditioned JEPA versus
conventional state-space and Dreamer-style comparison.  It does not itself
authorize model training or planner integration.

A scientific STOP closes further frozen-DINO oracle-future readout variants,
retries, seed changes, architecture changes, gate tuning, and use of the 3 TB
observational pool as a substitute for demonstrated planning headroom.  A
future task-coupled or embodiment-supervised mechanism is new research, not a
continuation or rescue of this route.

## One-shot output and custody

One separately reviewed authority may create only:

`.generated/dev/go2_dinov2_dense_shared_spatial_readout_calibration_v1/attempt_v1`

The exact output inventory is:

1. `reservation.json`;
2. `primary_compatibility_receipt.json`;
3. `pca_readout_checkpoint.pt`;
4. `evaluation.json`;
5. `replay_compatibility_receipt.json`;
6. `replay.json`;
7. `result.json`;
8. `terminal.json`.

The primary process must load strict metadata under its own compatibility
receipt, fit PCA and all six models using train data only, and write the frozen
checkpoint before loading the evaluation cache.  After it writes
`evaluation.json`, the runner launches a distinct fresh Python replay CLI that
receives only bound file paths and expected hashes, never in-memory primary
objects.  The replay publishes its own compatibility receipt before its strict
loader returns, rehashes both token caches and the checkpoint, rebuilds PCA,
reinitializes and retrains all six models, canonical-compares their states to
the checkpoint, and independently recomputes action selections, summaries,
bootstrap intervals, and gates without importing the primary evaluator's
aggregation, bootstrap, gate, or verdict implementation.  The parent may write
`result.json` and `terminal.json` only after canonical comparison with
`replay.json` succeeds.

The attempt may not open RGB, invoke an encoder, collect or generate data,
access held-out or sealed material, retry, resume, overwrite, promote, deploy,
or create another attempt root.
