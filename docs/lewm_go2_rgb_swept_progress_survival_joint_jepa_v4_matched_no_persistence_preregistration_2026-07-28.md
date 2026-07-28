# RGB Swept-Progress Survival Joint-JEPA V4 Matched No-Persistence Control — Preregistration

- Status: frozen before control implementation or runtime access.
- Purpose: isolate the development-selection effect of the executed-action EMA
  latent-persistence objective `P`. This is a backward-membership ablation, not
  a new model, a replacement candidate, or an untouched-generalization test.

## Frozen full-arm evidence

- Full V4 source commit:
  `aaa47a138d0eeb78aa20d9524e67f813f7a74a41`.
- V4 result file/content SHA-256:
  `bf93c96cf020553be74d51847c6876e345cd6cc391b05cec186e36b20ca15aa4` /
  `27ecf4895dfea01a1e5bb4f6f13f3add6a182a8dfa4b9f8651204bd1e6222ad8`.
- V4 training-trace file/content SHA-256:
  `2ad16afd722ada26439c4ebfb2993330ec3abe1cbe4a75ced496a7c2a2b1580b` /
  `bb027f8af94f352aac3ca1291a84285e25df431ca90682660afc7e1b476d4c12`.
- V4 terminal-result document commit:
  `8b3a8063b087c81030189deadc6c5f6e1c7d44c3`.
- V4 was a valid `PASS_FULL_ARM`: all semantic, swept-progress, family, and
  inference-control gates passed. It did not establish a `P` treatment effect
  and authorized only this identical-decoder matched control.

## Exact matched arm

- Construct a fresh V4 model from the accepted N320 encoder-only input. Use the
  exact V4 architecture, inherited base semantic head, 37,123-parameter
  residual local decoder, visibility mask, constructor seed `20260712`, decoder
  seed `20260713`, execution seed `20260728`, and deterministic runtime rules.
- Never name, read, hash, load, copy, resume, or warm-start the V4 terminal
  checkpoint or any rejected predecessor checkpoint/runtime state.
- Inherit byte-for-science-identical V4 RGB/data/labels, action order, sweep
  masks, schedule and schedule prefix, microbatch order, optimizer groups and
  hyperparameters, clipping, EMA momentum, semantic loss `S`, survival loss
  `U`, ranking loss `R`, coefficient-`0.5` occupied auxiliary `O`, evaluator,
  thresholds, controls, bootstrap, and family aggregation.
- Use the exact V4 runtime and hardware binding: one visible AMD Radeon AI PRO
  R9700, `HIP_VISIBLE_DEVICES=0`, with the deterministic-algorithm, benchmark,
  and TF32 settings unchanged.
- Preserve exactly 1,000 updates / 16,000 presentations, four size-four
  microbatches and backward calls per update, one optimizer step, and one EMA
  update per update. Preserve the exact accounting and finite-gradient checks.
- Execute the same online current/next encodes, semantic decoder, all-action
  predictor and survival head, EMA current/next target encodes, persistence
  diagnostic, and every evaluation/control forward. Compute and trace `P`
  exactly, even though it has no backward membership.

## Sole treatment delta

- Full V4 used the direct backward scalar `S + P + U + R + O`.
- The control must construct its backward scalar directly as
  `L_backward = S + U + R + O`.
- Do not implement this as `P * 0`, subtraction from the full loss, gradient
  cancellation, detach of another term, rescaling, loss renormalization, or a
  learning-rate/clip adjustment. `P` is simply absent from backward membership.
- Trace each update with exactly `S`, `P_diagnostic`, `U`, `R`, `O`,
  `L_full_diagnostic`, and `L_backward`. Verify to floating-point roundoff that
  `L_full_diagnostic = S + P_diagnostic + U + R + O` and
  `L_backward = S + U + R + O`.
- Keep every online parameter trainable and in its inherited optimizer group.
  The predictor and swept-progress head continue receiving gradient through
  `U` and `R`; the encoder/lift receive `S`, `U`, `R`, and `O`; the base and
  residual semantic decoder receive `S` and `O`.
- The target encoder and target BEV lift remain no-gradient and optimizer-
  excluded. Their identical forwards and one EMA update after every optimizer
  step remain mandatory matching operations even though they cannot influence
  `L_backward`. Add no semantic target head.
- Source tests must prove nonzero predictor gradient from `U/R`, zero gradient
  contribution from `P_diagnostic`, unchanged parameter/optimizer membership,
  target exclusion, exact loss identities, and exact terminal accounting.

## Reconstructed-initialization witness

- Before optimizer construction, record a canonical SHA-256 over the complete
  freshly reconstructed V4 tensor state. The canonical payload is the
  lexicographically ordered list of state-dict entries containing each name,
  exact dtype, shape, and SHA-256 of its contiguous CPU tensor bytes. Record its
  canonical-JSON SHA-256 and prove two fresh CPU reconstructions from the same
  accepted N320 state produce the identical payload and digest.
- Before update 1, record `target_hard_sync_count = 1`, `ema_update_count = 0`,
  an empty optimizer state, and a canonical receipt for the exact optimizer
  parameter-group inventories and hyperparameters.
- After the four update-1 backward calls but before the first optimizer step,
  the accumulated component means must equal the frozen V4 trace row exactly:
  - `S = 1.313827022910118`;
  - `P_diagnostic = 1.0`;
  - `U = 0.9792981296777725`;
  - `R = 1.0`;
  - `O = 1.026371382176876`.
- Any initial-state, inventory, or update-1 component mismatch terminates the
  control with a complete failure receipt and no retry. It may not be repaired
  by tolerance widening, state substitution, warm start, or schedule change.

## Historical matched-control deviation

- Older repository conventions constructed one CPU initialization, serialized
  it once, copied it byte-identically into both arms, and ran both arms within
  one reserved attempt. Full V4 already completed before this conditional arm,
  and its receipt did not publish a complete initial-state hash. A fresh
  deterministic reconstruction cannot literally prove the old serialized-
  clone/same-reserved-attempt condition.
- This V4-specific conditional continuation explicitly supersedes only those
  two historical requirements. It does not claim to satisfy them. The frozen
  source/seeds, reconstructed state receipt, and exact update-1 functional
  witness are the bounded replacement evidence.
- Consequently, this comparison is a matched development diagnostic on a
  repeatedly used selection role. It cannot establish causal generalization,
  seed robustness, held-out performance, or navigation performance.

## Evaluation and treatment predicate

- Freeze the full-V4 reference as this exact ordered family vector, copied once
  into this preregistration from the already reviewed terminal receipt:
  `large_enclosed_maze=0.8896189747752248`,
  `local_composite_motifs=0.9384050589932943`,
  `loop_alias_stress=0.8938629676334595`,
  `medium_enclosed_maze=0.8772593292124542`,
  `open_obstacle_field=0.8934829059829059`,
  `rough_local_dynamics=0.9430145611963794`,
  `small_enclosed_maze=0.922340425531915`, and
  `visual_sensor_stress=0.9229020111832612`.
- The canonical JSON payload with schema
  `lewm_v4_full_reference_family_utility_v1`, ordered `family_order`, and the
  parallel unrounded `normalized_chosen_prefix_utility` vector has SHA-256
  `8ba8d6126e922f6a36038304e3444d0d21ee69350fef4acd3828265754810e1e`.
  The control runner must use this embedded immutable reference and must not
  reopen the generated V4 result, trace, or checkpoint.
- Train exactly once to terminal update 1000. The control may not select an
  update or use calibration, selection, controls, or V4 results to alter
  training. Evaluate it once using the exact V4 calibration/selection order,
  metrics, semantic/progress gates, eight family gates, and four inference
  controls.
- Report the control's absolute gates and every full-V4-minus-control semantic,
  progress, calibration, control, scene, and family delta. These are reports,
  not new qualification, selection, or promotion gates.
- For each of the same eight fixed selection scenes/families, let `d_f` be full
  V4 minus control normalized chosen-prefix utility. The positive treatment
  predicate is true only when all are true:
  1. the equal-scene mean of `d_f` is strictly positive;
  2. the 10,000-replicate paired-scene bootstrap with seed `20260728` has a
     strictly positive lower 95% bound (sorted zero-based index `249`); and
  3. at least six of eight `d_f` values are strictly positive.
- The bootstrap is frozen to the exact resampling algorithm inherited from
  `paired_control_comparison_v1` in source SHA-256
  `870022fc84ad391c97c3fe06da83357d8575408a7d57874aa0aac118ace9deb2`:
  form the ordered float64 eight-value delta vector above, construct
  `numpy.random.default_rng(20260728)`, draw
  `rng.integers(0,8,size=(10000,8))`, take each indexed draw's arithmetic mean,
  sort the 10,000 means ascending, and report element `249` as the lower bound.
- Both receipts, accounting, initial witness, evaluation, and all compared
  values must be complete and finite. A structural or access failure makes the
  treatment predicate invalid, not positive.
- If and only if the predicate passes, the exact allowed conclusion is:
  **“`P` improved development selection utility under this fixed deterministic
  training schedule.”** No broader JEPA or generalization wording is allowed.
- A false predicate is negative evidence for benefit from `P`, not permission
  to rerun, change the seed, alter the objective, or promote the control.

## One-shot lifecycle and authority

- Authorize one fresh write-once control attempt only. There is no retry,
  resume, extension, intermediate selection, alternate seed, loss variant,
  checkpoint substitution, or result-conditioned intervention.
- The control is diagnostic only. It cannot qualify, disqualify, replace,
  initialize, average with, calibrate for, or promote the V4 full arm, even if
  its absolute metrics are better.
- Neither the control nor either treatment-predicate outcome authorizes G2,
  navigation, sealed, held-out, production, deployment, promotion, or final-
  evaluation access.
