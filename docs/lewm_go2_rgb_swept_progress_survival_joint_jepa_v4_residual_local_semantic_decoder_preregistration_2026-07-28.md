# RGB Swept-Progress Survival Joint-JEPA V4 Residual Local Semantic Decoder — Preregistration

- Status: frozen before V4 implementation or runtime access.
- Purpose: falsify one materially different learned perception mechanism, not
  tune the closed occupied-safety coefficient family or qualify navigation.

## Evidence and decision

- V3 was a valid `FAIL_FULL_ARM`: free recall was `0.846040` versus the fixed
  `0.85` floor, while balanced accuracy `0.845907`, occupied recall `0.745270`,
  unknown recall `0.946411`, and rough occupied recall `0.725384` passed.
- Of the remaining misclassified free cells, `72,934` were predicted OCCUPIED
  and `4,139` UNKNOWN. V1/V2/V3 coefficient `0/1/0.5` moved this same boundary
  but never made all semantic gates pass together. Coefficient tuning is closed.
- V3 retained strong action-conditioned evidence: utility `0.908511`, selected
  zero-prefix rate `0.037594`, concordance `0.863508`, all family gates, and all
  persistence, shuffled-action, wrong-RGB, and action-prior controls passed.
- Every loss worsened from updates 801--900 to 901--1000. Do not extend V3 or
  infer that more identical optimization will repair the semantic boundary.
- The inherited semantic readout is one per-cell linear `Conv2d(64,3,1)`. V4
  tests whether a small nonlinear, spatially local residual readout can use
  information already present in the jointly learned BEV latent without
  changing the successful JEPA predictor or moving another loss coefficient.
- This is a semantic-readout/decoder test, not an encoder test and not evidence
  by itself that JEPA features improved. Any JEPA treatment claim still requires
  the later matched no-JEPA arm.

## Sole scientific delta

- Replace only the semantic-head module with the sum of:
  1. the exact inherited base `Conv2d(64,3,kernel_size=1,bias=True)`; and
  2. `Conv2d(64,64,kernel_size=3,padding=1,bias=True)` -> exact
     `GELU(approximate="none")` ->
     `Conv2d(64,3,kernel_size=1,bias=True)`.
- Initialize the residual branch deterministically under an isolated RNG seed
  equal to `config.initialization_seed + 1` (exactly `20260713` under the frozen
  config). Restore caller/global RNG state so every inherited V3 component
  retains its frozen initialization.
- Initialize the final residual projection's weight and bias to exact zero.
  The base head retains the exact V3 initialization, so V4 semantic logits at
  initialization are exactly equal to V3 semantic logits for every latent.
- The final residual projection can receive gradient in the first backward
  pass. Because it initially gates the branch at zero, the preceding 3x3
  context convolution may begin receiving nonzero gradient only after the
  first optimizer step; this is expected and is not a detached or frozen stage.
- The residual branch adds exactly `37,123` trainable parameters: `36,864`
  3x3 weights + `64` biases + `192` 1x1 weights + `3` biases. Add no
  normalization, dropout, coordinates, privileged inputs, or other head.
- Apply the exact inherited visibility mask after summing base and residual
  logits. Invisible-cell logits remain the frozen forced-UNKNOWN values.

## Frozen joint training and evaluation

- Start one fresh model from the accepted N320 encoder-only initialization.
  Never read, hash, load, copy, resume, or warm-start any rejected V1, V2, or V3
  checkpoint or runtime state.
- Inherit exact V3 RGB/data/labels, action order, sweep masks, schedule, seeds,
  optimizer hyperparameters and parameter groups, clipping, EMA, losses,
  evaluator, thresholds, controls, bootstrap, and family gates.
- Freeze the occupied-vs-rest auxiliary coefficient at exactly `0.5`; preserve
  its logit definition, row-present-class balancing, current/next averaging,
  `log(2)` normalization, and separate trace key `O`. No further coefficient
  experiment is permitted.
- Jointly train the online encoder, BEV lift, base semantic projection,
  residual semantic branch, action predictor, and survival head from update 1
  under the inherited `S + P + U + R + O` objective. The residual parameters
  belong to the inherited lift/semantic optimizer group; add no optimizer group
  or training phase.
- Keep the EMA target exactly unchanged: target encoder and target BEV lift
  only. There is no target semantic head, semantic-head EMA, detach, frozen
  online stage, post-hoc fit, or separately trained predictor.
- Execute exactly 1,000 updates / 16,000 presentations with the inherited four
  size-four microbatches, one optimizer step, and one EMA step per update.
- Evaluate selection only after terminal update 1000. PASS requires every
  existing conjunctive semantic, swept-progress, per-family, and control gate;
  no threshold may move and no calibration diagnostic may select a model.

## One-shot lifecycle and authority

- Authorize exactly one fresh write-once V4 attempt. Once update 1 begins there
  is no retry, resume, schedule extension, or intermediate-checkpoint selection.
- Do not test another kernel, width, activation, initialization, coefficient,
  seed, duration, warm start, or residual-head variant. Failure closes this
  residual local semantic-decoder mechanism.
- A complete full-arm PASS authorizes only the exact matched no-JEPA training
  arm needed before any JEPA treatment-effect claim. A V4 failure authorizes no
  matched no-JEPA run.
- Neither outcome authorizes G2, navigation, sealed, held-out, production,
  deployment, promotion, or final-evaluation access.
