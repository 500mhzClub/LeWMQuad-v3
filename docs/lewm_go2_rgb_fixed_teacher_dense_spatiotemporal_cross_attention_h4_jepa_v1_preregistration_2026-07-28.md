# Go2 RGB fixed-teacher dense spatiotemporal cross-attention H4 JEPA V1 preregistration — 2026-07-28

## Status and question

- This document preregisters one development-only falsification. It grants no
  source-freeze, data, checkpoint, GPU, navigation, held-out, sealed,
  promotion, or deployment authority.
- Question: can a jointly learned RGB/action JEPA beat persistence when every
  future query can attend directly to the complete three-frame spatial token
  history, rather than receiving a GRU-compressed recurrent state?
- The hypothesis is motivated by the committed V3 result at `9d2898d`: V3
  retained rank and action sensitivity emerged without a synthetic ranking
  loss, but ordered history was harmful in all eight families and the
  selected prediction remained 1.4388 times persistence at H4.

## Evidence and closed boundaries

- V3 source commit: `011b6651f922a9bc548b86b0046bcbb15b33beb2`.
- V3 terminal-result commit: `9d2898d017345d8844952d359ae886ba5577242d`.
- The recurrent-H4 V1/V2/V3 branch is closed. No recurrent module, retry,
  resume, checkpoint, trace, or tensor from that branch may be used.
- The local-correspondence, warp, dense cost-volume, rigid-BEV, learned-BEV,
  spatial-transport, masked-tubelet, and retrieval branches remain closed.
- This mechanism contains no recurrence, BEV, warp, offset field, optical-flow
  surrogate, cost volume, candidate retrieval, or transport template. Dense
  attention is used only to expose uncompressed historical tokens to learned
  future queries; it does not implement a hand-coded spatial transform.

## Immutable input contract

- Each row is the existing seven-frame RGB/six-action sequence
  `e0,p0,e1,p1,e2,p2,e3,p3,e4,p4,e5,p5,e6`.
- Online inputs are RGB `e0:e2`, past actions `p0:p1`, and proposed future
  actions `p2:p5`. Future RGB `e3:e6` is visible only to the stopped-gradient
  fixed-teacher target path.
- The nine-action vocabulary and order remain exactly:
  `arc_left`, `arc_right`, `backward`, `forward_fast`, `forward_medium`,
  `forward_slow`, `hold`, `yaw_left`, `yaw_right`.
- No pose, odometry, depth, optical flow, occupancy, traversability, map,
  waypoint, reward, collision, navigation, simulator-state, or semantic label
  is an input or training target.
- Training index is exactly
  `.generated/go2_recurrent_h4_rgb_sequence_index_v1/train.jsonl`, SHA-256
  `f3f4dbe9ddd830427cc86bd27b0adb0b0fd0cebf64e937626088711748d9dd6b`,
  ordered-row identity SHA-256
  `c730c3f2a6afd36c351d461feb3ca122d96db38a8a6df269085273ff526dbdaa`,
  10,024,000 bytes and 16,000 ordered rows.
- Validation index is exactly
  `.generated/go2_recurrent_h4_rgb_sequence_index_v1/val.jsonl`, SHA-256
  `86ab3130e5ba3468bd7f7f3e3cb1759d0e4a30d2326496e06845b4af7cb66880`,
  ordered-row identity SHA-256
  `89f05e2c77261b387b7f4763b502adafaf1dfbdf32f08830b08852143a8f75ef`,
  1,278,976 bytes and 2,048 ordered rows.
- RGB leaves resolve only beneath
  `.generated/datagen_full/render_textured_v03`. The existing main-pool census
  receipt remains bound by file SHA-256
  `aac85f1016dca12e57e0cf612cd51a745becb2941adf361c0b4a752fe10a5408`
  and byte count 54,695.
- Train/validation scene and RGB-leaf disjointness, family balance, image
  decoding, resize, and normalization remain exactly the committed V3
  contract. Index regeneration, reordering, filtering, or substitution is
  forbidden.

## Immutable JEPA mechanism

### Encoders

- `T` is an exact copy of the accepted N320 encoder prefix from checkpoint
  SHA-256 `ece874b53941e841fffc61b724a86d4383b881549afa453b746dd5d68aba11b0`
  with semantic content SHA-256
  `9dcca536943f89acfd7d463fdab591e19a030ef3dc8f3f19a050b1b10025fc2b`.
  The checkpoint byte count is exactly 13,777,100.
  It remains in evaluation mode, stopped-gradient, outside the optimizer, and
  unchanged for the complete attempt. There is no EMA.
- `O` is independently loaded from the same accepted N320 encoder state and
  is jointly trainable with the complete attention predictor.
- Both produce a 16-by-16 lattice of 192-dimensional patch tokens. Define
  `zT_t = normalize(T(e_t))` and `zO_t = normalize(O(e_t))` tokenwise.
  The raw online `e2` tokens are retained only for bit-exact persistence
  reconstruction.

### Dense historical context

- Each historical patch token is
  `zO_t[j] + learned_spatial[j] + learned_time[t]` for `t in {0,1,2}` and
  `j in {0,...,255}`.
- One shared learned nine-action embedding is used for `p0:p5`. Past-action
  tokens for `p0` and `p1`, with distinct transition-step embeddings, are
  interleaved between their adjacent frame-token blocks.
- The resulting 770 tokens pass through exactly two standard pre-norm
  `TransformerEncoderLayer` blocks with width 192, six heads, feed-forward
  width 768, GELU, dropout zero, batch-first layout, and no causal or spatial
  transport mask. This context may mix every historical patch and past-action
  token, but receives no future RGB.

### Future action queries and prediction

- The shared nine-action embedding maps `p2:p5` to width 192. For each
  horizon `h in {1,2,3,4}`, the ordered prefix through `p(1+h)` is placed in a
  fixed four-position vector, unused suffix positions are exact zeros, and a
  shared `Linear(768,192)-GELU-Linear(192,192)` path encoder produces `a_h`.
- Horizon `h` begins with 256 query tokens
  `zO_2[j] + learned_spatial[j] + learned_horizon[h] + a_h`.
- Each horizon is decoded independently; horizons neither recurse nor attend
  to one another. The same two pre-norm `TransformerDecoderLayer` blocks are
  shared across horizons, with width 192, six-head query self-attention,
  six-head cross-attention to all 770 context tokens, feed-forward width 768,
  GELU, dropout zero, and batch-first layout.
- One shared `LayerNorm-Linear(192,192)` delta head is the only prediction
  head. Its final linear weight and bias are initialized to exact zero. No
  other module or path is zero-gated.
- It emits one direct cumulative delta `delta_hat_h` per horizon. Predictions
  are not recursively accumulated:
  `z_hat_h = normalize(raw_O(e2) + ||raw_O(e2)|| * delta_hat_h)`.
  The norm and scaling are tokenwise.
- Therefore update zero is exact online persistence for real, wrong-action,
  hold, reset-history, and reordered-history calls, independent of the query
  activations.

### Joint objective

- Define `D(u,v)` as the mean over batch, horizon/frame, and patch tokens of
  the sum of squared feature differences.
- Fixed-teacher target delta is
  `delta_star_h = stopgrad(zT_(2+h) - zT_2)`.
- The complete loss is exactly:
  - `L_delta = D(delta_hat, delta_star)`, weight 1.0;
  - `L_align = D(zO_{0,1,2}, stopgrad(zT_{0,1,2}))`, weight 1.0;
  - `L_total = L_delta + L_align`.
- This is one joint JEPA backward pass through the online encoder, historical
  context, action path, cross-attention decoder, and delta head. The fixed
  teacher receives no gradient.
- Absolute future-latent distance remains an evaluation diagnostic with exact
  training coefficient zero, and the API-compatible variance diagnostic is
  identically zero. Persistence/history hinges, wrong-action ranking, action
  classification, inverse dynamics, retrieval, reconstruction, semantic,
  navigation, and auxiliary control losses are not constructed. None of these
  diagnostics or losses contributes to `L_total`.

## Initialization and optimizer

- The only checkpoint input is the accepted N320 initialization above. V1,
  V2, V3, and every other predictor/checkpoint tensor input is forbidden.
- All attention, embedding, path, and head modules are freshly initialized
  once under seed `20260727`. Standard PyTorch transformer/linear
  initialization is frozen by the reviewed source and runtime; learned
  spatial/time/horizon/action/transition-step embeddings use normal
  initialization with mean zero and standard deviation 0.02. Only the final
  delta linear is exact zero.
- AdamW groups are disjoint and cover every trainable parameter exactly:
  - online encoder: learning rate `1e-4`;
  - historical context plus spatial, time, and past-transition embeddings:
    learning rate `3e-4`;
  - shared action and horizon embeddings, future-action path, query decoder,
    and delta head: learning rate `3e-4`.
- Weight decay is `1e-4`, betas are `(0.9,0.999)`, epsilon is `1e-8`, and
  each group is independently clipped to gradient norm 1.0 after one backward
  pass and before each optimizer step.
- Execution is float32 without autocast. Python, Torch, and CUDA RNGs use seed
  `20260727`; cuDNN benchmarking is disabled.

## Frozen schedule and cap

- Fresh output root:
  `.generated/go2_rgb_fixed_teacher_dense_spatiotemporal_cross_attention_h4_jepa_v1/probe_v1`.
- Exactly 1,000 optimizer updates, effective batch size 16, and exactly 16,000
  ordered training-sequence presentations. There is one backward pass and one
  optimizer step per update, with no accumulation.
- Validation uses the exact 2,048 rows at updates 0, 250, 500, 750, and 1,000.
  Validation does not alter model or RNG state.
- Active GPU time is capped at 90 minutes. There is no early scientific stop,
  schedule extension, second seed, retry, or resume.
- Checkpoints may be written only at updates 250, 500, 750, and 1,000 for
  receipt completeness. They are write-only during the attempt and cannot be
  used as runtime inputs.

## Evaluation and selection

- Metrics, normalization by target `e2` persistence change, scene-then-family
  macro aggregation, 1,000-replicate deterministic scene bootstrap, rank
  calculation, and the following evaluation-only controls remain exactly V3:
  - cyclic-plus-one wrong future actions;
  - all-`hold` future actions;
  - fixed-teacher `e2` persistence;
  - reordered `e1,e0,e2` history with reordered past actions;
  - reset `e2,e2,e2` history with `hold` past actions.
- An eligible checkpoint must be a trained observation at update 250, 500,
  750, or 1,000; have all registered values finite; have target and online
  effective-rank ratios at least 0.10; and have target and online near-zero
  variance fractions at most 0.05.
- Select exactly one eligible checkpoint by minimum scene-then-family macro
  mean real-action normalized error over H1-H4. There is no per-gate or
  per-family checkpoint selection. If none is eligible, the decision is STOP.

## Exact PASS gate

PASS requires every conjunct below; otherwise the decision is STOP:

- exactly 1,000 updates and 16,000 presentations completed;
- every registered value at every observation is finite;
- fixed-teacher target rank and near-zero fraction drift by at most `1e-6`
  from update zero and satisfy the target rank/variance floors at every
  observation;
- every update-zero action, hold, persistence, and history gap has absolute
  value at most `1e-5`;
- an eligible selected checkpoint exists;
- selected H4 real error improves by at least 10% relative to update zero;
- selected H1-H3 real errors are each strictly below update zero;
- every selected persistence gap is strictly positive;
- selected H4 persistence gap is at least 0.10, its bootstrap lower bound is
  strictly positive, at least six of eight families are strictly positive,
  and no family is below -0.02;
- selected H4 action gap is at least 0.05, its bootstrap lower bound is
  strictly positive, H1-H3 action gaps are nonnegative, at least six of eight
  families are strictly positive, and no family is below -0.02;
- selected H4 history gap is at least 0.03, its bootstrap lower bound is
  strictly positive, and at least six of eight families are strictly positive;
- selected H4 hold gap is strictly positive;
- the fixed-teacher state SHA-256 is identical before and after training, its
  EMA-update count is zero, and it never enters an optimizer group or receives
  a gradient.

Threshold equality passes only where `at least` or `nonnegative` is stated.
All `strictly positive` and `strictly below` relations remain strict.

## One-shot, custody, and terminal rules

- Before execution, additive source, focused synthetic tests, recursive source
  closure, independent science/custody review, frozen source commit, and a
  separate one-shot execution authorization are required. This preregistration
  alone authorizes none of them.
- Source/synthetic development may not open indexes, source RGB, the N320
  checkpoint, any predecessor checkpoint, or any generated runtime receipt.
  One disposable full-geometry, batch-16 GPU smoke using only deterministic
  synthetic tensors is permitted before source freeze to verify shape,
  backward reachability, and peak memory. It may write no model or runtime
  artifact, grants no scientific evidence, and does not reserve or consume the
  one-shot output root.
- Once the execution output root is reserved, every termination consumes the
  sole attempt. No operational replacement, retry, resume, repair run, second
  seed, threshold change, loss change, schedule change, or same-mechanism V2 is
  permitted.
- No held-out, test, sealed, legacy V4 sealed role, G2-G8 navigation,
  production, promotion, or deployment input may be opened or run.
- STOP permanently closes this dense cross-attention mechanism and further
  deterministic dense-H4 predictor-architecture variants. No failed
  checkpoint may be opened or reused; the next scientific category must
  reformulate the predictive target state or model uncertainty.
- PASS establishes development RGB/action JEPA substrate feasibility only.
  The selected checkpoint remains inaccessible unless a separate reviewed
  downstream qualification explicitly authorizes its exact use; PASS does not
  authorize navigation or held-out evaluation.
