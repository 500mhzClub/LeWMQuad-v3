# RGB Swept-Progress Survival Joint-JEPA V7 — Hierarchical CNN Encoder Preregistration

- Status: frozen before V7 implementation, source review, runtime/data access,
  or training.
- Clean scientific baseline: V4 source / full-arm result commits
  `aaa47a138d0eeb78aa20d9524e67f813f7a74a41` /
  `8b3a8063b087c81030189deadc6c5f6e1c7d44c3`.
- Closed successors: V5 near-field hazard ranking and V6 fine-RGB BEV fusion.
- Purpose: falsify exactly one materially different learned RGB encoder inside
  the joint JEPA. This is not a decoder, loss, threshold, calibration, data,
  token-overlap, fine-feature fusion, detached-readout, or separately trained
  predictor experiment.

## Evidence and hypothesis

- V4 passed all 24 development checks, but its fresh physical calibration
  produced zero passing tuples out of 2,016. Selection free precision was
  `0.92923`, near-obstacle detection `0.26853`, useful-free recall `0.85067`,
  and near-obstacle exclusion `0.72147`. Calibration improved NLL and Brier
  score but could not create the missing physical obstacle/free ordering.
- V5's explicitly gate-aligned ranking loss was active and fell substantially,
  yet it moved the class operating point and failed the unchanged development
  gate. Loss, coefficient, margin, and range refinement is closed.
- V6's learned pixel-scale branch received gradients and trained throughout its
  complete capped run, but its loss curve was almost identical to V4's and it
  passed only 23/24 checks. The additive fine-feature/fusion family is closed.
- V4--V6 retain the same patch-token VisionEncoder. Earlier overlapping
  tokenization changed its patch stem but retained a flat `16x16` ViT token
  lattice and failed. V4's successful residual mechanism was confined to the
  semantic decoder. Neither is this wholesale encoder replacement.
- V7 tests one remaining representation hypothesis: a translation-equivariant,
  overlapping, hierarchical local encoder may preserve and compose obstacle
  edges, floor texture, and near-field shape more usefully than the inherited
  patch-token transformer before the unchanged projective BEV lift must order
  occupied versus traversable cells. The hypothesis concerns learned features,
  not extra input information or post-hoc confidence fitting.

## Sole scientific delta from clean V4

- Wholesale replace both the online and EMA-target `VisionEncoder` with the
  exact compact hierarchical CNN below. Retain normalized `112x112` RGB as its
  only input and preserve the inherited `[B,257,192]` token interface.
- Remove the inherited patch embedding, learned CLS token, positional
  embedding, transformer blocks, and final LayerNorm. Do not copy any accepted
  N320 VisionEncoder parameter into the CNN. The accepted N320 lineage remains
  unchanged for every inherited component to which it applies; no produced or
  rejected experiment checkpoint may initialize V7.
- Preserve the exact V4 deformable BEV lift, residual-local semantic decoder,
  action-conditioned predictor, swept-progress survival head, target BEV lift,
  visibility handling, action vocabulary, and all other model behavior.

## Exact hierarchical CNN

- Use biased convolutions throughout. Every convolution has dilation one,
  groups one, and zero padding where padding is specified. Every GroupNorm is
  affine with `eps=1e-5`. Every activation is exact
  `GELU(approximate="none")`. Add no dropout, pooling, attention, positional or
  coordinate channel, learned token, stochastic depth, extra normalization,
  projection, or auxiliary head.
- Construct and apply modules in this exact order:
  1. `Conv2d(3,48,kernel_size=5,stride=2,padding=2,bias=True)` ->
     `GroupNorm(6,48)` -> GELU, producing `56x56`.
  2. Two consecutive 48-channel residual blocks. Each block is
     `Conv2d(48,48,3,1,1,bias=True)` -> `GroupNorm(6,48)` -> GELU ->
     `Conv2d(48,48,3,1,1,bias=True)` -> `GroupNorm(6,48)` -> add the
     unchanged block input -> GELU.
  3. `Conv2d(48,96,kernel_size=3,stride=2,padding=1,bias=True)` ->
     `GroupNorm(8,96)` -> GELU, producing `28x28`.
  4. Two consecutive 96-channel residual blocks with the exact block topology
     above, using `Conv2d(96,96,3,1,1,bias=True)` and `GroupNorm(8,96)`.
  5. `Conv2d(96,192,kernel_size=3,stride=2,padding=1,bias=True)` ->
     `GroupNorm(12,192)` -> GELU, producing `14x14`.
  6. Two consecutive 192-channel residual blocks with the exact block topology
     above, using `Conv2d(192,192,3,1,1,bias=True)` and
     `GroupNorm(12,192)`.
  7. Bilinearly interpolate the `14x14` map to exactly `16x16` with
     `align_corners=False`, then apply
     `Conv2d(192,192,kernel_size=1,stride=1,padding=0,bias=True)`.
- Flatten the final `[B,192,16,16]` map in PyTorch row-major spatial order and
  transpose it to `[B,256,192]`. Compute CLS as the arithmetic spatial mean of
  those final 256 projected tokens, giving `[B,1,192]`, and prepend it to return
  `[B,257,192]`. The CLS token has no parameter. The unchanged V4 BEV lift
  continues to consume only the 256 spatial tokens after the prepended CLS.

## Initialization and parameter identity

- Initialize the complete online CNN freshly on CPU under isolated seed
  `20260715`, constructing modules in the exact order listed above and using
  the standard PyTorch `Conv2d` and `GroupNorm` initializers. Restore the
  caller/global RNG state afterward so every inherited V4 component retains
  its frozen initialization.
- Create the target CNN as one exact `deepcopy` of the initialized online CNN;
  do not initialize it independently. Freeze it immediately, keep it in
  evaluation mode, and update it only through the inherited target hard-sync
  and EMA rules.
- The online CNN has exactly `1,994,880` trainable parameters:

  | Component | Parameters |
  |---|---:|
  | Initial 5x5 convolution + 48-channel GroupNorm | `3,744` |
  | Two 48-channel residual blocks | `83,520` |
  | 48-to-96 downsampling convolution + GroupNorm | `41,760` |
  | Two 96-channel residual blocks | `332,928` |
  | 96-to-192 downsampling convolution + GroupNorm | `166,464` |
  | Two 192-channel residual blocks | `1,329,408` |
  | Final 1x1 projection | `37,056` |
  | **Total** | **`1,994,880`** |

- Put all online CNN parameters in the exact inherited encoder optimizer and
  clipping group, each exactly once. Target CNN parameters remain frozen and
  absent from every optimizer. Add no optimizer group or training phase.

## Frozen joint training and falsification

- Inherit V4's exact RGB preprocessing, development data and roles, labels,
  action order, sweep masks, sampling order, batch construction, optimizer
  groups and hyperparameters, clipping, EMA momentum and cadence, seeds other
  than the isolated CNN seed, evaluator, metrics, controls, bootstrap, and
  thresholds.
- Jointly train the online CNN, unchanged BEV lift, V4 base and residual
  semantic decoder, action predictor, and survival head from update one under
  the exact V4 `S + P + U + R + O` objective. Preserve `O` at coefficient
  `0.5`; do not carry V5 loss `H` or the V6 fine-RGB branch.
- Execute exactly one fresh run of 1,000 optimizer/EMA updates, four size-four
  microbatches and backward graphs per update, and 16,000 presentations. There
  is no retry, resume, extension, alternate seed, intermediate-checkpoint
  selection, or width, depth, normalization, activation, interpolation, or
  architecture variant.
- Record the encoder output contract, exact parameter partitions, online CNN
  gradient receipt, frozen target receipt, update/presentation accounting, and
  unchanged loss accounting. Evaluate selection only at terminal update 1000.
- The terminal model must first pass the complete unchanged V4 24-check
  full-arm development gate. Any development failure closes the hierarchical
  CNN encoder family without calibration, checkpoint use, or G2.
- If and only if all 24 checks pass, package that terminal checkpoint as a
  development-only pre-calibration candidate and perform one fresh,
  separately source-frozen execution of the unchanged physical-evidence
  calibration protocol: one four-parameter fit on the calibration role, the
  unchanged 2,016-tuple search there, and one unchanged application to the
  selection role.
- Physical success requires at least one passing tuple, selection free
  precision at least `0.99`, near-obstacle detection at least `0.95`,
  useful-free recall at least `0.90`, and near-obstacle exclusion at least
  `0.95`. Physical failure closes this CNN family without a width/depth retry
  or related CNN variant.

## Authority and stopping

- This preregistration authorizes only implementation, independent source
  review, and one source-frozen capped V7 attempt followed conditionally by the
  unchanged fresh physical calibration. It grants no retry, resume, replacement
  attempt, schedule extension, checkpoint reuse, or variant.
- A physical pass authorizes only preparation and review of a separate one-shot
  G2 binding. It does not itself open G2 or qualify navigation.
- No G2, navigation, held-out, sealed, production, deployment, promotion,
  rejected-checkpoint, original-V4-runtime, V5-runtime, or V6-runtime access is
  authorized by this preregistration.
