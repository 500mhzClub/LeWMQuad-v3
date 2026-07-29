# RGB Swept-Progress Survival Joint-JEPA V9 Content-Adaptive Dense Local Token Lift — Preregistration

- Date: 2026-07-29, before V9 source, data access, GPU work, or model output.
- Status: one architecture-only capped falsification is selected. This record
  grants source implementation, focused tests, independent review, and—only
  after a frozen execution binding—one fresh run. It grants no retry, resume,
  checkpoint reuse, calibration unless earned, G2, navigation, held-out,
  sealed, production, deployment, or promotion access.

## Evidence and decision

- Clean V4 (`8b3a8063b087c81030189deadc6c5f6e1c7d44c3`) proved that the
  112-square global ViT, residual-local semantic decoder, and jointly trained
  action predictor can pass all 24 development checks.
- V4 physical calibration (`1f96caec54e5afa10882cd1e969518164f6dcf1e`)
  then found zero passing threshold tuples and only `0.26853` near-obstacle
  detection on the disjoint selection role. Aggregate argmax accuracy was not
  conservative physical separability.
- V6 and V8 supplied finer RGB evidence while retaining the same four-sample
  lift. Both improved occupied evidence but missed the unchanged FREE gate;
  V8 ended at 23/24, with FREE recall `0.8491042820`. V8's committed result
  (`8ad589760d90a13f21635df118508b346ac9c3a7`) identifies the inherited
  sparse image-token-to-BEV aggregation as the strongest remaining common
  bottleneck.
- The prior center-projective column-attention, radial/ray, and
  projective-height families are not reopened. V9 uses one existing ground
  anchor and no height hypotheses, camera ray, range/depth lattice, polar
  factorization, or vertical-column bias. Global learned BEV queries and
  prototypes are also excluded.
- V9 changes one mechanism: four content-independent deformable samples become
  content-adaptive attention over a fixed dense local token neighborhood.

## Frozen architecture

- Start from clean V4, not V6, V7, V8, or any runtime checkpoint.
- Preserve 112x112 RGB, patch size 7, the 16x16 final ViT token lattice,
  width/depth/heads `192/6/6`, the accepted N320 encoder-only initialization,
  the 64-channel 64x64 BEV, residual-local semantic decoder, action predictor,
  swept-progress survival head, EMA target, visibility logits, and all output
  schemas.
- Replace only `GeometryAnchoredDeformableBevLiftV1` aggregation. Reuse its
  exact fixed camera, metric grid, ground plane `z=-0.333 m`, per-cell ground
  projection, anchor visibility, 192-to-64 token projection, learned null
  evidence, and two local residual refinement blocks.
- Around each projected ground anchor, construct exactly 25 bilinear token
  samples on the Cartesian product of token offsets
  `x,y in {-2,-1,0,1,2}`. Convert a token offset to normalized image space by
  multiplying by `2/16`. Ordering is row-major by `y`, then `x`; the exact
  centre `(0,0)` is index 12.
- A support sample is valid only when the inherited ground anchor is visible
  and the proposed normalized coordinate lies in `[-1,1]^2`. The centre is
  the query. All valid support samples are keys and values.
- Apply one learned 64-dimensional, four-head scaled dot-product attention
  block with dropout zero and head width 16. Query, key, value, and output
  projections are biased 64-to-64 linear maps. Invalid supports are excluded
  before softmax. An invalid cell receives the inherited learned null evidence
  and must remain finite.
- Add the attention output residually to the centre sample, reshape to
  `[B,64,64,64]`, then apply the two unchanged V4 refinement blocks with the
  unchanged null-evidence mask after each block.
- The online and target lifts have identical inventories at initialization.
  The target lift is detached, hard-synchronized exactly once, and updated
  only by the unchanged EMA rule.

## Initialization boundary

- Construct a clean V4 model from the same accepted N320 encoder state and the
  same sweep masks.
- Every inherited model tensor outside the removed `raw_offsets` and
  `weight_logits` must be bit-exact to that clean construction before update
  one, including encoder, token projection, refinement blocks, semantic
  decoder, predictor, survival head, null evidence, and all fixed buffers.
- Initialize only the new attention projections under an isolated CPU RNG
  seed `20260729`, restoring the caller RNG exactly. No predecessor runtime
  tensor, rejected checkpoint, intermediate checkpoint, or trace may be read.
- V9 is not required to reproduce V4's initial latent because its aggregation
  is the scientific intervention. It must prove exact initial online/target
  equality and zero target gradients.

## Frozen learning and evaluation

- Reuse V4's exact Raw-V13 role files, endpoint order, fixed negatives,
  labels, action vocabulary, schedule prefix, batching, seed, optimizer
  groups and hyperparameters, clipping, EMA momentum, bootstrap, evaluator,
  and all 24 gate thresholds.
- Preserve the exact joint objective `L=S+P+U+R+O`, including the coefficient
  `0.5` occupied auxiliary. No new auxiliary, weighting, calibration term,
  threshold, class bias, label, pose, depth, ray, flow, or navigation signal
  is added.
- Train encoder, online V9 lift, semantic decoder, predictor, and survival head
  jointly. Every new online attention tensor must receive a finite nonzero
  gradient by update 2 and remain optimizer-owned exactly once. Target
  parameters remain frozen and outside the optimizer.
- One attempt only: exactly 1,000 optimizer/EMA updates, four B=4
  microbatches per update, and 16,000 ordered presentations. The terminal
  update-1000 state is the only scientific decision point. There is no retry,
  resume, second seed, extension, best-checkpoint selection, or repaired
  replacement for a scientific failure.
- The unchanged 24-check development gate is conjunctive. V9 passes only if
  all 24 checks pass, including balanced accuracy `>=0.80`, FREE recall
  `>=0.85`, occupied recall `>=0.70`, rough occupied recall `>=0.65`, unknown
  recall `>=0.90`, utility `>=0.85`, zero-prefix `<=0.05`, concordance
  `>=0.75`, every family floor, and all causal-control requirements.
- A complete 24/24 pass earns exactly one use of the existing development-only
  physical calibration procedure on the V9 terminal artifact. Any failed
  development check closes V9 before checkpoint admission or calibration.
- Physical calibration must pass its unchanged conservative gate before G2 can
  be considered. A development pass alone is not perception qualification.

## Falsification and stopping rule

- Success would show that V4's final global tokens contain useful local
  evidence that the four content-independent samples failed to collect, while
  retaining a genuine jointly trained JEPA predictor.
- Failure closes this exact 5x5 support, attention width/head count,
  initialization, seed, and cap. Do not try 3x3/7x7 windows, another head
  count, a vertical strip, a ray/height version, a second seed, a longer run,
  or a V8-resolution combination.
- This experiment neither changes nor opens the existing held-out benchmark.
  G2 through G8 remain ordered and closed unless their preceding gates are
  earned.
