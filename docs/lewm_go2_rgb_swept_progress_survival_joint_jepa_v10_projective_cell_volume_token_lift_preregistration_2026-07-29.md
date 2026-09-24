# RGB Swept-Progress Survival Joint-JEPA V10 Projective Cell-Volume Token Lift — Preregistration

- Date: 2026-07-29, before V10 source, data access, GPU work, or model output.
- Status: one fresh, capped, architecture-only falsification is selected. This
  record grants source implementation, focused source-only tests, independent
  review, and—only after a separate frozen execution binding—one development
  run. It grants no retry, resume, rejected-checkpoint reuse, G2, navigation,
  held-out, sealed, production, deployment, or promotion access.

## Decision and causal evidence

- V9 passed all 24 unchanged development checks, proving that the N320-seeded
  encoder, shared 64-channel BEV state, semantic decoder, survival head, and
  jointly trained action predictor remain viable.
- V9 then failed the unchanged physical-evidence gate. On selection, FREE
  precision was `0.928906`, near-OCCUPIED detection `0.322811`, useful FREE
  recall `0.842727`, and near-obstacle exclusion `0.744179`; the calibration
  role contained zero passing threshold tuples.
- The post-result source audit found a direct representation/target mismatch.
  V9 admits a BEV cell only when its ground-center projection is in the camera
  frustum. Every one of its 25 token-neighborhood samples and its final
  semantic logits inherit that gate. A visible vertical obstacle can therefore
  be labelled OCCUPIED while the model forcibly emits UNKNOWN because the
  cell's ground point lies below the image.
- The committed observable-physical-v3 label contract makes FREE from visible
  floor support and OCCUPIED from first-visible 3D obstacle-surface witnesses;
  hidden evidence remains UNKNOWN. The committed cell-square geometry audit
  found 373 known-label occurrences missed by center support. Its static
  25-point support recovered 369; only 4 of 129,021 known occurrences remained
  unsupported. This supplies a causal mechanism and a `99.9969%` empirical
  static known-support ceiling, above the `95%` physical-recall requirement.

## History-aware novelty boundary

- The 3D geometry itself is not new. Earlier fixed and attitude-dynamic
  `projective_cell_square_attention_v1` mechanisms already used the center and
  four closed 0.10 m cell corners at five registered heights. The old dynamic
  Cartesian N32 model used learned BEV queries attending globally to all image
  tokens under a Gaussian geometry bias and was closed after its strict fit
  gate failed.
- V9 also explicitly excluded a later ray/height variant. This preregistration
  supersedes that narrow stopping rule only because the completed V9 physical
  calibration exposed, and the subsequent source audit localized, the hard
  ground-visibility mismatch. It does not reopen V9, reuse its checkpoint, or
  repeat the old global-query decoder.
- The new composition is narrow: directly bilinear-sample the 25 registered 3D
  support projections into V9's projected token map; form a content-derived
  masked mean; then apply V9's unchanged four-head Q/K/V/O attention only over
  those 25 samples. There is no learned query table, global token search,
  Gaussian bias, ray/depth target, ordinal head, pose input, or new loss.

## Frozen V10 mechanism

- Construct from the same clean V4 source architecture, accepted N320
  encoder-only initialization, sweep masks, and fresh-component seeds. Do not
  read any V4–V9 experiment checkpoint, trace, intermediate tensor, or model
  output.
- Preserve 112x112 RGB, patch size 7, the 16x16 final ViT token lattice,
  encoder width/depth/heads `192/6/6`, 64-channel 64x64 BEV, residual-local
  semantic decoder, predictor, survival head, EMA target, and fixed camera.
- Replace only the V9 support geometry and its center-dependent aggregation.
  For every 0.10 m BEV cell, use horizontal points in this order:
  `(0,0)`, `(-0.05,-0.05)`, `(-0.05,0.05)`, `(0.05,-0.05)`,
  `(0.05,0.05)` metres relative to the cell center. For each horizontal point,
  use heights in ascending order `(-0.333,-0.133,0.067,0.267,0.467)` metres,
  giving exactly 25 horizontal-major supports.
- Use the fixed level-camera contract: origin `(0.326,0,0.043)` m, zero mount
  RPY, horizontal/vertical FOV `78.323/62.8370386364` degrees, and inclusive
  near plane `0.05` m. Project each support independently. A support is valid
  exactly when its depth and normalized image coordinates lie in the closed
  frustum. Invalid sampling coordinates are exactly `(2,2)`.
- The static cell-valid mask is the OR over all 25 support-valid bits. It must
  contain exactly 2,062 cells and have row-major uint8 SHA-256
  `4ebbafb6d4dd5fb13b96df978abfa7b81bc2f879b2ba6dec2fcda38dec54e60b`.
  The fixed `<=2 m` range contains 1,016 cells, of which exactly 222 are
  cell-volume-valid. The 0.10 m cell square is observation support, not the
  Go2 planning/body footprint and performs no configuration inflation.
- Project final patch tokens from 192 to 64 channels exactly as V9. Bilinearly
  sample all 25 projected positions with `grid_sample`, zero padding, and
  `align_corners=False`. The query/base feature is the arithmetic mean of the
  valid sampled features; invalid samples contribute exact zero before the
  valid-count division.
- Reuse V9's 64-dimensional, four-head Q/K/V/O projections, head width 16,
  initialization seed `20260729`, zero dropout, invalid-support masked
  softmax, and output projection without modification. Add the attention
  output residually to the masked mean.
- Apply the two inherited residual-local refinement blocks. After the initial
  lift and each block, every all-invalid cell is reset to inherited learned
  null evidence. The semantic output mask uses cell-volume validity, not
  ground-center visibility; unsupported cells emit exact inherited UNKNOWN
  logits `(0,-20,-20)`.
- Mirror the mechanism and buffers in the EMA target. Online and target copies
  are exact at initialization; only the online encoder/lift/decoder/predictor
  are optimizer-owned, and the target remains gradient-free.

## Frozen learning, cap, and gates

- Preserve V9/V4's exact Raw-V13 data identities, roles, endpoint order,
  labels, negatives, action vocabulary, input tensorization, schedule prefix,
  batching, optimizer, clipping, EMA momentum, controls, bootstrap, evaluator,
  and thresholds.
- Preserve the exact joint objective `L=S+P+U+R+O`, including the occupied
  auxiliary coefficient `0.5`. No FREE/OCCUPIED BCE replacement, margin,
  mining, new head, threshold loss, or coefficient change is allowed in V10.
- Train the encoder, V10 lift, semantic decoder, predictor, and survival head
  jointly from update one. Every online Q/K/V/O tensor must receive finite
  nonzero gradients by update two; every target gradient count remains zero.
- Execute at most once: exactly 1,000 optimizer/EMA updates, four B=4
  microbatches per update, and 16,000 ordered presentations. The terminal
  update-1000 state is the only decision point. No retry, resume, second seed,
  extension, or checkpoint selection is authorized.
- The unchanged 24-check development gate is conjunctive. A 24/24 pass earns
  only one separately preregistered use of the unchanged physical calibration
  and 2,016-tuple threshold protocol. Physical qualification still requires
  FREE precision `>=0.99`, near-OCCUPIED detection `>=0.95`, useful FREE
  recall `>=0.90`, and near-obstacle exclusion `>=0.95`.
- G2 remains closed unless both stages pass. Held-out and sealed material
  remain unopened.

## Falsification and stopping rule

- Success would show that V9's learned RGB tokens contain usable vertical
  obstacle evidence, but its ground-center-gated image-to-BEV routing prevented
  that evidence from entering the shared JEPA state.
- Failure closes this exact fixed-level 25-point direct-sampling composition,
  attention initialization, loss, seed, schedule, and cap. Do not tune support
  heights, corners, heads, thresholds, or run length. A successor is justified
  only by a new causal diagnosis and improving evidence, not by repeating this
  configuration.
