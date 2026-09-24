# RGB Swept-Progress Survival Joint-JEPA V8 — Native-Aspect High-Resolution ViT Preregistration

- Status: frozen before V8 implementation, source binding, or execution.
- Clean scientific baseline: V4 source / full-arm result commits
  `aaa47a138d0eeb78aa20d9524e67f813f7a74a41` /
  `8b3a8063b087c81030189deadc6c5f6e1c7d44c3`.
- Latest closed mechanism: V7 result commit
  `6ddf34eab09c1b54512156b04fbe0ac34b0b3195`.
- Purpose: test exactly one materially different perception mechanism inside
  the jointly trained RGB-only JEPA, capped at 1,000 updates and 16,000
  presentations. It is not a new dataset, loss, predictor, decoder, threshold,
  schedule, seed, control, or separately trained encoder experiment.

## Evidence and hypothesis

- V4 passed all 24 development checks, but no tuple in its unchanged 2,016-tuple
  physical calibration passed. Its selection near-obstacle detection was only
  `0.26853`, despite free precision `0.92923` and useful-free recall `0.85067`.
- V6's late fine-RGB residual branch trained but passed only 23/24 checks and
  did not materially improve the loss trajectory or physical separability.
- V7's fresh hierarchical CNN passed only 19/24 checks. It lost much of V4's
  rough-obstacle and action-conditional performance, so CNN widths, depths,
  schedules, seeds, and CNN/ViT hybrids are closed.
- Every prior relevant transformer path still processed a `16x16` spatial
  lattice: overlap V1 changed the patch kernel, multiresolution decoded after
  the transformer, V6 added a late branch, and older 224-pixel encoders used
  patch 14. No committed run gave all six accepted V4 transformer blocks a
  native `24x32` patch-7 lattice.
- V8 tests whether the main remaining perception bottleneck is spatial detail
  discarded before global attention. It preserves V4's strongest learned
  representation and joint action-JEPA behavior while tripling real spatial
  tokens from 256 to 768.

## Pre-preregistration feasibility access

- One already-authorized TRAIN RGB was opened only to establish the raw camera
  dimensions before choosing this mechanism. No label, model output, metric,
  evaluation-role image, selection image, calibration image, checkpoint,
  held-out material, or sealed material was opened.
- Endpoint identity:
  `000183c9287ab160a5a00cea0a7fd30d59d21776c9bdca69e9f1aa5c0e11c48f`.
- Relative path:
  `.generated/go2_render_selected_v04/scenes/scene_703b25447899b393/rgb/frame_005313_env_33.png`.
- Role `train`; PNG mode `RGB`; Pillow size `(224,168)`; `10,621` bytes; file
  SHA-256 `edfa33c29513ec80a2e24d20cb35e15aeec1b9b64e571e6b989bb9fc5ddd0ce2`.
  The endpoint index was opened. No access timestamp or separate ledger ID was
  recorded, and none is inferred here.
- A data-free synthetic R9700 feasibility benchmark used random tensors only.
  Six V4 transformer blocks at batch four and 769 tokens, including four
  forwards and backward, measured about `0.0768 s` and `713 MB` peak. This
  supports the unchanged microbatch size without mixed precision,
  checkpointing, or schedule changes.

## Sole scientific delta from clean V4

- Decode the same bound source RGB bytes at their native `224x168` resolution.
  Do not resize, crop, pad, upscale, augment, or change ImageNet normalization.
  The loader must require Pillow size `(224,168)` and return exact float32
  `[3,168,224]`. Preserve roles, identities, hashes, caching, ledgers, access
  counters, labels, and wrong-RGB mappings.
- Retain V4's accepted N320 patch projection, learned CLS token, all six global
  transformer blocks, final normalization, width 192, six heads, MLP ratio
  four, patch size seven, and dropout zero exactly.
- Native patching yields height 24 by width 32: 768 row-major spatial tokens
  plus CLS, with output shape `[B,769,192]`.
- Preserve every accepted encoder tensor bit-exactly except the spatial
  positional embedding. Copy the CLS position exactly. Reshape the 256 spatial
  positions to `[1,192,16,16]`, bicubically interpolate once on CPU float32 to
  `[1,192,24,32]` with `align_corners=False` and `antialias=False`, flatten in
  row-major order, and install the result as the trainable spatial positions.
  Add no other initialization or random draw.
- The resulting online encoder has exactly `2,845,824` trainable parameters,
  `98,304` more than V4 solely because of the additional positional entries.
- Adapt only the input lattice of the inherited deformable BEV lift. Its token
  map becomes `[B,192,24,32]`; every parameter, projective anchor, field of
  view, four-sample topology, raw offset, weight, null evidence tensor, and
  refinement tensor remains exact V4 state.
- Preserve the exact V4 normalized sampling displacement and arithmetic for
  the proposed grid:
  `legacy_offsets = 2*tanh(raw_offsets)` followed by
  `normalized_offsets = legacy_offsets*(2/16)`. Separately receipt equivalent
  native token-cell offsets as `tanh(raw_offsets)*[4,3]`, ordered x/width then
  y/height. This retains the same normalized search radius and validity masks;
  it does not shrink the search field with the denser rectangular grid.
- Construct a clean V4 model first, replace the online encoder and lift as
  above, and create exact frozen target copies. Retain exactly one initial hard
  sync, the unchanged `0.996` EMA, and zero initial EMA updates.
- Preserve the complete V4 residual-local semantic decoder, BEV output,
  visibility handling, action-conditioned predictor, swept-progress survival
  head, action vocabulary, and all other model behavior.

Native `224x168` deliberately combines increased token density with removal of
V4's square aspect warp. Square `224x224` would isolate density more narrowly,
but would vertically resample non-existent pixels and cost substantially more.
For the repository's physical-navigation goal and one-shot budget, native
camera geometry is preferred. A failed V8 will not be followed by the square
variant.

## Frozen joint training and falsification

- Inherit V4's exact development data and roles, labels, schedule and order,
  batch construction, optimizer groups and hyperparameters, clipping, masks,
  losses, coefficients, seeds, EMA cadence, evaluator, controls, bootstrap,
  metrics, and all 24 gate thresholds.
- Jointly train the online high-resolution encoder, unchanged lift parameters,
  semantic decoder, action predictor, and survival head from update one under
  the exact V4 `S + P + U + R + O` objective, with `O=0.5`. There is no frozen
  encoder phase or separately trained predictor/readout.
- Execute exactly one fresh terminal run: 1,000 optimizer and EMA updates,
  four size-four microbatches and backward graphs per update, and exactly
  16,000 presentations. There is no retry, resume, extension, alternate seed,
  intermediate selection, square follow-up, or positional/interpolation/
  resolution variant.
- Before execution, bind and independently review the exact model, native
  loader adapter, training wrapper, executor, and focused tests. Tests must
  cover exact encoder migration, deterministic position interpolation,
  `24x32` row-major tokens, V4-identical normalized sampling grids and masks,
  target copy/freeze/EMA, optimizer partitions, online encoder gradients,
  native-loader no-resize pixel identity, and unchanged V4 nonreplacement
  state.
- Evaluate only terminal update 1000. The terminal model must pass all 24
  unchanged V4 full-arm development checks. Any failure closes native
  high-resolution ViT without checkpoint use, physical calibration, or G2.
- If and only if all 24 checks pass, stage one separately source-frozen run of
  the unchanged V4 physical-evidence calibration: the same four-parameter fit,
  same 2,016 tuples, and same thresholds. Physical failure closes this family.

## Authority and stopping

- This preregistration authorizes only implementation, source review, one
  source-frozen capped V8 development run, and—conditionally on 24/24—one fresh
  unchanged physical calibration.
- It grants no retry, resume, replacement attempt, longer training, checkpoint
  reuse, architecture sweep, G2, navigation, held-out, sealed, production,
  deployment, or promotion access.
- No rejected V4, V5, V6, or V7 checkpoint/runtime may be opened. A physical
  pass would authorize only preparation and review of a separate G2 binding;
  it would not itself open G2 or qualify navigation.
