# Go2 RGB single-frame multi-block masked spatial JEPA V1 preregistration

Date: 2026-07-31

## Decision and claim boundary

V18/V4 falsified the spatial-token delay-line as a joint perception-and-memory
mechanism.  It did not show that the RGB encoder lacks useful spatial signal.
This successor therefore removes learned temporal state, action conditioning,
object-space lifting, and navigation from the experiment.  It asks one smaller
question: can a jointly trained same-image masked JEPA make the existing RGB
encoder predict genuinely missing spatial content without collapsing or
discarding its useful raw representation?

A pass qualifies only a learned perception interface for a later, separately
trained learned-memory/world-model stage.  It does not qualify navigation,
generalization, G2, or any held-out maze.

## Frozen data and accounting

- Training uses the already frozen corrected-H6 V2 `train.jsonl` role:
  16,000 rows, 1,000 scenes, and eight balanced development families.
- Checkpoint selection uses only its scene-disjoint `val.jsonl` role:
  2,048 rows and 150 scenes.  Held-out, sealed, test, probability-calibration,
  and navigation roles remain inaccessible.
- The exact H6 bindings are:
  - train: `.generated/go2_recurrent_h4_rgb_sequence_index_v2_schedule_integrity/train.jsonl`,
    16,000 rows, 10,328,000 bytes, file SHA-256
    `aee2a54cddd849162648f9b8cfd54a0a28a25bd0705b6482e6af7435c85f4d77`;
  - validation: `.generated/go2_recurrent_h4_rgb_sequence_index_v2_schedule_integrity/val.jsonl`,
    2,048 rows, 1,317,888 bytes, file SHA-256
    `83592e2fea5927802881f076a58a9710100bea017d658c1b978ba651369beac6`.
- Each row contributes only `row.current_rgb`, the registered `e2` endpoint.
  No action and no other history or future endpoint is loaded or exposed to the
  model.  The corrected-H6 order is retained exactly.
- A training presentation is one decoded RGB frame used by both the online and
  detached target branches.  Multiple blocks, tokens, encoder calls, controls,
  or evaluation views are not additional training presentations.
- Four consecutive microbatches of four give an effective batch of 16 and one
  optimizer/EMA update.  The maximum is 1,000 updates and 16,000 training
  presentations: update 250 = 4,000, update 500 = 8,000, update 750 = 12,000,
  and update 1,000 = 16,000.
- The RGB-only place panel is bound by
  `.generated/go2_memory_role_place_triplet_index_v1/manifest.json`, 42,308
  bytes, file SHA-256
  `a5997d93838419cabaaf8e262db70ed51f6f928195f1a312cadc4768f74ca6ca`,
  and its exact 320-row checkpoint-selection index
  `.generated/go2_memory_role_place_triplet_index_v1/checkpoint_selection.jsonl`,
  473,508 bytes, file SHA-256
  `a628a1047b6f15223a4fd7d30c5c87fa1914efef0955d70d9bd2f5330c77dcb0`.
  Its selection proof is discarded by the existing loader and never reaches
  the model.  Place is only a lightweight relevance/retention gate; temporal,
  action, and navigation metrics remain outside this perception experiment.

## Initialization

- The online encoder is initialized only from the reviewed N320 encoder state:
  `VisionEncoder`, 112x112 RGB, non-overlapping 7x7 patch embedding, 16x16 =
  256 spatial tokens, width 192, six blocks, six heads, MLP ratio four, and
  dropout zero.
- Before deserialization, the runtime must validate the N320 gate at
  `.generated/go2_observable_camera_ray_fit_v4/n320_compute_scaled_v1/gate.json`,
  7,960 bytes, file SHA-256
  `4943b4060e88296503c09fc714e55e40fd762527cfccb70a3a341f0df800efe6`
  and content SHA-256
  `76ce5ab703560d171f7c84684b90eed18e8b4cdcc2d8ed3eff6d48496f4de67b`.
  The only tensor input is
  `.generated/go2_observable_camera_ray_fit_v4/n320_compute_scaled_v1/checkpoint.pt`,
  13,777,100 bytes, file SHA-256
  `ece874b53941e841fffc61b724a86d4383b881549afa453b746dd5d68aba11b0`
  and content SHA-256
  `9dcca536943f89acfd7d463fdab591e19a030ef3dc8f3f19a050b1b10025fc2b`.
- No evidence head, predictor, optimizer, rejected checkpoint, V18 state, or
  predecessor runtime tensor is migrated.  The online encoder remains fully
  trainable, so this is a joint JEPA adaptation rather than a separately
  trained predictor on a frozen encoder.
- Encoder construction is RNG-neutral before strict N320 loading.  New
  predictor parameters are initialized once from seed `20260731` in their
  registered construction order.
- The target encoder is a hard copy of the online encoder at update zero,
  permanently frozen and in evaluation mode, then updated exactly once after
  every successful optimizer step with EMA momentum 0.996.

## Genuine spatial masking

- H6 training/validation RGB uses the existing reviewed V27 decoder exactly:
  require a 224x224 RGB PNG; crop `(left=0, top=28, right=224, bottom=196)` to
  224x168; bilinear-resize with Pillow `Image.Resampling.BILINEAR` to 112x112;
  convert uint8 to float32 in `[0,1]`; then channel-normalize with ImageNet
  mean `(0.485,0.456,0.406)` and standard deviation
  `(0.229,0.224,0.225)`.  Place RGB uses its existing reviewed loader exactly:
  require a 224x168 RGB PNG, perform no crop, then apply the same bilinear
  resize, float conversion, mean, and standard deviation.  No random crop,
  flip, color transform, blur, mix, or other augmentation is permitted.
- The model input is this fixed-normalized RGB only.  It receives no actions,
  sequence axis, pose, depth, geometry, labels, scene identity, memory state,
  or navigation outcome.
- Each presentation has exactly four disjoint 4x4 target blocks, one within
  each 8x8 quadrant of the 16x16 patch grid.  Quadrants are ordered
  `(top-left, top-right, bottom-left, bottom-right)` with bases
  `((0,0),(0,8),(8,0),(8,8))`.
- For each role (`train` or `val`), exact zero-based row index, and quadrant
  index, form the ASCII byte string
  `lewm_go2_rgb_single_frame_multiblock_masked_spatial_jepa_v1|mask|20260801|<role>|<row_index>|<quadrant_index>`.
  SHA-256 it; interpret digest bytes 0:4 and 4:8 as unsigned big-endian
  integers; their remainders modulo five are respectively the inclusive row
  and column offsets in `[0,4]`.  Enumerate each 4x4 block row-major, sort the
  64 target flat indices ascending, and define the 192 visible indices as the
  sorted complement in `[0,255]`.  This fixed rule is the complete mask
  schedule; it never depends on RGB, scene, action, update, or RNG state.
- The 7x7 patch kernel equals its stride.  Online patch embeddings at the 64
  target positions are discarded before the first online transformer block.
  Only CLS plus the 192 gathered visible embeddings and positions enter online
  self-attention.  Masking after a full encoder forward is forbidden.
- The EMA target alone encodes the full same RGB frame.  Only its 64 target
  positions are gathered and stop-gradient normalized for the loss.

## Predictor and objective

- The predictor owns a trainable 256x192 positional table initialized by
  copying the N320 spatial positional table, two width-192 `ViTBlock`s with
  six heads and MLP ratio two, LayerNorm, and one 192-to-192 output projection.
  Before its first block, scatter the 192 encoded visible tokens into their
  original row-major slots of a 256-token sequence and the learned mask token
  into each of the 64 target slots, then add the predictor position for each
  original slot.  After both blocks and normalization, gather the same sorted
  64 target slots and apply the output projection.  Append-order assembly or a
  predictor sequence in any order other than original row-major spatial order
  is forbidden.
- The sole training objective is mean half squared distance between L2-
  normalized predicted tokens and stop-gradient normalized EMA target tokens.
  The loss is averaged over rows and target tokens.
- All online encoder and predictor parameters are jointly optimized.  The
  target encoder is absent from the optimizer and must have zero gradient
  tensors.  There is no reconstruction decoder, whitening/SIGReg loss,
  contrastive training negative, action loss, temporal loss, place head,
  physical head, or object-space auxiliary.
- AdamW uses encoder learning rate `1e-4`, predictor learning rate `3e-4`,
  betas `(0.9, 0.999)`, epsilon `1e-8`, weight decay `1e-4`, float32, no
  autocast, and a global online gradient-norm clip of 1.0.  Both learning
  rates are constant for the complete attempt; there is no scheduler, warmup,
  decay, or adaptive learning-rate rule.

## Fixed observations and controls

Observations occur at updates 0, 250, 500, 750, and 1,000.  The complete
2,048-row validation role is streamed in fixed order.  Evaluation masks use
the same stateless rule with a disjoint identity namespace and never enter
training.

For every row, the frozen evaluator records normalized-token energy for:

1. `correct`: its prediction against same-RGB EMA targets;
2. `wrong_target`: the same prediction against same-position EMA targets from
   a deterministic different-scene row in the same family;
3. `wrong_context`: a prediction from that different-scene RGB with the
   original mask against the original EMA target; and
4. `position_mean`: the original prediction against the fixed validation
   cohort's per-position mean normalized EMA target.

The wrong-image donor for validation row `i` is selected metadata-only from
rows with the same family, a different scene, and a different `current_rgb`.
Choose the candidate `j` minimizing
`(((j-i) mod 2048), j)`, with zero modular offset forbidden.  Absence of an
eligible donor is terminal.  The same frozen donor mapping is used for
`wrong_target` and `wrong_context` at every observation.

All reported population means first average rows within each scene, then
scenes equally within each family, then the eight families equally.  The
primary ratio is `macro_mean(correct) / max(macro_mean(control), 1e-12)`;
a nonpositive control mean is terminal.  Paired row advantages are
`control - correct`, summarized using the same scene/family equalization.
For uncertainty, independently for each family resample its scene means with
replacement to the same scene count, average within family, then average the
eight families.  Use 2,000 replicates from a CPU `torch.Generator`, seeds
`20260802`, `20260803`, and `20260804` for wrong-target, wrong-context, and
position-mean respectively, sort ascending, and take zero-based element 50
(`int(0.025 * 2000)`) as the lower 95% bound.  Seeds and resampling are
identical across observation updates.

Representation health is measured on full-frame raw spatial tokens from both
the online and EMA encoders, not on a projector: per-position-centered
effective rank, cross-sample variance, and within-image spatial diversity.
For raw tensor `x[N,256,192]`, let `mu[p,d] = mean_n(x[n,p,d])`.
Cross-sample variance is the population mean of
`(x[n,p,d]-mu[p,d])^2`; within-image diversity is the population mean of
`(x[n,p,d]-mean_p(x[n,p,d]))^2`.  Effective rank uses the 192x192 covariance
of all `N*256` rows of `x-mu`, divides by `N*256-1`, clamps eigenvalues at
zero, and returns `exp(-sum(q*log(max(q,1e-12))))` for eigenvalue proportions
`q`; a zero total eigenvalue mass has rank zero.  Values are reported as
absolute measurements and update-zero retention.

The place gate uses a normalized flattened full spatial-token key from
the online anchor and EMA positive/negative/candidate images.  It reports
positive-versus-negative advantage, bootstrap bound, within-scene R@5 versus
exact chance, scenes above chance, and update-zero retention.  No fitted head
or place loss is permitted.  Its bootstrap uses the same scene/family method,
except that the one-scene-per-family place panel resamples rows within each
family to its original row count before the eight family means are averaged.
It uses 2,000 replicates, seed `20260805`, and sorted element 50 at every
update.

## Gates and continuation

- Update zero is an integrity/baseline observation only.  It must prove exact
  masks and shapes, hard-synced target state, target freeze/eval/zero-grad,
  optimizer exclusion, finite nonzero tensors, train/validation scene and RGB
  disjointness, registered access only, and zero held-out/sealed access.
- Any nonfinite value, target mutation outside the EMA rule, accounting error,
  unauthorized input, or target gradient is terminal.
- A catastrophic representation collapse is terminal at update 250 or later:
  either online or target per-position-centered effective rank, cross-sample
  variance, or spatial diversity below 25% of its update-zero value.
- Subject to integrity and health, update 250 always proceeds to update 500.
  At update 500 or 750, `improves` means the maximum of the three registered
  primary control ratios decreases by at least 0.001 absolute from the
  preceding observation, or the minimum positive-family count across those
  controls increases.  Continue if it improves, or if all three ratios are
  already below 1.0 with at least four positive families each.  Otherwise stop
  rather than spend the remaining cap on a mechanism with no useful
  separation.
- A checkpoint is perception-qualified only if it passes every integrity and
  catastrophic-health gate; correct/wrong-target, correct/wrong-context, and
  correct/position-mean ratios are each at most 0.90; all three paired
  bootstrap lower 95% advantages are strictly positive; all three controls
  have at least six of eight positive families; and online and target raw
  rank, variance, and diversity each retain at least 50% of update zero.
- The same checkpoint must retain navigation-relevant place substrate: mean
  scene-equal R@5/chance multiple at least 80% of update zero, R@5 at least
  1.5 times exact chance in at least six of the eight place scenes, and EMA
  target place-key effective rank at least 2.0 and at least 80% of its
  update-zero value.  These are deliberately light perception floors, not a
  demand that a single frame solve maze navigation.
- Among qualifying observed checkpoints, select the lowest maximum primary
  ratio, then lower correct energy, then the earlier update, using immutable
  metric sidecars only.  A snapshot is not itself evidence of qualification.

## Checkpoints, attempt policy, and next authority

- Immutable complete continuation snapshots and metric sidecars are written
  at updates 250, 500, 750, and 1,000.  They are never reopened by observers
  or by same-attempt selection.
- Exactly one fresh attempt is permitted after source review and narrow-export
  certification.  There is no automatic retry, alternate seed, resume,
  schedule extension, or science-identical replacement.  Mechanical failures
  found before reservation are fixed normally; a scientific miss closes V1.
- A pass permits only a separately preregistered learned-memory/world-model
  integration experiment on development data.  It does not authorize G2,
  navigation, probability calibration, held-out/sealed access, production,
  promotion, or deployment.
