# Go2 categorical-radial N32 V4 explicit-hierarchy binding

Date: 2026-07-11

Status: frozen before V4 implementation and before any V4 dataset-backed model
output.

## License and question

The exposure-matched width-24 N32 V2 and width-32 N32 V3 candidates both
failed the immutable all-family fit gate:

- V2 result file/content SHA-256:
  `0a5f8a822d7fec8287a30103125fca1a4927f0413e2f0906db431cef54ec2265` /
  `e070cc96d69b76e1f85f533fa1d94221225963a2b66a491f0c2a867c008b97ef`;
- V3 result file/content SHA-256:
  `0f3eb212afe54a38d7a81a1fc51ca544dfab667a94a836be742d3ea3e2298d85` /
  `ec8dd8450fb34bee3a5ba1c5a5b532339d281241560c8ed9ac07a48d2c2bea4e`.

V2's conditional FREE/OCCUPIED confusion was effectively solved while its
remaining errors were principally KNOWN versus UNKNOWN. A common KNOWN bias
cannot satisfy rough UNKNOWN and open-field OCCUPIED recall simultaneously;
the proof SHA-256 is
`e214bb80bcccf9ae5051231d90f7a5d8c2bfa33ca799e7db3eb969698fa2108a`.

The preregistered 320-frame camera-pose audit then rejected dynamic projection
as the next change. Its result file/content SHA-256 values are
`2c7efba897054ea0067db58f020e70dc5f3c5804785c74cbda4a8b76e0210b9d` /
`6a9d05a0fb92289334cf39bb6947a2022a05a7c1892e8bb1c5a7156f9ca227f4`;
its immutable report SHA-256 is
`e1a0c7e8c161827c5d8a1e2088135d8d986cbce9f9f7c02aa43d78d37a0be5e8`.
Rough median per-frame p50 displacement was `0.25533` token and
rough-minus-pooled-nonrough was `-0.02611`, so neither ordering threshold
passed.

V4 therefore tests one question: does an explicit probability factorization
remove the final-head gradient coupling between KNOWN/UNKNOWN and
FREE/OCCUPIED-given-KNOWN at width 24?

## Sole model intervention

The causal base is `CategoricalRadialPerceptionFullRay` from N32 V2. Preserve:

- input 112 x 112, patch 7, ordered 16 x 16 tokens;
- encoder width/depth/heads `192/6/6`;
- token feature width 24 and context width 64;
- five registered vertical anchors and fixed level-camera projection;
- 64 x 256 body-polar lattice and immutable Cartesian factorization;
- angular context and full-ray dilations `(1,2,4,8,16,32)`;
- UNKNOWN/FREE/OCCUPIED class order and support mask.

Replace only `polar_head: Conv2d(64,3,1)` with
`polar_head: Conv2d(64,2,1)`. The raw channel order is:

1. `k`: log odds `P(KNOWN) / P(UNKNOWN)`;
2. `o`: log odds `P(OCCUPIED|KNOWN) / P(FREE|KNOWN)`.

For every polar cell, compute before Cartesian gathering:

```text
log_unknown  = logsigmoid(-k)
log_known    = logsigmoid(k)
log_free     = log_known + logsigmoid(-o)
log_occupied = log_known + logsigmoid(o)
joint        = stack(log_unknown, log_free, log_occupied)
joint        = joint - logsumexp(joint, class_dim)
```

Use stable `logsigmoid`; `log(sigmoid(...))` is forbidden. The raw-factor API
returns finite `[B,2,64,256]`. `polar_logits()` returns normalized finite
three-class log probabilities `[B,3,64,256]`. Conversion occurs before the
unchanged Cartesian gather. Public `forward()` and `occupancy_logits()` retain
shape `[B,3,64,64]`; supported cells are normalized log probabilities and
unsupported cells retain the existing deterministic UNKNOWN sentinel.

The registered parameter count is `2,887,002`, exactly 65 fewer than V2's
`2,887,067`. The state key set remains 133 entries. Only two shapes change:

- `polar_head.weight`: `[3,64,1,1] -> [2,64,1,1]`;
- `polar_head.bias`: `[3] -> [2]`.

All 131 same-key/same-shape entries must be bit-identical to the
seed-corresponding untrained V2 initialization. Three softmax logits and two
hierarchical factors each have two probability degrees of freedom per cell,
but their affine feature-to-probability function families differ; V4 does not
claim initial-output or fixed-feature functional equivalence.

## Initialization

For each registered seed:

1. configure the unchanged deterministic/best-effort execution state;
2. save the CPU RNG state;
3. instantiate an untrained V2 width-24 reference;
4. restore the saved state and instantiate V4;
5. copy all 131 same-key/same-shape V2 entries into V4;
6. leave only the two shape-changed V4 head tensors at their deterministic
   PyTorch default initialization.

Do not apply class-prior bias matching, a V2-head analytic transform, zero
initialization, trained V2/V3 weights, or any other prior-matching scheme. The
implementation manifest must freeze construction order, both seed-specific V2
reference and V4 state hashes, both schedule hashes, and the complete state
comparison before dataset-backed V4 output.

## Loss and metrics

Reuse the existing V2 `direct_hierarchical_loss` unchanged on the reconstructed
three joint log probabilities. Do not substitute a new direct-BCE trainer.
Algebraically:

- `logsumexp(log_free,log_occupied) - log_unknown == k`;
- `log_occupied - log_free == o`;
- the UNKNOWN/KNOWN loss has zero gradient to `o`;
- the conditional FREE/OCCUPIED loss has zero gradient to `k`.

Floating-point cross-gradients must be bounded by a frozen test tolerance. Both
losses still update the shared context and encoder; V4 claims separation only
at the two output channels.

The existing float64 evaluator consumes the joint log probabilities as logits.
Its log-softmax is idempotent, so joint/hierarchical NLL, confusion matrices,
class recall/precision, average precision, distance-bin recall, MAP/tie order,
and wrong-view controls remain semantically unchanged. N32 V4 performs no
calibration, threshold fitting, vector scaling, or factor scaling.

## Data, schedule, and controls

Copy N32 V2 exactly:

- physical-dataset-role train fit frames: 320;
- direct FP32 training batch: 80, no microbatching or accumulation;
- four batches per epoch, 2,000 updates, 500 epochs;
- 160,000 frame presentations, 500 per fit frame;
- AdamW, weight decay `1e-4`, betas `(0.9,0.999)`, epsilon `1e-8`, no AMSGrad;
- global gradient clip 1.0 once before each optimizer step;
- one-indexed cosine learning rate from `2e-4` at update 1 to `1e-5` at update
  2,000, no warmup;
- no AMP/autocast, compilation, quantization, augmentation, EMA, averaging,
  early stopping, retry, or best-step selection;
- evaluation every 100 updates, with the terminal decision fixed at updates
  1,800, 1,900, and 2,000;
- identical seed-specific full-epoch batch schedules and role-global/same-scene
  wrong-view controls.

The only registered seeds are `20260710` and `20260711`. Seed 20260711 remains
forbidden unless the immutable seed-20260710 V4 result passes fit and both
train-role holdout decisions and its precommitted file hash is supplied before
seed-11 device/model construction.

Current physical-dataset role governs access. Legacy rollout split is recorded
provenance only and cannot filter, rank, calibrate, or select V4 rows. Before an
exact terminal fit pass, the runner may open only the 320 fit images and 20 fit
label shards. It must open zero holdout, checkpoint-selection,
probability-calibration, physical-nontrain, G2, or sealed payload/model output.

Both wrong-view conditions remain mandatory:

- role-global shuffled RGB;
- same-scene wrong-view RGB.

## Gates and conditional access

At each of updates 1,800, 1,900, and 2,000, the aggregate and all five family
reports must simultaneously pass every unchanged gate:

- hierarchical balanced NLL `<=0.03`;
- UNKNOWN/KNOWN balanced accuracy `>=0.99`;
- FREE/OCCUPIED-given-KNOWN balanced accuracy `>=0.99`;
- UNKNOWN, FREE, and OCCUPIED recall each `>=0.98`;
- FREE recall `>=0.95` in every supported registered distance bin from 1 m;
- each wrong-view-minus-correct NLL delta `>=0.25`.

Earlier passes cannot stop training or qualify the stage. Any failed terminal
evaluation makes the seed unfavorable and requires zero same-scene/cross-scene
holdout byte opens and model outputs.

Only an exact terminal fit pass authorizes one in-memory evaluation of both
committed physical-train-role holdouts. Preserve the immutable failed patch-7
comparator, family-macro arithmetic, class-regression limits, and strict same-
scene/cross-scene improvement rules from N32 V2. One fully favorable seed
authorizes only seed 20260711. Only two favorable seeds can license shared-JEPA
candidate construction; V4 cannot pass G2 or license calibration/runtime.

## Smoke, outputs, and finalizer

The three-update seed-20260710 smoke is nonauthoritative, nonaggregatable, uses
a distinct schema/path, and is unconditionally fit-only even if synthetic
three-step metrics pass. It cannot occupy either canonical result path.

Authoritative exclusive outputs are:

- `.generated/go2_categorical_radial_n32/v4/seed_20260710_result.json`;
- `.generated/go2_categorical_radial_n32/v4/seed_20260711_result.json`.

Add new immutable model, pure-decision, runner, finalizer, result, smoke, stage,
and two-seed schemas. Historical V2/V3 model/runner/finalizer sources must not
be edited because their result provenance binds those hashes.

The pure finalizer must remain torch-free and independently validate source and
input hashes, strict JSON types/domains, factor semantics, parameter/state
metadata, both initial states, schedules, controls, terminal gates, conditional
access, holdout decisions, seed-11 authorization, and exclusive immutable
evidence. Only two precommitted favorable results may emit
`categorical_radial_full_train_candidate_licensed=true`.

## Required pre-run evidence

Before the authoritative seed-20260710 command, freeze:

- new model, pure decision, runner, finalizer, test, and complete transitive
  source hashes;
- exact parameter/state counts, the two changed shapes, and 131 bit-identical
  initial state entries for both seeds;
- both V2-reference and V4 initial-state hashes and both schedule hashes;
- factor order/formula hash, class order, output semantics, geometry,
  factorization, support, and full-ray reachability;
- proof that supported joint log probabilities normalize and remain finite for
  factor logits through at least `+/-1e4`;
- arbitrary three-class probability round-trip through `k` and `o`;
- unchanged-loss versus direct weighted binary reference values/gradients,
  including no-known-mask and cross-gradient cases;
- equivalent-score evaluator parity, Cartesian support/sentinel behavior,
  encoder/context/both-factor gradients, and one encoder call per image;
- exact schedule/gates, causal controls, fit-only smoke access, conditional
  holdout non-access, seed ordering, finalizer tamper rejection, and immutable
  output-path checks.

## Failure interpretation

- Fit failure after this exact run rejects final-head factorization as
  sufficient; it does not license more exposure, calibration, a width retry, or
  post-hoc family thresholds.
- Fit pass plus holdout failure licenses a scene/view generalization change,
  not selection on holdout values.
- One favorable seed licenses only the registered replication seed.
- Two favorable seeds license shared-JEPA construction, not G2 or G3.

No checkpoint-selection, probability-calibration, physical-nontrain, G2,
sealed, shared-JEPA, or runtime payload/model output may be opened by this
diagnostic.
