# Go2 categorical radial N32 execution binding

Date preregistered: 2026-07-10 22:49 BST

Status: active; written before any categorical-radial N32 model output

## Purpose

This document binds the executable interpretation of the N32 fit-panel and
scene-disjoint holdout stage already registered by the categorical-radial
micro-fit protocol. It resolves frame-versus-transition batching, the negative
patch7 comparator, conditional holdout access, wrong-view construction, and
the two-seed decision without changing any registered threshold.

This remains a train-role diagnostic. A pass licenses construction of a full
training candidate only after two seeds agree. It does not pass G2, select or
calibrate a checkpoint, authorize a non-train read, or support a JEPA, memory,
exploration, or navigation claim.

## Immutable inputs

- frozen train-only panel:
  `.generated/go2_physical_micro_overfit/patch7_v1/panel.json`;
- panel file/content SHA-256:
  `c3f44c6b1147efbb6a5fbc2294c6431c72e25da877cab6884972d25c1ffdb16c` /
  `f3e5198b81ac48c06f6c8e4b21e8bf24d62200e3830b1d6685d949a668349d5f`;
- fit/same-scene/cross-scene row SHA-256:
  `5a75e202c8f7a803aafaad093c7f474137dd2e69f50ecdb7fb4e97765afb659d` /
  `d32713086c042d20f94825aa362c27a07bef6fd0e0cce0aa5846bb67bf8dc465` /
  `3565f7f7844f3aeee28b0433aa6dc77d553a9ebb831cf9af20b6d392c5416817`;
- categorical-radial ladder manifest:
  `.generated/go2_categorical_radial_micro_overfit/v1/ladder_manifest.json`;
- ladder manifest file/content SHA-256:
  `967812399045b29e8be316f2f87bc16f02d681b0ea01884513c6b4f29bbe4b12` /
  `00a3ad1263af16e3b858f7e7522df7b108a49301d25fa805148e82b36cb52f8e`;
- authoritative full-ray V3 ladder result:
  `.generated/go2_categorical_radial_micro_overfit/v3/seed_20260710_ladder_result.json`;
- V3 result file/content SHA-256:
  `7a5f67bacb2e3df67421bcff13b15d1fa3e00d99f3b2af52c52b0b6ce14617a8` /
  `517313139077027176c471f829f57148684d3df0def6096ce7702d3bbba46ce1`;
- immutable negative patch7 reference:
  `.generated/go2_physical_micro_overfit/patch7_v1/seed_20260710_result.json`;
- patch7 result file/content SHA-256:
  `6e2aacd18fe1d692fb6ad682b41132563dcbcdb95c7b7ce719f407baf6c91a8c` /
  `32d848d3df68e670ddb4cc24436981f62a1aa5562b89e6d6719ecb113f66b749`;
- categorical-radial protocol file SHA-256:
  `ef23ee607d0976d67adf33591f5af78652da4305811a563d94bd8539abc9d404`.

The runner must hash every evidence JSON/document input before deserialization,
recompute each content hash, validate its schema and decision, and rehash all
evidence inputs and sources after execution. This does not permit eager hashing
of every image/shard payload path referenced by the panel: fit payloads are
verified before fit access and holdout payloads only after a fit pass authorizes
that panel. It must not require today's mutable execution-ledger hash to equal
the historical hash embedded in the panel. The panel itself and its embedded
historical provenance remain immutable and are validated as a unit.

The ladder manifest's stored exact 960-frame roundtrip, zero outside-support
count, and zero mapping-collision audit must validate without reopening panel
labels. The immutable patch7 result's already-finalized all-panel
`post_selection_support_audit` must also validate exactly. The runner may
recompute fit support as fit labels become authorized, but it must not pre-open
a holdout label to repeat that audit.

## N32 unit and order

`N32` means 32 frozen transitions per family per panel. With five families,
the fit panel contains 160 transitions and 320 unique endpoint frames. Each
same-scene and cross-scene holdout also contains 160 transitions and 320
endpoint frames. This is not a 32-frame experiment.

The categorical head is a single-frame model. Training therefore uses batch
size 4 **frames**, not four paired transitions/eight frames from the legacy
JEPA runner. Canonical frames are produced by the frozen `frame_records`
helper: expand every row in panel order to `current` then `next`, then sort by
`(global_row, side)`. Current and next endpoints are independently shuffled
after expansion and do not remain paired.

Each stage uses a seed-local `torch.Generator` and the frozen ladder
`_next_batch_indices` rule. When the remaining order cannot provide four
indices, append one `torch.randperm(320)` and consume its first four. Since 320
is divisible by four, this is exactly 80 batches per epoch. The faithful stage
consumes 25 epochs; the ceiling consumes 62.5. A ceiling restart recreates the
generator from the same seed, so its first 2,000 minibatch identities equal
the faithful stage's.

## Model and loss

The candidate is exactly `CategoricalRadialPerceptionFullRay`, with 2,887,067
parameters and dilations `(1, 2, 4, 8, 16, 32)`. Each optimizer branch restarts
from the seed-specific initial state, never the N=16 final weights. For seed
20260710 the runner must reproduce initial-state SHA-256
`8b149b57ae4bb305a2306a4dde2cab5f57a46f1c3760837593ed4d9862491278`.
Faithful and ceiling initial hashes must match within a seed.

Training uses the categorical ladder's direct balanced hierarchical loss with
the unchanged UNKNOWN/known and FREE/OCCUPIED weights. It does not inherit the
legacy JEPA wrapper's extra `2.0 * occupancy_loss` scalar. No JEPA, predictor,
equivariance, action, calibration, or threshold loss is enabled.

Preprocessing is exactly the V3 ladder path: RGB with no crop, PIL bilinear
resize to 112x112, division by 255, then ImageNet mean
`(0.485, 0.456, 0.406)` and standard deviation `(0.229, 0.224, 0.225)`.
Each endpoint reads its own side-specific 64x64 label and supervision mask.
Training and evaluation use FP32 with no autocast, AMP, compilation, or
quantization. The model is in `train()` for optimizer steps and `eval()` under
`torch.no_grad()` for every reported metric.

## Optimizer branches

The original N32 branches remain fixed and do not use the V2/V3 ladder cosine:

1. `production_faithful`: AdamW, constant learning rate `2e-4`, weight decay
   `1e-4`, batch 4 frames, gradient clipping 1.0, 2,000 updates, evaluation
   every 100 updates.
2. `ceiling_optimizer`: run if and only if faithful fails its terminal **fit**
   gate; restart the exact initial state and minibatch generator; AdamW,
   constant learning rate `1e-4`, zero weight decay, batch 4 frames, gradient
   clipping 1.0, 5,000 updates, evaluation every 100 updates.

Every invoked branch consumes its full budget. There is no early stop, retry,
EMA, checkpoint averaging, best-step selection, schedule change, or alternate
batch size. A faithful fit pass followed by a holdout failure does not permit a
ceiling run.

Both branches include every trainable model parameter and use AdamW defaults
`betas=(0.9, 0.999)`, `eps=1e-8`, and `amsgrad=false`. Gradient norm clipping
occurs after backward and immediately before `optimizer.step()`.

## Fit gate and controls

Every fit evaluation reports raw metrics both pooled and independently for all
five families. It reuses the frozen physical accumulator and fit gate:

- balanced hierarchical NLL <= 0.03;
- UNKNOWN/known and FREE/OCCUPIED balanced accuracy >= 0.99;
- UNKNOWN, FREE, and OCCUPIED recall >= 0.98;
- FREE recall >= 0.95 in 1-2 m, 2-3 m, and >=3 m bins;
- role-global and same-scene wrong-view NLL minus correct-RGB NLL >= 0.25.

An evaluation passes only if the aggregate and every family pass every check.
The terminal branch decision requires exactly the final three evaluations:
steps 1,800/1,900/2,000 for faithful or 4,800/4,900/5,000 for ceiling. All
three must pass. Earlier single or three-consecutive passes are diagnostic only.

For every seed and panel namespace, controls are constructed exactly by the
frozen `frame_records`, `attach_role_global_shuffle`, and
`attach_same_scene_wrong_view` helpers. The role-global permutation must have
zero same-image, same-scene, and same-transition pairs. The same-scene control
must have zero same-image, same-transition, and different-scene pairs. Seed
20260711 uses its own seed-20260711 deterministic permutations; it does not
reuse seed-20260710 permutations.

Each helper is called once on the complete canonical 320-frame panel with
namespace exactly `fit`, `same_scene_holdout`, or `cross_scene_holdout`.
Family accumulators slice by the **target** frame family; no family-specific
permutation is constructed. Evaluation uses target batch size 4 and concatenates
correct, role-global, and same-scene inputs into one 12-image model call. The
model call is FP32; the three conditions are accumulated separately by the
frozen physical accumulator, which promotes logits to float64 for metrics.

## Conditional holdouts

No same-scene or cross-scene holdout image, label, or model output may be
opened during a branch that has not passed its terminal fit gate. If faithful
passes, evaluate both holdouts once at faithful step 2,000 and do not run the
ceiling. If faithful fails, expose no faithful holdout output; run the ceiling,
and evaluate holdouts once at ceiling step 5,000 only if its terminal fit gate
passes. If both branches fail fit, holdout model-output access remains zero.

Each authorized holdout runs and records all three correct/role-global/
same-scene conditions through the same evaluator. Holdout adjudication uses
only correct-RGB family metrics; the two controls remain diagnostic and cannot
change the comparison. It does not fit a threshold, select a checkpoint, or
use pooled metrics.

## Immutable patch7 comparator

The comparator is exactly:

```text
stages.production_faithful.patch7_16x16.final_panels[panel]
  .families[family].conditions.correct_rgb
```

from the immutable seed-20260710 patch7 result. This is its final step-2,000
AdamW `2e-4`, weight-decay `1e-4` checkpoint with final-state SHA-256
`fba4e91b333d57a813fb94edb13b215064d03da2830aae9d0ae4b34685cd38c1`.
Both categorical seeds and either qualifying categorical branch compare to
this one reference; the legacy ceiling is never used.

That patch7 checkpoint failed its own fit gate, so it was not a qualifying
common stage in the old patch14-versus-patch7 adjudicator. The categorical
protocol intentionally supersedes that old common-pass requirement by naming
the immutable **faithful patch7 reference** as a negative comparator. Complete
holdout metrics exist because the old runner evaluated its final panels, and
the old finalizer code is capable of validating every stored final panel. The
new runner and finalizer independently validate the loaded result structure and
metrics. This clarification does not promote the failed patch7 model.

For each holdout, compute candidate minus reference at full JSON precision:

- NLL ratio = mean of five candidate family NLLs divided by mean of five
  reference family NLLs; do not average five ratios;
- far-FREE delta = mean of the five family candidate-minus-reference >=3 m
  FREE recalls;
- each class macro delta = mean of five family candidate-minus-reference
  recalls.

Runner and finalizer must call one shared pure arithmetic helper and reduce in
the canonical frozen `FAMILIES` order so their full-precision decisions are
bit-identical.

The candidate must satisfy on both holdouts:

- macro NLL ratio <= 0.80;
- macro far-FREE delta >= +0.10;
- every macro class-recall delta >= -0.01;
- no individual family/class recall delta < -0.01;
- strictly lower NLL and strictly higher far-FREE recall in all 5/5
  cross-scene families and at least 4/5 same-scene families; ties fail.

The reference macro NLL/far-FREE values are `0.3219876256599372` /
`0.46708481911812594` on same-scene and `0.4054638461731662` /
`0.4665871805991353` on cross-scene. These constants are assertions against
the loaded immutable result, not substitutes for recomputation.

## Seed and promotion decision

Seed 20260710 runs first. It is favorable only if one fit branch passes and
both holdout gates pass. Only that result licenses seed 20260711. Every per-seed
artifact uses schema `lewm_go2_categorical_radial_n32_result_v1`, records
`authoritative=true`, `aggregation_eligible=true`, `promotion_eligible=false`,
and sets its own full-train license false. Canonical result paths are
`.generated/go2_categorical_radial_n32/v1/seed_20260710_result.json` and
`.generated/go2_categorical_radial_n32/v1/seed_20260711_result.json`.

The seed-20260711 CLI must receive both `--seed-20260710-result` and
`--expected-seed-20260710-sha256`. Before device/model construction it hashes,
deserializes, fully validates, and independently recomputes a favorable primary
result with common panel, ladder, V3, comparator, model, source, and access
provenance. It rehashes the primary after seed-20260711 execution. Seed 20260710
rejects either authorization argument; seed 20260711 rejects their absence.
The primary result authorizes the run only. It does not force or select the
second seed's optimizer branch.

If seed 20260710 is favorable, seed 20260711 repeats the identical mechanism
and decision. A pure finalizer then receives both immutable result paths and
precommitted file SHA-256 values, hashes before deserialization, recomputes all
terminal aggregate/family fit gates and holdout deltas from stored raw metrics,
validates the complete curve cadence and conditional stage/holdout/access
structure, and rehashes afterward. Neither per-seed input may self-license.
Both seeds must use the same qualifying optimizer stage string
(`production_faithful` or `ceiling_optimizer`) and pass every gate. Only then
may schema `lewm_go2_categorical_radial_n32_two_seed_result_v1` emit
`categorical_radial_full_train_candidate_licensed=true`. Two individually
valid favorable results on different branches produce the valid classification
`two_seed_inconclusive`; branch disagreement is not a parser error. This is not
a G2 or runtime license.

## Non-authoritative smoke

The GPU smoke is seed-20260710-only, batch 4 frames, faithful 3 updates,
ceiling 3 updates, and evaluation interval 1. It uses a distinct
`lewm_go2_categorical_radial_n32_smoke_result_v1` schema and records
`authoritative=false`, `aggregation_eligible=false`, `promotion_eligible=false`,
and full-train license false. The faithful smoke will normally fail its three
fit evaluations, so the ceiling path is exercised from a fresh identical
initial state and minibatch prefix. Conditional holdout rules remain active;
holdout paths are exercised with passing synthetic fixtures in unit tests, not
by weakening the GPU smoke gate. The finalizer rejects every smoke artifact.

## Isolation and implementation gates

Only the frozen panel supplies image and label-shard paths; the runner also
receives the bound evidence JSON/documents named above. It must not open the
global row index, checkpoint-selection, probability-calibration, G2, or any
other non-train artifact. It records image, label-shard, and model-output access
by fit and conditional holdout panel. Source NPZ files may materialize
unselected train rows at archive level; the optimizer indexes only the 320
selected fit frames. The approved global artifact set is 960 train images and
45 train shards: fit 320/20, same-scene 320/20, cross-scene 320/25.

The finalizer requires exact full-curve evaluation steps, fixed budgets,
conditional presence/absence of ceiling and holdouts, event reconciliation,
stored-decision equality with pure recomputation, common source/input/model
mechanism, exact seed set, and atomic immutable writes. Frozen source-bound V1,
V2, and V3 files must not be edited; the N32 contract, runner, finalizer, and
tests live in new files.

Before GPU output, focused tests must cover immutable input/source drift,
frame ordering and frame-level batches, identical branch restarts/minibatch
prefixes, direct loss scaling, terminal-three decisions, faithful/ceiling
branching, conditional holdout non-access, wrong-view zero-match controls,
all-family gates, reference extraction, macro-versus-pooled arithmetic, strict
tie behavior, second-seed blocking, finalizer recomputation, atomic writes,
and smoke non-promotability.
