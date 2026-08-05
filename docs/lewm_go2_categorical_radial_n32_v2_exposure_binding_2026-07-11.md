# Go2 categorical-radial N32 V2 exposure-matched execution binding

Date: 2026-07-11

Status: active; written before any N32 V2 dataset-backed model output

## Purpose and scope

The original N32 run failed before holdout access. Its maximum branch trained
320 fit frames in batches of 4 for 5,000 updates: 62.5 effective epochs. The
successful N=16 V3 rung trained 16 frames in batches of 4 for 2,000 updates:
500 effective epochs, with four optimizer updates and one evaluation every 25
epochs. It also used the fixed V3 cosine schedule that the original N32
branches explicitly did not use.

This V2 diagnostic transfers those dimensionless optimizer conditions to N32.
It changes no model tensor, factorization, label, panel row, control, metric,
gate, comparator, or holdout rule. It is train-role-only and cannot pass G2 or
license runtime promotion.

The authoritative failed V1 result is bound by:

- path:
  `.generated/go2_categorical_radial_n32/v1/seed_20260710_result.json`
- file SHA-256:
  `2f079925000ebbcd06843c413f4dcfd07fce93358482dd05512735af69cbc946`
- content SHA-256:
  `ef023faff0e49888ca673cfab5fca0c1110852e49312ce339ecb7f03ab3a8d5b`
- result note SHA-256:
  `4848c61e72be3b81bb4fe4ad0e545f9c3e6031df353c9d25f15a9dcd5109ddfd`

All immutable panel, ladder, V3, patch7-reference, physical protocol, model,
and factorization bindings from the V1 N32 contract remain in force.

## Sole intervention

The sole intervention is an exposure- and schedule-matched optimizer transfer:

- fit frames: 320;
- direct training batch: 80 independently indexed endpoint frames;
- batches per epoch: 4;
- optimizer updates: 2,000;
- effective epochs: 500;
- frame presentations: 160,000, exactly 500 per fit frame;
- evaluation interval: 100 updates, exactly 25 epochs;
- terminal decision: the exact final three evaluations at steps
  1,800, 1,900, and 2,000;
- optimizer: AdamW, weight decay `1e-4`, betas `(0.9, 0.999)`, epsilon `1e-8`,
  `amsgrad=false`;
- gradient clip: global norm 1.0, once immediately before each optimizer step;
- learning rate at one-indexed update `u` in `1..U`, `U=2000`:

```text
1e-5 + 0.5 * (2e-4 - 1e-5) * (1 + cos(pi * (u - 1) / (U - 1)))
```

The first update is exactly `2e-4`; the final update is exactly `1e-5`. There
is no warmup, retry, early stopping, best-step selection, EMA, checkpoint
averaging, AMP/autocast, compilation, or quantization.

Training uses one direct batch-80 forward and backward call. Gradient
accumulation and microbatching are forbidden. A synthetic, dataset-free GPU
feasibility check before this binding observed a 14,780,751,872-byte peak for a
batch-80 forward/backward and produced no checkpoint or research metric.

## Exact order and randomness

For each registered seed, create a CPU `torch.Generator` with
`manual_seed(seed)`. Generate consecutive full `randperm(320)` epochs. Split
each permutation, without replacement, into four ordered batches of 80. No
batch may cross an epoch boundary because 320 is exactly divisible by 80.
Store every batch index and its canonical SHA-256 in the result. The finalizer
must independently regenerate and compare the complete schedule.

The only registered seeds are 20260710 and 20260711. Model initialization,
wrong-view permutations, deterministic-algorithm request/effective state, and
all other randomness follow the V1 N32 contract. Seed 20260711 is forbidden
unless the immutable seed-20260710 V2 result is favorable and its file hash is
supplied to the runner before device/model construction.

## Evaluation and gates

Fit evaluation remains target batch 4 and combined model batch 12 across the
correct, role-global-shuffled, and same-scene-wrong-view conditions. Metrics
remain float64 accumulations over the full 320-frame panel. The aggregate and
each of the five family gates remain byte-for-byte semantically unchanged:

- hierarchical balanced NLL <= 0.03;
- UNKNOWN/KNOWN and FREE/OCCUPIED-given-KNOWN balanced accuracy >= 0.99;
- UNKNOWN, FREE, and OCCUPIED recall >= 0.98;
- every supported FREE distance bin from 1 m onward >= 0.95;
- both wrong-view minus correct NLL deltas >= 0.25.

The fixed stage passes only if the aggregate and all five family gates pass at
all three terminal evaluations. Earlier passes do not shorten training and do
not qualify the stage.

If and only if the terminal fit gate passes, the same-scene and cross-scene
train-role holdouts may be artifact-verified, decoded, and evaluated once while
the qualifying model remains in memory. Their immutable failed-faithful patch7
comparator, equal-weight family macro arithmetic, class-regression bounds, and
strict 4/5 same-scene plus 5/5 cross-scene improvement requirements are exactly
those in the V1 N32 contract. A fit failure must record zero holdout image,
shard, and model-output access.

## Results and authorization

Authoritative per-seed outputs are exclusive immutable writes at:

- `.generated/go2_categorical_radial_n32/v2/seed_20260710_result.json`
- `.generated/go2_categorical_radial_n32/v2/seed_20260711_result.json`

They use a new schema and describe one optimizer stage named
`exposure_matched_v3_cosine`. Each per-seed result is aggregation-eligible but
cannot self-license full training, G2, runtime, or promotion.

A non-authoritative seed-20260710 GPU smoke uses batch 80, three updates, and
evaluation interval 1 under a distinct schema. It remains nonaggregatable and
cannot access holdouts unless its unchanged terminal gate genuinely passes.

A pure finalizer must receive the two canonical seed paths and precommitted
file hashes, independently validate current immutable evidence and transitive
source hashes, regenerate schedules and controls, enforce strict JSON types and
metric domains, reconcile all access counts, recompute every fit and holdout
decision, and rehash both inputs after validation. Only two favorable seeds may
emit `categorical_radial_full_train_candidate_licensed=true`.

## Failure interpretation

- Fit failure after this exact 500-epoch transfer is the first clean license
  for a representation/capacity intervention at N32.
- Fit pass plus holdout failure licenses a scene/view generalization
  intervention, not more fit optimization selected on holdout values.
- One favorable seed licenses only the registered replication seed.
- Two favorable seeds license construction of the shared-JEPA full-training
  candidate. They do not pass G2 or license G3.

No checkpoint-selection, probability-calibration, non-train, G2, or sealed
payload/model output may be opened by the runner or finalizer.

