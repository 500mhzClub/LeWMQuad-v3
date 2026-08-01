# Existing-pool three-arm world-model V1 integrity replacement V2

Date: 2026-08-01

Status: **PREREGISTERED SOURCE-INTEGRITY REPLACEMENT; NOT EXECUTION AUTHORITY**

## 1. Purpose and closed predecessors

This is one science-identical integrity replacement for the consumed
`world_model_existing_pool_three_arm_v1_integrity_replacement_v1/attempt_v1`.
It corrects a second pre-training source defect reached only after V1
successfully validated the predecessor loader correction. It is not a retry,
resume, refill, continuation, or runtime-artifact reuse.

The full scientific experiment remains the original three-arm contract in:

- `docs/lewm_go2_world_model_existing_pool_three_arm_v1_preregistration_2026-08-01.md`;
- `docs/lewm_go2_world_model_existing_pool_three_arm_v1_plan_2026-08-01.json`.

Integrity replacement V1 and its prior failure chain are closed by:

- `docs/lewm_go2_world_model_existing_pool_three_arm_v1_integrity_replacement_v1_preregistration_2026-08-01.md`;
- `docs/lewm_go2_world_model_existing_pool_three_arm_v1_integrity_replacement_v1_terminal_pretraining_source_failure_result_2026-08-01.json`;
- expected SHA-256
  `a96f63aeb119163cd24e17272bfbf5228206c498d706578162c310841423ac1b`;
- expected byte count: 7,008;
- status: `PASS_COMPLETE_TERMINAL_PRETRAINING_SOURCE_FAILURE_AUDIT`.

The V2 plan, independent review, and authority must bind that exact audit.
Both earlier attempt roots and every artifact beneath them are quarantined and
ineligible as V2 inputs.

## 2. Established V1 replacement failure

V1 replacement completed fresh packing, passed the corrected 187-tensor
predecessor loader, constructed the real temporal substrate, and then failed
while building the first arm optimizer. It completed no update-zero
evaluation, training update, optimizer step, result, checker, or scientific
verdict.

The shared temporal substrate is intentionally frozen before independent arm
allocation. `ArmCore` then:

- created two direct `nn.Parameter` clones, which defaulted trainable;
- deep-copied predictor modules, action/time embeddings, and the GRU;
- inherited `requires_grad=false` on those 34 deep-copied parameter tensors.

The exact registered arm inventory is 36 parameter tensors: 30 predictor and
6 memory tensors. Name coverage, allocation, dtype, and nonempty role checks
were reached; the final fail-closed partition rejected the 34 frozen arm
tensors before optimizer construction.

This is not evidence about data, objective, architecture learnability, or
generalization. It validates only that the V1 checkpoint-loader correction
crossed its previously failing boundary.

## 3. Sole new source correction

After all independent `ArmCore` parameter copies are complete, V2 must call
`ArmCore.requires_grad_(True)`. The correction must:

- make all 36 arm parameter tensors trainable;
- preserve all parameter values bit-exactly;
- preserve names, shapes, dtypes, count, module structure, and initialization;
- preserve the 30-predictor/6-memory partition;
- preserve identical initialization across conditioned/blind/shuffled arms;
- preserve independent allocation across arms;
- leave the shared encoder and target frozen;
- leave optimizer groups and every hyperparameter unchanged.

The prior loader integrity correction is retained unchanged: the exact bound
spatial-V1 checkpoint must contain 186 finite strided float32 tensors plus
scalar strided `torch.long ema_update_count=1000`; the real temporal
constructor enforces its exact migration boundary, hard-syncs a fresh target,
and resets temporal EMA to zero.

A source-only regression must execute:

`full synthetic spatial-V1 state -> worker loader -> real temporal
constructor -> frozen substrate -> real ArmCore -> real partition/optimizer`.

It must prove the 187/108/79 predecessor inventory, all 36 arm parameters
trainable float32, exact 30/6 partition, parameter-value preservation,
identical initialization, independent allocation, and frozen shared encoder
and target. It must fail against the pre-correction behavior and open no real
checkpoint, pack, RGB, held-out, sealed, or protected payload.

## 4. Science-identical contract

Everything outside the two enumerated integrity corrections is frozen
unchanged from the original preregistration and V1 replacement, including:

- all four input hashes and visible positions;
- all arm semantics and order;
- shared frozen encoder/target and independently allocated heads;
- seeds, hashed schedule, presentations, batches, and 700 updates;
- optimizer groups, learning rates, warmup, cosine schedule, weight decay, and
  gradient clipping;
- validation/train-fit panels, metrics, bootstrap, thresholds, and gate order;
- requested-action factual-only claim scope;
- wall/GPU/free-space caps and exact 57-file success inventory;
- receipt-only checker custody; and
- development-only, non-citable, no-network, no-navigation, no-held-out, no
  promotion status.

No data, model value, loss, threshold, schedule, objective, or scientific
accounting change is permitted.

## 5. Fresh V2 lifecycle

The sole V2 identity is:

- attempt ID:
  `world_model_existing_pool_three_arm_v1_integrity_replacement_v2/attempt_v1`;
- output root:
  `/home/andrewknowles/Workspace/LeWMQuad-v3/.generated/dev/world_model_existing_pool_three_arm_v1_integrity_replacement_v2/attempt_v1`;
- maximum attempts: one;
- fresh absent root required;
- reservation consumes the attempt;
- retry, resume, overwrite, refill, partial reuse, and automatic extension:
  false.

V2 must rebuild its pack from the original bound inputs. It may not read or
link either consumed attempt's pack, audit payloads, snapshots, RNG state, or
partials. Their terminal JSON documents are identity evidence only.

V2 authority/plan/review/reservation/result/check/worker-failure/terminal
schemas must be V2-specific. The supervisor must reject the original and V1
replacement roots, IDs, plans, authorities, and schemas.

Before reservation V2 requires a committed V1 failure audit, frozen V2
source/plan, independent review with no findings and no authority, separately
committed caller-bound authority, exact live binding equality, absent root,
idle authorized device, and at least 16 GiB free.

If V2 fails after reservation, it is terminally consumed. This document grants
no further replacement.

## 6. Custody and result boundary

Sealed, held-out, G2-G8, navigation, production, promotion, and network access
remain forbidden. The checker may open only the result, two audit JSON
receipts, and 24 measurement JSON receipts. Packs, RGB, checkpoints, and
snapshots remain identity-only to the checker.

A successful V2 produces the original experiment's terminal metrics under a
V2 lifecycle. It still requires an independent terminal audit and durable
handoff and does not itself authorize promotion or protected evaluation.
