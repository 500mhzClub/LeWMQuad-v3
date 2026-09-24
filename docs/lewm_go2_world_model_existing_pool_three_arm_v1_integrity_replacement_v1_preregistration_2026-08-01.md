# Existing-pool three-arm world-model V1 integrity replacement V1

Date: 2026-08-01

Status: **PREREGISTERED SOURCE-INTEGRITY REPLACEMENT; NOT EXECUTION AUTHORITY**

## 1. Purpose and predecessor closure

This is one science-identical integrity replacement for the consumed
development attempt
`world_model_existing_pool_three_arm_v1/attempt_v1`. It exists only to
correct a pre-training checkpoint-loader validation defect. It is not a retry,
resume, refill, continuation, or reuse of that attempt.

The scientific experiment remains exactly the experiment preregistered in:

- `docs/lewm_go2_world_model_existing_pool_three_arm_v1_preregistration_2026-08-01.md`;
- `docs/lewm_go2_world_model_existing_pool_three_arm_v1_plan_2026-08-01.json`;
  plan SHA-256
  `f64b8029c9f49e3d66ce5f94e901d4000fd72f7cb40dd748fbf33e3c45254504`,
  4,360 bytes.

The original attempt is terminally closed by:

- `docs/lewm_go2_world_model_existing_pool_three_arm_v1_terminal_pretraining_source_failure_result_2026-08-01.json`;
- expected SHA-256
  `e2c219352e9ad770a232641fe0c5a7bdd8d154c61f85b5afa76b0d132856b70f`;
- expected byte count: 5,796;
- status: `PASS_COMPLETE_TERMINAL_PRETRAINING_SOURCE_FAILURE_AUDIT`.

The replacement plan, independent review, and authority must bind that audit
exactly. The original root and every artifact beneath it are quarantined and
ineligible as replacement inputs.

## 2. Established failure

The original worker verified the exact predecessor file binding, deserialized
it, found `model_state_dict`, and then failed before model construction,
update-zero evaluation, or any training update. Its loader required every
state tensor to be `torch.float32`.

The frozen spatial-V1 source publishes the complete `model.state_dict()`.
That exact inventory has 187 tensors: 186 finite float32 tensors and the
required persistent scalar-long `ema_update_count`. A prior independently
reviewed temporal-V1 run opened the same checkpoint once and recorded:

- checkpoint SHA-256
  `f5aac23cf275d73b92ce5609a583dea89f6686a624d4889d9762740535aab873`;
- checkpoint update 1,000;
- 187 total state tensors;
- 108 migrated tensors;
- 79 rejected tensors, comprising 78 stale target-encoder tensors plus
  `ema_update_count`;
- fresh target hard synchronization, fresh EMA count zero, and migration PASS.

That witness is
`.generated/go2_rgb_recurrent_patch_memory_temporal_jepa_v1/attempt_v1/receipts/terminal_access.json`,
SHA-256
`72e36e3d40a4e46bd3d03a42958257cbc6d1650d40f32a7ea4566c4af1d55113`,
10,802 bytes.

The original attempt therefore tested no data, objective, architecture, or
generalization hypothesis.

## 3. Sole scientific-source correction

Before temporal construction, the replacement loader's state-entry validator
must accept exactly:

- every state entry other than `ema_update_count`: string key, tensor value,
  exact `torch.float32`, all finite;
- `ema_update_count`: exact key, scalar `torch.long`, exact value 1,000.

That validator rejects every other integer, boolean, half, double, complex,
sparse/non-strided, non-tensor, or nonfinite entry. The accounting buffer is
preserved in the complete loaded state and is never cast or silently dropped.
The complete loader-to-constructor chain accepts no missing or extra state
keys: exact key-inventory enforcement remains the constructor's downstream
responsibility.

The existing temporal constructor remains the authoritative downstream
inventory and migration boundary. It must still require the exact complete
spatial-V1 key inventory and exact shapes/dtypes for all 108 migrated
online/predictor tensors; reject stale `target_encoder.*` and
`ema_update_count` from migration; hard-synchronize a fresh target from the
migrated online encoder; and initialize the new temporal EMA counter to zero.
The rejected target tensors remain string-keyed finite float32 values and
their exact bytes and shapes are transitively fixed by the predecessor file
binding; they are deliberately not copied or otherwise used.

A source-only regression must execute the complete chain:

`full synthetic spatial-V1 state -> serialized checkpoint -> worker loader ->
real temporal constructor`.

It must prove the 187/108/79 inventory split, exact migrated online/predictor
values, stale-target exclusion, target hard synchronization, fresh EMA zero,
and rejection of counter value 999. It must open no real checkpoint, RGB,
pack, held-out, sealed, or protected payload.

## 4. Science-identical contract

Everything not named in §3 is frozen unchanged from the original
preregistration and plan, including:

- the four exact predecessor/H6 input bindings;
- RGB positions 0 through 3 and requested-action IDs at positions 0 through 2;
- train/validation scene disjointness and all support diagnostics;
- conditioned, blind, and shuffled arm semantics and order;
- frozen shared predecessor encoder and hard-synchronized target;
- three independently allocated, identically initialized heads;
- seed 20260731 and the exact hashed train-order schedule;
- batch/microbatch geometry, optimizer groups, learning rates, warmup, cosine
  schedule, weight decay, gradient clipping, and 700 updates;
- update 0/100/.../700 validation panels and update-700 full-train fit panel;
- all metrics, Bayesian bootstrap provenance, thresholds, gate precedence, and
  requested-action factual-only claim boundary;
- 43,200-second wall and 36,000-second GPU caps;
- exact 57-file worker output inventory;
- receipt-only checker restrictions; and
- development-only, non-citable, no-navigation, no-G2-G8, no-held-out, no
  promotion status.

The source RGB pool, indices, predecessor checkpoint, model initialization,
data order, losses, thresholds, and scientific accounting may not change.

## 5. Fresh replacement lifecycle

The sole replacement identity is:

- attempt ID:
  `world_model_existing_pool_three_arm_v1_integrity_replacement_v1/attempt_v1`;
- output root:
  `/home/andrewknowles/Workspace/LeWMQuad-v3/.generated/dev/world_model_existing_pool_three_arm_v1_integrity_replacement_v1/attempt_v1`;
- maximum attempts: one;
- root must be absent before reservation;
- reservation consumes the attempt;
- retry, resume, overwrite, refill, partial reuse, and automatic extension:
  false.

The replacement must freshly rebuild its pack from the same bound indices and
permitted RGB leaves. It may not read or link the original attempt's pack,
audits, partial arms, snapshots, failure state, RNG state, or any other runtime
payload. The original terminal JSON documents are identity evidence only.

The external supervisor must reject the original attempt ID, root, plan
schema, authority schema, and authority file. Replacement result, receipt
check, reservation, worker-failure, and terminal-supervision schemas must be
replacement-specific while the scientific measurement and audit schemas
remain unchanged.

Before reservation, the replacement requires:

1. a committed terminal failure audit;
2. a frozen replacement source-and-plan commit;
3. an independent exact source review with no remaining findings and no
   execution authority;
4. a separately committed, caller-bound one-shot replacement authority;
5. exact live-file equality to all source, plan, review, runtime, input, and
   predecessor-failure bindings;
6. the fresh root to remain absent and at least 16 GiB free.

If the replacement fails after reservation, it is terminally consumed. This
document grants no second replacement.

## 6. Custody and result boundary

Sealed, held-out, G2-G8, navigation, production, and promotion access remain
forbidden. No network access is allowed. The checker may open only the result,
two audit JSON receipts, and 24 measurement JSON receipts. It may identity-
check but never open packs, RGB, checkpoints, or snapshots.

A successful replacement yields the original three-arm factual experiment's
terminal metrics and decision under replacement-bound lifecycle receipts. It
does not by itself authorize a scientific claim, navigation use, held-out
evaluation, or promotion. A separate terminal independent audit and durable
result handoff are required.
