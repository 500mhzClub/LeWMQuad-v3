# RGB Memory-Role Factorized Joint-JEPA V1 Integrity Replacement V1

Date: 2026-07-30

Status: preregistered science-identical integrity replacement only. This
document authorizes the narrow source correction and source-only tests below.
It does not itself authorize RGB or checkpoint access, GPU use, training,
navigation, memory integration, calibration, G2, held-out, sealed, benchmark,
promotion, production, or deployment activity.

## Trigger and terminal predecessor

The original V1 one-shot is consumed, terminal, and may not be retried or
resumed. Its complete immutable result is frozen in commit
`291a7bcfaf95f24d5c84bd3d590afd54556d5b3d` at
`docs/lewm_go2_rgb_memory_role_factorized_joint_jepa_v1_terminal_infrastructure_failure_result_2026-07-30.json`,
with file SHA-256
`80eaeb508a988b54e655df5b530fa3adab6a89bb13b6f5c45902ac851bc464f4`
and byte count `6060`. The replacement source closure, review, clean-export
certification, and later one-shot authority must bind that exact result before
replacement execution.

The original attempt reserved successfully but stopped during the update-0
observation before any completed observation, update, presentation, optimizer
step, EMA step, or checkpoint publication. Its exact terminal exception was
`PlaceTripletContractError: image must be exact 224x224 RGB PNG`, with
exception-message SHA-256
`0f2bbeae0fc2f5a7b5cc1d7918652ac08a4154de071a13d710d86e05135d6493`.
It therefore produced no scientific evidence about the registered mechanism.

The root cause is proven and mechanical. The place-triplet decoder froze
`SOURCE_IMAGE_SIZE = (224, 224)`, while the three manifest-bound RGB references
in the first development triplet are exact 224-by-168 RGB PNGs. This is the
same post-crop source geometry supplied by the inherited H6 RGB route before
its unchanged bilinear resize to 112 by 112. The failure occurred at that
shape check, not in the model or loss.

This is one distinct integrity replacement under fresh lifecycle roots, not a
retry or resume of the closed original output.

## Sole permitted corrections

Only the following source changes are authorized:

- In `lewm/datasets/go2_memory_role_place_triplets_v1.py`, change only
  `SOURCE_IMAGE_SIZE` from `(224, 224)` to `(224, 168)` and change the matching
  contract-error text from `224x224` to `224x168`. Preserve the PNG, RGB,
  no-follow path, regular-file, byte, SHA-256, decode, bilinear 112-by-112
  resize, ImageNet normalization, tensor-shape, dtype, and finiteness checks.
- Correct only synthetic/source-test image fixtures and exact message
  assertions that encoded the erroneous 224-by-224 source size. Add one
  focused regression proving that exact 224-by-168 RGB PNG input follows the
  unchanged bilinear 112-by-112 decode path and that other source dimensions
  remain rejected.
- Add minimal in-memory place-RGB reference accounting. Increment one attempted
  count immediately before each anchor, positive, or negative reference load;
  increment success only after that reference passes read, SHA-256, decode, and
  tensor validation; otherwise increment failure before propagating the same
  exception. Publish the three aggregate fields
  `place_rgb_reference_attempt_count`,
  `place_rgb_reference_success_count`, and
  `place_rgb_reference_failure_count` in both normal terminal-access receipts
  and exception failure-access snapshots. Require attempts to equal successes
  plus failures. These counters may cause no additional file open, decode,
  rehash, ordering change, or exception suppression.
- Change only the preregistration/result evidence selectors, lifecycle schema
  prefix, and exact fresh source/output roots needed to distinguish this
  replacement.

No other adapter, model, data, index, schedule, training, evaluation, gate, or
receipt behavior may change.

## Frozen scientific identity

Preserve the original V1 preregistration at
`docs/lewm_go2_rgb_memory_role_factorized_joint_jepa_v1_preregistration_2026-07-30.md`
and its split-integrity amendment at
`docs/lewm_go2_rgb_memory_role_factorized_joint_jepa_v1_split_integrity_amendment_2026-07-30.md`
exactly, including:

- the shared V18 RGB/object-space encoder, 64-dimensional place key,
  32-by-16-by-16 local-control state, both predictors, stop-gradient EMA target,
  and one joint optimization process;
- the accepted N320 initialization and every constructor, projection, and role
  seed, including V18 seeds `20260712` and `20260729` and role seed `20260731`;
- the exact physical, corrected H6 V2 local, and frozen place-triplet data,
  indexes, ordered rows, train/selection split, fields, and hashes;
- one AdamW optimizer, its parameter groups, learning rates, betas, epsilon,
  weight decay, route-gradient normalization, one optimizer step, and one EMA
  step per completed update;
- all physical, local MSE, cyclic wrong-action `0.05` hinge, place cosine, and
  place-negative `0.10` hinge losses, coefficients, margins, diagnostics, and
  terminal conjunctive gates;
- the exact 4+2+2 microbatch schedule, batch size four, update ordering,
  observations only at updates 0, 100, and 400, and all accounting multipliers;
  and
- exactly 400 maximum updates and 12,800 maximum presentations, comprising
  6,400 physical, 3,200 local, and 3,200 place presentations.

There is no alternate seed, data substitution, row cycling, hyperparameter
search, architecture change, loss or gate reinterpretation, extra observation,
extension, checkpoint initialization, or predecessor runtime reuse.

## One-shot replacement identity

- Schema/evidence prefix:
  `lewm_go2_rgb_memory_role_factorized_joint_jepa_v1_integrity_replacement_v1`.
- Exact fresh attempt root:
  `.generated/go2_rgb_memory_role_factorized_joint_jepa_v1_integrity_replacement_v1/attempt_v1`.
- Exact clean source root:
  `/home/andrewknowles/Workspace/LeWMQuad-v3-memory-role-factorized-joint-jepa-v1-integrity-replacement-v1-source`.
- Both roots must initially be absent. Exactly one replacement attempt is
  allowed. Retry, resume, recovery, extension, and a second integrity
  replacement are false.
- The original output remains closed. Only its committed terminal
  infrastructure-failure result may cross into the replacement as source-only
  identity evidence; no original runtime file, model state, tensor, optimizer,
  RNG state, trace, or checkpoint may be opened or reused.

## Lifecycle and authority boundary

Freeze this preregistration before implementation. Then require focused
source-only tests, a recursive source closure, independent source review, an
exact enumerated clean-export certification, and a separately committed
hash-bound one-shot execution authority before any replacement reservation,
RGB or checkpoint access, GPU use, or training.

Any source, authority, reservation, custody, exception, accounting, or
scientific gate failure terminalizes the sole replacement with complete
immutable receipts and no failed checkpoint. A terminal pass may publish only
the preregistered update-400 development checkpoint and receipts. It does not
authorize learned-memory integration, navigation, calibration, G2, held-out,
sealed, benchmark, promotion, production, or deployment access.
