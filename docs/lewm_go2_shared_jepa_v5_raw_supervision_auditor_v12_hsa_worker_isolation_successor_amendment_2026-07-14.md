# Shared JEPA V5 Raw-Supervision Auditor V12 HSA Worker-Isolation Successor Amendment

Date: 2026-07-14

Amendment author: `/root`

Status: **source construction and different-agent review only; no exact authority**

## Trigger and terminal predecessor

The source-only Auditor V11 candidate was frozen at these exact identities:

| Role | Path | SHA-256 |
|---|---|---|
| V11 source | `lewm/datasets/go2_shared_jepa_v5_raw_supervision_auditor_v11.py` | `ae4b1633a75e15340772d8bedf3b03af73261779117a7206e7d85abb8ebc2dda` |
| V11 CLI | `scripts/audit_go2_shared_jepa_v5_raw_supervision_v11.py` | `e4786f719b901134f32b2fed25674ce214bc018d1affb61f8b8293532cbb9265` |
| V11 test | `lewm/tests/test_go2_shared_jepa_v5_raw_supervision_auditor_v11.py` | `eb4c3a8addea99dd2f995d8672e7fa9791c8463704d7ced40a555be41c8b3828` |
| V11 handoff | `docs/lewm_go2_shared_jepa_v5_raw_supervision_auditor_v11_author_handoff_2026-07-14.md` | `d60fa55757f26d7193e608a08128fc7604d9fc9ad6e7b143890f7a422222fa54` |

Its different-agent review is the canonical BLOCK at
`docs/lewm_go2_shared_jepa_v5_raw_supervision_auditor_v11_independent_review_2026-07-14.json`,
file SHA-256
`169494633f8b9bd50ceac40436e6ef1b168624b8a8c487fedb2033a9c137f3db`
and canonical content SHA-256
`f45610c0db743bfd6ec655bd7d9c3f1e1f3578a3b57e7b55f7d2fcf029d76a94`.
The review independently reproduced `227/227` passing source and behavioral
tests, including Builder V9 byte parity and one/six-worker science parity. It
found one infrastructure defect: `_set_worker_environment()` cleared
`CUDA_VISIBLE_DEVICES`, `HIP_VISIBLE_DEVICES`, `ROCR_VISIBLE_DEVICES`, and
`GPU_DEVICE_ORDINAL`, but left hostile inherited `HSA_VISIBLE_DEVICES=0`
unchanged. The regression exited nonzero and observed HSA still equal to `0`.

No V11 exact audit ran. No V11 authorization, success report, failure report,
dataset-use authority, training authority, RGB decode, accelerator use, or
navigation authority exists. The V11 BLOCK is terminal for those bytes.

## Preserved scientific and audit contract

V12 is an additive, standalone audit-only successor. It must preserve every
accepted V11 behavior exactly, including:

- the immutable Builder V9 dataset and manifest;
- all V9 and V10 authorization, source, review, and terminal-failure bindings;
- the V11 Builder-parity correction based on all pair occurrences and the
  complete ten-field endpoint-context map;
- Builder V9 endpoint `content_sha256` provenance and the exact one-field
  `frame_key`;
- exact JSONL, geometry, raycast, evidence, raster, dtype, shape, byte-order,
  eight-array, inventory, count, and sample commitments;
- the original decoded raw-manifest boundary;
- one-versus-six spawned-worker canonical result and array-byte parity;
- atomic no-replace success/failure publication, predecessor preservation,
  fsync, terminality, and failure binding;
- one fresh audit attempt, no retry, no rebuild, no fallback, no alternate
  exact entry, and no downstream authority.

V12 may not change the dataset, sample set, audit samples, recomputation,
floating-point operations, raycast, rasterization, comparisons, success
criteria, report fields other than the required V12 namespace/bindings, or any
publication semantics. It may not import or call a predecessor exact entry.

## Sole operational correction

The closed V12 accelerator selector tuple must be exactly:

```text
CUDA_VISIBLE_DEVICES
HIP_VISIBLE_DEVICES
ROCR_VISIBLE_DEVICES
GPU_DEVICE_ORDINAL
HSA_VISIBLE_DEVICES
```

`_set_worker_environment()` must assign the empty string to all five selectors
and assign `1` to each frozen native math-thread variable. Every production
spawn initializer and every production task must call
`_set_worker_environment()` before authorization and before any task payload or
source is opened. This must hold even when the parent process supplies hostile,
nonempty values for all five selectors.

`HSA_OVERRIDE_GFX_VERSION` is not a device selector and is not added to the
closed selector tuple. The exact CPU launcher must nevertheless unset it, and
must explicitly set all five selectors empty before importing the auditor.

## V12 source and proof namespace

The fixed V12 implementation author is
`/root/raw_v11_builder_auditor_diff`.

The production closure is exactly:

1. `lewm/datasets/go2_shared_jepa_v5_raw_supervision_auditor_v12.py`;
2. `scripts/audit_go2_shared_jepa_v5_raw_supervision_v12.py`.

The proof closure is exactly:

1. `lewm/tests/test_go2_shared_jepa_v5_raw_supervision_auditor_v12.py`;
2. `docs/lewm_go2_shared_jepa_v5_raw_supervision_auditor_v12_author_handoff_2026-07-14.md`.

The canonical review is
`docs/lewm_go2_shared_jepa_v5_raw_supervision_auditor_v12_independent_review_2026-07-14.json`.
The canonical authorization is
`docs/lewm_go2_shared_jepa_v5_raw_supervision_audit_v12_authorization_2026-07-14.json`.
The only possible output leaves are the immutable V9 dataset path plus
`.audit_v12.json` or `.audit_v12.failed.json`.

The only production exact entry is:

```python
execute_exact_audit_v12(*, authorization_sha256: str, workers: int)
```

Workers remain strict non-boolean integers in `[1, 6]`; the one authorized
attempt must use exactly six spawned workers.

## Required author and reviewer proof

All tests are source-only, CPU-only, accelerators-hidden, native-thread-one,
and use temporary synthetic roots. They must not open the canonical dataset,
RGB, `.generated` payloads, checkpoints, GPUs, G2, held-out, or navigation
artifacts.

The author and future reviewer must:

1. independently rehash the V11 candidate and BLOCK evidence above;
2. rerun all `227` V11/retained tests unchanged;
3. prove V12 and V11 ASTs differ only by the frozen V12 namespace, expanded
   authority lineage, and the one HSA selector correction;
4. set all five selectors to hostile nonempty values, call
   `_set_worker_environment()`, and require all five to be empty afterward;
5. run the actual spawn initializer under hostile inheritance and require all
   selectors empty before its authorization check;
6. exercise every production worker task under hostile inheritance and prove
   it clears all selectors before authorization and before any opener;
7. run the real synthetic replay with one and six spawned workers while the
   parent exports hostile selector values, requiring identical canonical result
   bytes and identical eight-array bytes;
8. prove every initializer and task independently reauthorizes after hiding
   accelerators;
9. prove invalid, incomplete, repeated, reordered, noncanonical, aliased,
   symlinked, hard-linked, or changed authority fails before mapped-target or
   dataset access;
10. prove success/failure publication preserves the dataset, V9 failure, V10
    failure, V11 BLOCK, and absence of all predecessor success leaves;
11. prove source exposes no callback seam, mutable authority registry,
    monkeypatchable predecessor exact entry, retry, fallback, or alternate
    production entry; and
12. run `py_compile`, whitespace checks, the focused V12 suite, the complete
    V11 suite, and every applicable retained V10/V9 Builder/Auditor suite.

Any science-byte, report-arithmetic, dataset, worker-count, or publication
difference outside the explicitly versioned authority/report names is a BLOCK.

## V12 authorization closure

The future canonical V12 authorization must contain an ordered, unique source
map with exactly these roles and literal paths:

1. `amendment`: this V12 amendment;
2. `v9_build_authorization`: frozen V9 build authorization;
3. `v9_builder_source`: frozen Builder V9 source;
4. `v9_builder_review`: frozen passing Builder V9 review;
5. `v9_dataset_manifest`: immutable V9 dataset manifest;
6. `v9_terminal_failure`: immutable V9 auditor failure;
7. `v10_amendment`: frozen Auditor V10 amendment;
8. `v10_auditor_source`: frozen Auditor V10 source;
9. `v10_auditor_cli`: frozen Auditor V10 CLI;
10. `v10_auditor_test`: frozen Auditor V10 test;
11. `v10_auditor_handoff`: frozen Auditor V10 handoff;
12. `v10_auditor_review`: frozen passing Auditor V10 review;
13. `v10_audit_authorization`: frozen Auditor V10 authorization;
14. `v10_terminal_failure`: immutable Auditor V10 failure;
15. `v11_amendment`: frozen Auditor V11 amendment;
16. `v11_auditor_source`: frozen Auditor V11 source;
17. `v11_auditor_cli`: frozen Auditor V11 CLI;
18. `v11_auditor_test`: frozen Auditor V11 test;
19. `v11_auditor_handoff`: frozen Auditor V11 handoff;
20. `v11_auditor_block`: canonical Auditor V11 BLOCK review;
21. `auditor_source`: frozen Auditor V12 source;
22. `auditor_cli`: frozen Auditor V12 CLI;
23. `auditor_test`: frozen Auditor V12 test;
24. `auditor_handoff`: frozen Auditor V12 handoff;
25. `auditor_review`: passing different-agent Auditor V12 review.

It must deep-validate both predecessor authorizations, both terminal failure
receipts, the complete V11 source/BLOCK chain, the immutable V9 manifest and
inventory, the V12 candidate/review, and absence of every V9/V10/V11 success
leaf. Its schema must set `exact_audit_v12_authorized=true` and every earlier
audit/build/rebuild/retry/RGB/dataset-use/training/selection/calibration/G2/
held-out/runtime/navigation/hardware/production/promotion/deployment authority
false. Phase one must validate the complete closed structure without opening a
mapped target; phase two alone may open the exact bound targets.

The future reviewer must start with `/root/` and differ from all of:

- amendment author `/root`;
- V12/V11 implementation author `/root/raw_v11_builder_auditor_diff`;
- V9 Builder implementation author
  `/root/raw_v7_successor_author/auditor_v7_author`;
- V9 Auditor implementation author
  `/root/camera_v5_independent/camera_v7_pre_freeze_review/v7_review_artifact_schema`;
- V9 Builder reviewer `/root/raw_v8_auditor_reviewer`;
- V9 Auditor reviewer `/root/raw_v8_builder_reviewer`;
- V10 implementation author `/root/raw_v9_auth_hash_witness`;
- V10 reviewer `/root/raw_v10_independent_review`;
- V11 reviewer `/root/camera_v10_later_rung_plan/v11_adapter_design`;
- the future V12 authorization publisher; and
- the future V12 authorization fingerprint witness.

The fixed V12 authorization publisher is `/root`, which is also the amendment
author and is already ineligible to review the candidate. The authorization
fingerprint witness must differ from the publisher, amendment author,
implementation author, and reviewer. An agent that authors or reviews any V12
candidate byte may not serve as the authorization fingerprint witness.

## Sequence and non-authority

1. Freeze this source-free amendment before any V12 source exists.
2. The fixed implementation author constructs only V12 source, CLI, tests, and
   handoff without canonical-data, exact, RGB, `.generated`, or GPU access.
3. A different eligible agent independently reviews the frozen candidate and
   publishes one canonical `PASS` or `BLOCK`.
4. Only `PASS` permits `/root` to publish the separate V12 authorization.
5. A distinct agent independently reproduces the complete authorization file
   SHA-256.
6. Only then may one exact V12 audit run, serialized against every `.generated`
   mutator, CPU-only, six spawned workers, one native math thread per process,
   all five device selectors empty, and `HSA_OVERRIDE_GFX_VERSION` unset.
7. A terminal failure grants no retry; another defect requires another
   source-free additive successor and namespace.
8. A PASS report still does not authorize dataset use or training. Those
   require a separate later authorization.

This amendment grants only V12 source construction and different-agent review.
It grants no exact execution, retry, rebuild, dataset use, training, selection,
calibration, G2, held-out, runtime, navigation, hardware, production,
promotion, or deployment authority.
