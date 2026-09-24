# Camera-ray N5 hierarchical-first-hit V11 implementation handoff

Date: 2026-07-14

Implementation author: `/root/camera_v10_later_rung_plan`

Status: **source and synthetic CPU closure complete; independent review required; no exact authority**

## Frozen authority

The source-free amendment is:

`docs/lewm_go2_observable_camera_ray_fit_v4_n5_hierarchical_first_hit_v11_retained_gate_schema_adapter_successor_amendment_2026-07-14.md`

File SHA-256:

`369a4428ebc574899106c78ee6b90416afdacb99ed6b6ca47cf620cbe4eeed3e`

The amendment author is `/root/camera_v10_gate_loss_diagnosis`. The
implementation author differs from the amendment author. A canonical reviewer
must start with `/root/` and differ from both authors.

## Frozen production closure

| Role | Path | File SHA-256 |
| --- | --- | --- |
| Retained hierarchical loss/model | `lewm/models/observable_camera_ray_evidence_v4_hierarchical_first_hit_v9.py` | `52bc99f0ba59c2cf7444221931169ba57af61f343308b85625877c7a257adffd` |
| V11 policy | `lewm/benchmarks/go2_observable_camera_ray_fit_v4_n5_hierarchical_first_hit_v11.py` | `75b017d73181baaffb8e05931e0af7b53b4fd24b8a8b77740009fc7297e43cd5` |
| V11 trainer | `scripts/train_go2_observable_camera_ray_fit_v4_n5_hierarchical_first_hit_v11.py` | `99de094c1df010f17c26d6f6109ff256a658d74f7799275bf572eae6afa5a1ae` |
| V11 verifier | `scripts/verify_go2_observable_camera_ray_fit_v4_n5_hierarchical_first_hit_v11.py` | `7cf4d8e7649cd735156bf1e92b6f12b49754f804832a2af7c3ffc2b7229ddf51` |
| V11 executor | `scripts/execute_go2_observable_camera_ray_fit_v4_n5_hierarchical_first_hit_v11.py` | `401b46296fd367e2945d8e53844c0e80242ee1dc5bd5412f2a89f43fe4f22bc9` |

The retained V4 ladder gate remains byte-identical at SHA-256
`aa51413edfea10a2d7c04b034033c83c78c27b1c08d2be1413f5917dc32e36ad`.

## Frozen proof closure

| Role | Path | File SHA-256 |
| --- | --- | --- |
| Synthetic lifecycle and complete N5 gate fixture | `lewm/tests/n5_hierarchical_first_hit_v11_synthetic_execution.py` | `baab30b06011aa80b922f1399d0d53666f7c08eb7a9991d4e4c30d4a04cb7d2a` |
| Hierarchical science tests | `lewm/tests/test_go2_observable_camera_ray_fit_v4_n5_hierarchical_first_hit_v11.py` | `7818a89395eccb779094fd5ddc26107584ccc228fee3e8d4698896149f55749c` |
| Lifecycle, adapter, and subprocess tests | `lewm/tests/test_go2_observable_camera_ray_fit_v4_n5_hierarchical_first_hit_v11_lifecycle.py` | `80347c7b773733bdfc34b3f683d66614f712449c29287bb26ff6272c46efcc6c` |

This handoff is the fourth proof file. The reviewer must hash its final bytes
and bind that hash in the canonical review.

## Implemented boundary

V11 preserves the V10/V9 scientific treatment: the same model and hierarchical
loss, five-frame panel, seed `20260710`, AdamW settings, 4,000 full-panel
updates, 20,000 frame exposures, schedule SHA-256
`fb5a6c13708944b6ce514960c11c063eece704676276b352bd5233080f4fd380`,
four equal loss weights, 41 diagnostics, final-update-only checkpoint, matched
and wrong-RGB evaluation, metrics, and retained thresholds.

The sole behavioral change is one policy-owned adapter:

`adapt_hierarchical_evaluation_for_retained_v4_gate`

It validates the original hierarchical schema, deep-copies without mutation,
replaces only `hierarchical_first_hit_nll` with
`ordered_first_hit_nll` in the private copy, preserves the numeric object/value,
proves metric/control/mapping equality, and invokes the actual frozen loss
validator. There is no `_shadow_evaluation` or second transformation.

The only frozen-gate entry is:

`reconstruct_retained_v4_gate`

Only this helper invokes frozen `_validated_metric_evaluation` and `_gate_stage`.
The verifier child, parent metric-receipt validator, inline V11 finalizer, smoke
child, and smoke parent all use it. Published result and receipt evaluations
retain `hierarchical_first_hit_nll`; the compatibility view is never published.

V11 no longer monkeypatches or calls the retained V1 finalizer. Inline
finalization rereads and validates the metric receipt, independently reconstructs
the unchanged 26-check gate, and publishes only through the owned transaction.
A passing gate can authorize later-rung design/review only. Later-rung execution,
retry, second seed, N16, shared-JEPA training, held-out, G2, navigation, runtime,
hardware, production, and promotion remain false.

The direct V11 result validator retains fail-closed structure without leaking a
legacy-name view outside the gate stack. It validates the exact top-level schema,
self-hash, scope, source review, one-attempt record, subset/target bindings, all
exact inputs, model/checkpoint, full hierarchical training/evaluation records,
R9700 resource receipt, determinism receipt, complete access ledger, and licenses.

## Proof results

The final author command hid every accelerator selector, set all native math
threads to one, and ran:

```text
test_go2_observable_camera_ray_fit_v4_n5_hierarchical_first_hit_v11.py
test_go2_observable_camera_ray_fit_v4_n5_hierarchical_first_hit_v11_lifecycle.py
test_go2_observable_camera_ray_fit_v4_ladder_gate.py
```

Result: **190 passed in 18.92 seconds**.

The closure proves:

- raw retained loss validation reproduces the V10 schema failure;
- adapter nonmutation and exactly one remove/add operation per control;
- missing, legacy/both, extra, boolean, nonnumeric, nonfinite, negative,
  inconsistent-total, malformed-control, mapping, and metric cases fail closed;
- a complete frozen-signature N5 fixture traverses the actual retained validator
  and unchanged gate with exactly 26 checks and a passing decision;
- passing and failing decisions equal direct retained-gate decisions bit-for-bit;
- AST checks leave raw frozen-gate calls only in the shared policy helpers;
- runtime spies execute synthetic verifier-child, parent-validator, and inline
  finalizer paths and observe exactly one adapter invocation in each;
- result-validation mutation parity rejects rehashed scope, review, attempt,
  input, model, resource, determinism, access, and license changes;
- normalized trainer/model/loss/evaluation science matches the retained source;
- all inherited no-follow, inotify journal, one-shot, isolation, diagnostic,
  cleanup, and all-false failure-license regressions pass; and
- the actual V11 executor launches a real `-I -B --verification-child`
  subprocess on hidden CPU. The serialized complete hierarchical fixture uses
  the same production adapter and reconstruction helper in child and parent;
  all 11 phase failures and timeout, signal, nonzero, malformed, oversized,
  stderr, and schema-crossing cases pass.

No canonical experiment RGB/data, checkpoint, numeric result, held-out, G2,
navigation, runtime, or hardware payload was opened. No GPU operation or exact
execution was launched. The V11 `.generated` output root and canonical review
file were absent when this handoff was frozen.

## Independent review

The next action is a different-agent source review only. The reviewer must:

1. rehash the amendment, every production source, every proof, this handoff,
   the V10 predecessor closure, the three terminal V10 receipts, and the frozen
   ladder gate;
2. independently inspect the adapter, shared gate-helper call graph, direct
   result validator, inline finalizer, one-attempt namespace, and licenses;
3. rerun the hidden-CPU V11, frozen-gate, runtime-spy, and real subprocess
   closures without canonical data or GPU access; and
4. publish
   `docs/lewm_go2_observable_camera_ray_fit_v4_n5_hierarchical_first_hit_v11_independent_review_2026-07-14.json`
   as `PASS` or `BLOCK` last.

Only a canonical different-agent `PASS` binding the exact frozen bytes may
authorize the sole fresh V11 attempt at
`.generated/go2_observable_camera_ray_fit_v4/n5_hierarchical_first_hit_v11/attempts/seed_20260710/n5`.
This handoff grants no exact, data, RGB, checkpoint, GPU, retry, later-rung
execution, full training, navigation, production, or promotion authority.
