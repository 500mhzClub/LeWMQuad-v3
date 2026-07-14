# Camera-ray N5 gate-aligned raster NLL V12 successor amendment

Date: 2026-07-14

Amendment author: `/root/camera_v10_later_rung_plan`

Fixed implementation author: `/root/camera_v12_gate_aligned_implementer`

Status: **source-free scientific successor amendment; source construction and
different-agent review only; no exact execution authority**

## 1. Scope and authority

This amendment freezes one scientific successor to the terminal Camera V11
N5 result. It authorizes only:

1. additive V12 source and synthetic-proof construction by the fixed
   implementation author named above; and
2. a source-only review by an eligible different agent after all implementation
   bytes are frozen.

This amendment does not authorize opening or using the V11 checkpoint, opening
canonical RGB or dataset payloads, running a GPU, launching exact training,
repairing V11, retrying V11, changing a threshold, selecting a checkpoint,
running another seed or rung, or performing held-out, G2, navigation, runtime,
hardware, production, promotion, or later-rung work.

The implementation author must be exactly
`/root/camera_v12_gate_aligned_implementer`. The implementation author must not
write the canonical review. The canonical reviewer must have a `/root/` agent
path and must differ from `/root`, the amendment author, and the implementation
author. A later exact execution agent, if separately authorized by a canonical
review `PASS`, must also differ from the implementation author and reviewer.

No implementation or review may infer authority from the V11 checkpoint. The
V11 checkpoint is terminal development evidence only. Its existence and the
binding declared by `completed.json` do not authorize reading it.

## 2. Frozen V11 authority and source closure

The V11 amendment, handoff, and different-agent review are bound as follows.

| Role | Path | File SHA-256 | Content SHA-256 |
| --- | --- | --- | --- |
| V11 amendment | `docs/lewm_go2_observable_camera_ray_fit_v4_n5_hierarchical_first_hit_v11_retained_gate_schema_adapter_successor_amendment_2026-07-14.md` | `369a4428ebc574899106c78ee6b90416afdacb99ed6b6ca47cf620cbe4eeed3e` | not applicable |
| V11 implementation handoff | `docs/lewm_go2_observable_camera_ray_fit_v4_n5_hierarchical_first_hit_v11_implementation_handoff_2026-07-14.md` | `27c879957605b9bd25a98d76b12d394f71250aba476e65f2647875e2b09ae506` | not applicable |
| V11 independent review | `docs/lewm_go2_observable_camera_ray_fit_v4_n5_hierarchical_first_hit_v11_independent_review_2026-07-14.json` | `f8b4c8705d05af84ab9ccabd39efcf4cdbfd625b40b44fb3703973aeedfc1836` | `8410cfb50be1673017f71151001720cfe5e1660c92486c5773d0ad286df5d656` |

The V11 production sources are frozen at these file identities.

| Role | Path | File SHA-256 |
| --- | --- | --- |
| Retained hierarchical first-hit model/loss | `lewm/models/observable_camera_ray_evidence_v4_hierarchical_first_hit_v9.py` | `52bc99f0ba59c2cf7444221931169ba57af61f343308b85625877c7a257adffd` |
| V11 policy | `lewm/benchmarks/go2_observable_camera_ray_fit_v4_n5_hierarchical_first_hit_v11.py` | `75b017d73181baaffb8e05931e0af7b53b4fd24b8a8b77740009fc7297e43cd5` |
| V11 trainer | `scripts/train_go2_observable_camera_ray_fit_v4_n5_hierarchical_first_hit_v11.py` | `99de094c1df010f17c26d6f6109ff256a658d74f7799275bf572eae6afa5a1ae` |
| V11 verifier | `scripts/verify_go2_observable_camera_ray_fit_v4_n5_hierarchical_first_hit_v11.py` | `7cf4d8e7649cd735156bf1e92b6f12b49754f804832a2af7c3ffc2b7229ddf51` |
| V11 executor | `scripts/execute_go2_observable_camera_ray_fit_v4_n5_hierarchical_first_hit_v11.py` | `401b46296fd367e2945d8e53844c0e80242ee1dc5bd5412f2a89f43fe4f22bc9` |

The V11 proof closure is frozen at these file identities.

| Role | Path | File SHA-256 |
| --- | --- | --- |
| Synthetic execution proof | `lewm/tests/n5_hierarchical_first_hit_v11_synthetic_execution.py` | `baab30b06011aa80b922f1399d0d53666f7c08eb7a9991d4e4c30d4a04cb7d2a` |
| Science tests | `lewm/tests/test_go2_observable_camera_ray_fit_v4_n5_hierarchical_first_hit_v11.py` | `7818a89395eccb779094fd5ddc26107584ccc228fee3e8d4698896149f55749c` |
| Lifecycle tests | `lewm/tests/test_go2_observable_camera_ray_fit_v4_n5_hierarchical_first_hit_v11_lifecycle.py` | `80347c7b773733bdfc34b3f683d66614f712449c29287bb26ff6272c46efcc6c` |

The retained shared sources and threshold contract reviewed for V11 remain
frozen. V12 must import or preserve them without scientific modification except
for the sole additive objective defined in Section 5.

| Path | File SHA-256 |
| --- | --- |
| `docs/lewm_go2_observable_camera_ray_fit_v4_ladder_v2_partition_amendment_2026-07-12.md` | `1e65f8884b1b8e0ad2219ddad54f79f9fabae514bfcaa048b29c8113b076ac1f` |
| `docs/lewm_go2_observable_camera_ray_fit_v4_target_partitions_2026-07-12.json` | `4ca8ef7f427f525e591a107496ef3b42c2586a9e47f7b8a7a0fd5710ca0d248a` |
| `lewm/benchmarks/go2_observable_camera_ray_evidence_v4.py` | `708d368e461fe60aacb860dda5b0cbfd1acaf43e5cb3ae18a77bb48de739fb85` |
| `lewm/benchmarks/go2_observable_camera_ray_fit_v4_ladder_gate.py` | `aa51413edfea10a2d7c04b034033c83c78c27b1c08d2be1413f5917dc32e36ad` |
| `lewm/benchmarks/go2_observable_camera_ray_fit_v4_metrics.py` | `6a0e40f9dcb496831553dc5bbc6d1efcdf6d82676d6f18aa20e417f8de4fa6a0` |
| `lewm/models/observable_camera_ray_evidence_v4.py` | `6238f7fb2b9c0c5201c9d7ebb5343ceef72fa97b423dddb466465b6c594cc882` |
| `lewm/models/observable_camera_ray_evidence_v4_training.py` | `c0f3f944883987950edb7579a9e108171486122a9a3ae9d84d2a1abb6ac015ed` |
| `scripts/launch_go2_observable_camera_ray_fit_v4_v2.py` | `65c58e36cb97d155a58ec1cbc93a1f2f42a75e62f049b5d8e874481a435a614b` |
| `scripts/train_go2_observable_camera_ray_fit_v4_v2.py` | `c9d22fb38acdf5fd3099271661dc65bb9cea989426a3b6021ad28649d6dd74d3` |
| `scripts/verify_go2_observable_camera_ray_fit_v4_target_partitions.py` | `4624dd761901808c72b37eb256b360e3db61c9b8f61337879547ed38836a3eed` |

## 3. Frozen V11 terminal evidence

The single reviewed V11 attempt completed at the process and publication level,
then failed the unchanged numeric gate. These are the only terminal V11 payloads
that V12 source construction and review may inspect. `checkpoint.pt` is
deliberately excluded.

| Role | Path | File SHA-256 | Content SHA-256 |
| --- | --- | --- | --- |
| Reservation | `.generated/go2_observable_camera_ray_fit_v4/n5_hierarchical_first_hit_v11/attempts/seed_20260710/n5/reservation.json` | `d561b26fb455f2ef7dc5f4bba49156cd8c86c738ec7dc53694afa0abca2a3b29` | `b8bc34b916c5a53a6ae5e4b28d7e280262edef32c51acf6b294aabd9655344f2` |
| Result | `.generated/go2_observable_camera_ray_fit_v4/n5_hierarchical_first_hit_v11/attempts/seed_20260710/n5/result.json` | `73c6521cc1fc9431d7de18f812e1341c425b371539da7671aa37964355a3242d` | `bf5731fc94b1af1614ff8fff5b48c4e598672115e193ba48b903c66af12c27e5` |
| Completion | `.generated/go2_observable_camera_ray_fit_v4/n5_hierarchical_first_hit_v11/attempts/seed_20260710/n5/completed.json` | `663c8225b3c8dd51b4b0928a8d138ac43778f4401d3feff9a15a2cbbbf361f0d` | `cd725130dd5a5b9faa4030c50d65e7ffa5aa6e4f818d6ca8d63d66df3cd1259d` |
| Metric verification | `.generated/go2_observable_camera_ray_fit_v4/n5_hierarchical_first_hit_v11/metric_verifications/seed_20260710_n5.json` | `1ea93e82db6bfbc649e06eb9ea54999d3c7501e0205021d3ca1a5c4d3b18635e` | `845c0463dfb8f7d4b0488101a25be0301bdb55e7358994606f09057a7ed6fd4b` |
| Gate | `.generated/go2_observable_camera_ray_fit_v4/n5_hierarchical_first_hit_v11/gates/seed_20260710_n5.json` | `74792618b41ca9084782aaa7fad5cbd18ce43b0276a16aaa51403b5812fcdbf4` | `cb1bbe7eb4805cf29243c01b17f233083cec378ddc3df8b7682603e791f14293` |

The V11 gate status is exactly `terminal_numeric_failure`: 25 of 26 checks
passed. The sole failed check is:

```text
name       = matched.raster_nll
value      = 0.07255925759673118
comparison = less_than_or_equal
threshold  = 0.06
passes     = false
```

The matched raster balanced accuracy was `0.9939025862808951`; class recalls
for UNKNOWN, FREE, and OCCUPIED were respectively
`0.9894560565651553`, `0.9922517022775299`, and `1.0`. The target counts were
16,123 UNKNOWN, 4,259 FREE, and 98 OCCUPIED cells. The confusion rows
UNKNOWN/FREE/OCCUPIED by predicted columns UNKNOWN/FREE/OCCUPIED were:

```text
[[15953, 0, 170],
 [0, 4226, 33],
 [0, 0, 98]]
```

All 203 hard errors were non-OCCUPIED cells predicted OCCUPIED. This supports,
but does not prove, excess occupied probability on common cells. V11 did not
publish per-class or per-family raster NLL, confidence histograms, Brier score,
ECE, temperature, or reliability bins, so the terminal aggregates cannot prove
whether the NLL excess is diffuse underconfidence or a smaller number of
high-confidence errors.

Every V11 gate license is false, including checkpoint use, retry, second seed,
N16, shared-JEPA training, holdout, G2, selection, calibration change,
navigation, runtime, hardware, later-rung design/review, later-rung execution,
production, and promotion. The result, metric verification, and gate are not
authoritative, aggregation eligible, or promotion eligible. V12 must not turn
any V11 license true.

## 4. First-principles reduction diagnosis

Let the existing differentiable soft raster expose normalized class
probabilities `P` in class order UNKNOWN, FREE, OCCUPIED, and let `y` be the
integer target raster label. Let:

- `R` be mean occupied-branch BCE over non-OCCUPIED targets;
- `O` be mean occupied-branch BCE over OCCUPIED targets;
- `U` be mean conditional-free-branch BCE over UNKNOWN targets; and
- `F` be mean conditional-free-branch BCE over FREE targets.

The retained V11 raster objective is state balanced:

```text
H = 0.25 * (R + O + U + F)
```

The retained gate raster NLL is an all-cell micro average. For this frozen N5
partition it is:

```text
G = (20382 / 20480) * R
  + (   98 / 20480) * O
  + (16123 / 20480) * U
  + ( 4259 / 20480) * F
```

Thus V11 trains `R:O` at 1:1 while the gate observes approximately 208:1, and
trains `U:F` at 1:1 while the gate observes approximately 3.786:1. Balanced
accuracy uses only argmax classes; NLL also penalizes insufficient probability
on the correct class. High balanced accuracy and failed NLL are therefore
compatible. The V11 evaluation value `H = 0.02024492286145687` and gate value
`G = 0.07255925759673118` use different reductions and are not interchangeable.

## 5. Sole V12 scientific delta

V12 must add exactly one gate-aligned differentiable scalar. It must not remove,
replace, rescale, or otherwise change any V11 loss.

For a float32 `class_probabilities` tensor with shape `(B, 3, H, W)` and integer
`target_raster_labels` with shape `(B, H, W)`, define:

```text
epsilon = torch.finfo(class_probabilities.dtype).eps
P_target = class_probabilities.gather(
    1,
    target_raster_labels.to(dtype=torch.long)[:, None],
).squeeze(1)
G = (-P_target.clamp_min(epsilon).log()).mean()
```

The V12 training objective is exactly:

```text
L_V11 = 0.25 * hierarchical_first_hit_nll
      + 0.25 * target_bin_offset_smooth_l1
      + 0.25 * ground_clear_distance_state_balanced_bce
      + 0.25 * derived_raster_hierarchical_bce

L_V12 = L_V11 + 0.25 * derived_raster_cell_nll
      = L_V11 + 0.25 * G
```

The five coefficients are each exactly `0.25`; they are not renormalized to sum
to one. `G` uses every target cell once with no class weighting, state
balancing, family weighting, label smoothing, temperature, calibration,
confidence penalty, focal term, invented logits, or post-hoc transform. It must
be computed from the same soft-raster class probabilities evaluated by the
retained metric accumulator.

The retained state-balanced raster loss `H` is required. Replacing `H` with `G`
is forbidden because it would sharply reduce rare OCCUPIED protection and risk
the passing OCCUPIED-recall and balanced-accuracy checks. A schedule-only
extension is also outside this amendment: it would add exposure while
continuing to optimize the mismatched reduction.

No other scientific behavior may change.

## 6. Frozen experiment contract

V12 preserves all of the following V11 values exactly:

- model class and architecture: `ObservableCameraRayEvidenceV4Model`;
- retained hierarchical first-hit model/loss and soft rasterizer;
- fresh model initialization with no predecessor state loaded;
- dataset role `train`, fit size 5, and the same frozen five-frame subset and
  target partition commitments;
- one frame from each of the five registered families;
- seed `20260710` as a paired RNG control;
- full-panel training batch size 5;
- exactly 4,000 AdamW optimizer updates and 20,000 frame exposures;
- learning rate `1e-4`, weight decay `1e-4`, float32, no autocast, and gradient
  clipping at norm `1.0`;
- deterministic schedule algorithm and schedule SHA-256
  `fb5a6c13708944b6ce514960c11c063eece704676276b352bd5233080f4fd380`;
- diagnostic updates 1, 100, 200, ..., 4,000;
- final-update-only checkpoint selection, with no best-step selection;
- evaluation batch size 1;
- matched image mapping `[0, 1, 2, 3, 4]`;
- wrong-RGB-with-target-calibration mapping `[1, 2, 3, 4, 0]`;
- independent verifier recomputation from the final V12 checkpoint;
- the retained frozen gate, all 26 checks, and every threshold, including
  `matched.raster_nll <= 0.06`; and
- the V11 R9700-only, hidden-Raphael, native-thread, determinism, access-ledger,
  filesystem, isolated-child, transaction, and terminal-publication contracts.

The V11 checkpoint must not be opened, copied, hashed again, loaded, warmed
from, selected, compared at tensor level, or used as a model input. Reusing the
same seed is a paired experimental control, not checkpoint reuse or a V11 retry.

## 7. Native V12 records and diagnostics

Training records must publish all five native component scalars and the exact
V12 total at every frozen diagnostic update. They must also retain a separate
V11-base total so arithmetic and the sole additive delta can be checked without
ambiguity.

Matched and wrong-RGB evaluation must publish both:

1. the retained four V11 evaluation component values and retained V11-base
   total for gate compatibility; and
2. a native V12 objective record containing `derived_raster_cell_nll`, the
   retained V11-base total, and the exact V12 total.

The native V12 objective record must never be silently converted into a V11
record. If a private compatibility view is required by the frozen gate, one
policy-owned, nonmutating helper must deep-copy the native evaluation, exclude
only additive V12 objective diagnostics from the compatibility view, preserve
the retained V11 loss record and all metrics/control/mappings byte-for-value,
prove the original record unchanged, and then invoke the actual frozen
validator. No V12 scalar may be published under a false V11 meaning.

For each evaluation control, V12 must additionally publish non-gating aggregate
raster NLL diagnostics:

- overall cell count, NLL sum, and mean;
- for each UNKNOWN/FREE/OCCUPIED target class: count, NLL sum, and mean, with a
  null mean only when count is zero;
- for each registered scene family: count, NLL sum, and mean; and
- the existing 3x3 confusion and class recalls.

Class counts and family counts must each partition the overall count exactly;
class NLL sums and family NLL sums must each reconstruct the overall NLL sum;
and every mean must reconstruct its sum divided by count. These diagnostics are
aggregate-only and must not publish per-cell probabilities. They add no gate,
threshold, selection rule, calibration, or adaptation.

The training and independent verifier must compute `G` independently. The
verifier must not reuse the trainer's scalar, result metrics, confusion, NLL
sum, or diagnostics. Finalization must use only the independently verified
receipt and the unchanged 26-check gate.

## 8. Frozen source and proof namespace

The fixed implementation author may create only the additive V12 closure below
plus its implementation handoff:

```text
lewm/models/observable_camera_ray_evidence_v4_gate_aligned_raster_nll_v12.py
lewm/benchmarks/go2_observable_camera_ray_fit_v4_n5_gate_aligned_raster_nll_v12.py
scripts/train_go2_observable_camera_ray_fit_v4_n5_gate_aligned_raster_nll_v12.py
scripts/verify_go2_observable_camera_ray_fit_v4_n5_gate_aligned_raster_nll_v12.py
scripts/execute_go2_observable_camera_ray_fit_v4_n5_gate_aligned_raster_nll_v12.py
lewm/tests/n5_gate_aligned_raster_nll_v12_synthetic_execution.py
lewm/tests/test_go2_observable_camera_ray_fit_v4_n5_gate_aligned_raster_nll_v12.py
lewm/tests/test_go2_observable_camera_ray_fit_v4_n5_gate_aligned_raster_nll_v12_lifecycle.py
docs/lewm_go2_observable_camera_ray_fit_v4_n5_gate_aligned_raster_nll_v12_implementation_handoff_2026-07-14.md
```

The canonical different-agent review path is:

```text
docs/lewm_go2_observable_camera_ray_fit_v4_n5_gate_aligned_raster_nll_v12_independent_review_2026-07-14.json
```

The future exact output namespace, which must remain absent during source
construction and review, is:

```text
.generated/go2_observable_camera_ray_fit_v4/n5_gate_aligned_raster_nll_v12/
```

The sole future attempt path is frozen as:

```text
.generated/go2_observable_camera_ray_fit_v4/n5_gate_aligned_raster_nll_v12/attempts/seed_20260710/n5
```

No alias, alternate output root, fallback path, predecessor directory, or
second attempt is authorized.

## 9. Required source and synthetic proof

Before an independent review may pass, hidden-accelerator CPU tests must prove:

1. the exact `gather -> clamp_min(float32 epsilon) -> log -> all-cell mean`
   formula, input shapes/types/devices, finite scalar and gradients, and
   nonmutation;
2. exact equality, within the frozen numeric tolerance, between differentiable
   `G` and the retained metric accumulator's `nll_sum / count`;
3. parity between one batch of five and five batch-one aggregates;
4. the analytical `R/O/U/F` decomposition, including the frozen V11 target
   counts and missing-class synthetic cases;
5. retained V11 `H`, hierarchical first-hit NLL, offset, ground loss, model,
   rasterizer, controls, target derivation, and thresholds remain unchanged;
6. exact arithmetic for the four retained `0.25` terms, additive `0.25 * G`,
   V11-base total, and V12 total at every diagnostic and evaluation record;
7. per-class and per-family diagnostic partition, sum, mean, finite-value, and
   mutation rejection;
8. trainer and verifier independently compute `G`, with runtime spies proving
   no trainer scalar or result metric is reused;
9. the actual frozen gate still reconstructs exactly 26 checks with unchanged
   names, values, thresholds, pass/fail decisions, and threshold-contract hash;
10. passing and failing synthetic fixtures, including high-balanced-accuracy
    but failed-NLL behavior, remain fail closed;
11. exact result, receipt, completion, gate, license, source-hash, review-hash,
    and self-hash schemas reject missing, extra, boolean, nonnumeric, nonfinite,
    negative, inconsistent, or mutated values;
12. no V11 checkpoint path can be opened or supplied; fresh initialization is
    required at runtime and in every durable receipt;
13. the same no-follow, inotify journal, exclusive reservation, one-shot,
    isolated verifier, transactional publication, failure cleanup, and
    all-false failure-license lifecycle inherited from V11;
14. an actual `sys.executable -I -B --verification-child` smoke traverses the
    production V12 request/response, independent `G` recomputation, native
    diagnostics, compatibility boundary, and unchanged 26-check gate without
    canonical data or a GPU; and
15. timeout, signal, nonzero exit, malformed/oversized response, unexpected
    stderr, schema crossing, source drift, review drift, and every lifecycle
    phase failure terminalize once and authorize nothing.

The implementation handoff must freeze the amendment hash, every V12 source and
proof hash, retained V11 bindings, test commands and counts, real-smoke receipt,
and confirmation that the V12 output root, canonical review, exact data, RGB,
checkpoint, and accelerators were not accessed.

## 10. Independent review and future one-attempt lifecycle

The eligible reviewer must independently:

1. hash this amendment, the full V11 authority/source/proof closure, all five
   terminal V11 JSON receipts above, every V12 source/proof, and the V12 handoff;
2. reproduce the reduction analysis and inspect the sole scientific diff;
3. verify the fixed implementation author and reviewer separation;
4. inspect native result semantics and any private gate-compatibility helper;
5. rerun all source, parity, gradient, diagnostic, lifecycle, and real-child
   CPU proofs with every accelerator selector hidden;
6. prove that no V11 checkpoint, canonical RGB/data, GPU, exact output root, or
   unowned payload was opened; and
7. publish the canonical review as `PASS` or `BLOCK` last.

Only a canonical different-agent `PASS` binding the exact frozen V12 bytes may
authorize one future exact attempt. That attempt must use a fresh model, the
frozen V12 namespace, seed, subset, schedule, controls, and R9700-only resource
contract. It must be launched once by a separately named execution agent. No
retry is permitted under success, numeric failure, runtime failure, verifier
failure, publication failure, timeout, or interruption.

The exact attempt must finish with durable reservation and exactly one terminal
success/completion or failure receipt. If training publishes a result, an
independent metric receipt and unchanged 26-check gate must be published before
terminal interpretation. A gate `PASS` is evidence only: every checkpoint-use,
retry, second-seed, N16, shared-JEPA, held-out, G2, selection, calibration,
navigation, runtime, hardware, later-rung, production, promotion, and downstream
license must remain false. Any future work requires a new source-free amendment
and different-agent review.

## 11. Explicit prohibitions

V12 must not use threshold relaxation, temperature scaling, label smoothing,
post-hoc calibration, checkpoint reuse, warm start, optimizer-state reuse,
V11 tensor comparison, best-step selection, extra training updates, another
seed, another subset, another wrong-RGB mapping, metric repair, result repair,
gate bypass, a synthetic executor as production authority, or an in-process
verifier fallback.

This amendment is not an execution receipt, source review, implementation
handoff, exact authorization, scientific success claim, or downstream license.
