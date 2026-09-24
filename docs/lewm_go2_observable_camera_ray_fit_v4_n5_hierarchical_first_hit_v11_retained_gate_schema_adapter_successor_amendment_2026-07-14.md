# Camera-ray N5 hierarchical-first-hit V11 retained-gate schema-adapter successor amendment

Date: 2026-07-14

Amendment author: `/root/camera_v10_gate_loss_diagnosis`

Fixed implementation author: `/root/camera_v10_later_rung_plan`

Status: **source construction and different-agent review only; no exact authority**

## Trigger, classification, and authority boundary

Camera V10 completed its sole fresh seed-`20260710`, N=5 training operation,
published a result and completion, and independently recomputed an evaluation
from its checkpoint in the isolated verifier child. The verifier then failed
at phase `gate_reconstruction` before any numerical threshold decision because
the V10 evaluation schema and the retained V4 ladder-gate loss schema use
different names for the first-hit loss.

The V10 trainer, result validator, and independent verifier use
`hierarchical_first_hit_nll`. The frozen retained V4 gate requires the retired
field name `ordered_first_hit_nll` before it will validate the otherwise
unchanged evaluation record. The failure is therefore a closed
schema-integration failure at the retained-gate boundary. It is not evidence
that the trained model passed or failed the 26 numerical checks.

V10 is terminal. Its checkpoint, result, and completion were deleted only
after a durable diagnostic was published. Their numeric payloads are absent,
unlicensed, and may not be recovered, reconstructed, inferred, or used. V10
may not be retried or repaired in place.

This amendment creates one additive, science-identical V11 successor whose
only permitted behavioral change is an audited, nonmutating compatibility view
at every call into the retained V4 metric gate. V11 must train from a fresh
initialization in a new one-attempt namespace. No V11 policy, trainer,
verifier, executor, test, handoff, review, output root, checkpoint, result,
metric receipt, or gate existed when this source-free amendment was drafted.
This amendment grants no exact
execution, data or RGB opening, checkpoint use, GPU use, later fit rung,
second seed, shared-JEPA training, selection, calibration, G2, held-out,
runtime, navigation, hardware, production, promotion, deployment, or retry
authority.

## Frozen V10 source and review closure

The following V10 bytes are immutable predecessor evidence.

| Role | Path | File SHA-256 |
| --- | --- | --- |
| V10 amendment | `docs/lewm_go2_observable_camera_ray_fit_v4_n5_hierarchical_first_hit_v10_durable_verifier_lifecycle_amendment_2026-07-14.md` | `1d4e4e315c880ef8b1362093f41b1d1cb5cabf6052c13886fac0a9fe2573501f` |
| Hierarchical loss/model source | `lewm/models/observable_camera_ray_evidence_v4_hierarchical_first_hit_v9.py` | `52bc99f0ba59c2cf7444221931169ba57af61f343308b85625877c7a257adffd` |
| V10 policy | `lewm/benchmarks/go2_observable_camera_ray_fit_v4_n5_hierarchical_first_hit_v10.py` | `9ff40daadcda1962de2d9d54def09b7ec5a128c0f7f3f14ee2449367f15481d5` |
| V10 trainer | `scripts/train_go2_observable_camera_ray_fit_v4_n5_hierarchical_first_hit_v10.py` | `ec22c49855fe310f43bc72132a53e867604126db096e1064451e56f080259b1a` |
| V10 verifier | `scripts/verify_go2_observable_camera_ray_fit_v4_n5_hierarchical_first_hit_v10.py` | `4ea17008e7805aba63a50415e8e9aefed31ebf70f1ccf803ec7e64e29a72cdbc` |
| V10 executor | `scripts/execute_go2_observable_camera_ray_fit_v4_n5_hierarchical_first_hit_v10.py` | `5387fbce1eb4c7c8cd1628fcf97c33a6bc7d15f8afd748a65760124d8f7002b4` |
| Retained V4 ladder gate | `lewm/benchmarks/go2_observable_camera_ray_fit_v4_ladder_gate.py` | `aa51413edfea10a2d7c04b034033c83c78c27b1c08d2be1413f5917dc32e36ad` |
| V10 synthetic lifecycle support | `lewm/tests/n5_hierarchical_first_hit_v10_synthetic_execution.py` | `843a75dc295451190af43c255475cbe6541d6d305b448f5dde9bc173fcbb76d5` |
| V10 science tests | `lewm/tests/test_go2_observable_camera_ray_fit_v4_n5_hierarchical_first_hit_v10.py` | `d27e6a6e98d5fdec9d70b446d4f6f760b87cf0057ea0299db2131f252561f1a5` |
| V10 lifecycle tests | `lewm/tests/test_go2_observable_camera_ray_fit_v4_n5_hierarchical_first_hit_v10_lifecycle.py` | `59f5a1a784586dba97170890a356e73b8b4005fb14b65f640437465289ba60a6` |
| V10 implementation handoff | `docs/lewm_go2_observable_camera_ray_fit_v4_n5_hierarchical_first_hit_v10_implementation_handoff_2026-07-14.md` | `97a87facbef35c00b3fd7fe055ce04b603d8eba6cd0cb73cf5b399d3f8b45cb3` |
| V10 independent review | `docs/lewm_go2_observable_camera_ray_fit_v4_n5_hierarchical_first_hit_v10_independent_review_2026-07-14.json` | `0aa327d5f7fbf6eb291b4f7b23f0ecd71b2bf4ecd18a30f09ea2b788ecb26286` |

The V10 independent-review content SHA-256 is
`5012c2e15f23070112a329d21735e1778c2e44fb829f965fbc8df48e35852137`.
It records a different-agent source-review pass and binds every listed V10
source and proof.

## Frozen terminal V10 evidence

Only these three V10 attempt files survive and may be opened as terminal static
evidence:

| Role | Path | File SHA-256 | Content SHA-256 | Bytes |
| --- | --- | --- | --- | ---: |
| Reservation | `.generated/go2_observable_camera_ray_fit_v4/n5_hierarchical_first_hit_v10/attempts/seed_20260710/n5/reservation.json` | `30f381e3442ac94f88de48b918e7edc95d13aae8e227d1efbba22bb39166d4e5` | `4b709f9e5dc3cc7f1dee7d748f4cca7bad598d282159e9cacad1ca3d6b27e1df` | 12720 |
| Durable verifier diagnostic | `.generated/go2_observable_camera_ray_fit_v4/n5_hierarchical_first_hit_v10/attempts/seed_20260710/n5/verification_failure.json` | `2db93f56f3c3277c4d93833dcd0a281e96fad6cc86b8b8b0da1e547391c775b5` | `e05fe4c1b5a4b53d91019126ceb32c483f02ef3dcf7f1e16754a86c8f0b9841c` | 6797 |
| Terminal failure | `.generated/go2_observable_camera_ray_fit_v4/n5_hierarchical_first_hit_v10/attempts/seed_20260710/n5/failed.json` | `cfd10f1e384b996389b297feb0a9d199285aded3341dd8880e164941bdfc8005` | `1110fee5bba4735a17023694c5f508594c31e4ec5c2e7081c67d62f6cd5f5b24` | 1759 |

The diagnostic binds child return code `70`, no signal, phase
`gate_reconstruction`, exception class `ValueError`, request content SHA-256
`989974f5769c422f3cabba776bd54e674eae2371487af0aa70bca692df9321a0`,
child-envelope content SHA-256
`8cf615d19fe2d28aa782dc1de3ab7a3642b026f78fbaea85ec7aa7868afb7688`,
and captured-stdout SHA-256
`c30e1f110abb3ffceaec889dff4251d3319368119d1513a0256f65c44af7152e`.
The sanitized message is `V# matched evaluation loss fields changed`.

The durable diagnostic also records bindings for deleted, unlicensed files:

| Deleted role | File SHA-256 | Content SHA-256 | Bytes |
| --- | --- | --- | ---: |
| Checkpoint | `a5da01de0e75f64621dacf29c6dc3f17af9177626b51dab4131cab356a40e99d` | `467100d4ce1b7c28b9cc7693e0b95ae8c85929376cbbe070e0f65dff17b05a84` | 13787980 |
| Result | `6f4a8bbd16e1fb14ccfdde5fc97f89f16772730ed79939b42a12f013cd12a824` | `785b1c61080ed7ac002a13623b39045b4f95e9aed5c2145e209661c0ed9bb51a` | 108360298 |
| Completion | `b2b5a5cb599483b8c8708e0674baa912d1e532856b410cf4a9c0a66d186295dc` | `9100d50a198735961027cb7fcacfcb35e04b1a2f4ffdcdfd20951620abc904e1` | 1250 |

These deleted bindings prove only that artifacts existed before diagnostic
publication and owned cleanup. They license no path search, undelete, cache
inspection, value recovery, score claim, initialization, checkpoint use,
calibration, selection, repair, or supervision.

## Frozen scientific treatment

V11 must preserve V10 and V9 science exactly:

- the frozen hierarchical first-hit loss/model source above;
- `ObservableCameraRayEvidenceV4Model`, parameterization, raw outputs, camera
  calibration, physical target derivation, and differentiable rasterizer;
- the same exact five-frame train panel and target partition;
- seed `20260710`, fresh initialization, AdamW, learning rate `1e-4`, weight
  decay `1e-4`, gradient clipping norm `1.0`, float32, and no autocast;
- exactly 4,000 full-panel optimizer updates and 20,000 frame exposures under
  schedule SHA-256
  `fb5a6c13708944b6ce514960c11c063eece704676276b352bd5233080f4fd380`;
- exactly four losses at weight `0.25`: `hierarchical_first_hit_nll`,
  `target_bin_offset_smooth_l1`,
  `ground_clear_distance_state_balanced_bce`, and
  `derived_raster_hierarchical_bce`;
- diagnostics at update 1 and every 100 updates through update 4,000;
- final-update-only checkpoint selection, with no best-update, early-stop,
  averaging, retry, repair, or gate-based selection;
- matched-RGB and wrong-RGB-with-target-calibration batch-one evaluation,
  independent verifier reconstruction, all metric accumulators, target
  partition signatures, quantile evidence, confusion matrices, and family
  metrics; and
- the frozen retained gate bytes, all 26 N5 checks, thresholds, arithmetic,
  pass/fail semantics, and license consequences.

No V11 result, checkpoint, completion, metric receipt, or published evaluation
may report the new hierarchical objective under the retired ordered objective
name. Any change to data, objective values, model, optimizer, schedule,
checkpoint selection, metrics, threshold values, gate arithmetic, or numerical
post-processing requires a separate source-free scientific amendment.

## Sole permitted change: retained-gate loss-schema compatibility view

V11 must define one shared, pure, audited helper with semantics equivalent to
`adapt_hierarchical_evaluation_for_retained_v4_gate`. The exact spelling may be
frozen by the implementation handoff, but there must be exactly one production
implementation of the transformation.

For each of `matched_rgb` and
`wrong_rgb_with_target_calibration`, the helper must:

1. validate the complete original evaluation under the V11 hierarchical loss
   schema before adapting it;
2. deep-copy the evaluation and never mutate the caller's object or any nested
   loss, metric, mapping, confusion, quantile, or family record;
3. require `hierarchical_first_hit_nll` and reject a missing key, a preexisting
   `ordered_first_hit_nll`, both keys, an extra key, a nonnumeric value, a
   boolean, NaN, infinity, a negative value, or inconsistent `total`;
4. remove only the key `hierarchical_first_hit_nll` from the private copy and
   insert `ordered_first_hit_nll` with the exact same numeric object/value;
5. preserve every other loss value and `total` exactly, with no cast,
   rounding, recomputation, scaling, calibration, clipping, repair, or
   reordering dependency;
6. prove the original and compatibility-view metric subtrees have identical
   canonical SHA-256 values and that controls, image mappings, and mapping
   hashes are byte-semantically unchanged; and
7. validate the compatibility-view loss record with the actual frozen retained
   V4 loss validator before returning it.

The compatibility view is an explicit schema alias solely for the retained
gate's legacy precondition. It does not reinterpret the hierarchical loss as
the old objective and may not escape the gate-validation call stack. The V11
result and independently recomputed evaluation remain hierarchical and their
canonical hashes are computed before adaptation. The metric receipt may record
the adapter version and canonical hash of the ephemeral compatibility view,
but `recomputed_evaluation` itself must remain the original hierarchical
record.

Every production entry into the frozen retained gate must consume the output
of this one helper, never the raw V11 evaluation. This includes at minimum:

1. isolated-child metric receipt construction;
2. parent validation of the child's metric receipt and numeric gate;
3. parent finalization or any retained-finalizer metric-receipt revalidation;
   and
4. any source-review or smoke path that claims to reconstruct the retained
   gate.

The author may implement a versioned V11 parent finalizer or inline reviewed
V11 finalization in the lifecycle-owning executor. It may not call the V10/V1
retained finalizer path that passes raw hierarchical evaluations to the frozen
gate. It may not modify, monkeypatch, replace, or mutate the frozen ladder-gate
module, its globals, thresholds, validator, or source bytes. The V11 call graph
must contain no reachable invocation of `_validated_metric_evaluation` on raw
V11 evaluation data.

After the frozen gate validates the compatibility view, its returned matched
and wrong metric bundles feed the unchanged `_gate_stage`. Since `_gate_stage`
uses only metric bundles, the adapter may not affect any numerical gate value.

## V11 source and proof namespace

The fixed implementation author is `/root/camera_v10_later_rung_plan`, which
differs from the amendment author. The implementation must freeze exactly one
V11 production closure consisting of:

1. the retained hierarchical loss/model source above;
2. `lewm/benchmarks/go2_observable_camera_ray_fit_v4_n5_hierarchical_first_hit_v11.py`;
3. `scripts/train_go2_observable_camera_ray_fit_v4_n5_hierarchical_first_hit_v11.py`;
4. `scripts/verify_go2_observable_camera_ray_fit_v4_n5_hierarchical_first_hit_v11.py`;
5. `scripts/execute_go2_observable_camera_ray_fit_v4_n5_hierarchical_first_hit_v11.py`.

Any additional V11 production finalizer module must be named and added to the
frozen closure by an amendment before it is created. Integrating the
parent-finalization adapter inside the V11 executor does not require an extra
module.

The required proof closure is:

1. `lewm/tests/n5_hierarchical_first_hit_v11_synthetic_execution.py`;
2. `lewm/tests/test_go2_observable_camera_ray_fit_v4_n5_hierarchical_first_hit_v11.py`;
3. `lewm/tests/test_go2_observable_camera_ray_fit_v4_n5_hierarchical_first_hit_v11_lifecycle.py`;
4. `docs/lewm_go2_observable_camera_ray_fit_v4_n5_hierarchical_first_hit_v11_implementation_handoff_2026-07-14.md`.

The canonical review is
`docs/lewm_go2_observable_camera_ray_fit_v4_n5_hierarchical_first_hit_v11_independent_review_2026-07-14.json`.
The future reviewer must start with `/root/` and differ from the amendment
author and `/root/camera_v10_later_rung_plan`. The implementation author may
not self-review. Only a canonical different-agent `PASS` that binds every
source, proof, predecessor, and terminal receipt may authorize one exact V11
attempt.

## Exact namespace and retained lifecycle

The only possible later-authorized V11 output root is

`.generated/go2_observable_camera_ray_fit_v4/n5_hierarchical_first_hit_v11`

and it permits exactly one fresh attempt at
`attempts/seed_20260710/n5`. V9 and V10 roots are read-only terminal evidence.
V11 must not open any predecessor checkpoint/result/completion path or numeric
payload.

The V10 lifecycle and process-boundary protections remain unchanged: one
isolated synchronous canonical-path operation; exclusive one-shot reservation;
component-wise no-follow descriptors; closed owned-directory transactions;
exact inotify event provenance; permanent journal poison; source/RGB rehashes;
fresh compute-only verifier child; parent-only publication; bounded canonical
success/failure envelopes; durable `verification_failure.json` fsynced before
cleanup; diagnostic survival; preserve-artifacts behavior if diagnostic
publication fails; no fallback; no repair; and terminal all-false licenses on
failure.

If exact execution is later authorized, GPU work is pinned only to discrete
GPU0/R9700, the Raphael iGPU is forbidden, no HSA override is permitted, RGB
workers remain in `[1,5]`, native math threads remain one, and V11 is serialized
against every other `.generated` mutator.

## Mandatory adapter and gate proofs

Before review `PASS`, CPU-only tests with all accelerators hidden must prove:

1. **Failure reproduction:** a structurally V11-valid hierarchical loss record
   is rejected by the raw retained V4 loss validator with `evaluation loss
   fields changed`, while the compatibility view is accepted.
2. **Nonmutation:** canonical bytes and hashes of the original evaluation are
   identical before and after adaptation; recursively, all original nested
   objects retain their values.
3. **Only-key delta:** a structural diff between original and compatibility
   view contains exactly two operations per control: remove
   `hierarchical_first_hit_nll` and add `ordered_first_hit_nll` with the exact
   same value. Metrics, total, controls, mappings, and mapping hashes are
   identical.
4. **Negative cases:** missing hierarchical key, preexisting legacy key, both
   keys, extra fields, booleans, nonnumbers, NaN, infinities, negatives,
   changed totals, malformed controls, wrong mapping, and malformed metric
   records all fail closed before a retained gate decision.
5. **Full retained gate:** a complete production-ineligible synthetic N5
   evaluation with frozen target-partition counts passes through the shared
   adapter and the actual retained `_validated_metric_evaluation`, then the
   actual unchanged `_gate_stage`; it produces exactly 26 checks. The existing
   frozen ladder-gate test fixture semantics for binary, raster, family,
   distance, depth-quantile, and target-count records may be reproduced, but
   no canonical result or predecessor numeric payload may be opened.
6. **Both controls:** matched and wrong-RGB records are adapted independently,
   retain the frozen mappings `[0,1,2,3,4]` and `[1,2,3,4,0]`, and yield the
   same target-partition signature as the unadapted metric subtrees.
7. **All production call sites:** AST and runtime-spy tests prove the exact
   child, parent receipt validator, and parent finalizer each invoke the single
   adapter once and pass only its returned private view to the retained gate.
   A raw-evaluation bypass must fail a regression test.
8. **Unchanged decision:** for passing and failing synthetic metric bundles,
   the V11 gate output equals the frozen retained gate output bit-for-bit after
   the name-only view; thresholds and failure lists are unchanged.
9. **Published schema:** result and metric-receipt evaluations expose
   `hierarchical_first_hit_nll`, reject `ordered_first_hit_nll`, and never
   publish the compatibility view as scientific output.
10. **Frozen bytes:** the retained loss/model and ladder-gate file hashes equal
    the values in this amendment, and all 26 threshold constants rehash
    unchanged.

## Mandatory real production-helper subprocess proof

The V11 review must execute a real fresh
`sys.executable -I -B ... --verification-child` boundary on CPU with every
accelerator hidden. A real subprocess that only exercises a separate linear
model and a custom finite/difference check is insufficient.

The production-ineligible smoke must invoke the same shared production
adapter used by exact child and parent calls and must validate its output with
the actual frozen retained V4 loss validator. It must also run the complete
synthetic 26-check fixture through the same production gate-reconstruction
helper used by exact execution. Separate smoke-only gate reconstruction logic
is forbidden.

The V10 transport and failure suite remains mandatory: real serialization,
stdin/stdout, `-I -B`, PID/parent and environment binding, canonical request
and response, fresh CPU state load, matched and wrong-input computation,
success response validation, injected failure at every closed phase, timeout,
signal, nonzero, malformed, oversized, stderr, bounded captures, sanitized
failure envelopes, exact/smoke schema mutual rejection, and temporary-tree
removal. At least one test must launch the actual V11 executor as a fresh
process without monkeypatching `subprocess.run`.

Smoke requests, checkpoints, evaluations, responses, and gates are fixed
`production_eligible=false`, use only in-memory values or a private temporary
directory, publish nothing, and accept no source review, path, seed, data,
checkpoint, output, or authority argument.

## Retained science and lifecycle proof

The implementation author and future reviewer must additionally prove:

- normalized V10/V11 AST identity for model construction, all four loss
  computations, target derivation, training, evaluation, checkpoint metadata,
  metric accumulation, wrong-RGB mapping, schedule, resource controls, and
  phase-tagged lifecycle outside the named adapter and finalization call sites;
- exact schedule hash, 4,000 updates, 20,000 exposures, 41 diagnostics, four
  equal hierarchical loss names/weights, and final-update-only state;
- independent trainer/verifier evaluation equality after a fresh CPU state
  roundtrip without canonical data access;
- every applicable V10 no-follow transaction, journal, isolation, failure,
  cleanup, diagnostic, source-binding, and all-false-license regression;
- fixed V11 paths and schemas, one fresh exclusive attempt, and absence of V10
  checkpoint/result/completion or numeric access;
- no dynamic plugin/import path, caller-supplied adapter, callback, backend,
  alternate opener, runtime monkeypatch of the retained gate, threshold change,
  test outcome promotion, GPU1 path, or retry API; and
- a different-agent source review that reruns the real subprocess and full
  retained closures and publishes `PASS` or `BLOCK` last.

All author and review tests are CPU-only, use temporary roots, hide all
accelerators, cap native math threads at one, and open no canonical experiment
RGB/data, checkpoint, model output, V11 output, G2, held-out, runtime, hardware,
production, or navigation path.

## Execution sequence

1. Freeze this source-free amendment before any V11 source or proof is
   created.
2. `/root/camera_v10_later_rung_plan` constructs and freezes the exact V11
   source/proof closure without exact, data, RGB, checkpoint, or GPU work.
3. A future reviewer distinct from both authors rehashes all predecessor and
   V11 bytes, executes the real production-helper subprocess proof and all
   retained CPU closures, and publishes canonical `PASS` or `BLOCK` last.
4. Only `PASS` may authorize one fresh V11 attempt on GPU0, serialized against
   every `.generated` mutation.
5. V11 success requires the independently recomputed evaluation to traverse
   the audited adapter in child and parent, then pass all 26 unchanged checks.
   Any infrastructure or numeric failure is terminal and grants no V11 retry.
6. A full unchanged numerical pass may license design and different-agent
   review of the next preregistered fit rung. It does not itself authorize that
   rung, another seed, shared-JEPA training, G2, held-out, navigation, runtime,
   hardware, production, promotion, or deployment.

## Explicit non-authority

This amendment licenses only V11 source construction by the fixed
implementation author and a later different-agent review. It does not
authorize exact execution, canonical data/RGB opening, checkpoint loading,
GPU use, V9 or V10 retry, predecessor artifact recovery, threshold or
calibration change, model or capacity change, N16/N320, a second seed,
shared-JEPA training, selection, G2, held-out navigation, runtime, hardware,
production, promotion, deployment, or any numerical claim.
