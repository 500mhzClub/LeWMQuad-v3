# Camera-ray N5 gate-aligned raster NLL V12 implementation handoff

Date: 2026-07-14

Implementation author: `/root/camera_v12_gate_aligned_implementer`

Status: **source and synthetic CPU closure complete; independent review required; no exact authority**

## Frozen authority

The source-free amendment is:

`docs/lewm_go2_observable_camera_ray_fit_v4_n5_gate_aligned_raster_nll_v12_successor_amendment_2026-07-14.md`

File SHA-256:

`77de8c69b1bef69ab3d1b976567eb20371f53d47d81af757ef8c7fdaade93c1b`

The amendment author is `/root/camera_v10_later_rung_plan`. The fixed
implementation author matches the amendment and differs from the amendment
author. A canonical reviewer must start with `/root/` and differ from `/root`,
the amendment author, and the implementation author.

## Frozen production closure

| Role | Path | File SHA-256 |
| --- | --- | --- |
| V12 gate-aligned raster NLL | `lewm/models/observable_camera_ray_evidence_v4_gate_aligned_raster_nll_v12.py` | `735563f811c5d7b9efb9e37dca8348825a8467bd0a059f83ab94d41d45d57662` |
| V12 policy | `lewm/benchmarks/go2_observable_camera_ray_fit_v4_n5_gate_aligned_raster_nll_v12.py` | `ad8a77c4f201f00891e7e6b45c395966eaa8f3723a3b2720d26eeb0b1ca23fc6` |
| V12 trainer | `scripts/train_go2_observable_camera_ray_fit_v4_n5_gate_aligned_raster_nll_v12.py` | `91018ecd28483fbbc3399eea70d720a9b327e7e03b4920dbe349ca9b81603d54` |
| V12 verifier | `scripts/verify_go2_observable_camera_ray_fit_v4_n5_gate_aligned_raster_nll_v12.py` | `f8814836c1073f13c563ba11035f806a0faa70be9a0d44b7d3e900350b1a8baf` |
| V12 executor | `scripts/execute_go2_observable_camera_ray_fit_v4_n5_gate_aligned_raster_nll_v12.py` | `4e4c45c85827ad4db6e65a4f02557fd6c5b1e9d97ada4ac4577cb0b6b099b521` |

The retained hierarchical first-hit model/loss remains byte-identical at
SHA-256 `52bc99f0ba59c2cf7444221931169ba57af61f343308b85625877c7a257adffd`.
The retained V4 ladder gate remains byte-identical at SHA-256
`aa51413edfea10a2d7c04b034033c83c78c27b1c08d2be1413f5917dc32e36ad`.

## Frozen proof closure

| Role | Path | File SHA-256 |
| --- | --- | --- |
| Synthetic lifecycle and native V12 gate fixture | `lewm/tests/n5_gate_aligned_raster_nll_v12_synthetic_execution.py` | `1cbcb80d3f6bec5b9ce536d6b4fa9bad645d170a4be6b4e8d1b261ad5f5dc453` |
| V12 loss, parity, gradient, and diagnostic tests | `lewm/tests/test_go2_observable_camera_ray_fit_v4_n5_gate_aligned_raster_nll_v12.py` | `98a11ec91865ff106dd943a6b6468ca227018d92db4a346fcbbe9497a7d8d099` |
| V12 lifecycle, schema, gate, and subprocess tests | `lewm/tests/test_go2_observable_camera_ray_fit_v4_n5_gate_aligned_raster_nll_v12_lifecycle.py` | `77b5d05373613220a0de1d78236659f0c038e9f1a91f3a4efbf5cbcaa73936c1` |

This handoff is the fourth proof file. The reviewer must hash its final bytes
and bind that hash in the canonical review.

## Retained V11 closure

The V12 policy binds the exact V11 amendment, handoff, review, model, policy,
trainer, verifier, executor, synthetic proof, science tests, and lifecycle
tests frozen by the V12 amendment. In particular, the V11 production sources
remain:

| Role | File SHA-256 |
| --- | --- |
| V11 policy | `75b017d73181baaffb8e05931e0af7b53b4fd24b8a8b77740009fc7297e43cd5` |
| V11 trainer | `99de094c1df010f17c26d6f6109ff256a658d74f7799275bf572eae6afa5a1ae` |
| V11 verifier | `7cf4d8e7649cd735156bf1e92b6f12b49754f804832a2af7c3ffc2b7229ddf51` |
| V11 executor | `401b46296fd367e2945d8e53844c0e80242ee1dc5bd5412f2a89f43fe4f22bc9` |

The policy also binds the five amendment-declared terminal V11 receipt file
and content hashes. It interprets them only as a 25/26 terminal numeric failure
at `matched.raster_nll = 0.07255925759673118 > 0.06`, with no retry or checkpoint
input authority. The implementation author did not open those `.generated`
receipts or the V11 checkpoint during source construction.

## Implemented scientific boundary

V12 preserves the V11 model, target derivation, hierarchical first-hit loss,
offset loss, state-balanced ground loss, state-balanced hierarchical raster
loss, soft rasterizer, five-frame subset, seed `20260710`, 4,000 AdamW updates,
20,000 frame exposures, full-panel batch size 5, schedule SHA-256
`fb5a6c13708944b6ce514960c11c063eece704676276b352bd5233080f4fd380`,
gradient clipping at `1.0`, final-update-only selection, batch-one evaluation,
both RGB controls, and all 26 retained thresholds.

The sole scientific delta is:

```text
G = mean(-log(gather(class_probabilities, target_class).clamp_min(float32_eps)))
L_V12 = L_V11 + 0.25 * G
```

All five coefficients are exactly `0.25`; the objective is not renormalized.
The retained four-term V11 base total is published separately at every frozen
diagnostic and evaluation. The retained `losses` record remains four-term V11
semantics. The native `native_v12_objective` record publishes `G`, the V11 base
total, and the exact V12 total without relabeling a V12 scalar as V11.

The trainer computes `G` from the same soft-raster class probabilities used by
the retained metric accumulator. The verifier has a separate inline
`gather -> clamp -> log -> mean` implementation and recomputes model outputs,
`G`, metrics, confusion, class/family diagnostics, and both controls from the
final V12 checkpoint. It never reuses a trainer scalar, result metric,
confusion, NLL sum, or diagnostic.

Each control publishes aggregate-only overall, UNKNOWN/FREE/OCCUPIED, and
registered-family NLL count/sum/mean records. Validators require both class and
family partitions to reconstruct the overall count and sum, every mean to
reconstruct sum/count, and the overall mean/count/sum to agree with the retained
metric accumulator. Missing classes require zero count, zero sum, and null mean.

The sole private compatibility helper is:

`adapt_native_v12_evaluation_for_retained_v4_gate`

It validates and deep-copies the native record, removes only additive V12
objective diagnostics from the private copy, replaces only
`hierarchical_first_hit_nll` with `ordered_first_hit_nll`, proves the caller and
retained metrics/control/mappings unchanged, and invokes the actual frozen loss
validator. Only `reconstruct_retained_v4_gate` invokes the frozen metric and
gate internals. Child verification, parent receipt validation, finalization,
and the real CPU smoke all use this boundary. The compatibility view is never
published.

Fresh model initialization is mandatory. No V11 checkpoint path can be passed,
opened, loaded, copied, hashed, or used as state. Metric verification may open
only the final checkpoint created by the sole future V12 attempt.

## Proof results

The final author command hid every accelerator selector, set all native math
threads to one, disabled external pytest plugins, and ran:

```text
lewm/tests/test_go2_observable_camera_ray_fit_v4_n5_gate_aligned_raster_nll_v12.py
lewm/tests/test_go2_observable_camera_ray_fit_v4_n5_gate_aligned_raster_nll_v12_lifecycle.py
lewm/tests/test_go2_observable_camera_ray_fit_v4_ladder_gate.py
```

Result: **202 passed in 18.77 seconds**.

The count partitions as 23 V12 science tests, 159 V12 lifecycle tests, and 20
frozen ladder-gate tests.

The closure proves:

- exact float32 gather/clamp/log/all-cell-mean arithmetic, finite gradients,
  nonmutation, input validation, and epsilon behavior;
- literal batch-5 versus five batch-1 parity and equality to retained metric
  `nll_sum / count`;
- exact four retained `0.25` terms plus additive `0.25 * G`, without removal or
  renormalization;
- the frozen 16,123 UNKNOWN, 4,259 FREE, 98 OCCUPIED `R/O/U/F` decomposition and
  missing-class behavior;
- class/family partition, sum, mean, finite-value, mutation, and merge checks;
- independent trainer/verifier recomputation with a runtime spy proving the
  verifier does not call or reuse the trainer scalar;
- native-record validation, private compatibility nonmutation, and unchanged
  26-check names, values, thresholds, decisions, and gate implementation;
- a high-balanced-accuracy fixture still fails closed solely at
  `matched.raster_nll` when NLL exceeds `0.06`;
- exact result, training, evaluation, receipt, completion, gate, source, review,
  self-hash, resource, access-ledger, and all-false license schemas;
- inherited no-follow, exclusive reservation, inotify journal, recovery,
  transactional publication, failure cleanup, one-shot, no-retry, timeout,
  signal, malformed/oversized response, unexpected stderr, source/review drift,
  and every verifier-phase failure behavior; and
- an actual `sys.executable -I -B --verification-child` subprocess traversing
  independent `G`, native diagnostics, the compatibility boundary, and the
  unchanged 26-check gate on hidden CPU.

The real smoke summary reported:

```text
real_subprocess=true
isolated=true
no_bytecode=true
accelerators_hidden=true
independent_v12_raster_nll_recomputed=true
native_class_family_diagnostics_recomputed=true
native_to_retained_compatibility_boundary_exercised=true
retained_gate_check_count=26
phase_failures_validated=11/11
process_cases_validated=timeout,signal,nonzero,malformed,oversized,stderr
publication_performed=false
```

## Access and authority closure

During implementation and proof execution:

- every accelerator selector was empty and no GPU or iGPU operation ran;
- no canonical experiment data, RGB payload, `.generated` receipt payload,
  checkpoint, numeric result, held-out, G2, navigation, runtime, hardware, or
  production payload was opened;
- the V11 checkpoint was not opened, copied, hashed, loaded, or used;
- the future V12 output root did not exist;
- the canonical V12 independent-review file did not exist;
- no exact training, exact verification, finalization, or publication ran; and
- no canonical review was written by the implementation author.

Every V12 gate license remains false even if the numeric gate passes, including
checkpoint use, retry, second seed, N16, shared-JEPA, held-out, G2, selection,
calibration, navigation, runtime, hardware, later-rung design/execution,
production, promotion, and downstream work.

## Independent review

The next action is a different-agent source review only. The reviewer must:

1. rehash the amendment, full V11 closure, five terminal V11 receipts, every V12
   source/proof, this handoff, and the retained gate;
2. independently reproduce the reduction analysis and inspect the sole loss
   delta, native record semantics, diagnostic partitions, compatibility helper,
   fresh-init boundary, finalizer, and licenses;
3. rerun all hidden-CPU source, parity, gradient, lifecycle, and real-child
   proofs without canonical data, checkpoint, `.generated` payload, or GPU
   access; and
4. publish
   `docs/lewm_go2_observable_camera_ray_fit_v4_n5_gate_aligned_raster_nll_v12_independent_review_2026-07-14.json`
   as `PASS` or `BLOCK` last.

Only a canonical different-agent `PASS` binding these exact frozen bytes may
authorize the sole fresh V12 attempt at
`.generated/go2_observable_camera_ray_fit_v4/n5_gate_aligned_raster_nll_v12/attempts/seed_20260710/n5`.
This handoff grants no exact, data, RGB, checkpoint, GPU, retry, second seed,
later rung, full training, navigation, production, or promotion authority.
