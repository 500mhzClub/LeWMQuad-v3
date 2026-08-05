# Shared JEPA V5 full-training V3 successor amendment

Date: 2026-07-14

Status: **source-free contract frozen before V3 implementation, preflight,
payload access, accelerator use, training, selection, calibration, or G2;
different-agent implementation review required**

Implementation author: `/root/full_training_v3`

## Purpose and authority

This additive amendment retains the independently reviewed Shared JEPA V5
full-training V2 lifecycle and closes the five gaps recorded in
`docs/lewm_go2_ready_to_benchmark_handoff_2026-07-14.md`:

1. use the terminal Builder V9 and Raw Auditor V13 provenance chain instead of
   the obsolete Builder/Auditor V1 bindings;
2. train the Camera successor objective that retained hierarchical first-hit
   supervision and added gate-aligned all-cell raster NLL;
3. freeze the nonlinear reduction at real microbatch size four;
4. publish a strictly pre-G2 candidate schema rather than the post-G2 Shared
   V5 checkpoint schema; and
5. require the future Camera V13 two-seed N5/N16/N32/N320 ladder and primary
   N320 checkpoint without inventing their hashes before those artifacts
   exist.

This document licenses only additive source implementation and source-only
CPU/synthetic proof. It authorizes no preflight execution, canonical
`.generated` open, dataset/RGB/label/checkpoint open, accelerator use, exact
training, selection, calibration, G2, held-out, navigation, runtime, hardware,
production, promotion, deployment, or retry. A different eligible agent must
review the exact V3 implementation closure before any later authority can be
constructed.

## Frozen V2 base

The following reviewed V2 bytes are the structural base. V3 may not mutate
them.

| Artifact | SHA-256 |
|---|---|
| V2 design amendment | `b521d2885b5dca1a72838282fbb8e193a21ec0f2db0e0a5950074506fba1f66d` |
| V2 design PASS record | `6a53a3c9d72da6499714883676f49a62d0c3ba61c2d2ccde741f1654e6f089d4` |
| V2 policy | `e0c3409ce104d954e40aa73ae5bd5b79ec3daa77564e90c6be183c2fbc19f680` |
| V2 preflight executor | `fbc6d63394625d2c3ccc79821d9a07b507fdfb95e02ee1768ed6325857531eff` |
| V2 preflight verifier | `1453a6a6134c25cad21d41f44628e4cc8e1e041ae8994d570413ebb1101e09e3` |
| V2 exact executor | `698fb92f2f854365f2d0bfbf6f034b1c3f04704a8d6227fceff7c3ed275fc271` |
| V2 exact trainer | `bdd8e4b1c24e855f3e3ff535a195f2c370c4ffdadc48eb9e83b214b53362f23b` |
| V2 exact verifier | `d8950c8bf23b0bd5494c7c864f2f2543d533b0bc07af3f70287291227c872543` |
| V2 implementation review JSON | `2ce422c2821491f936af9b47a5898f90969723338195d7f2069902357297132a` |

V3 retains the V2 one-shot reservation ordering, payload-free preflight,
R9700-only exact device contract, fixed schedule and optimizer, matched
initialization, development-role isolation, selection/calibration gates,
diagnostic-only no-JEPA arm, immutable output, complete actual-open ledger,
independent reconstruction, no retry, and closed G2/held-out/runtime/production
boundaries except where this amendment explicitly supersedes a field.

## Terminal raw-supervision chain

V3 binds the already completed Raw V13 evidence without reopening it during
source implementation:

| Role | Path or identity | SHA-256 |
|---|---|---|
| Builder V9 source | `lewm/datasets/go2_shared_jepa_v5_raw_supervision_builder_v9.py` | `2388c1138d9b03ea6e385cc0250c81a1869a40cab62507d02f709ef39197c664` |
| Builder V9 review | `docs/lewm_go2_shared_jepa_v5_development_raw_supervision_builder_v9_independent_review_2026-07-13.json` | `c39eb2787c37f8cab064de75355b3af56971ef98209d329e4789eb383c1dc60f` |
| Auditor V13 source | `lewm/datasets/go2_shared_jepa_v5_raw_supervision_auditor_v13.py` | `fddc678187f082a0a245ff5868ca5d944cba4adc2703d3b97088d57451deb4b7` |
| Auditor V13 review | `docs/lewm_go2_shared_jepa_v5_raw_supervision_auditor_v13_independent_review_2026-07-14.json` | `f3705d1a300204a3e4f7e52b31fae5401b56bbe8de018972ebe66f046c9b2343` |
| Auditor V13 authorization | `docs/lewm_go2_shared_jepa_v5_raw_supervision_audit_v13_authorization_2026-07-14.json` | `8a12c5f8d6c6e64a418052cf01177dd25049d6d373f7e87cd52c5d2a5b2bf587` |
| Auditor V13 fingerprint witness | `docs/lewm_go2_shared_jepa_v5_raw_supervision_audit_v13_authorization_fingerprint_2026-07-14.json` | `882bf8877b12874998ad0f4d179d89ebe8d7db048ffdf3ddc03d4ea38ea5b846` |
| Dataset manifest file | canonical V9 manifest | `e102b3c64e99029f118597353966edaaaddbc11efe49b9081d5d7a9c9d974360` |
| Dataset manifest content | canonical V9 manifest content | `74ae5799919ff4d9a06f56d98929cb4cb702d64db52ecdfc93cfa9a8e82fb35a` |
| Raw V13 PASS file | canonical terminal report | `0680e1680f30c45feda60498792c3f208c28313e8f087dfbdd1c5807bcf1fe76` |
| Raw V13 PASS content | canonical terminal report content | `0c16e368c9de258d0fbf46e3123d7a3cfcdf60162fd9efa6440d4a7773056aca` |

The exact V3 trainer must independently reopen and validate the complete
Builder V9 review, Auditor V13 review/authorization/fingerprint, dataset
manifest, and Raw V13 PASS after durable exact reservation and before the
first dataset leaf. It must require exact schemas, paths, hashes, PASS verdict,
the fixed population counts, all downstream authority booleans false in the
raw report, and exact transitive source-map commitments. A matching file hash
alone is insufficient.

This V3 amendment supplies the new, narrow dataset-use grant that Raw V13
deliberately did not supply. The grant is valid only inside one reviewed V3
exact attempt, only for the three frozen development roles, only after the
above chain and the V3 execution manifest pass, and only for the fixed training,
selection, and calibration operations. It does not grant RGB use outside the
bound raw-supervision leaves, G2, held-out, navigation, runtime, hardware,
production, promotion, deployment, or retry.

## Correct Camera objective

The V3 loss adapter must bind these unchanged model/loss sources:

| Source | SHA-256 |
|---|---|
| Shared V5 model | `b438295d7ec5cb0897cc953a229f461da7fca16322c4c936555d37833a36e4b9` |
| Hierarchical first-hit V9 | `52bc99f0ba59c2cf7444221931169ba57af61f343308b85625877c7a257adffd` |
| Gate-aligned raster NLL V12 | `735563f811c5d7b9efb9e37dca8348825a8467bd0a059f83ab94d41d45d57662` |

For one real frame microbatch of exactly `B=4`, derive targets and the soft
raster once, then compute these scalar terms separately:

```text
H = hierarchical_first_hit_nll_v9
O = target_bin_offset_smooth_l1
G = ground_clear_distance_state_balanced_bce
R = derived_raster_hierarchical_bce
N = derived_raster_cell_nll_v12

frame_total_B4 = 0.25 * (H + O + G + R) + 0.25 * N
```

`ordered_first_hit_nll` is forbidden from V3 backward loss and diagnostics.
The current and next frame losses must each be computed from their own real
`B=4` tensors. Only after both scalars exist may they be combined:

```text
camera_pair_B4 = 0.5 * current_frame_total_B4
               + 0.5 * next_frame_total_B4
promoted_backward_B4 = established_jepa_total_B4 + camera_pair_B4
matched_backward_B4  = camera_pair_B4
```

The existing Shared V5 configuration weight remains `1.0`; V3 must reject any
other camera weight. Current and next terms may not be concatenated before a
nonlinear balanced/grouped loss. Missing hit/no-hit, represented-depth,
ground-state, or raster classes retain the exact source loss semantics; no
synthetic padding or cross-frame pooling is allowed.

Each optimizer update consumes four ordered microbatches of four. It computes
four complete backward scalars independently and performs:

```text
update_backward = 0.25 * (loss_B4_0 + loss_B4_1 + loss_B4_2 + loss_B4_3)
```

Equivalently, each scalar is divided by four immediately before its backward
call. It is forbidden to concatenate the 16 samples and compute any nonlinear
loss once, weight a microbatch by represented-class counts, or pool raw
numerators/denominators across microbatches. Trace diagnostics are the same
four-scalar arithmetic mean. CPU proof must include a constructed case where
the correct four-scalar mean differs from nonlinear synthetic-B16 pooling and
must require the former exactly.

## Camera V13 ladder remains unresolved

V3 source may freeze the required roles and intended canonical namespace, but
must not invent a file hash, content hash, gate PASS, review PASS, or checkpoint
hash for a future artifact. The blocked source-time execution manifest must
leave every Camera V13 future field `null` and list it as unresolved:

- Camera V13 exact source-review PASS file and content hashes;
- the one-shot Camera V13 N5 gate file and content hashes;
- a later source-free ladder preregistration and different-agent review;
- the two-seed `N=5,16,32,320` ladder PASS file and content hashes; and
- the primary seed `20260710`, `N=320` checkpoint file hash.

The later ladder must use fresh initialization at every rung and seed, no warm
start, no observed-gate tuning, both seeds `20260710` and `20260711`, and all
four rungs. Only seed `20260710`, `N=320` may migrate. Resolving a field requires
an additive reviewed binding; source implementation itself grants no Camera
execution or checkpoint authority.

## Strict pre-G2 checkpoint boundary

V3 replaces `qualified_checkpoint.pt` with
`pre_g2_candidate_checkpoint.pt`. Its schema is fixed to
`lewm_go2_shared_jepa_v5_full_training_v3_pre_g2_candidate_checkpoint_v1`.
The object must contain exactly:

- schema, lifecycle stage, model config, deployment-state hash;
- exact development selection and promoted-arm calibration objects;
- `checkpoint_kind = "pre_g2_candidate"`;
- `development_only = true`;
- `independent_exact_reconstruction_required = true`;
- `g2_attempted = false`, `g2_gate_receipt = null`;
- `post_g2_qualified = false`, `runtime_ready = false`;
- held-out/runtime/navigation/hardware/production/promotion/deployment
  authority false; and
- canonical content hash plus the filtered deployment state dictionary.

The Shared model's post-G2 `CHECKPOINT_V5_SCHEMA` is forbidden here. The V3
trainer cannot call this artifact qualified or runtime-ready. The independent
exact verifier may prove the complete attempt and thereby make the immutable
candidate eligible as a later one-shot G2 input; it may not rewrite the
checkpoint or convert its schema. Only the reviewed G2 runner/finalizer/
publisher lifecycle may emit a post-G2 V5 checkpoint after an actual G2 PASS.

## V3 namespaces and process separation

The sole V3 preflight and exact roots are:

```text
.generated/go2_shared_observable_camera_ray_jepa_v5/full_training_v3_preflight
.generated/go2_shared_observable_camera_ray_jepa_v5/full_training_v3
```

They are distinct from V2 and may never reuse a V2 reservation, process,
descriptor, receipt, state, or partial tree. All V2 descriptor-relative,
exclusive, no-follow, fsync, immutable-completion, failure, and no-retry rules
remain literal with V3 schemas and paths. Preflight remains payload-free. The
exact reservation remains standard-library-only and precedes Torch, GPU,
model, checkpoint, manifest payload, dataset leaf, RGB, label, worker,
inference, backward, optimizer, or calibration access.

## Source implementation closure

The additive V3 candidate must contain exactly these production roles plus one
source-only author test:

```text
lewm/benchmarks/go2_shared_jepa_v5_full_training_v3_policy.py
lewm/models/shared_observable_camera_ray_jepa_v5_full_training_v3_loss.py
scripts/preflight_go2_shared_jepa_v5_full_training_v3.py
scripts/verify_go2_shared_jepa_v5_full_training_v3_preflight.py
scripts/execute_go2_shared_jepa_v5_full_training_v3.py
scripts/train_go2_shared_jepa_v5_full_training_v3.py
scripts/verify_go2_shared_jepa_v5_full_training_v3.py
lewm/tests/test_go2_shared_jepa_v5_full_training_v3_implementation.py
```

The policy, executors, and verifiers must remain fail-closed. There is no
dynamic backend/module/callback/test switch. Neural imports remain nested
behind their fixed reservation boundaries. The source test may use CPU Torch
and synthetic tensors only, with accelerators hidden and all native math
threads fixed to one. It must not execute preflight, exact reservation,
training, canonical verification, or open any `.generated` path.

## Required different-agent review

An eligible reviewer must differ from `/root`, `/root/full_training_v3`, the
V2 implementation author, the Raw V13 implementation author/reviewer, the
Camera V13 implementation author/reviewer/executor, and any future exact V3
executor. Review must independently rehash this amendment and every V3 role;
verify exact V2 preservation; validate the Raw V13 chain and narrow dataset
grant without opening payloads; prove hierarchical first-hit and raster NLL
gradient inclusion; prove current/next B4 separation and four-scalar update
reduction; reject nonlinear B16 pooling; prove the strict pre-G2 schema;
confirm every Camera hash remains unresolved; and rerun all source-only CPU
tests with accelerators hidden.

A PASS may authorize only a later payload-free V3 preflight design step. It
does not itself authorize preflight execution, exact reservation, dataset or
checkpoint access, accelerator use, training, selection, calibration, G2,
held-out, navigation, runtime, hardware, production, promotion, deployment,
or retry.
