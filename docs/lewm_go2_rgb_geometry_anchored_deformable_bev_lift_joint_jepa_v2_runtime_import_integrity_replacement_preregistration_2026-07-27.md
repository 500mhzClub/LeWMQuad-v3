# Geometry-Anchored Deformable BEV Lift Joint-JEPA V2 runtime-import integrity replacement

Date: 2026-07-27

Status: preregistered for source implementation, source-only testing, independent
review, and a later separately authorized one-shot execution. This document does
not itself authorize runtime execution.

## Decision basis

The only authorized V1 attempt is consumed and permanently closed. Its committed
terminal audit is
`docs/lewm_go2_rgb_geometry_anchored_deformable_bev_lift_joint_jepa_v1_runtime_import_terminal_audit_2026-07-27.json`
at commit `605198aa253b0ec98bccfd81af7cdb68dd48b48e`, raw SHA-256
`59ee565175ab9bb3718ada88a7d195fd85cc7855b8355e1d0151f5a6ec01332d`,
content SHA-256
`5427c28a4a4624cc7d786d909e87fda68b8716290456f0bbecf287118cf87f5f`,
and 7,693 bytes.

That audit establishes a valid zero-exposure operational import failure. The
attempt stopped at the reserved stage with `ModuleNotFoundError: No module named
'lewm'`. It performed zero updates, presentations, objectives, backward calls,
optimizer updates, EMA updates, data loads, checkpoint loads, trace writes, or
GPU work. It neither supports nor rejects the scientific mechanism.

The root cause is one lazy-import lifetime defect. The V1 runner temporarily
placed the repository root on `sys.path` while source-loading the matched runner,
then removed it before calling that runner's `_load_runtime()`. The delayed
absolute imports from `lewm` therefore had no import root in the exact isolated
`-I -B` interpreter.

Standing user authority permits working through obvious operational defects
while metrics have not rejected the scientific idea. V2 is therefore exactly
one science-identical integrity replacement, not a retry, resume, repair of the
V1 root, or a scientific successor.

## Sole permitted implementation delta

V2 may change only the repository-root lifetime around the existing
`_load_post_reservation_stack` operation:

1. snapshot the complete original `sys.path`;
2. make the canonical repository root available while source-loading the
   matched runner, calling its unchanged `_load_runtime()`, source-loading the
   unchanged schedule adapter, and source-loading the unchanged V1 model;
3. restore the exact original `sys.path` on both success and exception; and
4. retain the existing pre/post source rehash checks.

No runtime input, checkpoint, tensor, accelerator, held-out, sealed, or V1
runtime artifact may be opened by this change or its source review. Before V2
execution authorization, the exact frozen V2 post-reservation stack loader must
be exercised in the reviewed isolated runtime with `-I -B`. That preflight may
import the reviewed Python/Torch source stack in CPU-only mode, but it must load
no development data or checkpoint, query no accelerator, create no output root,
and restore `sys.path` exactly. This review-time preflight is the required
falsification of the operational fix; the actual one-shot runner still reserves
its new output root before importing Torch or opening runtime inputs.

The inherited complete six-file post-reservation operational-failure receipt
set (`metrics`, `artifact`, `access`, `result`, `failure`, and `completed`), plus
the reservation receipt, remains mandatory for any V2 failure. No receipt may
be synthesized from V1, and V1 output is never a V2 input.

## Frozen scientific identity

V2 must preserve the frozen reviewed V1 source, including the exact V1 model,
and deep-equal all scientific contracts. In particular it preserves:

- RGB-only 112-pixel input, patch size 7, 16-by-16 tokens, width 192;
- geometry-anchored per-cell projection, exactly four bounded local bilinear
  samples, learned offsets and mixing weights, invalid-anchor UNKNOWN handling,
  two local 3-by-3 refinement blocks, and latent width 64;
- the three-way UNKNOWN/FREE/OCCUPIED semantic head and local two-block 3-by-3,
  nine-action predictor, with no global lift or predictor bypass;
- the exact N320-compatible initialization, parameter values, parameter draw
  order, target hard sync, and target EMA momentum 0.996;
- the exact development RGB/raster sources, role partitions, rows, mappings,
  actions, pair identities, and input hashes;
- seed, deterministic controls, schedule order and hashes, batch and microbatch;
- updates 1-400 semantic warmup and updates 401-1000 genuine joint online
  representation/predictor JEPA training against the stop-gradient EMA target;
- every semantic, prediction, retrieval, contrastive, persistence, and shared
  gradient term, weight, reduction, comparator, and threshold;
- every update-0, update-100, update-400, phase-switch, shared-gradient, update-
  1000, mechanism, custody, and numerical gate;
- AdamW groups, learning rates, weight decay, clipping, precision, and update
  ordering; and
- one attempt, no retry or resume, at most 1,000 updates, 16,000 presentations,
  and 30 active GPU minutes.

The frozen V1 science-contract SHA-256 is
`f839076bf7f9db9e9f211703323436f4b607cca21e2e60fb228e4d174c699fa3`.
The frozen V1 model, objective, optimizer, schedule, and gate-threshold hashes
are respectively
`595d91a6fc9ae985378ff480780bf7ad5a9beeb3c7f35ab012c010bb74162f39`,
`93c73c1f1a91de70699f634821159d4d544431b45faa469202016fa0b9fd7ba8`,
`2bb70f943838b656540b3dac3b6e0f30bb384547180270274abfc5077e264b34`,
`bc0ad45c06171cff7533fbfcb054e5afecf6086de0a58060c35cb5ca0256c2e3`,
and `0c485c0bccb88873c0ff76a1061a315420b6c27c4865b259d3b4c6f374862bd0`.

The frozen V1 source manifest is commit
`638fc22118f19e24e9a580b79873833d10fd51f8`, raw SHA-256
`5f5a8931ca9563628c3d1356bb202013830251ec64afca9fee2719c5fd3976a7`,
content SHA-256
`003e149244dba7fc336457240831929dab3228defcfd79225b7a98a76df59582`,
source-bindings SHA-256
`a1e0787f566f2c06c7d9e45e30d5d5053be79c1ab8e6d24691c2959a4a5e2d54`,
74 sources, and 22,933 bytes. The independent V1 source review is commit
`325ecbca05306c060a3ebb686afca2b45643e924`, raw SHA-256
`a11ffc2bafa2e59860d414a0bea64464ff4081e351509e1ef3cf679a9b94d783`,
content SHA-256
`a53ad943e0e8d213b0d948bd16caf228d6f7dfa283eaae24216d2b04ce8bd0c3`.
The consumed V1 authorization is commit
`41a61bb29e0239e54ff76a3cb8384b0062f87783`, raw SHA-256
`c7647c65738298ad68e29173f8a7ebe797322c13ffcb1965f5199dcf7ac7eddb`,
content SHA-256
`31b0e8b6b14719966852fe2995566c6b904d86a6357b20ce9f0bd9159f36ac53`.
It is evidence of V1 custody only and grants no V2 execution authority.

## V2 identity and lifecycle

The V2 experiment ID is
`geometry_anchored_deformable_bev_lift_joint_jepa_v2_runtime_import_integrity_replacement`.
Its sole output root is
`.generated/go2_rgb_geometry_anchored_deformable_bev_lift_joint_jepa_v2_runtime_import_integrity_replacement/attempt_v1`.
The root must be absent before authorization and reservation. Any reservation or
partial attempt consumes it permanently. The V1 root must never be reopened,
reused, renamed, copied, or treated as initial state.

Source implementation may add only a lean versioned governance contract,
runner wrapper containing the import-lifetime correction, launcher wrapper,
recursive closure checker, and focused test module. It must not add or modify a
model implementation. The closure is the exact 74-source V1 closure plus those
five additive V2 files.

Execution requires, in order: a committed V2 source freeze and recursive
manifest; an independent source-and-science-identity review that records the
successful isolated import preflight; a distinct independent authorization for
exactly one attempt; confirmation that the new root is absent; and an exact
launcher invocation bound to the frozen review and authorization hashes.

A passing V2 perception probe still does not authorize G2 navigation, held-out
or sealed access, production, promotion, deployment, checkpoint use, retry,
resume, a second seed, or any parameter/loss/schedule/timing successor. It
authorizes only a separate terminal audit and scientific decision.

No generated input, checkpoint, tensor, runtime output, trace, accelerator,
navigation, held-out, sealed, or rejected material was opened to write this
preregistration.
