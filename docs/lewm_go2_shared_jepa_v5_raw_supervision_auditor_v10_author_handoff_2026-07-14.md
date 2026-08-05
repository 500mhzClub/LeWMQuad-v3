# Shared JEPA V5 raw-supervision Auditor V10 author handoff

Date: 2026-07-14

Implementation author: `/root/raw_v9_auth_hash_witness`

Status: **FROZEN AUTHOR CANDIDATE; NO REVIEW, AUTHORIZATION, OR EXACT AUTHORITY**

## Governing amendment

This audit-only successor implements the frozen manifest-representation amendment:

| Artifact | SHA-256 |
| --- | --- |
| `docs/lewm_go2_shared_jepa_v5_raw_supervision_auditor_v10_manifest_representation_successor_amendment_2026-07-14.md` | `02100ee004a572209866a3eb9356441600944b2da2d9b1010282ab992ad02a81` |

The amendment grants source construction and different-agent review only. It does
not grant an exact audit attempt, dataset use, retry, rebuild, training, G2,
held-out, runtime, hardware, production, promotion, or deployment authority.

## Frozen implementation

The three implementation artifacts frozen by the author are:

| Role | Artifact | SHA-256 |
| --- | --- | --- |
| `auditor_source` | `lewm/datasets/go2_shared_jepa_v5_raw_supervision_auditor_v10.py` | `3c87dc7878f2e0ae9c54e9b05f1183339b9839568832c0e2fcb6ce75dda984d9` |
| `auditor_cli` | `scripts/audit_go2_shared_jepa_v5_raw_supervision_v10.py` | `695653257d4aeccef162f3e8f30fd0eba88a090f29cb811481898b4680fe3866` |
| `auditor_test` | `lewm/tests/test_go2_shared_jepa_v5_raw_supervision_auditor_v10.py` | `af084a3d097ae66db14f68db7c700843f1bc4007515eb07e444a5018036f177d` |

This handoff is the fourth V10 review-candidate role, `auditor_handoff`. Its file
SHA-256 is intentionally computed after these bytes are frozen and bound by the
different-agent review and later V10 audit authorization; it is not embedded in
itself.

## Sole science correction

Auditor V10 mechanically inherits Auditor V9's audit science, worker policy, and
closed no-replace publication transaction. The only replay correction keeps the
two representations distinct:

1. `_read_exact_source_json` returns the original decoded `dict` as
   `raw_scene_manifest`.
2. Its canonical JSON fingerprint is checked across parsing, direct raw-object
   validation, and render-contract validation.
3. The exact original object is passed directly to
   `_validate_raw_scene_object_records`; no `to_dict`, `asdict`, copy, tuple/list
   conversion, or JSON round trip is used.
4. The parsed immutable `SceneManifest` remains the only value used for semantic
   hashing, rendered geometry, ray evidence, rasterization, and array comparison.

The focused tests reproduce the terminal V9 `wall boxes are not a list` failure
with a real parsed manifest. They prove direct raw-object identity, reject tuple,
mapping-proxy, generator, reconstructed, and mutated representations, and verify
that erasing exactly the raw parameter and integrity guards reduces both changed
functions to the frozen V9 AST.

## Audit-only authority consumer

The source does not create or publish an authorization. It consumes only the
future canonical file:

`docs/lewm_go2_shared_jepa_v5_raw_supervision_audit_v10_authorization_2026-07-14.json`

The strict V10 authorization schema binds eleven ordered, unique roles:

1. `amendment`
2. `v9_build_authorization`
3. `v9_builder_source`
4. `v9_builder_review`
5. `v9_dataset_manifest`
6. `v9_terminal_failure`
7. `auditor_source`
8. `auditor_cli`
9. `auditor_test`
10. `auditor_handoff`
11. `auditor_review`

Phase one validates structure, canonical content self-hash, exact paths, all
authority booleans, frozen hashes, reviewer identity, and the four-role review
candidate without opening a mapped target. Phase two opens only bound targets,
deep-validates the frozen V9 build authorization, and reopens/hashes all nine V9
source-map entries. The accepted capsule separately carries V9 build provenance
for the unchanged dataset manifest and V10 execution authority for the live
transaction and report.

The literal leaves are additive:

- success: `development_raw_supervision_v1.audit_v10.json`
- terminal failure: `development_raw_supervision_v1.audit_v10.failed.json`

The existing V9 terminal failure receipt is required read-only evidence and is
not a V10 namespace collision. Success and failure remain downstream-false and
grant no retry.

## Author verification

All executed tests used CPU-only imports and pytest temporary roots with third
party plugin autoload disabled, one native math thread, and CUDA/HIP/ROCR
visibility empty.

| Check | Result |
| --- | --- |
| Focused Auditor V10 author suite | `25 passed` |
| Frozen Auditor V9 author suite | `61 passed` |
| Frozen Auditor V9 independent QA suite | `7 passed` |
| V10 science-region AST versus V9 | exact after `V10/v10` authority rename |
| Two corrected boundary functions versus V9 | exact after erasing only the permitted raw delta |
| All closed-publication transaction methods versus V9 | exact after context/type rename |
| One-worker versus six-worker synthetic result bytes | exact |
| Spawn initializer/task reauthorization and CPU-only environment | PASS |
| V10 temp-root failure publication and V9 receipt preservation | PASS |
| `py_compile` for source, CLI, and test | PASS |
| CLI fixed argument surface | PASS |
| Candidate trailing-whitespace check | PASS |

No Builder, Auditor, trainer, exact command, canonical dataset, corpus, RGB,
checkpoint, G2, held-out, runtime, navigation, hardware, production, or
accelerator workload was run during author verification. No canonical
`.generated` byte or V9 receipt was modified.

## Required next step

A different eligible agent must hash this handoff, independently review the exact
four-file candidate, and publish a canonical `PASS` or `BLOCK` review. Only a
`PASS` review permits `/root` to publish a separate V10 audit authorization.
Only after another agent independently reproduces that authorization file's
SHA-256 may the single exact V10 audit attempt run.
