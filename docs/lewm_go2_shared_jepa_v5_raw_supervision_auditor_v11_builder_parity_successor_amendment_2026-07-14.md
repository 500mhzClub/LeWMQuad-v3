# Shared JEPA V5 raw-supervision Auditor V11 Builder-parity successor amendment

Date: 2026-07-14

Amendment author: `/root`

Status: **audit-source construction and different-agent review only; no exact authority**

## Purpose and terminal predecessor results

Builder V9 completed its one authorized exact execution and atomically published
the fixed development raw-supervision dataset at:

`.generated/go2_shared_observable_camera_ray_jepa_v5/development_raw_supervision_v1`

The immutable dataset manifest has status
`complete_pending_independent_audit`, file SHA-256
`e102b3c64e99029f118597353966edaaaddbc11efe49b9081d5d7a9c9d974360`,
and canonical content SHA-256
`74ae5799919ff4d9a06f56d98929cb4cb702d64db52ecdfc93cfa9a8e82fb35a`.

The one authorized Auditor V9 execution terminally failed before source replay
because it sent `SceneManifest.to_dict()` tuple-valued fields to a validator of
raw decoded JSON lists. Its immutable terminal receipt is:

`.generated/go2_shared_observable_camera_ray_jepa_v5/development_raw_supervision_v1.audit_v9.failed.json`

with file SHA-256
`863630579e6d8f8ac222ff7ce5ba04ff3e7901885b606dcb6bcfd7a07fe7722f`,
content SHA-256
`aaf342f7df88796e0d03259e964ed51e42ebd1faecb33bbfe9ea9cfd0d5e2c72`,
and status `terminal_failed_no_dataset_authority`.

Audit-only V10 correctly repaired that representation boundary by retaining the
original decoded raw scene-manifest mapping for raw JSON validation and the
parsed immutable `SceneManifest` for semantic hashing and geometry. Its one
authorized execution then exposed a second, independent predecessor defect. It
terminally failed during the exact source-replay dataflow with:

`KeyError: 'sidecar_row_identity_sha256'`

Its immutable terminal receipt is:

`.generated/go2_shared_observable_camera_ray_jepa_v5/development_raw_supervision_v1.audit_v10.failed.json`

with file SHA-256
`2c391550df540d233ded11bfcf1531dbbb29663a51918fb60e7d8cf4146d0996`,
content SHA-256
`66370ec52ae06bef81ab75a47cce481830067b88ec2d579ed41a4a58a7cecc83`,
and status `terminal_failed_no_dataset_authority`.

Both receipts are canonical, newline-terminated, self-hash-valid, downstream
false, and `retry_authorized=false`. Neither failure is a scientific result or
an observed dataset mismatch. Auditor V9 and Auditor V10 are terminal and may
not be modified, rerun, repaired in place, or reinterpreted as a dataset fail.
No V10 success report exists.

No Auditor V11 source, CLI, test, handoff, review, authorization, report,
failure receipt, or exact attempt existed when these amendment bytes were
frozen.

## Frozen V9 evidence

| Role | Path | File SHA-256 |
| --- | --- | --- |
| V9 Builder/Auditor amendment | `docs/lewm_go2_shared_jepa_v5_raw_supervision_builder_auditor_v9_linearization_successor_amendment_2026-07-13.md` | `6fba5de8d7f04d85bd87e084096ae269c3d3dd6368a6db0b0f8f149c1c5cf773` |
| Builder V9 source | `lewm/datasets/go2_shared_jepa_v5_raw_supervision_builder_v9.py` | `2388c1138d9b03ea6e385cc0250c81a1869a40cab62507d02f709ef39197c664` |
| Builder V9 CLI | `scripts/build_go2_shared_jepa_v5_development_raw_supervision_v9.py` | `f239a4ef7c067a71f991b30e14bd5c8632c31be3173780fc25b3d9801fff79ee` |
| Builder V9 test | `lewm/tests/test_go2_shared_jepa_v5_raw_supervision_builder_v9.py` | `541d1957df0a3da18c2b529cd2d7ca721d7e657c8ebcced2a37931d502cab7bc` |
| Builder V9 handoff | `docs/lewm_go2_shared_jepa_v5_development_raw_supervision_builder_v9_author_handoff_2026-07-13.md` | `b6cdf34fa933214e1bb603681f4638f2226e093dad42705445fd8084d6442efd` |
| Builder V9 review | `docs/lewm_go2_shared_jepa_v5_development_raw_supervision_builder_v9_independent_review_2026-07-13.json` | `c39eb2787c37f8cab064de75355b3af56971ef98209d329e4789eb383c1dc60f` |
| Auditor V9 source | `lewm/datasets/go2_shared_jepa_v5_raw_supervision_auditor_v9.py` | `ebe0c6a31cf027b8b0bc049257079a5e0ab0493b12aabeb96bf50f02990bbc14` |
| Auditor V9 CLI | `scripts/audit_go2_shared_jepa_v5_raw_supervision_v9.py` | `76f0b2b29eff8df6905fed142cc622eb0fa8024c397a3c7efb54e58cc36f67ba` |
| Auditor V9 test | `lewm/tests/test_go2_shared_jepa_v5_raw_supervision_auditor_v9.py` | `10951cc2e622281f72ec2a20114ccca184af7624a95fef4683c83dc6839992d1` |
| Auditor V9 handoff | `docs/lewm_go2_shared_jepa_v5_raw_supervision_auditor_v9_author_handoff_2026-07-13.md` | `819d1857bf315f775f45c4a16db994f333d7174c5c20f5cb762f93d04b30a3a5` |
| Auditor V9 review | `docs/lewm_go2_shared_jepa_v5_raw_supervision_auditor_v9_independent_review_2026-07-13.json` | `bd5e46680345977d1b8b08e0060e5b693701165e14490d477b0f7a64bdc856d6` |
| V9 build authorization | `docs/lewm_go2_shared_jepa_v5_raw_supervision_build_authorization_2026-07-13.json` | `7878c807fda5f7057addc302afeca77cf6163683f0b2c5b6d4e4ac2701f86792` |
| V9 dataset manifest | `.generated/go2_shared_observable_camera_ray_jepa_v5/development_raw_supervision_v1/manifest.json` | `e102b3c64e99029f118597353966edaaaddbc11efe49b9081d5d7a9c9d974360` |
| V9 terminal receipt | `.generated/go2_shared_observable_camera_ray_jepa_v5/development_raw_supervision_v1.audit_v9.failed.json` | `863630579e6d8f8ac222ff7ce5ba04ff3e7901885b606dcb6bcfd7a07fe7722f` |

The V9 build authorization has canonical content SHA-256
`d16d0237694e8ed3b9741f6051390de0f759226b49dafa3d8a2ebe95ae7d7df9`
and ordered source-map SHA-256
`a57d8ee6685627aaca1d9c52bbbf35c13a73ecadf852c788fc210566727c57da`.
The Builder V9 review has canonical content SHA-256
`49d8024ae48211cc4fc7d7c2fb674c7ddc7adb38abccace1eb8c6bbc4f10b0df`.
The Auditor V9 review has canonical content SHA-256
`78df31dac5d030ec1f92e024d66ea1b44d56287e91bc3af89d7681b50abcd604`.

## Frozen V10 evidence

| Role | Path | File SHA-256 |
| --- | --- | --- |
| V10 amendment | `docs/lewm_go2_shared_jepa_v5_raw_supervision_auditor_v10_manifest_representation_successor_amendment_2026-07-14.md` | `02100ee004a572209866a3eb9356441600944b2da2d9b1010282ab992ad02a81` |
| Auditor V10 source | `lewm/datasets/go2_shared_jepa_v5_raw_supervision_auditor_v10.py` | `3c87dc7878f2e0ae9c54e9b05f1183339b9839568832c0e2fcb6ce75dda984d9` |
| Auditor V10 CLI | `scripts/audit_go2_shared_jepa_v5_raw_supervision_v10.py` | `695653257d4aeccef162f3e8f30fd0eba88a090f29cb811481898b4680fe3866` |
| Auditor V10 test | `lewm/tests/test_go2_shared_jepa_v5_raw_supervision_auditor_v10.py` | `af084a3d097ae66db14f68db7c700843f1bc4007515eb07e444a5018036f177d` |
| Auditor V10 handoff | `docs/lewm_go2_shared_jepa_v5_raw_supervision_auditor_v10_author_handoff_2026-07-14.md` | `9635d4fa891e9734a6245b8cde3d6eaf8934bb8b9b7c90db9365aa6207e2c959` |
| Auditor V10 review | `docs/lewm_go2_shared_jepa_v5_raw_supervision_auditor_v10_independent_review_2026-07-14.json` | `e03eb3d0d48ca2696b417fe82c221720ace68f8ea535acc2d9399ea080fedffb` |
| V10 audit authorization | `docs/lewm_go2_shared_jepa_v5_raw_supervision_audit_v10_authorization_2026-07-14.json` | `146e0bbf029d28fdf883bfc357b1ddbbce955f86bda00508c6091cb01db4800a` |
| V10 terminal receipt | `.generated/go2_shared_observable_camera_ray_jepa_v5/development_raw_supervision_v1.audit_v10.failed.json` | `2c391550df540d233ded11bfcf1531dbbb29663a51918fb60e7d8cf4146d0996` |

The V10 review has canonical content SHA-256
`725e4bd70c7eca9b0542ee0d51c8cf2d4897db3014b3dc9f2ceda4ea8c4acf6a`.
The V10 audit authorization has canonical content SHA-256
`8bab96369a5633cb82266fef6ec54964a3c25f27dc0877fde550721f3b6af981`
and ordered source-map SHA-256
`1b6ffd40b72c7d02dba24d2035ac3442af361b8b804e7df4273f5e73d1cda79b`.
All eleven bound V10 source-map files matched those frozen hashes after the
terminal attempt. The V10 failure receipt binds that exact authorization, the
unchanged V9 manifest, and the unchanged V9 terminal receipt.

## Complete diagnosis

Auditor V10 retained the correct raw-mapping repair. The original decoded scene
manifest is passed directly to every raw JSON/schema validator and remains
fingerprinted before and after parsing and validation. The parsed immutable
`SceneManifest` remains the value for semantic hashing, typed geometry,
rendered boxes, raycasting, rasterization, and array comparison. V11 must retain
that correction exactly.

The V10 replay failure came from a latent V9 auditor dataflow which the V10
source-equivalence proof intentionally retained:

1. Auditor V10 `_source_record_for_endpoint` searches pair occurrences and
   chooses the minimum `(global_row, side)` occurrence.
2. It returns a reduced reconstructed mapping which omits
   `sidecar_row_identity_sha256`.
3. `_recompute_one_exact_sample` indexes that absent key while constructing
   `FrameBuildInputV4`, producing the terminal `KeyError` before
   `build_frame_evidence_v4` runs.
4. The same replay constructs a four-field `frame_key` containing role, family,
   scene, and endpoint identity. Builder V9 used only
   `{"endpoint_identity_sha256": digest}`.

Adding the selected pair's `sidecar_row_identity_sha256` to the reconstructed
mapping would remove the exception but would not reproduce Builder V9. A real
published pair proves the hashes are semantically distinct: pair sidecar row
identity `0e0b3943d66b8265c5a696f107f104c3a09b964e77db12a89b2a584500c3040d`,
current endpoint content
`f95e4cb1559aa1b58ab2f516388ba60ea24171602d695e392d16a9dc8e8f45a2`,
and next endpoint content
`328744523e758fea7763f9fa09e91d5173ee7d85a86cb4b4072ba18ef932b914`.

Builder V9 is the immutable construction oracle. It:

1. indexes every plan endpoint by `identity_sha256` and rejects duplicates;
2. visits every current and next occurrence in every plan pair;
3. reconstructs this exact ten-field endpoint context from each occurrence:
   `scene_id`, `family`, `episode_id`, `reset_count`, `episode_step`,
   `frame_index`, `env_index`, `timestamp_ns`, `image_sha256`, and
   `image_path_metadata_only`;
4. requires every repeated occurrence of one endpoint to reconstruct exactly
   the same context;
5. rejects absent pair references, conflicting repeated contexts, and orphan
   endpoints;
6. selects the source frame by the strict
   `(frame_index, env_index, timestamp_ns)` key and validates source episode and
   camera provenance against that all-occurrence context;
7. constructs `FrameBuildInputV4` with exactly
   `frame_key={"endpoint_identity_sha256": digest}`;
8. supplies exactly `endpoint["content_sha256"]` to the inherited field named
   `sidecar_row_identity_sha256`; and
9. converts evidence and raster output to the eight arrays in the frozen
   `ARRAY_LAYOUT`.

The endpoint content hash is preserved in every published endpoint index row as
`plan_endpoint_content_sha256`; Auditor V10 already verifies it against the
metadata-plan endpoint. The pair sidecar row hash is a different pair-level
provenance value and must not be substituted into the Builder-exact frame
construction.

`FrameBuildInputV4.sidecar_row_identity_sha256` and `frame_key` do not alter the
numeric ray evidence arrays, but exact audit must reproduce the frozen Builder
inputs rather than exploit that implementation detail. The V10 failure occurred
before evidence recomputation. The already-published V9 dataset remains the
only authorized audit input and must not be rebuilt.

## V11 scope and concrete paths

V11 is an **audit-only, additive successor**. It reuses the immutable Builder V9
dataset and may not rebuild, mutate, normalize, rename, repair, delete, or
republish any V9 dataset byte, either predecessor receipt, or predecessor
authorization.

The concrete V11 artifacts are:

| Role | Path |
| --- | --- |
| Amendment | `docs/lewm_go2_shared_jepa_v5_raw_supervision_auditor_v11_builder_parity_successor_amendment_2026-07-14.md` |
| Auditor source | `lewm/datasets/go2_shared_jepa_v5_raw_supervision_auditor_v11.py` |
| Auditor CLI | `scripts/audit_go2_shared_jepa_v5_raw_supervision_v11.py` |
| Auditor test | `lewm/tests/test_go2_shared_jepa_v5_raw_supervision_auditor_v11.py` |
| Author handoff | `docs/lewm_go2_shared_jepa_v5_raw_supervision_auditor_v11_author_handoff_2026-07-14.md` |
| Different-agent review | `docs/lewm_go2_shared_jepa_v5_raw_supervision_auditor_v11_independent_review_2026-07-14.json` |
| Exact authorization | `docs/lewm_go2_shared_jepa_v5_raw_supervision_audit_v11_authorization_2026-07-14.json` |
| Input dataset | `.generated/go2_shared_observable_camera_ray_jepa_v5/development_raw_supervision_v1` |
| Success report | `.generated/go2_shared_observable_camera_ray_jepa_v5/development_raw_supervision_v1.audit_v11.json` |
| Terminal failure | `.generated/go2_shared_observable_camera_ray_jepa_v5/development_raw_supervision_v1.audit_v11.failed.json` |

Production V11 must be standalone. It may mechanically inherit V10 audit
science, raw-mapping correction, filesystem threat controls, and closed
no-replace transaction, but it must not import or call a predecessor auditor or
Builder in production. Tests may import the frozen Builder V9 only as a direct
oracle. The only exact production entry is:

```python
execute_exact_audit_v11(*, authorization_sha256: str, workers: int)
```

Workers remain strict non-boolean integers in `[1, 6]`, use `spawn`, expose no
accelerator, and set every native math thread variable to one. V11 has one fresh
attempt, no retry, no fallback, and no alternate entry.

## Sole Builder-parity correction

V11 must construct the complete endpoint-context map once from the full frozen
metadata plan before exact sample tasks are created. Its production semantics
must be mechanically identical to Builder V9 `_pair_endpoint_contexts` after
only exception/type/name adaptation:

- reject repeated endpoint identities;
- visit all pairs and both endpoint sides, without selecting one occurrence;
- reject a pair reference to an absent endpoint;
- reconstruct the exact ten-field Builder context from the plan endpoint and
  pair;
- require repeated occurrences to reconstruct equal contexts;
- reject every conflicting repeated context;
- require the final context key set to equal the endpoint identity key set; and
- reject every orphan endpoint.

The task for an exact sampled endpoint receives this already-validated Builder
context directly. It must not reconstruct provenance from a chosen minimum,
first, last, or arbitrary pair occurrence. Pair occurrence-only fields must not
be conflated with Builder endpoint context.

Source-frame matching and extraction use the exact Builder context. The
recomputed `FrameBuildInputV4` must use:

```python
frame_key={"endpoint_identity_sha256": endpoint_digest}
sidecar_row_identity_sha256=str(endpoint["content_sha256"])
```

The endpoint content value must be a strict SHA-256, remain canonically
self-valid, and equal the already-validated published endpoint row's
`plan_endpoint_content_sha256`. Pair `sidecar_row_identity_sha256` must remain a
separately validated pair-level field and must never feed the Builder frame
input.

V11 must retain V10's original-decoded-raw-mapping correction exactly. It must
retain all V10 sample selection, cardinality, source-byte allowlist, semantic
hashing, geometry, raycast, raster, dtype, shape, byte comparison, digest,
threshold, report, and publication rules. No scientific rule may otherwise
change.

The eight replay outputs remain, in exact order:

1. `camera_origin_body_m.f4`
2. `camera_basis_body_fru.f4`
3. `ground_plane_z_body_m.f4`
4. `ground_support_in_frustum.u1`
5. `ground_support_clear_to_target.u1`
6. `pixel_hit_mask.u1`
7. `pixel_first_hit_distance_m.f4`
8. `raster_labels.u1`

## V11 report and transaction closure

The V10 closed publication transaction is retained after mechanical V11
authority and namespace renaming. Before and continuously through publication,
V11 must retain no-follow descriptors and watches for:

- the immutable V9 dataset and full manifest-inventoried contents;
- the V9 and V10 terminal receipts;
- the V9 build and V10 audit authorizations;
- every authorization-bound source and review;
- the publication directory and ancestor chains; and
- the owned V11 candidate.

The V10 success path must remain absent. V11 may publish exactly one additive,
canonical, newline-terminated, no-replace, fsynced leaf: the V11 success report
or the V11 terminal receipt. Cleanup may remove only a V11-owned unpublished
candidate. It may not remove or replace the dataset, either predecessor receipt,
or any predecessor authorization.

A V11 success report remains downstream-false and does not itself authorize
training, selection, G2, held-out, navigation, runtime, hardware, production,
promotion, or deployment. A V11 terminal receipt must bind the V11 execution
authorization, V9 manifest file/content hashes, V9 receipt file/content hashes,
V10 authorization file/content/source-map hashes, and V10 receipt file/content
hashes. It must set `retry_authorized=false` and every downstream authority
false.

## Required author and reviewer proof

All author and reviewer tests must use temporary roots, synthetic metadata, and
CPU-only execution. They must not open canonical `.generated`, corpus, RGB,
checkpoint, G2, held-out, runtime, hardware, or production paths.

The proof suite must:

1. reproduce the V10 missing-key failure through the real
   `_source_record_for_endpoint` to `_recompute_one_exact_sample` dataflow;
2. prove a repeated endpoint with identical reconstructed contexts is accepted;
3. mutate each of the ten Builder-context fields in a later occurrence and prove
   every conflict is rejected;
4. reject duplicate endpoints, absent pair references, and orphan endpoints;
5. prove no minimum/first/last pair occurrence is selected as endpoint
   provenance;
6. use deliberately unequal pair-sidecar and endpoint-content hashes and prove
   the frame input receives only endpoint content;
7. prove the frame key is exactly the one-field Builder V9 mapping;
8. retain the V10 test with an actual parsed manifest proving the original raw
   mapping, rather than `to_dict()` or another reconstruction, reaches the raw
   validator unchanged;
9. run a real synthetic end-to-end replay through source JSONL selection,
   source-frame extraction, parsed scene geometry, camera composition,
   `FrameBuildInputV4`, `build_frame_evidence_v4`, rasterization, and conversion
   of all eight arrays without mocking the recomputer boundary;
10. compare that replay directly against frozen Builder V9 as a test-only oracle
    and require identical endpoint context, frame key, endpoint-content
    provenance, dtype, shape, bytes, evidence content hash, and raster content
    hash;
11. compare V11 and V10 ASTs after authority renaming and permit only the frozen
    all-occurrence context, exact frame-key, endpoint-content provenance, V10
    terminal binding, and V11 namespace deltas;
12. compare V11 context construction mechanically to Builder V9
    `_pair_endpoint_contexts` after only allowed exception/type/name adaptation;
13. rerun every applicable retained V10 and V9 auditor test while replacing any
    obsolete AST assertion that would require preserving the diagnosed defect;
14. run the real synthetic replay with one worker and six spawned workers and
    require identical canonical result bytes and identical eight-array bytes;
15. prove every spawned initializer and task independently reauthorizes and
    hides accelerators;
16. prove the immutable dataset, V9 receipt, V10 receipt, and V10-success absence
    remain unchanged across both V11 success and failure publication tests;
17. prove V11 success/failure leaves are distinct, no-replace, fsynced,
    terminal, and correctly bound; and
18. prove production source exposes no test hook, mutable authority registry,
    predecessor import, Builder import, alternate opener, unrestricted path,
    skip, retry, fallback, rebuild, trainer, GPU, G2, held-out, navigation, or
    runtime entry.

The direct Builder V9 oracle is test-only. Tests must freeze its source hash and
fail closed if it differs. No test may use the canonical dataset or source
corpus to obtain parity.

## V11 authorization closure

The future canonical V11 authorization must use an ordered, unique source map
with exactly these roles and literal paths:

1. `amendment`: the frozen V11 amendment;
2. `v9_build_authorization`: the frozen V9 build authorization;
3. `v9_builder_source`: the frozen Builder V9 source;
4. `v9_builder_review`: the passing frozen Builder V9 review;
5. `v9_dataset_manifest`: the immutable V9 dataset manifest;
6. `v9_terminal_failure`: the immutable V9 failure receipt;
7. `v10_amendment`: the frozen V10 amendment;
8. `v10_auditor_source`: the frozen Auditor V10 source;
9. `v10_auditor_cli`: the frozen Auditor V10 CLI;
10. `v10_auditor_test`: the frozen Auditor V10 test;
11. `v10_auditor_handoff`: the frozen Auditor V10 handoff;
12. `v10_auditor_review`: the passing frozen Auditor V10 review;
13. `v10_audit_authorization`: the frozen V10 audit authorization;
14. `v10_terminal_failure`: the immutable V10 failure receipt;
15. `auditor_source`: the frozen Auditor V11 source;
16. `auditor_cli`: the frozen Auditor V11 CLI;
17. `auditor_test`: the frozen Auditor V11 test;
18. `auditor_handoff`: the frozen Auditor V11 author handoff; and
19. `auditor_review`: the passing different-agent V11 review.

The authorization must bind and deep-validate:

- the V9 build authorization file, canonical content, ordered source map, and
  all nine nested source-map targets;
- the V9 dataset manifest file and canonical content hashes and its complete
  inventory;
- the V9 terminal receipt file/content hashes, terminal status, authorization
  binding, and false retry/downstream fields;
- the V10 authorization file SHA-256
  `146e0bbf029d28fdf883bfc357b1ddbbce955f86bda00508c6091cb01db4800a`,
  canonical content SHA-256
  `8bab96369a5633cb82266fef6ec54964a3c25f27dc0877fde550721f3b6af981`,
  source-map SHA-256
  `1b6ffd40b72c7d02dba24d2035ac3442af361b8b804e7df4273f5e73d1cda79b`,
  and all eleven nested targets;
- the V10 terminal receipt file/content hashes, terminal status, V10
  authorization binding, V9 artifact bindings, and false retry/downstream
  fields;
- absence of the V10 success report;
- every V11 candidate and review hash;
- the literal input dataset and V11 success/failure paths; and
- the V11 implementation author and different-agent reviewer identities.

The authorization schema must set `exact_audit_v11_authorized=true` and all of
the following false:

- `exact_audit_v9_authorized`
- `exact_audit_v10_authorized`
- `exact_build_authorized`
- `exact_rebuild_authorized`
- `retry_authorized`
- `rgb_decode_authorized`
- `dataset_use_authorized`
- `training_authorized`
- `selection_authorized`
- `calibration_authorized`
- `g2_authorized`
- `heldout_authorized`
- `runtime_authorized`
- `navigation_authorized`
- `hardware_authorized`
- `production_authorized`
- `promotion_authorized`
- `deployment_authorized`

Any missing, changed, repeated, reordered, noncanonical, symlinked, hard-linked,
or ancestry-escaped bound artifact must fail closed before dataset access. Phase
one must validate authorization structure without opening mapped targets. Phase
two must open only the exact bound targets, deep-validate both predecessor
authorizations and receipts, and carry separate immutable V9 build provenance
and V11 execution authority.

Fixed V11 implementation author: `/root/raw_v11_builder_auditor_diff`.

The V11 reviewer must start with `/root/` and differ from `/root`,
`/root/raw_v11_builder_auditor_diff`, every V9 Builder/Auditor implementation
author and reviewer, every V10 implementation author and reviewer, and any
agent that publishes or independently fingerprints the V11 authorization.

## Sequence and non-authority

1. Freeze this source-free amendment before any V11 production source exists.
2. `/root/raw_v11_builder_auditor_diff` constructs and freezes only the V11
   source, CLI, tests, and handoff without exact or canonical-data access.
3. A different eligible agent independently reviews the complete frozen
   candidate and publishes one canonical `PASS` or `BLOCK` review.
4. Only `PASS` permits `/root` to publish the separate canonical V11 audit
   authorization.
5. An agent distinct from the amendment author, implementation author,
   reviewer, and authorization publisher independently reproduces the complete
   authorization file SHA-256.
6. Only then may the one six-worker exact V11 audit run, serialized with every
   `.generated` mutator, CPU-only, one native math thread per worker, and all
   accelerators hidden.
7. A terminal failure grants no retry. Any further defect requires another
   source-free additive successor amendment and new namespace.
8. Only a passing V11 report may support a later, separate dataset-use or
   training authorization.

This amendment grants source construction and different-agent review only. It
does not authorize an exact attempt, retry, rebuild, dataset use, training,
selection, calibration, G2, held-out, runtime, navigation, hardware,
production, promotion, or deployment.
