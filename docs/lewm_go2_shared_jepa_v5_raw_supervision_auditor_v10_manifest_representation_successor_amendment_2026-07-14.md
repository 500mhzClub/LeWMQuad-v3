# Shared JEPA V5 raw-supervision Auditor V10 manifest-representation successor amendment

Date: 2026-07-14

Amendment author: `/root`

Status: **audit-source construction and different-agent review only; no exact authority**

## Trigger and terminal V9 result

The one authorized Builder V9 execution completed and atomically published the
fixed dataset at
`.generated/go2_shared_observable_camera_ray_jepa_v5/development_raw_supervision_v1`
with status `complete_pending_independent_audit` and canonical content SHA-256
`74ae5799919ff4d9a06f56d98929cb4cb702d64db52ecdfc93cfa9a8e82fb35a`.

The one authorized Auditor V9 execution then terminally failed during exact
sample replay with `ValueError: scene manifest wall boxes are not a list`.
Its canonical terminal receipt is
`.generated/go2_shared_observable_camera_ray_jepa_v5/development_raw_supervision_v1.audit_v9.failed.json`,
file SHA-256
`863630579e6d8f8ac222ff7ce5ba04ff3e7901885b606dcb6bcfd7a07fe7722f`,
content SHA-256
`aaf342f7df88796e0d03259e964ed51e42ebd1faecb33bbfe9ea9cfd0d5e2c72`,
and status `terminal_failed_no_dataset_authority`. It grants no retry and no
dataset, training, G2, or production authority. Auditor V9 is terminal and may
not be changed, rerun, repaired in place, or reinterpreted as a scientific
failure.

The failure is a representation-boundary defect in the frozen V9 auditor, not
an observed mismatch in the built dataset. Builder V9 validates the decoded
raw scene-manifest JSON mapping at
`lewm/datasets/go2_shared_jepa_v5_raw_supervision_builder_v9.py:2971` before
parsing it. Auditor V9 instead parses that same mapping into
`SceneManifest`, whose sequence fields are tuples, and at
`lewm/datasets/go2_shared_jepa_v5_raw_supervision_auditor_v9.py:4088` passes
`scene_manifest.to_dict()` to the raw-JSON validator. Python
`dataclasses.asdict` preserves tuples, while the raw-JSON validator correctly
requires JSON arrays decoded as lists. The auditor therefore rejects its own
lossy representation round trip before comparing any sample arrays.

No Auditor V10 source, CLI, test, handoff, review, authorization, exact report,
or exact attempt existed when these amendment bytes were frozen.

## Frozen evidence

| Role | Path | SHA-256 |
| --- | --- | --- |
| Builder/Auditor V9 amendment | `docs/lewm_go2_shared_jepa_v5_raw_supervision_builder_auditor_v9_linearization_successor_amendment_2026-07-13.md` | `6fba5de8d7f04d85bd87e084096ae269c3d3dd6368a6db0b0f8f149c1c5cf773` |
| Builder V9 source | `lewm/datasets/go2_shared_jepa_v5_raw_supervision_builder_v9.py` | `2388c1138d9b03ea6e385cc0250c81a1869a40cab62507d02f709ef39197c664` |
| Builder V9 review | `docs/lewm_go2_shared_jepa_v5_development_raw_supervision_builder_v9_independent_review_2026-07-13.json` | `c39eb2787c37f8cab064de75355b3af56971ef98209d329e4789eb383c1dc60f` |
| Auditor V9 source | `lewm/datasets/go2_shared_jepa_v5_raw_supervision_auditor_v9.py` | `ebe0c6a31cf027b8b0bc049257079a5e0ab0493b12aabeb96bf50f02990bbc14` |
| Auditor V9 CLI | `scripts/audit_go2_shared_jepa_v5_raw_supervision_v9.py` | `76f0b2b29eff8df6905fed142cc622eb0fa8024c397a3c7efb54e58cc36f67ba` |
| Auditor V9 test | `lewm/tests/test_go2_shared_jepa_v5_raw_supervision_auditor_v9.py` | `10951cc2e622281f72ec2a20114ccca184af7624a95fef4683c83dc6839992d1` |
| Auditor V9 handoff | `docs/lewm_go2_shared_jepa_v5_raw_supervision_auditor_v9_author_handoff_2026-07-13.md` | `819d1857bf315f775f45c4a16db994f333d7174c5c20f5cb762f93d04b30a3a5` |
| Auditor V9 review | `docs/lewm_go2_shared_jepa_v5_raw_supervision_auditor_v9_independent_review_2026-07-13.json` | `bd5e46680345977d1b8b08e0060e5b693701165e14490d477b0f7a64bdc856d6` |
| V9 build authorization | `docs/lewm_go2_shared_jepa_v5_raw_supervision_build_authorization_2026-07-13.json` | `7878c807fda5f7057addc302afeca77cf6163683f0b2c5b6d4e4ac2701f86792` |
| Built dataset manifest | `.generated/go2_shared_observable_camera_ray_jepa_v5/development_raw_supervision_v1/manifest.json` | `e102b3c64e99029f118597353966edaaaddbc11efe49b9081d5d7a9c9d974360` |
| Auditor V9 terminal receipt | `.generated/go2_shared_observable_camera_ray_jepa_v5/development_raw_supervision_v1.audit_v9.failed.json` | `863630579e6d8f8ac222ff7ce5ba04ff3e7901885b606dcb6bcfd7a07fe7722f` |

The frozen V9 authorization file was independently fingerprinted by
`/root/raw_v9_auth_hash_witness`. The V9 dataset is not accepted merely because
its build completed; only a passing exact V10 audit can create dataset-use
authority.

## V10 scope

V10 is an **audit-only** additive successor. It must not rebuild, modify,
rename, normalize, repair, or delete any V9 dataset byte or the V9 terminal
receipt. It audits the already-published V9 dataset under a new namespace and
publishes only:

- success: `development_raw_supervision_v1.audit_v10.json`; or
- terminal failure: `development_raw_supervision_v1.audit_v10.failed.json`.

Auditor V10 is standalone. It may mechanically inherit the passing V9 audit
science and transaction, but production source may not import or call a legacy
auditor or a builder. Its only exact entry is:

    execute_exact_audit_v10(*, authorization_sha256: str, workers: int)

Workers remain exact non-boolean integers in `[1,6]`, use `spawn`, expose no
accelerator, and use one native math thread. There is one fresh V10 attempt,
no retry, and no fallback.

## Sole semantic correction

The exact replay path must keep two distinct values:

1. the decoded raw scene-manifest mapping, used for every raw JSON/schema and
   rendered-object-record validation; and
2. the parsed immutable `SceneManifest`, used for semantic hashing, typed
   geometry, rendered-box construction, and ray evidence recomputation.

The raw validator must receive the original decoded mapping directly. It must
never receive `SceneManifest.to_dict()`, `asdict`, a tuple-normalized copy, a
JSON round trip, or any other reconstructed representation. V10 must prove the
raw mapping is unmodified across validation and that the parsed semantic hash
still equals the frozen source inventory.

No sample selection, cardinality, source byte, geometry, raycast, raster,
array, digest, comparison, threshold, report field, publication transaction,
filesystem threat control, or scientific rule may change. V10 must retain all
V9 checks, including complete exact sample recomputation and byte/digest
comparison.

## V10 authorization closure

The future canonical V10 audit authorization must bind exactly:

- this amendment file and SHA-256;
- the frozen V9 build authorization file and SHA-256;
- the frozen V9 Builder source and passing review;
- the frozen V9 dataset manifest file SHA-256 and canonical content SHA-256;
- the frozen V9 terminal audit receipt file and content SHA-256;
- Auditor V10 source, CLI, test, handoff, and different-agent review hashes;
- the literal input dataset and V10 output paths; and
- `exact_audit_v10_authorized=true` with every build, retry, training, G2,
  held-out, runtime, hardware, production, promotion, and deployment field
  false.

The authorization must reject any missing or changed V9 failure receipt,
dataset manifest, dataset inventory byte, V9 source/review, or V10 source/review.
Its own canonical file hash must be independently reproduced by an agent other
than the amendment author before the sole exact V10 audit command.

Fixed implementation author: `/root/raw_v9_auth_hash_witness`.

The reviewer must start with `/root/` and differ from `/root`, the fixed
implementation author, both V9 implementation authors, and both V9 reviewers.

## Required proof

Author and different-agent reviewer tests use only temporary roots and must:

- reproduce the V9 list-to-tuple failure with an actual parsed manifest;
- prove V10 passes the unchanged original raw mapping to the raw validator;
- prove tuple, mapping proxy, generator, reconstructed, and mutated raw values
  are rejected rather than normalized;
- prove parsed-manifest semantic hashing and all exact recomputed arrays remain
  bit-identical to V9 after only the boundary correction;
- mechanically compare V9/V10 science and transaction ASTs after permitted
  authority renames and the one raw-mapping parameter addition;
- rerun every applicable retained V9 auditor test;
- prove one-worker and six-worker results are identical;
- prove the V9 dataset and V9 failure receipt are opened read-only, remain
  byte-identical, and cannot be removed or replaced by V10 cleanup;
- prove V10 success and failure report names are distinct, no-replace, fsynced,
  terminal, and correctly bound to the frozen V9 artifacts; and
- prove production source exposes no test hook, mutable authority registry,
  legacy import, alternate opener, unrestricted path, skip, retry, fallback,
  builder, trainer, GPU, G2, held-out, or runtime entry.

Tests must not open canonical `.generated`, corpus, RGB, checkpoint, G2,
held-out, runtime, hardware, or production paths and must not run an
accelerator.

## Sequence and non-authority

1. Freeze this source-free amendment.
2. The fixed non-root author constructs and freezes Auditor V10 source, CLI,
   tests, and handoff without exact/data work.
3. A different eligible agent publishes a canonical `PASS` or `BLOCK` review.
4. Only `PASS` permits `/root` to publish the separate canonical V10 audit
   authorization.
5. Only after a different agent reproduces that authorization file's frozen
   SHA-256 may the sole six-worker exact V10 audit run, serialized with every
   `.generated` mutator and with all accelerators hidden.
6. Only a passing V10 report may authorize this dataset for later training.

This amendment grants source construction and different-agent review only. It
does not authorize an exact audit, dataset use, rebuild, retry, training,
selection, calibration, G2, held-out, runtime, navigation, hardware,
production, promotion, or deployment.
