# Shared JEPA V5 development raw-supervision Builder V5 author handoff

Date: 2026-07-13

Implementation author: `/root/raw_builder_arch`

Status: **FROZEN SOURCE CANDIDATE FOR DIFFERENT-AGENT REVIEW; NO EXACT AUTHORITY**

## Bound amendment and predecessor

Builder V5 is authored against exactly:

| Artifact | SHA-256 |
| --- | --- |
| `docs/lewm_go2_shared_jepa_v5_raw_supervision_builder_auditor_v5_authorization_successor_amendment_2026-07-13.md` | `fe6a29a27eb0284ce84fcba409b530c6351befad18ee9d655f5f2e9b337d9e91` |

The amendment was the sole V5 artifact when frozen. It binds the complete
Builder V4 candidate and its independent publication-order BLOCK, including:

| Bound evidence | SHA-256 |
| --- | --- |
| Builder V4 source | `e46f42db3b5ed50581ed916d459e05f2dd9b73dcbdd906ea5d1991b7b61893e0` |
| Builder V4 independent QA | `116b81f65c6c6eb23ed8aba58e9fa2b62a0e0177c4c5e2a0c821c2d0aa8268e2` |
| Builder V4 BLOCK JSON file | `4c91d7ce09c97fea657ae279183c02f45da7911dbbd6178c5d311e938f602dc4` |
| Builder V4 BLOCK canonical content | `34cfd6b139434f3b09b98d4bb44a339b7c355b68027d7f1e192112191c404ea6` |
| Auditor V4 non-candidate source checkpoint | `d030122e24b7ab2d6da96dff7b88b4bec6ff028da2767e24d480069165654e0d` |

No V4 artifact supplies exact authority. The V5 amendment also grants no exact
build, exact audit, dataset-use, training, selection, calibration, G2,
held-out, navigation, runtime, hardware, production, or promotion authority.

## Frozen Builder V5 candidate

| Role | Artifact | SHA-256 |
| --- | --- | --- |
| `builder_source` | `lewm/datasets/go2_shared_jepa_v5_raw_supervision_builder_v5.py` | `8d85635a85d5a6a3575602a89f37a01f97acf03bd0059a8ae452b21ed4cddce2` |
| `builder_cli` | `scripts/build_go2_shared_jepa_v5_development_raw_supervision_v5.py` | `3116c2a5b429cf0fbed0674de91b0569d6ecf6e10c26cd6064a3bb0349e78019` |
| `builder_test` | `lewm/tests/test_go2_shared_jepa_v5_raw_supervision_builder_v5.py` | `6b49d5d5847e22cea413a7b72da34d5fbf221f876b89bfdf899804024c9d05d6` |

Production-ineligible test infrastructure, not an authority role:

| Artifact | SHA-256 |
| --- | --- |
| `lewm/tests/go2_shared_jepa_v5_raw_supervision_builder_v5_test_support.py` | `d98ee9ad399296617230aab85a58917a0c479d5ecec1f6c72ee053d5b5a0aefa` |

This handoff is the fourth ordered Builder V5 review candidate. Its file hash
is intentionally reported after creation and must be included verbatim in any
machine review.

## Publication-order repair

Builder V4 called `_revalidate_exact_before_publication` too early. The frozen
independent reproducer found exactly these eight forbidden calls between that
second pass and `_libc_renameat2`, in order:

```text
_precommitted_audit_sample
_with_content_sha256
_validate_staging_inventory
_write_json_exclusive
_validate_staging_inventory
_fsync_directory
fsync
fsync
```

Builder V5 moves the one complete second pass after all of that work. Its
pre-publication order is now:

1. construct shards, indexes, audit sample, counts, provenance, access ledger,
   and the canonical manifest object;
2. validate pre-manifest staging inventory;
3. exclusively write the manifest;
4. validate complete staging inventory;
5. fsync every staging directory, the staging descriptor, and the retained
   parent descriptor;
6. run `_revalidate_exact_before_publication` once; and
7. perform only `retained.validate()` and the exact owned staging-identity
   comparison before one `_libc_renameat2`.

No sample/manifest construction, serialization, write, traversal, inventory,
hash, staging fsync, parent fsync, or provenance work follows the second pass.
The already-reviewed post-rename refresh, published identity check, parent
fsync, return, and inode-owned cleanup remain unchanged.

The V5 author test parses both frozen sources. It positively reproduces V4's
exact eight-call violation, requires one V5 second pass and one rename, proves
the forbidden interval is empty, limits the interval to the two permitted
read-only checks (plus error construction on mismatch), and proves every
required staging/manifest/fsync call precedes the second pass.

## Retained standalone boundary

All Builder V4 authority and worker repairs remain intact:

- no retained Builder V1-V4 or Auditor module object, legacy exact entry,
  validator bridge, dynamic fallback, caller callback pool, or non-pure loader;
- pure phase one and one-parameter fixed-root phase two with canonical capsule
  revalidation before its first target opener;
- exact ordered nine-role V5 source map, both review cross-bindings, and all 44
  frozen predecessor/reviewed-science file hashes;
- immutable three-hash `AcceptedAuthorizationV5` receipt, never an accepted
  source mapping;
- keyword-only `execute_exact_build_v5(authorization_sha256, workers)` with
  fixed authorization/output paths and exact non-boolean workers in `[1,6]`;
- only literal internal process-pool targets, full V5 authority in the fixed
  spawn initializer before task receipt/deserialization, and repeated authority
  validation as each worker's first operation; and
- test root/reader/synthetic seams only in the production-ineligible helper,
  which production and the CLI never import.

OMP, OpenBLAS, MKL, and NumExpr threads are one. CUDA, HIP, ROCr, and GPU
ordinal visibility are empty. There is no GPU path.

## Retained science and artifact behavior

The V4-to-V5 source diff changes no reviewed camera-ray science or array
semantics. It contains only V5 authority/type/path rebinding, frozen predecessor
additions, the exact-entry/type name change, and the publication-order repair.
V5 retains:

- 5,172 pairs, 10,344 endpoint references, 9,460 unique endpoints, 88 scenes,
  and three development roles;
- one schedule/raycast operation per unique endpoint;
- reviewed V4 camera composition, full-RPY geometry, raycast, ground support,
  and rasterization;
- all eight frozen arrays/dtypes/shapes, scalar ground-plane `[N]`, and
  `64 x 64` three-state raster labels;
- strict direct pair/endpoint joins and all duplicate/missing/orphan/cross-
  context rejection;
- complete 354-record provenance, exact access ledger, complete second pass,
  and zero forbidden opens;
- deterministic one/six-worker bytes and canonical ordering; and
- retained filesystem descriptors and fingerprints, private `0700` sibling
  staging, single-link files, inode-owned cleanup, fsync, and atomic no-replace
  publication.

The canonical dataset path and dataset schema remain unchanged. Every
dataset-use and downstream license is false.

## Verification

All commands ran CPU-only with native thread caps of one and accelerator
visibility empty.

| Verification | Result |
| --- | --- |
| Builder V5 author suite, including paired V4/V5 AST ordering proof | `31 passed` |
| frozen Builder V4 plus Builder V5 author suites | `61 passed` |
| frozen Builder V1/V2 regression in isolated pytest base | `42 passed` |
| metadata V5 author plus independent QA | `45 passed` |
| reviewed V4 observable-camera-ray evidence | `20 passed` |
| source, CLI, role-test, and test-support `py_compile` | PASS |
| fixed predecessor/reviewed-science hashes | `44/44` reproduced |
| CLI `--help` and import-surface smoke | PASS |

The frozen Builder V4 independent QA intentionally remains `1 passed, 1
failed`; that failure is the bound BLOCK reproduced as a passing negative
control in the V5 author suite.

No exact entry was invoked. No authorization file, canonical output, failure
receipt, development source payload, RGB payload, legacy label, protected role,
G2/held-out data, model/checkpoint, runtime/hardware result, or GPU was opened or
created.

## Independent review target

A different agent must review the four exact Builder V5 candidate files against
the frozen amendment and publish only canonical PASS or BLOCK JSON at:

```text
docs/lewm_go2_shared_jepa_v5_development_raw_supervision_builder_v5_independent_review_2026-07-13.json
```

with schema:

```text
lewm_go2_shared_jepa_v5_raw_supervision_builder_v5_independent_review_v1
```

The ordered candidate is source, CLI, role test, and this handoff with exact
file hashes. A PASS approves Builder V5 source only; every exact, dataset-use,
downstream, runtime, hardware, production, and promotion authority remains
false. Any changed byte or failed reproducer requires BLOCK and a new additive
successor.
