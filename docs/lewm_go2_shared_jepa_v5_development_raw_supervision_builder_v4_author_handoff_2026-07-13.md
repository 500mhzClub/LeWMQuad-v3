# Shared JEPA V5 development raw-supervision Builder V4 author handoff

Date: 2026-07-13

Implementation author: `/root/raw_builder_arch`

Status: **FROZEN SOURCE CANDIDATE FOR DIFFERENT-AGENT REVIEW; NO EXACT AUTHORITY**

## Bound amendment

Builder V4 is authored against exactly:

| Artifact | SHA-256 |
| --- | --- |
| `docs/lewm_go2_shared_jepa_v5_raw_supervision_builder_auditor_v4_authorization_successor_amendment_2026-07-13.md` | `a535ee8de9a6002f5548f3c3894548ddb42cd9d077eccbb9ca922a41611ced83` |

That amendment was the sole V4 artifact when frozen. It binds the V3
structural invalidation, the stale/changed Auditor V3 closure, and the
compile-safe but unfrozen Builder V3 checkpoint. It grants no exact build,
exact audit, dataset-use, training, selection, calibration, G2, held-out,
navigation, runtime, hardware, production, or promotion authority.

## Frozen Builder V4 candidate

| Role | Artifact | SHA-256 |
| --- | --- | --- |
| `builder_source` | `lewm/datasets/go2_shared_jepa_v5_raw_supervision_builder_v4.py` | `e46f42db3b5ed50581ed916d459e05f2dd9b73dcbdd906ea5d1991b7b61893e0` |
| `builder_cli` | `scripts/build_go2_shared_jepa_v5_development_raw_supervision_v4.py` | `db14bb159b39204e7576b71f3b93409e13b9f28c5cb0d2e87a627557471c0901` |
| `builder_test` | `lewm/tests/test_go2_shared_jepa_v5_raw_supervision_builder_v4.py` | `80ca9d1d35b83fd29027ab297ac662c406dcdd15f68ac5aced9cc7419fef61c0` |

Production-ineligible test infrastructure, not an authorization role:

| Artifact | SHA-256 |
| --- | --- |
| `lewm/tests/go2_shared_jepa_v5_raw_supervision_builder_v4_test_support.py` | `b9c40dbdc63dd86b8603f9c008ab80b393d34c2228456879e24426b18e3cb433` |

This handoff is the fourth ordered Builder V4 review candidate. Its file hash
is intentionally reported after creation and must be added verbatim to any
different-agent machine review.

## Standalone authority boundary

Builder V4 owns its production implementation directly. It imports no Builder
V1, V2, or V3 module object and exposes no legacy exact entry, legacy validator,
compatibility bridge, dynamic fallback, caller function pool, or non-pure
metadata loader. The named V4 label primitives and metadata data types/constants
are the only inherited construction surface. Frozen metadata and inventory
loaders are imported locally only after complete V4 authority acceptance.

The two phases are closed:

1. phase one performs only canonical structural validation of the complete
   six-field authorization object, exact ordered nine-role source map, both
   review bindings, reviewer/author independence, and every cross-binding;
2. phase two takes only the frozen phase-one capsule, canonical-parses it,
   reruns phase one, requires exact capsule equality, descriptor-reads the fixed
   nine V4 targets in order, validates both canonical machine PASS reviews,
   and rehashes all 36 frozen predecessor/reviewed-science files.

Phase two has one parameter and no reader, root, path, mapping, callback,
parent-skip, exact flag, or test seam. It returns only the frozen three-hash
`AcceptedAuthorizationV4` receipt, never the accepted source mapping. The
production source never assigns or replaces any authority validator. Synthetic
root/reader support exists only in the separately named test module above, and
production never imports it.

The public exact entry is keyword-only
`execute_exact_build_v4(authorization_sha256, workers)`. Both paths are fixed
internally. Worker count is an exact non-boolean integer in `[1,6]` and is
checked before authority or data access.

## Worker boundary

All three process-pool sites use `spawn`, the one fixed
`_initialize_exact_worker`, and one literal internal target each:

- `_write_prepared_scene_job`;
- `_load_exact_scene_job`; and
- `_revalidate_exact_scene_sources`.

The initializer descriptor-reads the fixed authorization and completes full V4
phase one, fixed-nine phase two, machine-review validation, predecessor rehash,
and reviewed-science rehash before the process receives/deserializes a task.
Each worker repeats full V4 authority validation as its first operation before
using its already-deserialized values or opening any source. Pools accept no
caller function, callback, initializer, worker target, reader, root, path,
mapping, skip, or exact flag.

OMP, OpenBLAS, MKL, and NumExpr threads are one in parent and workers. CUDA,
HIP, ROCr, and GPU ordinal visibility are empty. There is no GPU path.

## Retained construction and publication behavior

The standalone engine preserves the V1-reviewed Builder science and V3
security checkpoint behavior:

- 5,172 pairs, 10,344 endpoint references, 9,460 unique endpoints, 88 scenes,
  and the three frozen development roles;
- one scheduled/raycast operation per unique endpoint;
- the reviewed V4 camera, full-RPY geometry, ray, ground-support, and raster
  functions;
- all eight frozen arrays, dtypes, scalar ground-plane `[N]` layout, and
  `64 x 64` three-state raster labels;
- strict direct pair/endpoint joins and duplicate/missing/orphan/cross-context
  rejection;
- the 354-record source receipt, exact access ledger, complete second metadata
  and source pass, and zero forbidden opens;
- canonical ordering and worker-count-independent bytes; and
- retained filesystem descriptors/fingerprints, private sibling staging,
  single-link regular files, inode-owned cleanup, fsync, and one
  `renameat2(RENAME_NOREPLACE)` publication.

The canonical dataset path and dataset schema remain unchanged. Every
dataset-use and downstream license remains false pending a later independent
dataset audit and separate authority.

## Verification

All commands ran CPU-only with native thread caps of one and accelerator
visibility empty.

| Verification | Result |
| --- | --- |
| Builder V4 author suite | `30 passed` |
| frozen Builder V1/V2 author regression, isolated pytest base | `42 passed` |
| metadata V5 author plus independent QA | `45 passed` |
| directly imported V4 observable-camera-ray evidence suite | `20 passed` |
| `py_compile` for source, CLI, role test, and test support | PASS |
| all Builder V4 fixed predecessor/reviewed-science hashes | `36/36` reproduced |
| CLI `--help` import smoke | PASS |

An additional combined camera-fit build/audit suite produced `69 passed, 3
failed`; its failures are in that separate subsystem's current implementation-
authorization state and repository-semantic import ordering. The direct V4
evidence suite above is clean, and none of the three failures executes or
implicates Builder V4.

No exact entry was invoked. No authorization file, canonical output, failure
receipt, development source payload, RGB payload, legacy label, protected role,
G2/held-out data, model/checkpoint, runtime/hardware result, or GPU was opened or
created.

## Independent review target

A different agent must review the four exact Builder V4 candidate files against
the frozen amendment and publish only canonical PASS or BLOCK JSON at:

```text
docs/lewm_go2_shared_jepa_v5_development_raw_supervision_builder_v4_independent_review_2026-07-13.json
```

with schema:

```text
lewm_go2_shared_jepa_v5_raw_supervision_builder_v4_independent_review_v1
```

The review's ordered candidate must be source, CLI, role test, and this handoff
with their exact file hashes. A PASS approves Builder V4 source only; every
exact, dataset-use, downstream, runtime, hardware, production, and promotion
authority remains false. Any changed byte or failed structural reproducer
requires BLOCK and a new additive successor.
