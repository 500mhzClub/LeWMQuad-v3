# Shared JEPA V5 raw-supervision Builder/Auditor V3 authorization successor amendment

Date: 2026-07-13

Amendment author: `/root/raw_builder_arch`

Status: **PRE-IMPLEMENTATION CONTRACT; NO EXACT AUTHORITY; NO DATA AUTHORITY**

## Purpose

This additive amendment is frozen before any Builder V3 or Auditor V3 source,
CLI, test, handoff, or review artifact is authored. It binds the independently
blocked Builder V2 lineage, records Auditor V2 as a security-ready but
structurally unlicensable base, and preregisters the only V3 source paths,
schemas, authorization ordering, and execution surface eligible for later
independent review.

This amendment grants no exact build, exact audit, dataset use, training,
selection, calibration, G2, held-out, navigation, runtime, hardware,
production, or promotion authority. No exact authorization or canonical output
may be created or opened while implementing or reviewing V3.

## Frozen Builder V2 BLOCK

The following Builder V2 author candidate remains immutable:

| Artifact | SHA-256 |
| --- | --- |
| `lewm/datasets/go2_shared_jepa_v5_raw_supervision_builder_v2.py` | `0ae5ddd836802ced1fcf7524b67970247dccace6787fd0acc7268cbae4d3e71c` |
| `scripts/build_go2_shared_jepa_v5_development_raw_supervision_v2.py` | `c11396874677c3cd3d0ef76353ea7de1449ef610d35f0b4256530a4f62b1d303` |
| `lewm/tests/test_go2_shared_jepa_v5_raw_supervision_builder_v2.py` | `6755044af535dc0c2de93f0f5bd79b01b140da33bc8ff2ec5b003ef592b50339` |
| `docs/lewm_go2_shared_jepa_v5_development_raw_supervision_builder_v2_author_handoff_2026-07-13.md` | `7f278c5c24a8e9d89c6b0e3ecb9252acd0edec5729bd9fdde5d72231848bc04f` |

The different-agent BLOCK evidence is:

| Evidence | SHA-256 |
| --- | --- |
| `lewm/tests/test_go2_shared_jepa_v5_raw_supervision_builder_v2_independent_review.py` | `2c34fec949ea43e03b3f7f3c97b8d8ddba0aad1c9192dfd8b00d3f646dd03d43` |
| `docs/lewm_go2_shared_jepa_v5_development_raw_supervision_builder_v2_independent_review_2026-07-13.md` | `e42a5876c2b9f564085b3f8e98eeb607f7c15a24e75b5534da79619db1f7ccad` |
| `docs/lewm_go2_shared_jepa_v5_development_raw_supervision_builder_v2_independent_review_2026-07-13.json` file | `726e03fdc6242ca3074f0b861dbc49565469212e566338fcd3d2756ced886e4a` |
| machine BLOCK canonical content | `6696b9b786d1dc5a21bbfb1ebe5224c37cff08d94f2dea64c09b5f3b014da0d6` |

Builder V2 correctly fixed the V1 early-open defect, exact role/path structure,
canonical review parsing, duplicate-key rejection, and review cross-bindings.
It is nevertheless ineligible for authorization because:

1. importing V2 retains `_v1.execute_exact_build_v1`, the independently blocked
   legacy exact entry;
2. its compatibility bridges temporarily overwrite the process-global V1
   authority validator with an accepting callback;
3. its authorization-named worker pool accepts and runs a caller-supplied
   function without reading its authorization argument; and
4. its production phase-two function accepts caller root, reader, and
   parent-rehash-skip injection parameters and returns an accepted mapping.

No V3 artifact may import Builder V2 or route execution through any V1/V2 exact
entry, validator, worker pool, or authority bridge.

## Frozen Auditor V2 base

Auditor V2 is retained unchanged as a security-ready implementation base:

| Artifact | SHA-256 |
| --- | --- |
| `lewm/datasets/go2_shared_jepa_v5_raw_supervision_auditor_v2.py` | `d57aacd4849ea3e79468618b73925418ad2035d47de636dc991afda777314b2a` |
| `scripts/audit_go2_shared_jepa_v5_raw_supervision_v2.py` | `4502ac44a451841af18e9f9eb545ef961bc81324ea84ce713e434c434e000ae9` |
| `lewm/tests/test_go2_shared_jepa_v5_raw_supervision_auditor_v2.py` | `45d60db1f1a7385b7941f8f52e01a923f056bb3f52cc85b7fec4097d54fa9399` |
| `docs/lewm_go2_shared_jepa_v5_raw_supervision_auditor_v2_author_handoff_2026-07-13.md` | `6a338b7c15c1fe23ab3680e80c4a30781369e29eebb33331e7ccff723cd4b7ab` |

Auditor V2 closes the known Auditor V1 trust, population, integer-coercion,
hard-link, exact/synthetic API, and publication defects. It is still
structurally unlicensable for a future exact run because its literal nine-role
authority names the now-blocked Builder V2 source closure. Altering those
frozen literals would invalidate Auditor V2. Auditor V3 must therefore be an
additive successor that preserves Auditor V2's security checks while naming
the V3 closure below.

This amendment does not presume or substitute for a different-agent Auditor V2
PASS. Auditor V2 is evidence and a reviewed implementation input only after its
own independent review; it is not a V3 authority target.

## Canonical V3 authority paths

The only eligible V3 authorization source map is an ordered list of exactly
nine objects, each with exactly `role`, `path`, and lower-case `sha256`:

| Order | Role | Literal repository-relative POSIX path |
| ---: | --- | --- |
| 1 | `builder_source` | `lewm/datasets/go2_shared_jepa_v5_raw_supervision_builder_v3.py` |
| 2 | `builder_cli` | `scripts/build_go2_shared_jepa_v5_development_raw_supervision_v3.py` |
| 3 | `builder_test` | `lewm/tests/test_go2_shared_jepa_v5_raw_supervision_builder_v3.py` |
| 4 | `builder_handoff` | `docs/lewm_go2_shared_jepa_v5_development_raw_supervision_builder_v3_author_handoff_2026-07-13.md` |
| 5 | `builder_review` | `docs/lewm_go2_shared_jepa_v5_development_raw_supervision_builder_v3_independent_review_2026-07-13.json` |
| 6 | `auditor_source` | `lewm/datasets/go2_shared_jepa_v5_raw_supervision_auditor_v3.py` |
| 7 | `auditor_cli` | `scripts/audit_go2_shared_jepa_v5_raw_supervision_v3.py` |
| 8 | `auditor_test` | `lewm/tests/test_go2_shared_jepa_v5_raw_supervision_auditor_v3.py` |
| 9 | `auditor_review` | `docs/lewm_go2_shared_jepa_v5_raw_supervision_auditor_v3_independent_review_2026-07-13.json` |

Roles and paths must be unique, appear in exactly this order, and equal these
literal canonical relative paths. Absolute paths, aliases, empty/dot/dot-dot
components, backslashes, repeated separators, alternate spellings, extra rows,
and missing rows are invalid before any target opener.

The nine-role lineage intentionally includes the builder handoff and omits an
auditor handoff. An explanatory Auditor V3 handoff may be frozen and reviewed,
but it is not a machine authorization role and cannot silently become a tenth
target.

## Canonical V3 schemas

The V3 machine authorization schema is:

```text
lewm_go2_shared_jepa_v5_raw_supervision_build_authorization_v3
```

It has exactly these six top-level fields:

```text
schema
exact_build_authorized_after_independent_reviews
builder_review
auditor_review
source_map
content_sha256
```

The exact-build flag must be literal `true`. `content_sha256` is the canonical
JSON SHA-256 of the other five fields.

Both nested review bindings use:

```text
lewm_go2_shared_jepa_v5_raw_supervision_implementation_review_binding_v3
```

and exactly these fields:

```text
schema
review_schema
verdict
reviewer
implementation_author
path
file_sha256
content_sha256
candidate
```

The Builder V3 machine review schema is:

```text
lewm_go2_shared_jepa_v5_raw_supervision_builder_v3_independent_review_v1
```

Its ordered candidate is Builder V3 source, CLI, test, and handoff. The fixed
implementation author is `/root/raw_builder_arch`.

The Auditor V3 machine review schema is:

```text
lewm_go2_shared_jepa_v5_raw_supervision_auditor_v3_independent_review_v1
```

Its ordered candidate is Auditor V3 source, CLI, and test. The fixed
implementation author is `/root/raw_auditor_author`.

Each machine review is canonical duplicate-key-free JSON plus one newline and
has exactly:

```text
schema
verdict
reviewer
implementation_author
candidate
authority
content_sha256
```

The verdict must be `PASS`; each reviewer must differ from its implementation
author; builder and auditor authors must differ; builder and auditor reviewers
must differ. The review `candidate` rows must exactly equal their ordered
source-map subset. The binding path/file hash must equal the corresponding
review source-map row, and its content hash must equal the review's canonical
self-hash.

The exact review authority map has one true field only:
`builder_source_approved` for the builder review or `auditor_source_approved`
for the auditor review. All of the following are exactly false:

```text
exact_build_authorized
exact_audit_authorized
dataset_use_authorized
training_authorized
selection_authorized
calibration_authorized
g2_authorized
heldout_authorized
runtime_authorized
navigation_authorized
hardware_authorized
production_authorized
promotion_authorized
```

## Builder V3 production boundary

Builder V3 must be a standalone production module with these properties:

1. It must not import or retain a legacy builder module object. Its module
   namespace must not expose V1/V2 modules, legacy exact entries, legacy
   validators, or unrestricted attribute fallbacks. It may import only named
   reviewed data types and pure construction primitives needed to preserve the
   frozen science and publication behavior.
2. It must never assign, replace, wrap, cache, or otherwise mutate any module-
   global authority validator. No accepted authority mapping may be returned to
   or supplied by a caller.
3. Production phase one is pure structural validation. It has no reader,
   repository-root, source-map, policy-map, or callback injection parameter.
4. Production phase two accepts only the frozen phase-one capsule. Before its
   first opener it must canonical-parse the capsule, rerun phase one, and require
   exact capsule equality. It has no reader, root, parent-skip, mapping,
   callback, or test injection parameter. It opens the fixed nine V3 targets,
   validates both machine PASS reviews, then rehashes every frozen parent and
   reviewed V4 source. It returns only an immutable minimal receipt containing
   the authorization file, content, and source-map hashes needed for manifest
   provenance, never the accepted mapping.
5. The exact authorization file is a fixed repository path and is the only
   file that may open before phase-one acceptance. Absent authority must reach
   no byte opener. Malformed authority may open only that fixed file and must
   reach no target, parent, metadata, inventory, or referenced-source opener.
6. Test root/reader/skip/mapping seams must exist only in a separately named
   production-ineligible module below `lewm/tests`; production must not import
   that helper. The helper is test infrastructure, not an authority target or
   an exact-capable entry.

## Builder V3 worker boundary

The production source may expose only two fixed internal pool paths: exact scene
loading and exact source revalidation. Neither pool accepts a caller function,
callback, initializer, worker target, repository root, authority mapping, or
reader. Each submits one literal internal worker function.

Every spawned worker must independently descriptor-read the fixed authorization
file and complete the full V3 phase-one, fixed nine-target phase-two, machine
review, frozen-parent, and reviewed-V4 validation in that worker process before
it opens or parses metadata, a scene source record, a development source, a
parent geometry/render contract, or any other exact operation. Repeated
fixed-source rehashing is required and is not an optimization target.

Worker count is an exact integer in `[1,6]`. The start method remains `spawn`.
OMP, OpenBLAS, MKL, and NumExpr threads are fixed to one. CUDA, HIP, ROCr, and
GPU ordinal visibility are empty in parent and workers. No GPU path is eligible.

## Retained construction contract

Builder V3 must preserve, byte-for-byte where applicable, the already passing
raw-supervision science and artifact contract:

- metadata V5 exact population: 5,172 pairs, 10,344 endpoint references, 9,460
  unique endpoints, 88 development scenes, and three development roles;
- each unique endpoint scheduled and raycast exactly once;
- reviewed V4 camera composition, full-RPY object geometry, raycast, ground
  support, and rasterization semantics;
- eight arrays with the frozen names, dtypes, and shapes, including scalar
  ground-plane `[N]` storage and `64 x 64` three-state raster labels;
- direct pair-to-endpoint joins with duplicate, missing, orphan, cross-role,
  cross-scene, and cross-family rejection;
- complete 354-record source provenance, exact access ledger, zero forbidden
  opens, and the complete second metadata/source pass immediately before
  publication;
- deterministic one-worker/six-worker byte identity and canonical ordering;
- private sibling staging mode `0700`, retained filesystem-root/parent
  descriptors, seven-field fingerprints, single-link regular files,
  inode-owned cleanup, fsync, and `renameat2(RENAME_NOREPLACE)`; and
- the unchanged canonical dataset path
  `.generated/go2_shared_observable_camera_ray_jepa_v5/development_raw_supervision_v1`
  with every dataset-use and downstream license false.

Exact execution, canonical output creation, failure-receipt creation, source
payload access, RGB decode, legacy-label access, G2 access, held-out access,
model/checkpoint access, runtime/hardware access, and GPU use remain prohibited
during V3 implementation and independent source review.

## Review and authorization sequence

The only eligible sequence is:

1. freeze this amendment and publish its file SHA-256;
2. independently author and freeze Builder V3 and Auditor V3 against this exact
   contract without exact execution;
3. different agents independently review each frozen implementation and publish
   canonical machine PASS or BLOCK JSON at the fixed V3 review paths;
4. only if both verdicts are PASS, author a separate canonical V3 authorization
   that binds the exact nine source rows and all review cross-bindings; and
5. only after a human separately supplies that authorization file's frozen
   SHA-256 may the sealed V3 CLI validate it and consider exact construction.

Any BLOCK, changed candidate byte, changed amendment byte, role/path/schema
mismatch, absent review, shared reviewer, false-authority mismatch, or missing
cross-binding leaves exact build and audit authority absent.
