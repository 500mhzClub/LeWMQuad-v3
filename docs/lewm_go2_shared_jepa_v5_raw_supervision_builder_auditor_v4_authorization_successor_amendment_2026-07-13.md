# Shared JEPA V5 raw-supervision Builder/Auditor V4 authorization successor amendment

Date: 2026-07-13

Amendment author: `/root/raw_builder_arch`

Status: **PRE-IMPLEMENTATION CONTRACT; NO EXACT AUTHORITY; NO DATA AUTHORITY**

## Purpose

This additive amendment is frozen before any Builder V4 or Auditor V4 source,
CLI, test, handoff, or review artifact exists. It records the structurally
invalid V3 attempt, retains the compile-safe but unfrozen Builder V3 checkpoint
only as implementation input, and preregisters the only V4 source paths,
schemas, review bindings, authority ordering, and production surfaces eligible
for later independent review.

This amendment grants no exact build, exact audit, dataset use, training,
selection, calibration, G2, held-out, navigation, runtime, hardware,
production, or promotion authority. No exact authorization, canonical dataset,
audit output, development payload, protected role, checkpoint, or accelerator
may be opened or created while implementing or reviewing V4.

## Frozen V3 predecessor and invalidation

The V3 pre-implementation amendment remains immutable:

| Artifact | SHA-256 |
| --- | --- |
| `docs/lewm_go2_shared_jepa_v5_raw_supervision_builder_auditor_v3_authorization_successor_amendment_2026-07-13.md` | `501062e2eba625cf4d7ab28810f2a629652c327c770366c07f3b788f3f6f8b2b` |

Auditor V3 never became a frozen review candidate. Its handoff described one
source identity, the source changed after the handoff, and the changed source
was left nonfunctional with unresolved legacy references. The complete bound
state is:

| Artifact or identity | SHA-256 |
| --- | --- |
| `docs/lewm_go2_shared_jepa_v5_raw_supervision_auditor_v3_structural_invalidation_2026-07-13.md` | `db86ea8bb72478b0f032068151a3c492660444b1fad21b33c700b658de33e213` |
| stale `docs/lewm_go2_shared_jepa_v5_raw_supervision_auditor_v3_author_handoff_2026-07-13.md` | `a3b66f150320aa790c2a9aa3c8aa0f437824cc619de12349448155559642fe23` |
| handoff-declared Auditor V3 source identity | `08cbbc8b7ae197ee100e3327adcd2c3921c90ba834d433f0fdf0a9ce348a9606` |
| changed unusable `lewm/datasets/go2_shared_jepa_v5_raw_supervision_auditor_v3.py` | `423164701e735c17dca10449434d4d96692180ee148d2a222c9af9b357a83043` |
| `scripts/audit_go2_shared_jepa_v5_raw_supervision_v3.py` | `f1258680802be18ad77ca4cf0fa1aacef5e941d9aca40fa68a6d7d8105892445` |
| `lewm/tests/test_go2_shared_jepa_v5_raw_supervision_auditor_v3.py` | `4e111e961ed3e8a7250f6c0cfbff4033c5cb6487c67cbbb9d65d389081e9fd19` |

The declared Auditor V3 candidate retained a legacy Auditor V1 module object
and its independently blocked exact entry, exposed caller repository and
dataset paths, accepted a caller root in production phase two, and inherited an
exact-flag seam. The partial edit did not close those defects. No Auditor V3
PASS or BLOCK review JSON exists or is eligible; V3 can never authorize an
exact build or audit.

## Builder V3 implementation input

Builder V3 stopped at a compile-safe technical checkpoint when Auditor V3 was
invalidated. It is not frozen, was not independently reviewed, and is not an
authorization candidate. It has no Builder V3 role test, author handoff, or
review JSON. The checkpoint bytes are recorded solely to make the V4 carry
forward explicit:

| Implementation input | SHA-256 |
| --- | --- |
| `lewm/datasets/go2_shared_jepa_v5_raw_supervision_builder_v3.py` | `3f5154b8c48125146944c740d8cf2b8d7859543ed04e4a2513ddf06a4108c88f` |
| `scripts/build_go2_shared_jepa_v5_development_raw_supervision_v3.py` | `5bf1e8114706596e0281787e849340ed2f87f4064e7efebfd338153cfaec7ad2` |
| production-ineligible `lewm/tests/go2_shared_jepa_v5_raw_supervision_builder_v3_test_support.py` | `df1f92d116f185398ec8b752a24240d94d4a42da0756501ee58853756313145e` |

The checkpoint owns its construction engine, uses fixed worker functions, puts
full authority validation in spawned-process initializers before task
deserialization, repeats validation at worker entry, keeps metadata loaders
local to already-authorized operations, and returns a minimal immutable
authority receipt. These are implementation inputs, not inherited authority.
V4 must rebind and independently test every property under the V4 closure.

## Canonical V4 authority paths

The only eligible V4 authorization source map is an ordered list of exactly
nine objects. Each object has exactly `role`, `path`, and lower-case `sha256`:

| Order | Role | Literal repository-relative POSIX path |
| ---: | --- | --- |
| 1 | `builder_source` | `lewm/datasets/go2_shared_jepa_v5_raw_supervision_builder_v4.py` |
| 2 | `builder_cli` | `scripts/build_go2_shared_jepa_v5_development_raw_supervision_v4.py` |
| 3 | `builder_test` | `lewm/tests/test_go2_shared_jepa_v5_raw_supervision_builder_v4.py` |
| 4 | `builder_handoff` | `docs/lewm_go2_shared_jepa_v5_development_raw_supervision_builder_v4_author_handoff_2026-07-13.md` |
| 5 | `builder_review` | `docs/lewm_go2_shared_jepa_v5_development_raw_supervision_builder_v4_independent_review_2026-07-13.json` |
| 6 | `auditor_source` | `lewm/datasets/go2_shared_jepa_v5_raw_supervision_auditor_v4.py` |
| 7 | `auditor_cli` | `scripts/audit_go2_shared_jepa_v5_raw_supervision_v4.py` |
| 8 | `auditor_test` | `lewm/tests/test_go2_shared_jepa_v5_raw_supervision_auditor_v4.py` |
| 9 | `auditor_review` | `docs/lewm_go2_shared_jepa_v5_raw_supervision_auditor_v4_independent_review_2026-07-13.json` |

Roles and paths must be unique, appear in exactly this order, and equal these
literal canonical relative paths. Absolute paths, aliases, empty, dot, or
dot-dot components, backslashes, repeated separators, alternate spellings,
extra rows, missing rows, and reordered rows are invalid before any target
opener. The nine-role lineage includes the builder handoff and omits an auditor
handoff. An explanatory Auditor V4 handoff may exist, but it is not an
authorization role and cannot become a tenth target.

## Canonical V4 schemas

The V4 machine authorization schema is:

```text
lewm_go2_shared_jepa_v5_raw_supervision_build_authorization_v4
```

It has exactly these six fields:

```text
schema
exact_build_authorized_after_independent_reviews
builder_review
auditor_review
source_map
content_sha256
```

The exact-build flag is literal `true`. `content_sha256` is the canonical JSON
SHA-256 of the other five fields.

Both nested review bindings use:

```text
lewm_go2_shared_jepa_v5_raw_supervision_implementation_review_binding_v4
```

and exactly:

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

The Builder V4 review schema is:

```text
lewm_go2_shared_jepa_v5_raw_supervision_builder_v4_independent_review_v1
```

Its ordered candidate is Builder V4 source, CLI, test, and handoff. The fixed
implementation author is `/root/raw_builder_arch`.

The Auditor V4 review schema is:

```text
lewm_go2_shared_jepa_v5_raw_supervision_auditor_v4_independent_review_v1
```

Its ordered candidate is Auditor V4 source, CLI, and test. The fixed
implementation author is `/root/raw_auditor_author`.

Each review is canonical, duplicate-key-free JSON plus one newline and has
exactly:

```text
schema
verdict
reviewer
implementation_author
candidate
authority
content_sha256
```

The verdict is `PASS`. Each reviewer differs from its implementation author;
the two implementation authors differ; and the two reviewers differ. Each
candidate row exactly equals its ordered source-map subset. Each binding's
path and file hash equal its review source-map row, and its content hash equals
the review's canonical self-hash.

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

## Shared standalone production boundary

Every rule in this section applies independently to both Builder V4 and
Auditor V4.

1. Each is a standalone production module. It must not import, retain, return,
   or expose any legacy builder or auditor module object, legacy exact entry,
   legacy authority validator, compatibility bridge, or unrestricted attribute
   fallback. Named frozen data types, constants, and pure science primitives
   may be imported when necessary; non-pure metadata or inventory loaders must
   be imported locally only after full V4 authority acceptance.
2. Neither module may assign, replace, wrap, cache, monkeypatch, or otherwise
   mutate any module-global authority validator. No accepted authority mapping,
   source map, policy map, root, reader, function, or callback may be supplied
   by or returned to a caller.
3. Production phase one is pure structural validation. It accepts the
   candidate object and fixed authorization file hash only. It has no reader,
   repository root, dataset path, source-map override, policy-map override,
   callback, skip, exact flag, or test injection parameter.
4. Production phase two accepts only the frozen phase-one capsule. Before its
   first opener it canonical-parses the capsule, reruns phase one, and requires
   exact capsule equality. It has no reader, root, path, parent-skip, mapping,
   callback, function, exact flag, or test injection parameter. It opens the
   fixed nine V4 targets in order, validates both machine PASS reviews, then
   rehashes every frozen predecessor and reviewed science source. It returns
   only an immutable minimal receipt containing the authorization file,
   content, and source-map hashes, never an accepted mapping.
5. The authorization file is a fixed repository path and is the only file that
   may open before phase-one acceptance. Absent authority reaches no byte
   opener. Malformed authority may open only that fixed file and reaches no
   review target, frozen parent, metadata, inventory, dataset, manifest, report,
   or referenced-source opener.
6. Every production public exact API takes only named frozen hashes and an
   exact integer worker count. Builder V4's exact entry takes only
   `authorization_sha256` and `workers`. Auditor V4's exact entry also takes
   only `authorization_sha256` and `workers`. Repository roots, authorization
   paths, dataset paths, output paths, readers, callbacks, mappings, skip flags,
   exact flags, and production/test mode switches are fixed internally and are
   never caller parameters.
7. Test root, path, reader, callback, skip, mapping, and synthetic-value seams
   may exist only in separately named production-ineligible modules below
   `lewm/tests`. Production modules and CLIs must not import those helpers. A
   helper is neither an authority target nor an exact-capable entry.

## Shared worker boundary

Both programs use only fixed internal worker functions and fixed internal pool
paths. No pool accepts a caller function, callback, initializer, worker target,
repository root, dataset path, output path, authority mapping, reader, skip, or
exact flag. Each pool submits one literal internal worker function.

Every spawned process must descriptor-read the fixed authorization file and
complete full V4 phase one, fixed nine-target phase two, both machine-review
checks, all frozen-predecessor checks, and all reviewed-science-source checks in
its process initializer before it receives or deserializes a task payload. The
fixed worker repeats full V4 authority validation as its first operation before
opening or parsing metadata, source records, development data, dataset shards,
manifests, report state, or parent contracts. Repeated fixed-file rehashing is
required and is not an optimization target.

Worker count is an exact, non-boolean integer in `[1,6]`; the start method is
`spawn`. OMP, OpenBLAS, MKL, and NumExpr threads are one. CUDA, HIP, ROCr, and
GPU ordinal visibility are empty in the parent and all workers. No GPU path is
eligible.

## Retained builder contract

Builder V4 preserves the reviewed raw-supervision science and artifact
contract byte-for-byte where applicable:

- metadata V5 population: 5,172 pairs, 10,344 endpoint references, 9,460 unique
  endpoints, 88 development scenes, and three development roles;
- each unique endpoint scheduled and raycast exactly once;
- reviewed V4 camera composition, full-RPY object geometry, raycast, ground
  support, and rasterization semantics;
- eight arrays with frozen names, dtypes, and shapes, including scalar
  ground-plane `[N]` storage and `64 x 64` three-state raster labels;
- direct pair-to-endpoint joins with duplicate, missing, orphan, cross-role,
  cross-scene, and cross-family rejection;
- complete 354-record source provenance, exact access ledger, zero forbidden
  opens, and complete second metadata/source validation immediately before
  publication;
- deterministic one-worker/six-worker byte identity and canonical ordering;
- private sibling staging mode `0700`, retained filesystem-root and parent
  descriptors, seven-field fingerprints, single-link regular files,
  inode-owned cleanup, fsync, and `renameat2(RENAME_NOREPLACE)`; and
- canonical dataset path
  `.generated/go2_shared_observable_camera_ray_jepa_v5/development_raw_supervision_v1`
  with every dataset-use and downstream license false.

## Retained auditor contract

Auditor V4 preserves Auditor V2's security and publication properties while
owning the audit engine directly and rebinding it to V4:

- independently reconstruct and validate the complete manifest, shard, file,
  pair, endpoint, role, scene, family, population, provenance, access-ledger,
  audit-sample, array-layout, dtype, shape, hash, and canonical-JSON contract;
- reject duplicate, missing, orphan, cross-role, cross-scene, cross-family,
  noncanonical, symlink, hard-link, special-file, replaced-root, replaced-parent,
  replaced-entry, unexpected-file, and integer/bool coercion cases;
- retain fixed-root and fixed-dataset descriptors plus seven-field fingerprints
  across the complete audit and publication transaction;
- use private sibling staging, full staging inventory, fsync, and
  `renameat2(RENAME_NOREPLACE)` to publish only the fixed canonical audit report;
- never call a legacy auditor exact entry, never accept an `exact` flag, and
  never expose synthetic audit mode through production; and
- keep every dataset-use and downstream license false. A source-audit PASS is
  evidence only and does not authorize dataset use or downstream work.

## Review and authorization sequence

The only eligible sequence is:

1. freeze this amendment and publish its file SHA-256;
2. independently author and freeze Builder V4 and Auditor V4 against these
   exact paths and schemas without exact execution;
3. different agents independently review each frozen implementation and
   publish canonical machine `PASS` or `BLOCK` JSON at the fixed V4 review
   paths;
4. only if both verdicts are `PASS`, author a separate canonical V4
   authorization binding the exact nine source rows and review cross-bindings;
   and
5. only after a human separately supplies that authorization file's frozen
   SHA-256 may the sealed V4 CLI validate it and consider exact construction or
   audit.

Any BLOCK, changed candidate byte, changed amendment byte, predecessor mismatch,
role/path/schema mismatch, absent review, shared reviewer, false-authority
mismatch, missing cross-binding, exposed legacy module or exact entry, caller
root/path/reader/callback/mapping/skip/exact seam, or worker authorization-order
failure leaves exact build and exact audit authority absent.
