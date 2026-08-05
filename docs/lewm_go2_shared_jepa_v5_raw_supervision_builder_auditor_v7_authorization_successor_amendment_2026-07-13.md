# Shared JEPA V5 raw-supervision Builder/Auditor V7 authorization successor amendment

Date: 2026-07-13

Amendment author: `/root/raw_v7_successor_author`

Status: **PRE-IMPLEMENTATION CONTRACT; NO EXACT AUTHORITY; NO DATA AUTHORITY**

## Purpose

This additive amendment is frozen before any Builder V7 or Auditor V7 source,
CLI, test, handoff, or review artifact exists. It binds the independently
blocked Builder V6 candidate, records Auditor V6 only as a compile-safe
non-candidate implementation input, and preregisters the sole V7 paths,
schemas, authority boundary, and ancestor-closed publication transaction
eligible for later independent review.

This amendment grants no exact build, exact audit, dataset use, training,
selection, calibration, G2, held-out, navigation, runtime, hardware,
production, or promotion authority. No exact authorization, canonical dataset,
audit output, development source payload, protected role, checkpoint, or
accelerator may be opened or created while implementing or reviewing V7.

## Frozen Builder V6 BLOCK

The V6 predecessor amendment and frozen Builder V6 author candidate are:

| Role | Artifact | SHA-256 |
| --- | --- | --- |
| `v6_amendment` | `docs/lewm_go2_shared_jepa_v5_raw_supervision_builder_auditor_v6_authorization_successor_amendment_2026-07-13.md` | `09ced36b2eab16585c759e65f7eda844f76006b93de013e5f7057fb9a8e7a137` |
| `builder_source` | `lewm/datasets/go2_shared_jepa_v5_raw_supervision_builder_v6.py` | `88c36063e257d9d163317abb15d7854f3da783e0ec15537da4c3d62b113740d7` |
| `builder_cli` | `scripts/build_go2_shared_jepa_v5_development_raw_supervision_v6.py` | `089aca4882f4f574be7972914c12c05acabf1cd898bea6f59422bf07b94f828d` |
| `builder_test` | `lewm/tests/test_go2_shared_jepa_v5_raw_supervision_builder_v6.py` | `acf5ca8cdd829d1c3c4ef44dbc4fe7e5d2f05a7dc7ec01662b60d9f27ececdd0` |
| `builder_handoff` | `docs/lewm_go2_shared_jepa_v5_development_raw_supervision_builder_v6_author_handoff_2026-07-13.md` | `d2cf130a9e2c902776327f6bd71a1b1f363a4dcfde6df0e2aba15edc3957e80b` |

The different-agent terminal BLOCK evidence is:

| Evidence | SHA-256 |
| --- | --- |
| `lewm/tests/test_go2_shared_jepa_v5_raw_supervision_builder_v6_independent_qa.py` | `2c74e3315be3443bab11a3b7896df4df29d8b233b634b7ab539123386bc0c89a` |
| `docs/lewm_go2_shared_jepa_v5_development_raw_supervision_builder_v6_independent_review_2026-07-13.json` file | `55d50a38f0c7d23e4ff537b124db3b9f24a24ea5b30413ff6be1ac381870c163` |
| machine BLOCK canonical content | `c639170b672180c8943e08efaff8d23063e8773488d1ff0f77beeb4ce44dd74b` |

Reviewer `/root` independently reproduced the candidate and issued `BLOCK`.
Builder V6 correctly maintains one continuous source-and-staging inotify
transaction through its complete final source pass, retained-inventory checks,
owned no-replace rename, and post-rename validation. It is blocked because the
continuous watch set does not include the canonical publication ancestors
above the immediate publication parent. After post-rename validation an
adversary can move such an ancestor and recreate the canonical path. The final
quiet operation drains the existing watches but does not revalidate the
retained publication ancestry, so the builder can return success with the
dataset reachable only under the moved alias and absent at the canonical path.

Builder V6 can never enter a V7 authorization map or exact execution path. V7
must mechanically preserve every passing V6 property while closing only this
publication-ancestry and terminal-success gap.

## Auditor V6 non-candidate checkpoint

Auditor V6 was authored while the Builder V6 review was in flight. The Builder
V6 BLOCK makes a dual-PASS V6 authorization impossible, so these bytes are
compile-safe implementation inputs only:

| Implementation input | SHA-256 |
| --- | --- |
| `lewm/datasets/go2_shared_jepa_v5_raw_supervision_auditor_v6.py` | `cf67c993427950c147860f9afe0e7661b2cb6841ccec27a867868cc34c7c00b8` |
| `scripts/audit_go2_shared_jepa_v5_raw_supervision_v6.py` | `de37e42d09d949ac5ca1cd8e4ebba2d32e757ef72cc769a151f814cc8fe84ffe` |
| `lewm/tests/test_go2_shared_jepa_v5_raw_supervision_auditor_v6.py` | `6cc84a493cb677437385efd3c00a8120b26748e8cabb2abd76d0f4825deaf764` |
| `docs/lewm_go2_shared_jepa_v5_raw_supervision_auditor_v6_author_handoff_2026-07-13.md` | `f7e0c1244eb55a826dfc90f7f633d88f4c3390ae3c8551949028a9757da4dc15` |

Auditor V6 already implements retained and continuously watched canonical
source and publication ancestors. V7 may promote that logic additively to the
V7 authority and report namespace, but none of the V6 files supplies a V7 role,
review, authorization, or exact authority.

## Canonical V7 authority paths

The only eligible V7 authorization source map is an ordered list of exactly
nine objects, each with exactly `role`, `path`, and lower-case `sha256`:

| Order | Role | Literal repository-relative POSIX path |
| ---: | --- | --- |
| 1 | `builder_source` | `lewm/datasets/go2_shared_jepa_v5_raw_supervision_builder_v7.py` |
| 2 | `builder_cli` | `scripts/build_go2_shared_jepa_v5_development_raw_supervision_v7.py` |
| 3 | `builder_test` | `lewm/tests/test_go2_shared_jepa_v5_raw_supervision_builder_v7.py` |
| 4 | `builder_handoff` | `docs/lewm_go2_shared_jepa_v5_development_raw_supervision_builder_v7_author_handoff_2026-07-13.md` |
| 5 | `builder_review` | `docs/lewm_go2_shared_jepa_v5_development_raw_supervision_builder_v7_independent_review_2026-07-13.json` |
| 6 | `auditor_source` | `lewm/datasets/go2_shared_jepa_v5_raw_supervision_auditor_v7.py` |
| 7 | `auditor_cli` | `scripts/audit_go2_shared_jepa_v5_raw_supervision_v7.py` |
| 8 | `auditor_test` | `lewm/tests/test_go2_shared_jepa_v5_raw_supervision_auditor_v7.py` |
| 9 | `auditor_review` | `docs/lewm_go2_shared_jepa_v5_raw_supervision_auditor_v7_independent_review_2026-07-13.json` |

Roles and paths are unique, appear in exactly this order, and equal these
literal canonical repository-relative POSIX paths. Absolute paths, aliases,
empty, dot, or dot-dot components, backslashes, repeated separators, alternate
spellings, reordered rows, missing rows, and extra rows reject before any
target opener. The builder handoff is the fourth builder candidate. An auditor
handoff may explain the implementation but is not a tenth authority role.

## Canonical V7 schemas and authors

The machine authorization schema is:

```text
lewm_go2_shared_jepa_v5_raw_supervision_build_authorization_v7
```

It has exactly `schema`,
`exact_build_authorized_after_independent_reviews`, `builder_review`,
`auditor_review`, `source_map`, and `content_sha256`. The exact-build flag is
literal `true`; the content hash is the canonical JSON SHA-256 of the other
five fields.

Both nested review bindings use:

```text
lewm_go2_shared_jepa_v5_raw_supervision_implementation_review_binding_v7
```

and exactly `schema`, `review_schema`, `verdict`, `reviewer`,
`implementation_author`, `path`, `file_sha256`, `content_sha256`, and
`candidate`.

The Builder V7 review schema is:

```text
lewm_go2_shared_jepa_v5_raw_supervision_builder_v7_independent_review_v1
```

Its ordered candidate is Builder V7 source, CLI, test, and handoff. Its fixed
implementation author is `/root/raw_v7_successor_author`.

The Auditor V7 review schema is:

```text
lewm_go2_shared_jepa_v5_raw_supervision_auditor_v7_independent_review_v1
```

Its ordered candidate is Auditor V7 source, CLI, and test. Its fixed
implementation author is `/root/raw_v7_successor_author/auditor_v7_author`.

Each review is duplicate-key-free canonical JSON plus one newline with exactly
`schema`, `verdict`, `reviewer`, `implementation_author`, `candidate`,
`authority`, and `content_sha256`. Verdict is `PASS`; reviewers differ from
their implementation authors and from each other; implementation authors
differ. Candidate rows exactly equal the ordered source-map subsets. Binding
path/file/content hashes exactly cross-bind the review row and canonical review
self-hash.

The authority map has only `builder_source_approved` or
`auditor_source_approved` true for its respective review. All exact build,
exact audit, dataset use, training, selection, calibration, G2, held-out,
runtime, navigation, hardware, production, and promotion fields are false.

## Standalone authority and API boundary

Every passing V6 boundary remains mandatory for both V7 programs:

1. Each production module is standalone and retains no legacy builder/auditor
   module object, exact entry, validator, compatibility bridge, or unrestricted
   fallback. Only named immutable data types/constants and pure reviewed
   science primitives may be imported. Non-pure loaders are local to
   already-authorized operations.
2. Neither module assigns, replaces, wraps, caches, or mutates a global
   authority validator. No caller supplies or receives an accepted authority
   mapping, root, path, reader, function, callback, initializer, skip, or exact
   flag.
3. Phase one is pure structural validation. Phase two accepts only the frozen
   phase-one capsule, canonical-parses and revalidates it before its first
   opener, opens the fixed nine V7 targets in order, validates both machine
   reviews, rehashes all frozen predecessors and reviewed science, and returns
   only the immutable authorization-file/content/source-map hash receipt.
4. The fixed authorization path is the only possible pre-phase-one opener.
   Absent authority reaches zero byte openers. Malformed authority opens only
   that file and reaches no review, parent, metadata, inventory, source,
   dataset, manifest, report, or output operation.
5. Builder and Auditor exact entries each accept only keyword-only
   `authorization_sha256` and `workers`. Roots, authorization paths, dataset
   paths, output paths, readers, callbacks, mappings, skips, exact flags, and
   test switches remain fixed internally.
6. Synthetic/test injection lives only in separately named
   production-ineligible modules below `lewm/tests`, which production never
   imports.

Worker count is an exact non-boolean integer in `[1,6]`. All pools use `spawn`,
one fixed process initializer, and literal internal workers. Every initializer
completes full V7 authority validation before task receipt/deserialization;
each worker repeats it first before using metadata or opening data. Native CPU
threads are one and CUDA/HIP/ROCr/GPU ordinal visibility is empty.

## Required V7 ancestor-closed publication transaction

V7 preserves the V6 closed transaction unchanged from full baseline creation
through final source revalidation, retained source and staging inventory
validation, queue drain, `renameat2(RENAME_NOREPLACE)`, and characterized owned
rename events. It adds the following complete publication-ancestry coverage and
terminal success boundary.

### Complete retained ancestry

The transaction adopts the already-retained descriptor chain from the
filesystem anchor through every canonical component of the publication parent.
Before final source validation begins, it binds each ancestry directory by
canonical absolute path, retained descriptor, parent descriptor and basename
where applicable, seven-field fingerprint, and device/inode identity. The
publication parent and staging root remain part of the full V6 source/staging
baseline. Every retained ancestry component is revalidated through both its
retained descriptor and its canonical parent-relative namespace entry.

Each ancestry component receives continuous inotify coverage before the final
pass and keeps it until final success or poison cleanup. The self-event mask
covers `IN_ATTRIB`, `IN_DELETE_SELF`, `IN_MOVE_SELF`, `IN_UNMOUNT`, and
`IN_IGNORED`. The immediate publication parent retains its stricter V6
directory mask for the owned staging/destination namespace. A path that is
also a strict V6 source or staging parent retains the union of those stricter
roles.

An ancestry watch is concerned only with the watched directory itself. Events
with a nonempty child name that arise solely from an ancestry role are ignored,
so unrelated sibling or child churn above the publication parent cannot poison
an otherwise valid build. Empty-name self mutation, watch loss, unmount,
move-self, delete-self, or changed retained/canonical fingerprint poisons the
attempt. Events for any merged strict source, staging, or publication-parent
role remain governed by that stricter role and are never ignored as unrelated
churn.

### Final success boundary

After the owned rename, exact owned-event drain, post-rename validation,
publication-parent refresh, and publication-parent fsync, V7 performs one
explicit terminal close while every descriptor and watch remains live:

1. drain the queue and accept no event except any still-pending exact owned
   rename event already accounted for by the characterized sequence;
2. revalidate the complete retained publication ancestry through descriptors
   and canonical namespace entries;
3. revalidate the complete retained source inventory, including hashes,
   fingerprints, parents, link counts, and canonical names;
4. revalidate the complete published inventory at the canonical destination,
   proving every directory and file is the retained former staging object and
   every byte hash, size, fingerprint, link count, relative path, and manifest
   binding is unchanged;
5. revalidate the publication parent and destination identities after those
   reads; and
6. drain the event queue again to quiescence, poison on every unexpected event,
   then repeat the retained ancestry and canonical destination identity check
   immediately before success is returned.

Moving, replacing, deleting, unmounting, or attribute-changing any retained
publication ancestor at any point from baseline through this boundary must
reject. Recreating an identical-looking canonical path never repairs a poison.
No watch or retained descriptor closes before the terminal checks pass.

On post-rename failure, cleanup may remove only the publication whose
destination fingerprint proves it is the exact retained, attempt-owned staging
inode. It must never follow the canonical recreated path or remove a
pre-existing/replaced path. Pre-rename cleanup remains inode-owned. Failure
receipts remain non-authoritative and cannot conceal the primary failure.

## Required V7 adversarial proof

Builder V7 author tests and different-agent review tests must bind the frozen
V6 hashes and reproduce its ancestor-move success bug. They must prove V7:

- rejects ancestor move, delete, replacement, unmount/watch loss, attribute
  mutation, and move-then-canonical-recreation before and after rename;
- rejects those mutations when injected during the terminal source inventory,
  published inventory, first drain, second drain, and final identity check;
- permits unrelated named child create/modify/rename/delete churn on an
  ancestry-only watch without poisoning;
- preserves strict rejection for the same named child churn when the watched
  directory also has a source, staging, or publication-parent role;
- rehashes and identity-checks every retained source and every published leaf
  after parent fsync and before return;
- retains exact successful owned-rename event characterization and all V6
  adversarial source/staging mutation, overflow, watch-loss, destination-race,
  cleanup-ownership, worker, authority, and science proofs; and
- contains no alternating final source/staging validation workaround and no
  production hook, callback, alternate root/path, reader, mapping, skip, or
  event source.

Tests use synthetic temporary roots only. No test may open canonical authority,
source payload, dataset, audit output, G2, held-out, runtime, hardware,
production, or promotion namespaces.

## Auditor V7 promotion

Auditor V7 promotes the compile-safe Auditor V6 implementation additively. It
must replace all V6 authority roles, schemas, author bindings, output report
namespace, provenance labels, accepted capsules, exact entry names, and
publication context types with the fixed V7 equivalents. It binds this V7
amendment and the frozen Builder V6 BLOCK/Auditor V6 inputs as predecessors.

Its already-implemented complete ancestor retention and self-watch behavior is
preserved. Auditor V7 author tests must prove the new nine-row authority/API
boundary, absent-authority zero-open behavior, fixed CPU worker policy,
continuous source/dataset/report-candidate and ancestor coverage, owned
no-replace publication, final ancestor rejection, and preservation of foreign
destinations. It may not import or call Auditor V6, Builder V7, or any legacy
exact entry.

## Retained science and publication contract

V7 preserves all other V6 passing properties byte-for-byte where applicable:

- metadata V5 population of 5,172 pairs, 10,344 endpoint references, 9,460
  unique endpoints, 88 development scenes, and three roles;
- exactly one schedule/raycast operation per unique endpoint;
- reviewed V4 camera composition, full-RPY geometry, raycast, ground support,
  and raster semantics;
- eight frozen arrays/dtypes/shapes, scalar ground-plane `[N]`, and `64 x 64`
  three-state raster labels;
- strict direct joins and duplicate/missing/orphan/cross-context rejection;
- complete 354-record provenance, exact access ledger, zero forbidden opens,
  and complete second pass;
- deterministic one/six-worker bytes and canonical ordering;
- retained filesystem descriptors, private `0700` sibling staging, single-link
  regular files, inode-owned cleanup, fsync, and one
  `renameat2(RENAME_NOREPLACE)`; and
- the unchanged canonical dataset path with every dataset-use/downstream
  license false.

## Review and authorization sequence

The only eligible sequence is:

1. freeze this amendment and publish its file SHA-256;
2. independently author/freeze Builder V7 and Auditor V7 without exact work;
3. different agents independently review the exact frozen candidates and
   publish canonical PASS or BLOCK JSON at the fixed V7 review paths;
4. only two PASS reviews permit a separate canonical V7 authorization binding
   all nine exact rows and review cross-bindings; and
5. only after a human separately supplies that authorization file's frozen
   hash may either fixed CLI validate it and consider exact work.

Any changed byte, BLOCK, role/path/schema mismatch, missing cross-binding,
publication-event anomaly, or authority-boundary regression requires a new
additive successor. This amendment itself is not an authorization.
