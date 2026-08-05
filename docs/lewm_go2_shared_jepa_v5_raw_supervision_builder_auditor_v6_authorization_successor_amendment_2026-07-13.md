# Shared JEPA V5 raw-supervision Builder/Auditor V6 authorization successor amendment

Date: 2026-07-13

Amendment author: `/root/raw_builder_arch`

Status: **PRE-IMPLEMENTATION CONTRACT; NO EXACT AUTHORITY; NO DATA AUTHORITY**

## Purpose

This additive amendment is frozen before any Builder V6 or Auditor V6 source,
CLI, test, handoff, or review artifact exists. It binds the independently
blocked Builder V5 candidate, records Auditor V5 only as a compile-safe
non-candidate implementation input, and preregisters the sole V6 paths,
schemas, authority boundary, and closed publication transaction eligible for
later independent review.

This amendment grants no exact build, exact audit, dataset use, training,
selection, calibration, G2, held-out, navigation, runtime, hardware,
production, or promotion authority. No exact authorization, canonical dataset,
audit output, development source payload, protected role, checkpoint, or
accelerator may be opened or created while implementing or reviewing V6.

## Frozen Builder V5 BLOCK

The coordinated V5 predecessor amendment is immutable:

| Artifact | SHA-256 |
| --- | --- |
| `docs/lewm_go2_shared_jepa_v5_raw_supervision_builder_auditor_v5_authorization_successor_amendment_2026-07-13.md` | `fe6a29a27eb0284ce84fcba409b530c6351befad18ee9d655f5f2e9b337d9e91` |

The frozen Builder V5 author candidate is:

| Role | Artifact | SHA-256 |
| --- | --- | --- |
| `builder_source` | `lewm/datasets/go2_shared_jepa_v5_raw_supervision_builder_v5.py` | `8d85635a85d5a6a3575602a89f37a01f97acf03bd0059a8ae452b21ed4cddce2` |
| `builder_cli` | `scripts/build_go2_shared_jepa_v5_development_raw_supervision_v5.py` | `3116c2a5b429cf0fbed0674de91b0569d6ecf6e10c26cd6064a3bb0349e78019` |
| `builder_test` | `lewm/tests/test_go2_shared_jepa_v5_raw_supervision_builder_v5.py` | `6b49d5d5847e22cea413a7b72da34d5fbf221f876b89bfdf899804024c9d05d6` |
| `builder_handoff` | `docs/lewm_go2_shared_jepa_v5_development_raw_supervision_builder_v5_author_handoff_2026-07-13.md` | `a8037613cca9c3879eb2dc8f9df847097a9053326ff973f01a79b3299aec9d26` |

The different-agent BLOCK evidence is:

| Evidence | SHA-256 |
| --- | --- |
| `lewm/tests/test_go2_shared_jepa_v5_raw_supervision_builder_v5_independent_qa.py` | `fc0ba7af24aeacf975a4b75855e830e9691475391979068385d9d256e8a66812` |
| `docs/lewm_go2_shared_jepa_v5_development_raw_supervision_builder_v5_independent_review_2026-07-13.json` file | `2687d43da0eb69c39b964ce72f5065fecceb5c2d28652589371d257711702307` |
| machine BLOCK canonical content | `5fd83545022f2109102407929a62568c557ff8a3a226faeab8a58c62b61201e9` |

Reviewer `/root` independently reproduced the frozen candidate and issued
`BLOCK`. Builder V5 correctly moved artifact construction, inventory, manifest
write, and fsync before the final complete source revalidation, and preserved
the prior authority, science, worker, and filesystem properties. It is blocked
because its sequential ordering still leaves an unguarded namespace:

1. staging is completely validated and fsynced;
2. `_revalidate_exact_before_publication` then reads and hashes all source
   inputs; and
3. only retained-parent and staging-root identity checks precede rename.

A staged leaf can be changed during the source revalidation and restored or
left changed before rename. The retained staging-root identity check does not
rehash its leaves, so V5 can publish bytes not represented by its validated
inventory and manifest. Reversing the two checks merely moves the same race to
the source side. Alternating a final source pass and a final staging pass cannot
close both namespaces at once and is explicitly forbidden as the V6 repair.

Builder V5 can never enter a V6 authorization map or exact execution path. V6
must mechanically preserve its passing properties while replacing the final
sequential checks with the single closed transaction below.

## Auditor V5 non-candidate checkpoint

Auditor V5 stopped immediately after the Builder V5 BLOCK. Only these
compile-safe implementation checkpoints exist:

| Implementation input | SHA-256 |
| --- | --- |
| `lewm/datasets/go2_shared_jepa_v5_raw_supervision_auditor_v5.py` | `6df29a2faea62191db3b48a93ce114adc23265458a2bb2986fa1a4c5ca732855` |
| `scripts/audit_go2_shared_jepa_v5_raw_supervision_v5.py` | `3f2b99ffbf3ab55f6d57c7686f95650f1086394739148978ab618e1b6d8e9b27` |

There is no frozen Auditor V5 role test, author handoff, independent review, or
authorization eligibility. These two files may be used only as implementation
inputs for an additive Auditor V6 successor. Neither can supply a V6 role hash
or review.

## Canonical V6 authority paths

The only eligible V6 authorization source map is an ordered list of exactly
nine objects, each with exactly `role`, `path`, and lower-case `sha256`:

| Order | Role | Literal repository-relative POSIX path |
| ---: | --- | --- |
| 1 | `builder_source` | `lewm/datasets/go2_shared_jepa_v5_raw_supervision_builder_v6.py` |
| 2 | `builder_cli` | `scripts/build_go2_shared_jepa_v5_development_raw_supervision_v6.py` |
| 3 | `builder_test` | `lewm/tests/test_go2_shared_jepa_v5_raw_supervision_builder_v6.py` |
| 4 | `builder_handoff` | `docs/lewm_go2_shared_jepa_v5_development_raw_supervision_builder_v6_author_handoff_2026-07-13.md` |
| 5 | `builder_review` | `docs/lewm_go2_shared_jepa_v5_development_raw_supervision_builder_v6_independent_review_2026-07-13.json` |
| 6 | `auditor_source` | `lewm/datasets/go2_shared_jepa_v5_raw_supervision_auditor_v6.py` |
| 7 | `auditor_cli` | `scripts/audit_go2_shared_jepa_v5_raw_supervision_v6.py` |
| 8 | `auditor_test` | `lewm/tests/test_go2_shared_jepa_v5_raw_supervision_auditor_v6.py` |
| 9 | `auditor_review` | `docs/lewm_go2_shared_jepa_v5_raw_supervision_auditor_v6_independent_review_2026-07-13.json` |

Roles and paths are unique, appear in exactly this order, and equal these
literal canonical repository-relative POSIX paths. Absolute paths, aliases,
empty, dot, or dot-dot components, backslashes, repeated separators, alternate
spellings, reordered rows, missing rows, and extra rows reject before any
target opener. The builder handoff is the fourth builder candidate. An auditor
handoff may explain the implementation but is not a tenth authority role.

## Canonical V6 schemas

The machine authorization schema is:

```text
lewm_go2_shared_jepa_v5_raw_supervision_build_authorization_v6
```

It has exactly `schema`,
`exact_build_authorized_after_independent_reviews`, `builder_review`,
`auditor_review`, `source_map`, and `content_sha256`. The exact-build flag is
literal `true`; the content hash is the canonical JSON SHA-256 of the other
five fields.

Both nested review bindings use:

```text
lewm_go2_shared_jepa_v5_raw_supervision_implementation_review_binding_v6
```

and exactly `schema`, `review_schema`, `verdict`, `reviewer`,
`implementation_author`, `path`, `file_sha256`, `content_sha256`, and
`candidate`.

The Builder V6 machine review schema is:

```text
lewm_go2_shared_jepa_v5_raw_supervision_builder_v6_independent_review_v1
```

Its ordered candidate is Builder V6 source, CLI, test, and handoff. The fixed
implementation author is `/root/raw_builder_arch`.

The Auditor V6 machine review schema is:

```text
lewm_go2_shared_jepa_v5_raw_supervision_auditor_v6_independent_review_v1
```

Its ordered candidate is Auditor V6 source, CLI, and test. The fixed
implementation author is `/root/raw_auditor_author`.

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

Every V5 passing boundary remains mandatory for both V6 programs:

1. Each production module is standalone and retains no legacy builder/auditor
   module object, exact entry, validator, compatibility bridge, or unrestricted
   fallback. Only named immutable data types/constants and pure reviewed science
   primitives may be imported. Non-pure loaders are local to already-authorized
   operations.
2. Neither module assigns, replaces, wraps, caches, or mutates a global
   authority validator. No caller supplies or receives an accepted authority
   mapping, root, path, reader, function, callback, initializer, skip, or exact
   flag.
3. Phase one is pure structural validation. Phase two accepts only the frozen
   phase-one capsule, canonical-parses and revalidates it before its first
   opener, opens the fixed nine V6 targets in order, validates both machine
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
6. Synthetic/test injection lives only in separately named production-
   ineligible modules below `lewm/tests`, which production never imports.

Worker count is an exact non-boolean integer in `[1,6]`. All pools use `spawn`,
one fixed process initializer, and literal internal workers. Every initializer
completes full V6 authority validation before task receipt/deserialization;
each worker repeats it first before using metadata or opening data. Native CPU
threads are one and CUDA/HIP/ROCr/GPU ordinal visibility is empty.

## Required V6 closed publication transaction

After all artifact construction, canonical manifest write, complete staging
inventory validation, and required fsyncs, Builder V6 must execute exactly one
Linux publication transaction. It starts before the final complete source
revalidation and remains live without a gap until publication succeeds or the
attempt is poisoned and cleaned up.

### Bound input and staging inventories

Before the final pass, the transaction builds a canonical source-leaf inventory
covering every byte source that the final revalidation can open:

- the fixed authorization file and all nine V6 authority targets;
- every `FROZEN_PARENT_HASHES` and reviewed-science source;
- the fixed metadata-plan implementation and its frozen parent/review inputs;
- the geometry contract and render-audit contract; and
- all four source paths for each of the 88 source records: frames, scene
  manifest, render plan, and render summary.

Duplicate canonical paths reject unless all expected bindings are identical.
Each source leaf is a single-link regular file opened without following a final
symlink. The inventory binds its canonical path, expected SHA-256, actual
SHA-256, byte size, and retained-descriptor seven-field fingerprint: device,
inode, mode, uid, gid, size, and nanosecond mtime. Every distinct parent
directory is retained and binds the same seven-field fingerprint plus its
canonical path.

The transaction separately walks the complete private staging tree after its
manifest exists. It binds every relative directory and regular-file path in
canonical sorted order, rejects symlinks and other file types, retains every
directory and file descriptor, and binds every descriptor fingerprint. Every
file additionally binds its byte SHA-256 and size. The complete staging
inventory must exactly reproduce the already-validated manifest inventory; no
omitted, extra, aliased, replaced, multiply linked, or non-regular leaf is
allowed. The retained publication-parent descriptor, staging-root descriptor,
and their fingerprints are part of the same baseline.

No final source or staging validation may begin until this full baseline is
complete and all watches below are installed. Baseline construction validates
all expected source hashes, so an extant change immediately before the baseline
rejects rather than becoming a new accepted baseline.

### Continuous Linux event coverage

V6 uses a private nonblocking close-on-exec Linux inotify instance. Before the
final pass it installs watches on every bound source leaf, every distinct
source parent directory, the publication parent, every staging directory, and
every staging leaf. Directory masks cover create, delete, move-from, move-to,
attribute, modify, close-write, delete-self, move-self, unmount, and ignored.
Leaf masks cover attribute, modify, close-write, delete-self, move-self,
unmount, and ignored. Read/open/access events are deliberately not requested.

Watch installation itself is fail-closed. Duplicate or reused watch descriptors
for distinct canonical watch identities, impossible descriptor/path identity,
watch loss, `IN_IGNORED`, `IN_UNMOUNT`, `IN_Q_OVERFLOW`, truncated or malformed
records, an unknown watch descriptor, unknown mask bits, or an undecodable or
noncanonical event name permanently poisons the attempt. A poisoned attempt can
never publish even if bytes are later restored.

Once the pre-final event queue is proven empty, the transaction runs the
complete second metadata/source pass while all source and staging descriptors
and watches remain live. It then validates, through the retained descriptors,
the full source inventory, full staging inventory, directory fingerprints,
publication-parent fingerprint, and staging-root identity exactly once. These
checks are one coordinated transaction validation, not alternating final source
and staging passes. It drains the event queue after validation; any source or
staging mutation, creation, deletion, replacement, rename, restoration,
attribute change, or write event poisons the attempt regardless of final hash.

### Atomic publication and exact event acceptance

Only an unpoisoned transaction may call the single
`renameat2(RENAME_NOREPLACE)` using the retained publication-parent descriptor
and the exact owned staging basename and canonical destination basename. The
inotify queue remains live through the call. After rename, V6 drains the queue
to quiescence and accepts only the kernel event sequence attributable to that
one owned rename: the publication-parent `IN_MOVED_FROM` for the exact staging
basename and `IN_MOVED_TO` for the exact destination basename with one equal,
nonzero cookie, plus only any documented staging-root self-move notification
from the same operation. Names, masks, cookies, watch identities, order, and
cardinality must match the frozen implementation's characterized Linux test.
Every other event poisons the attempt.

The post-rename validation compares the retained inventory descriptors and
fingerprints with the destination namespace, proves that the destination inode
is the exact previously owned staging inode, and refreshes/fsyncs the retained
parent. Watches and retained descriptors close only after the post-rename queue
is drained and all checks pass. Overflow, watch loss, descriptor reuse, a
source/staging event concurrent with rename, or post-rename mismatch is a hard
failure.

On post-rename failure, cleanup may remove only the publication whose
destination fingerprint proves it is the exact retained, attempt-owned staging
inode. It must never remove a pre-existing or replaced path. Pre-rename failure
uses the existing inode-owned staging cleanup. Failure receipts remain
non-authoritative and cannot conceal the primary failure.

## Required V6 adversarial proof

Builder V6 author tests and different-agent review tests must bind the frozen
V5 hashes and reproduce its publication failure. They must then prove V6 rejects
without publication for each of the following independently:

- staging-leaf modify, create, delete, replace, rename, and modify-then-restore;
- source-leaf modify, create, delete, replace, rename, and modify-then-restore;
- source-parent and staging-parent namespace mutations;
- mutations injected immediately before final validation, during the complete
  source pass, after retained-descriptor validation, and at rename;
- inotify overflow, ignored/watch loss, unknown watch descriptor, unknown event
  bit, malformed event, and attempted watch-descriptor reuse;
- destination pre-existence and replacement races; and
- post-rename mismatch with proof that only a fingerprint-proven owned
  destination is removed.

The tests must also characterize and freeze the exact successful owned-rename
event sequence on Linux, prove a clean publication succeeds, prove no alternating
source/staging terminal checks appear in the source AST, and prove the event
transaction begins before and ends after the final pass and rename. Test hooks
or synthetic path injection may exist only in production-ineligible test
support below `lewm/tests`; the production entry and production helper
signatures expose no hook, callback, alternate root/path, reader, mapping, skip,
or event source.

## Retained science and publication contract

V6 preserves all other V5 passing properties byte-for-byte where applicable:

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
2. independently author/freeze Builder V6 and Auditor V6 without exact work;
3. different agents independently review the exact frozen candidates and
   publish canonical PASS or BLOCK JSON at the fixed V6 review paths;
4. only two PASS reviews permit a separate canonical V6 authorization binding
   all nine exact rows and review cross-bindings; and
5. only after a human separately supplies that authorization file's frozen hash
   may either fixed CLI validate it and consider exact work.

Any changed byte, BLOCK, role/path/schema mismatch, missing cross-binding,
publication-event anomaly, or authority-boundary regression requires a new
additive successor. This amendment itself is not an authorization.
