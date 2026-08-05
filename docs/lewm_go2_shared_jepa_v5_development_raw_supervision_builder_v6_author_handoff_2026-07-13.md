# Shared JEPA V5 development raw-supervision Builder V6 author handoff

Date: 2026-07-13

Implementation author: `/root/raw_builder_arch`

Status: **FROZEN AUTHOR CANDIDATE; INDEPENDENT REVIEW REQUIRED; NO EXACT OR DATA AUTHORITY**

## Contract

This candidate implements only the Builder V6 role preregistered by:

| Artifact | SHA-256 |
| --- | --- |
| `docs/lewm_go2_shared_jepa_v5_raw_supervision_builder_auditor_v6_authorization_successor_amendment_2026-07-13.md` | `09ced36b2eab16585c759e65f7eda844f76006b93de013e5f7057fb9a8e7a137` |

The amendment was frozen before any V6 role source existed. It binds the
different-agent Builder V5 `BLOCK`, forbids an alternating final source/staging
repair, and requires one continuously watched and retained-descriptor-bound
transaction from the pre-final baseline through source validation, staging
validation, owned atomic rename, post-rename validation, and final event drain.

This handoff is not an authorization. No exact authorization, protected source,
canonical output, audit output, dataset payload, training input, checkpoint, or
GPU was opened or created while authoring or testing this candidate.

## Frozen candidate

| Role | Artifact | SHA-256 |
| --- | --- | --- |
| `builder_source` | `lewm/datasets/go2_shared_jepa_v5_raw_supervision_builder_v6.py` | `88c36063e257d9d163317abb15d7854f3da783e0ec15537da4c3d62b113740d7` |
| `builder_cli` | `scripts/build_go2_shared_jepa_v5_development_raw_supervision_v6.py` | `089aca4882f4f574be7972914c12c05acabf1cd898bea6f59422bf07b94f828d` |
| `builder_test` | `lewm/tests/test_go2_shared_jepa_v5_raw_supervision_builder_v6.py` | `acf5ca8cdd829d1c3c4ef44dbc4fe7e5d2f05a7dc7ec01662b60d9f27ececdd0` |

The production-ineligible test helper is:

| Artifact | SHA-256 |
| --- | --- |
| `lewm/tests/go2_shared_jepa_v5_raw_supervision_builder_v6_test_support.py` | `8c203eb2ccd2158f55f977fbe8ff5f6b84bfc95b630138d7f4d10fd3f73e8361` |

The fourth Builder V6 candidate role is this handoff. Its file hash must be
computed externally after freeze and included verbatim in any independent
review candidate and later nine-row authorization source map.

## V5 failure reproduced

The frozen Builder V5 source validated and fsynced staging, then ran its full
source pass, then checked only the staging-root identity before rename. The
different-agent reproducer mutates `pairs.jsonl` during that source pass. V5
does not rehash the staged leaf and publishes the changed bytes.

Builder V6 retains the V5 source and staging descriptors and installs inotify
watches before the final source pass. The same full-build reproducer is adapted
twice: once mutating a staged leaf and once mutating a source leaf during the
final pass. Both attempts are permanently poisoned, publish no destination, and
remove only their proven-owned staging directory.

## Closed publication transaction

`_ClosedPublicationTransaction` provides the V6 repair:

1. It reconstructs the exact source-leaf map from the fixed authorization,
   nine V6 role rows, 44 frozen-parent rows, nine reviewed-science rows, three
   metadata data inputs, two parent contracts, and all 88 scene records with
   four source leaves each. Conflicting duplicate bindings reject.
2. Every source and staging leaf is opened no-follow as a single-link regular
   file. The transaction retains its descriptor and binds SHA-256, size, and
   device/inode/mode/uid/gid/size/mtime fingerprint. Every source parent and
   every staging directory is likewise retained and fingerprint-bound.
3. The complete staging file/directory namespace is canonicalized and matched
   to the already-validated file inventory plus the exact canonical manifest.
4. A private `O_NONBLOCK|O_CLOEXEC` Linux inotify instance watches every source
   parent/leaf, publication parent, staging directory, and staging leaf. Watch
   creation uses no-follow, `IN_MASK_CREATE`, and descriptor/path uniqueness.
5. The watcher permanently poisons on any source or staging event, queue
   overflow, watch loss/ignored/unmount, unknown/reused watch descriptor,
   unknown event bit, malformed/truncated record, noncanonical name, poll/read
   failure, inventory mismatch, fingerprint mismatch, or hash mismatch.
6. One coordinated retained-descriptor validation covers the complete source
   and staging inventories after the final source pass. No alternating terminal
   source/staging pass exists.
7. Only an unpoisoned transaction calls one retained-parent
   `renameat2(RENAME_NOREPLACE)`. The characterized accepted event sequence is
   exactly publication-parent `IN_MOVED_FROM|IN_ISDIR`, publication-parent
   `IN_MOVED_TO|IN_ISDIR` with the same nonzero cookie, and staging-root
   `IN_MOVE_SELF`; names, order, masks, cookie, descriptors, and cardinality are
   exact.
8. All retained source/staging hashes, fingerprints, namespace rows, and owned
   destination identity are checked again after rename. The queue is drained
   after post-rename validation and again after parent fsync.
9. Any post-rename failure removes a destination only when its root identity is
   the exact retained attempt-owned staging inode. A replaced/pre-existing
   destination is never removed.

The fixed production entry remains keyword-only
`execute_exact_build_v6(*, authorization_sha256, workers)`. No alternate root,
path, reader, callback, authority mapping, validator, skip, exact switch, or
test hook is exposed. Workers remain exact non-boolean integers in `[1,6]`, use
`spawn`, validate authority before task receipt and again before source use,
set native thread counts to one, and hide CUDA/HIP/ROCr/GPU ordinal visibility.

## Adversarial coverage

The 56 Builder V6 author tests cover:

- the frozen V6 amendment and complete frozen V5 BLOCK evidence;
- exact V6 role paths, schemas, authors, two-phase authority, fixed opener
  ordering, absent-authority zero-open behavior, malformed-authority
  one-file-only behavior, and production API/import closure;
- the V5 staging race and V6 transaction span in the frozen source AST;
- a clean real-Linux owned rename and its exact three-event signature;
- source and staging modify/restore, create/delete, rename/restore, and
  replace/restore;
- extant pre-baseline hash change, mutation during retained-FD hashing, after
  final validation, inside rename, and after rename;
- overflow, ignored/watch loss, unknown watch descriptor, unknown event mask,
  malformed record, and attempted watch-descriptor reuse;
- destination creation race, replacement after rename, exact-owned cleanup,
  and refusal to remove an unowned replacement;
- deterministic synthetic shard bytes, direct joins, reviewed array layouts,
  strict workers, worker authority/environment, retained parent/read alias
  rejection, and every inherited V5 authority/science property.

## Verification

The frozen candidate passed:

```text
/usr/bin/pytest -q lewm/tests/test_go2_shared_jepa_v5_raw_supervision_builder_v6.py
56 passed in 0.61s

/usr/bin/pytest -q \
  lewm/tests/test_go2_shared_jepa_v5_raw_supervision_builder_v5.py \
  lewm/tests/test_go2_shared_jepa_v5_raw_supervision_builder_v4.py
61 passed in 0.47s

/usr/bin/pytest -q \
  lewm/tests/test_go2_shared_jepa_v5_raw_supervision_plan_v5.py \
  lewm/tests/test_go2_shared_jepa_v5_raw_supervision_plan_v5_independent_qa.py
45 passed in 2.44s

/usr/bin/pytest -q \
  lewm/tests/test_go2_shared_jepa_v5_raw_supervision_builder_v5_independent_qa.py::test_frozen_builder_v5_candidate_rehashes_exactly
1 passed in 0.05s
```

All four V6 Python artifacts also pass `py_compile`. All 44
`FROZEN_PARENT_HASHES` and all nine `REVIEWED_V4_SOURCES` reproduce exactly.
Normalized AST comparison against Builder V5 reports no missing inherited
function/class and no changed inherited function except
`_build_exact_prepared_dataset_v5`; it reports only the 12 new transaction
types/helpers. The raycast, raster, array, joining, metadata, source-reader,
authority-worker, and exact-entry bodies are otherwise unchanged after the
mechanical V5-to-V6 type/function rename.

Filesystem adversarial tests should run as one suite rather than concurrent
copies: retained ancestor fingerprints intentionally treat another suite's
creation of sibling pytest directories as a mutation and fail closed. The
independent metadata and predecessor suites were run concurrently where their
filesystem ownership did not conflict.

## Independent review request

A reviewer other than `/root/raw_builder_arch` must first rehash the amendment,
the three role artifacts above, this handoff, the non-role helper, and all
frozen predecessor evidence. Review must independently rerun or strengthen the
V5 reproducer and every V6 event/namespace/cleanup attack. It must also inspect
the exact source-input enumeration, retained-FD validation, inotify parser,
owned-rename event acceptance, exception cleanup, fixed authority opener order,
worker ordering, unchanged science, and absence of production injection seams.

The only eligible machine review path and schema are:

```text
docs/lewm_go2_shared_jepa_v5_development_raw_supervision_builder_v6_independent_review_2026-07-13.json
lewm_go2_shared_jepa_v5_raw_supervision_builder_v6_independent_review_v1
```

The candidate order is source, CLI, test, and this handoff. A `PASS` grants only
`builder_source_approved=true`; every exact-build, exact-audit, dataset-use,
training, selection, calibration, G2, held-out, navigation, runtime, hardware,
production, and promotion field remains false. A changed byte or any review
finding requires a new additive successor.
