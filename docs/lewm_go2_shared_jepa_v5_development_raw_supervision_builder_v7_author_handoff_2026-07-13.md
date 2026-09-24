# Shared JEPA V5 development raw-supervision Builder V7 author handoff

Date: 2026-07-13

Implementation author: `/root/raw_v7_successor_author`

Status: **FROZEN AUTHOR CANDIDATE; INDEPENDENT REVIEW REQUIRED; NO EXACT OR DATA AUTHORITY**

## Contract

This candidate implements only the Builder V7 role preregistered by:

| Artifact | SHA-256 |
| --- | --- |
| `docs/lewm_go2_shared_jepa_v5_raw_supervision_builder_auditor_v7_authorization_successor_amendment_2026-07-13.md` | `ebeb552a89792b63f10c7d9ab5c9c9abd96d74d6ae7cf39f709f0657708798fc` |

The amendment was frozen before any V7 role source existed. It binds the
different-agent Builder V6 `BLOCK`, retains Auditor V6 only as a compile-safe
input, and requires continuous self-mutation coverage for the complete retained
publication ancestry plus a final post-fsync source/published-inventory close.

This handoff is not an authorization. No exact authorization, protected source,
canonical output, audit output, dataset payload, training input, checkpoint, or
GPU was opened or created while authoring or testing this candidate.

## Frozen candidate

| Role | Artifact | SHA-256 |
| --- | --- | --- |
| `builder_source` | `lewm/datasets/go2_shared_jepa_v5_raw_supervision_builder_v7.py` | `c79e68a2dcccb0fba937a9e6cd0ab778fc267b99473163c6c3c0bdbe6d1ac2ab` |
| `builder_cli` | `scripts/build_go2_shared_jepa_v5_development_raw_supervision_v7.py` | `9fdecaac622f0e5c1022c6a6298557da0ad0d7effb4b350330536da177cbf432` |
| `builder_test` | `lewm/tests/test_go2_shared_jepa_v5_raw_supervision_builder_v7.py` | `cb03351928d9d9849736a53b57d338cab10ae60bb14de062e3218e01901e99da` |

The production-ineligible test helper is:

| Artifact | SHA-256 |
| --- | --- |
| `lewm/tests/go2_shared_jepa_v5_raw_supervision_builder_v7_test_support.py` | `588e75ae507484f70775e70b9cea50dbbe9529ce1e3b843722f090e2cc28b6b9` |

The fourth Builder V7 candidate role is this handoff. Its file hash must be
computed externally after freeze and included verbatim in any independent
review candidate and later nine-row authorization source map.

## V6 failure reproduced

The frozen Builder V6 transaction retains the complete publication descriptor
chain but watches only the immediate publication parent. Its final quiet method
only drains the existing watches. The author test publishes a synthetic
dataset, moves a canonical ancestor, recreates the old canonical directories,
and calls the frozen V6 final quiet method. V6 returns without error while the
canonical destination is absent and the dataset remains under the moved alias.

The same real-filesystem sequence is then applied to V7. Its ancestor watch
observes the move and its terminal retained/canonical ancestry comparison also
detects the replacement. The V7 attempt is permanently poisoned and cannot
report success.

## Ancestor-closed transaction

Builder V7 preserves the complete V6 source/staging transaction and changes its
terminal publication boundary as follows:

1. The transaction adopts every retained descriptor from the filesystem anchor
   through the canonical publication parent. It records each descriptor's
   seven-field transaction fingerprint and stable device/inode/mode/uid/gid
   identity.
2. Every ancestry component receives an inotify watch before the final source
   pass. The ancestry mask covers attribute, delete-self, move-self, unmount,
   ignored/watch-loss, no-follow, exclusive-unlink, and no watch replacement.
3. A directory that is also a source, staging, or publication parent keeps the
   stricter V6 directory role and mask. Only a named child event from a pure
   ancestry role may be ignored. Empty-name self events, queue anomalies, watch
   loss, and every event on a merged strict role remain fatal.
4. Retained ancestry validation compares canonical parent-relative names and
   open descriptors by stable identity. Directory size/time drift caused only
   by unrelated named child churn is permitted; mode/owner, device/inode,
   symlink, type, move, deletion, and replacement changes reject.
5. After exact owned rename events, post-rename inventory validation,
   publication-parent refresh, and parent fsync, `require_final_quiet` drains
   events and revalidates the complete retained source and canonical published
   inventories. Every leaf is rehashed through its retained descriptor.
6. The terminal close drains events again, revalidates the complete ancestry
   and destination identity, then performs a final drain before success. All
   descriptors and watches stay live throughout.
7. Post-rename cleanup remains restricted to a destination whose device/inode
   identity proves it is the exact attempt-owned staging root. Recreated or
   foreign canonical paths are never followed or removed.

The fixed production entry is keyword-only
`execute_exact_build_v7(*, authorization_sha256, workers)`. No alternate root,
path, reader, callback, authority mapping, validator, skip, exact switch, or
test hook is exposed. Workers remain exact non-boolean integers in `[1,6]`, use
`spawn`, validate authority before task receipt and again before source use,
set native thread counts to one, and hide CUDA/HIP/ROCr/GPU ordinal visibility.

## Authority and provenance

The source owns the fixed ordered nine-row V7 role map and V7 schemas. The
Builder author is `/root/raw_v7_successor_author`; the separately preregistered
Auditor author is `/root/raw_v7_successor_author/auditor_v7_author`. Reviewers
must differ from both authors and from each other.

`FROZEN_PARENT_HASHES` contains 55 rows. All reproduce exactly, including the
V7 amendment, V6 amendment, complete Builder V6 candidate, independent V6 QA
and BLOCK JSON, and all four compile-safe Auditor V6 inputs. All nine retained
reviewed-science source hashes also reproduce exactly.

## Adversarial coverage

The 65 Builder V7 author tests cover:

- the frozen V7 amendment and complete frozen Builder V6 BLOCK/Auditor V6
  evidence;
- exact V7 role paths, schemas, authors, two-phase authority, fixed opener
  ordering, absent-authority zero-open behavior, malformed-authority
  one-file-only behavior, and production API/import closure;
- direct reproduction of the V6 ancestor-move false success and V7 rejection;
- complete retained ancestry watch coverage from filesystem anchor through
  publication parent and exact ancestry self-event masks;
- permitted unrelated child churn for ancestry-only roles and strict rejection
  when the same path has a source/staging/publication role;
- attribute mutate-then-restore poisoning, ancestor move/recreation after
  rename, and mutation between the terminal inventory and identity checks;
- terminal rehash of every source and published leaf and rejection of a source
  mutation during that rehash;
- the full retained V6 source/staging mutation, owned-rename event, queue,
  watch-loss/unmount, destination-race, cleanup, authority, worker, and science
  proof set; and
- deterministic synthetic shard bytes, direct joins, reviewed array layouts,
  strict workers, worker authority/environment, retained parent/read alias
  rejection, and every inherited V6 passing property.

## Verification

All commands used one native CPU thread per library and empty accelerator
visibility. Filesystem transaction suites were serialized because frozen V6
intentionally treats sibling pytest-directory churn as an ancestor mutation.

```text
pytest -q lewm/tests/test_go2_shared_jepa_v5_raw_supervision_builder_v7.py
65 passed in 0.86s

pytest -q lewm/tests/test_go2_shared_jepa_v5_raw_supervision_builder_v6.py
56 passed in 0.56s

pytest -q \
  lewm/tests/test_go2_shared_jepa_v5_raw_supervision_builder_v5.py \
  lewm/tests/test_go2_shared_jepa_v5_raw_supervision_builder_v4.py
61 passed in 0.47s

pytest -q \
  lewm/tests/test_go2_shared_jepa_v5_raw_supervision_plan_v5.py \
  lewm/tests/test_go2_shared_jepa_v5_raw_supervision_plan_v5_independent_qa.py
45 passed in 2.40s

pytest -q \
  lewm/tests/test_go2_shared_jepa_v5_raw_supervision_builder_v6_independent_qa.py::test_frozen_builder_v6_candidate_rehashes_exactly
1 passed in 0.05s
```

All four V7 Python artifacts pass `py_compile`; the V7 candidate passes
`git diff --check`; and the production source/CLI contain no legacy builder
import, legacy exact entry, test import, or visible accelerator assignment.
Normalized top-level AST comparison against Builder V6 reports no removed
science or authority implementation. Intentional changes are confined to the
retained-publication-parent class/open function, the closed transaction class,
and the two new ancestry binding/identity helpers, plus mechanical V6-to-V7
type and exact-entry names.

## Independent review request

A reviewer other than `/root/raw_v7_successor_author` and
`/root/raw_v7_successor_author/auditor_v7_author` must first rehash the
amendment, the three role artifacts above, this handoff, the non-role helper,
and all frozen predecessor evidence. Review must independently reproduce the
V6 false success, attack each V7 terminal phase, inspect the event-role filter,
prove unrelated ancestry child churn cannot mask a strict-role event, verify
complete post-fsync source/published rehashing, and retain all V6 authority,
science, worker, event, and cleanup proofs.

The only eligible machine review path and schema are:

```text
docs/lewm_go2_shared_jepa_v5_development_raw_supervision_builder_v7_independent_review_2026-07-13.json
lewm_go2_shared_jepa_v5_raw_supervision_builder_v7_independent_review_v1
```

The candidate order is source, CLI, test, and this handoff. A `PASS` grants only
`builder_source_approved=true`; every exact-build, exact-audit, dataset-use,
training, selection, calibration, G2, held-out, navigation, runtime, hardware,
production, and promotion field remains false. A changed byte or any review
finding requires a new additive successor.
