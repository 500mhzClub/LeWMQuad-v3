# Shared JEPA V5 Raw Builder V7 independent review

Date: 2026-07-13

Reviewer: `/root`

Implementation author: `/root/raw_v7_successor_author`

Verdict: **PASS for Builder V7 source only**

## Authority boundary

This review covers only the frozen Builder V7 source, CLI, author test, and
handoff. It does not review Auditor V7 and cannot substitute for the separately
required Auditor V7 review. It authorizes no exact build, exact audit, dataset
use, training, selection, calibration, G2, held-out, navigation, runtime,
hardware, production, promotion, or deployment operation.

No canonical authorization or dataset was created. Review used temporary
synthetic files, CPU-hidden tests, source reads, and hashes only. It opened no
experiment payload, RGB, label array, checkpoint, model output, accelerator,
G2, held-out, runtime, hardware, or production input.

## Frozen candidate

| Role | Path | SHA-256 |
| --- | --- | --- |
| Amendment | `docs/lewm_go2_shared_jepa_v5_raw_supervision_builder_auditor_v7_authorization_successor_amendment_2026-07-13.md` | `ebeb552a89792b63f10c7d9ab5c9c9abd96d74d6ae7cf39f709f0657708798fc` |
| Builder source | `lewm/datasets/go2_shared_jepa_v5_raw_supervision_builder_v7.py` | `c79e68a2dcccb0fba937a9e6cd0ab778fc267b99473163c6c3c0bdbe6d1ac2ab` |
| Builder CLI | `scripts/build_go2_shared_jepa_v5_development_raw_supervision_v7.py` | `9fdecaac622f0e5c1022c6a6298557da0ad0d7effb4b350330536da177cbf432` |
| Builder author test | `lewm/tests/test_go2_shared_jepa_v5_raw_supervision_builder_v7.py` | `cb03351928d9d9849736a53b57d338cab10ae60bb14de062e3218e01901e99da` |
| Builder handoff | `docs/lewm_go2_shared_jepa_v5_development_raw_supervision_builder_v7_author_handoff_2026-07-13.md` | `b4fc01993b5d47e11f789192cfcf0d4e9a8ea5cce865ef0624cbce7e6379642d` |
| Production-ineligible author helper | `lewm/tests/go2_shared_jepa_v5_raw_supervision_builder_v7_test_support.py` | `588e75ae507484f70775e70b9cea50dbbe9529ce1e3b843722f090e2cc28b6b9` |
| Independent QA | `lewm/tests/test_go2_shared_jepa_v5_raw_supervision_builder_v7_root_independent_review.py` | `ce670f0743d28ea2f1881756741b46feada321c1a18876e222bf12b844eefbee` |

All candidate hashes match the author handoff. All 55 frozen parent hashes and
all nine reviewed V4 science-source hashes rehash exactly.

## Independent findings

### V6 exploit and V7 repair

The review reproduced the decisive V6 failure with real filesystem operations:
after owned publication and parent fsync, moving a canonical ancestor and
recreating an identical-looking canonical path lets V6
`require_final_quiet()` return success while the dataset exists only below the
moved alias and the canonical dataset is absent.

The identical attack against V7 fails closed. V7 keeps a retained descriptor
and watch for every component from the filesystem anchor through the
publication parent. The move emits a watched self event and the retained-versus-
canonical device/inode comparison also detects the replacement. Recreating the
path cannot clear the poison.

An independent second attack moved and recreated the ancestor after the full
terminal source/published inventory rehash but before the final identity
boundary. V7 rejected it. Source and published leaves are rehashed through
retained descriptors while all watches remain live, and the destination must
remain the original staging inode.

### Event handling

An ancestry-only watch ignores only a nonempty named-child event. Empty-name
self events, moved/deleted/attribute-changed ancestors, unmount, watch loss,
queue overflow, unknown descriptors, unknown masks, malformed names, and
strict-role events remain fatal. When a path also has a source, staging, or
publication-parent role, that strict role prevents the ancestry-only ignore.

`IN_IGNORED` is intentionally handled as an output-only event rather than a
subscription bit: Linux emits it automatically when a watch is removed. The
parser includes it in accepted event bits and poisons immediately on it. This
matches the kernel contract and is not a missing-watch gap.

### Authority and process boundary

The production entry is keyword-only
`execute_exact_build_v7(*, authorization_sha256, workers)`. Invalid worker
values reject before authority. Absent authority reaches no byte or metadata
opener; malformed authority opens only the fixed authorization leaf; phase one
is structural; phase two revalidates the capsule, opens exactly the fixed nine
roles in order, validates both canonical reviews, and rehashes every frozen
parent/science source.

No legacy builder module, legacy exact entry, auditor implementation, test
helper, injected root/path/reader/callback, mutable authority validator, or
accelerator-visible worker path is retained. Workers use spawn, validate
authority before task receipt and again before source use, set native math
threads to one, and hide HIP/CUDA/ROCr/GPU ordinal selectors.

### Science and publication

The retained population, direct joins, V4 calibration/raycast/ground/raster
semantics, eight array layouts, deterministic shard ordering, access ledger,
second source pass, and one/six-worker equivalence remain covered. Publication
uses private sibling staging, retained no-follow descriptors, single-link
regular leaves, fsync, and exactly one
`renameat2(RENAME_NOREPLACE)`. Cleanup follows only a matching owned inode and
does not remove a recreated or foreign destination.

## Verification evidence

All commands used one native thread per math library, disabled external pytest
plugins, and hid every accelerator selector.

```text
Builder V7 author suite:                         65 passed
Retained Builder V6/V5/V4 suites:              117 passed
Metadata V5 plus V6 frozen-candidate rehash:     46 passed
Independent Builder V7 QA:                       4 passed
Total observed tests:                          232 passed
py_compile for source/CLI/tests/helpers:         PASS
git diff --check for reviewed artifacts:         PASS
candidate and frozen-parent rehash:               PASS
canonical authorization/output mutation:          NONE
protected data/checkpoint/GPU access:              NONE
```

The independent QA specifically covers frozen hashes, the live V6 false
success, the corresponding V7 rejection, mutation between terminal boundaries,
fixed production signature, absence of legacy/test imports, and the complete
ancestor self-event/watch-loss handling contract.

## Decision

Builder V7 satisfies the frozen amendment and receives
`builder_source_approved=true` only. Every downstream authority remains false.
The canonical machine review JSON is published separately and last. Any change
to the four candidate bytes invalidates this PASS and requires a new additive
successor.
