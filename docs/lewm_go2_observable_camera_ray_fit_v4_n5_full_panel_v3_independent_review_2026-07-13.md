# V4 N5 full-panel V3 independent review

Date: 2026-07-13

Reviewer: `/root/raw_plan_v2_qa`

Verdict: **BLOCK**

V3 preserves the frozen numerical experiment, prior BLOCK evidence, recovery
regressions, and fail-closed canonical state.  It does not satisfy its promised
single isolated end-to-end operation: imported production helpers can create
completion, metric, and gate artifacts from caller-supplied values without
executing the reviewed sequence.

## Blocking findings

### 1. Stage writers are callable outside the documented operation

The production executor exports the constructible `AttemptReservation` class
and `_publish_success`.  A temporary-path conformance check constructed a
reservation, supplied arbitrary checkpoint bytes and a caller result mapping,
and reached `checkpoint.pt`, `result.json`, and `completed.json`.  The writer
performed no isolation, source-review, reservation-structure, result-structure,
or lifecycle-provenance check.

Copy, deepcopy, dataclass replacement, direct reconstruction, and mutation of
the reservation's caller-owned mapping all continued to produce accepted
bindings.  This contradicts the handoff statement that no caller-held execution
object exists.

The same issue is present in `_write_canonical_json`.  With the canonical
constants redirected to a temporary mirror, caller mappings reached both the
metric and gate writers.  The function checks the path but not the mapping's
origin, schema, derivation, or participation in the isolated sequence.

### 2. Publication loses the claimed directory identity

The reservation object stores only a pathname.  The device/inode pair captured
during `_reserve_exact_attempt` is not retained after that function returns.
In the independent temporary-path check, the claimed directory was renamed and
replaced before `_publish_success`; publication continued into the replacement.

The production claim does correctly fsync the seed root immediately after
rename and checks its local inode before handling a claim-time exception.  That
passing behavior does not bind later training failure, success publication,
metric verification, or finalization to the same claimed directory.

### 3. Canonical review aliases are accepted

`preflight_source_review` resolves both the supplied path and
`CANONICAL_SOURCE_REVIEW_PATH`, then compares the two resolved values.  A
canonical leaf alias therefore resolves identically on both sides and is
accepted.  The temporary review remained byte- and content-correct; the defect
is that the required canonical directory entry was not retained.

### 4. Source reads do not retain parent identity

`read_regular_bytes` checks the leaf and later opens it by absolute pathname
with `O_NOFOLLOW`.  The leaf flag does not bind parent components.  The
independent temporary-path check changed the parent directory identity between
the preliminary check and open while preserving the leaf inode and bytes; the
read continued.

This affects the source-review promise because the helper is used for retained
and successor source rehashing.  The existing size, modification-time, inode,
and SHA-256 checks remain useful but do not close the parent-identity gap.

## Frozen candidate

The author artifacts were rehashed and were not edited.

| Artifact | SHA-256 |
|---|---|
| implementation handoff | `c97b3f761955fb6d73469c53632c27388626ae75b010c317fe64b860f76bf8db` |
| V3 policy | `b0f5929aadfaeb9a10f2211db21297c7c01d10305e094a249e5ad8f27b8f46d3` |
| V3 executor | `8a8bec79bbbfdd2554e0625afc3d423ea9ec8e56baf1134f70d334efe357af66` |
| synthetic lifecycle support | `83af899f8479f6a3e98530da5af2c58b2b0fd25b48e29954ef77db08e5bf5c91` |
| V3 author tests | `730513d7607b02539b58cde883600a28e6d0e3592333a16d5df67ac3e092beee` |
| independent V3 tests | `b7d3669135f22311e13c840e04c4ec2ed583365fc77f7fce6c5c0ecc4e512395` |

Parent V2 review, BLOCK JSON, and independent-test file hashes reproduced as
`24953fc64da151a6ff1f4ad89e5465e1caae300223556702e0f5c8430d47ee04`,
`ddca89e467e4cc30e52bacf57b28c040465e712843fde465f472f3cc8b38fc73`,
and `a53c5e5d351784ff2a4824231998194e15040597897411c91e7727ec73a95e69`.
The frozen V1 and V2 BLOCK content identities remain
`99ded56d11b357ada724b238e750d1845bd0010d72a081f4819948b3e05163e7`
and `c4d93bbac0c849a2add12bb0ab69609cef0c58a6e203a02d6b806b3c7a41fd8a`.

Machine-readable BLOCK receipt:

- path:
  `docs/lewm_go2_observable_camera_ray_fit_v4_n5_full_panel_v3_independent_review_block_2026-07-13.json`;
- file SHA-256:
  `d1f859aea2a80f090c3ee09df5194f5b4bcfca22865f323de543f3b216b3e168`;
- canonical content SHA-256:
  `d84152d611631364e4c52114a753c36fdabd1cf69d5508d4cb25b5b93dd67f2f`.

## Passing evidence

The independent review also confirmed:

- the V3 author suite remains 19/19 passing;
- the retained numerical/downstream closure remains 48/48 passing;
- the frozen V2 review reproducer remains one pass and three expected V2
  failures;
- the synthetic executor is production-ineligible, recoverable after complete
  staging, and single-use;
- the production claim has an immediate rename-to-parent-fsync sequence and a
  claim-time inode ownership check;
- retained training revalidates exact inputs and all five RGB commitments
  before checkpoint serialization and publication;
- retained module bindings are restored in `finally` blocks;
- the isolated launcher retains `-I -B`, GPU0-only visibility, no ROCm override,
  and one native thread per family; and
- the frozen schedule remains 400 updates, 2,000 exposures, full N=5 panels,
  and SHA-256
  `62efec890e572623ab6d76e8c67337ee29badaf81638943ae56ed8da0a3a8634`.

These passing checks establish scientific regression preservation.  They do
not establish that every artifact can only arise from the one documented
operation.

## Verification

All pytest commands disabled external plugins, set OMP, MKL, OpenBLAS, and
NumExpr threads to one, and hid HIP, CUDA, ROCr, and HSA devices.

```text
V3 author suite:             19 passed in 0.89s
retained closure:            48 passed in 1.81s
V2 BLOCK reproducer:          1 passed, 3 failed in 0.06s
independent V3 conformance:   7 passed, 8 failed in 0.70s
py_compile:                  PASS
git diff --check:            PASS
```

The eight independent failures cover the five finding groups recorded in the
machine-readable BLOCK receipt.  All dynamic review work used temporary paths.

The canonical V3 PASS review JSON and canonical output root remained absent.
Exact optimization was not run.  No dataset, RGB, model, checkpoint, protected
role, G2, held-out, selection, calibration, runtime, hardware, navigation, or
production payload was opened.

## Required successor

An additive successor must preserve V1-V3 and all prior evidence while:

1. making completion, metric, gate, failure, reservation, and stage-transition
   writers unreachable from caller-supplied objects or mappings outside the one
   isolated operation;
2. deriving and validating every artifact internally from raw stage outcomes;
3. retaining the claimed directory descriptor/device/inode through training,
   terminalization, success publication, verification, and finalization;
4. opening the review and all source paths through descriptor-relative,
   no-follow component walks rooted at the canonical repository descriptor;
5. retaining the current post-training input/source/RGB rehash and independent
   checkpoint recomputation; and
6. passing the eight independent V3 checks plus the 19 author, 48 retained, and
   frozen predecessor suites under a new different-agent review.

Until then, source closure is not approved and exact attempt, retry, later-rung,
V5, G2, held-out, runtime, hardware, navigation, production, and promotion
authority remain false.
