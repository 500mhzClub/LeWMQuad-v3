# Shared JEPA V5 raw-supervision metadata plan V5 author handoff

Date: 2026-07-13

Implementation author: `/root/v4_full_panel_v3_author`

Status: **AUTHOR COMPLETE; DIFFERENT-AGENT REVIEW REQUIRED**

## Scope

V5 is an additive continuity successor to the frozen V4 BLOCK. V1-V4 source,
tests, handoffs, independent tests, reviews, and BLOCK evidence were not
edited. V5 changes only the descriptor-bound read of the one permitted frozen
source-index metadata file. It grants no dataset construction, training,
selection, calibration, G2, held-out, runtime, hardware, production, or
promotion authority.

No referenced frame list, scene manifest, render plan, render summary,
excluded sidecar, model, checkpoint, G2, held-out, runtime, hardware, or
production payload was opened during implementation or verification.

## Frozen V4 evidence

| Artifact | SHA-256 |
| --- | --- |
| `lewm/datasets/go2_shared_jepa_v5_raw_supervision_plan_v4.py` | `d6282a6ee561d34fbe20542f31acd8c7bee82badfa74d1d640930148a9951de2` |
| `lewm/tests/test_go2_shared_jepa_v5_raw_supervision_plan_v4.py` | `724f1c93023256015fe0d468c56591fab35512de79c1e0b0822e78bccdb4a0e0` |
| `docs/lewm_go2_shared_jepa_v5_raw_supervision_metadata_plan_v4_author_handoff_2026-07-13.md` | `4753d83517a41d2e70e8f25d7cb03ad3709f2d798d1f9f39eea358a527c91415` |
| `lewm/tests/test_go2_shared_jepa_v5_raw_supervision_plan_v4_independent_qa.py` | `5e079be910f5633c01df6d9afc2967715515b27293cc09f279eb71f373c40f78` |
| `docs/lewm_go2_shared_jepa_v5_raw_supervision_metadata_plan_v4_independent_review_2026-07-13.md` | `46d44155916b53dd850274b1c8704d0feb62ed7c1bd05f28391b1ea83ded9757` |
| `docs/lewm_go2_shared_jepa_v5_raw_supervision_metadata_plan_v4_independent_review_block_2026-07-13.json` | `6897064fb3752b0d9552c0db9b8bd81372a3ba891ff3e98ce7174a84e9e6c2d8` |

The frozen V4 independent suite continues to reproduce `9 passed, 2 failed`.
Those failures remain historical BLOCK evidence and are not edited or
reclassified by V5.

## Additive V5 candidate

| Artifact | SHA-256 |
| --- | --- |
| `lewm/datasets/go2_shared_jepa_v5_raw_supervision_plan_v5.py` | `67c4d325ddab3ac3405e231b78681f4b9ef17b4833ca199395f24ed7a8b82921` |
| `lewm/tests/test_go2_shared_jepa_v5_raw_supervision_plan_v5.py` | `384af6e2b254ea98d32fd7f4798beafe429a4cd83fee6e2903d0d1e8c84f9636` |

## Complete fingerprint continuity

V5 preserves V4's filesystem-root-anchored, descriptor-relative,
`O_DIRECTORY | O_NOFOLLOW` walk and its `O_NOFOLLOW | O_NONBLOCK` leaf open.
The original complete fingerprint is exactly:

```text
(st_dev, st_ino, st_mode, st_nlink, st_size, st_mtime_ns, st_ctime_ns)
```

V5 retains that seven-field value for:

- the filesystem-root named entry and its open descriptor;
- every component between the filesystem root and repository root;
- the repository root's original canonical-path and open-descriptor binding;
- every source-index parent component beneath the repository root; and
- the source-index leaf's named entry and open descriptor.

Each directory row is retained as
`(parent_fd, component, child_fd, original_fingerprint)`. The filesystem root,
repository root, and leaf retain their original fingerprints explicitly. One
complete-chain validator compares both the descriptor-relative named entry and
open descriptor to that exact original value. It is called immediately before
the first `os.read` and immediately after the final `os.read`, before accepting
or joining the bytes.

Directory timestamp or mode changes that preserve device, inode, and file kind
therefore reject. Symlink, directory, FIFO, hard-link, different-inode, and
same-inode relinked leaf substitutions also reject before a first byte is read
when present at the pre-read boundary. All descriptors close on success and
failure, and the frozen source-index SHA-256 remains the final byte check.

## Closed V4 findings

The exact two V4 independent mismatch cases are passing V5 regressions:

1. changing the immediate source-index parent directory's full fingerprint
   during the first read; and
2. changing an ancestor between the filesystem root and repository root during
   the first read.

Both preserve device, inode, directory type, open descriptors, and leaf bytes;
V5 rejects both after the read. The focused matrix also covers mode as well as
timestamp changes and the repository root itself. It passes `6/6`.

A separate pre-read regression changes the source-index parent immediately
before the first complete-chain validation and proves zero `os.read` calls.
Further tests bind the filesystem-root fingerprint, validate full
seven-field rows at both validation calls, replay the V3 transient ancestor
alias and same-inode leaf relink, and exercise every nonregular/linked leaf
kind.

## Preserved science and access boundary

V5 reproduces exactly:

- 5,172 selected pairs;
- 10,344 endpoint uses;
- 9,460 unique endpoints;
- 88 source records;
- plan content SHA-256
  `8004ab0d3aa6a2f5d576ba0ff4d6a75f50899152e542dc62b8d6e35f614921a3`;
- ordered-pair SHA-256
  `76810dba883f3aaffb92fccb593d382daf7edca74a9bb5559a977e7e88b7b5ea`;
- ordered-endpoint SHA-256
  `8130e961b7b5c04944b178fa4f73c1fa157776f7702ab5cdc213cf16c922f698`;
- inventory hashes `scene_role=f967364a...ed5b`,
  `frames=7512a041...623d`, `manifests=2bc5f468...44c5`,
  `plans=03590784...36e4`, and `summaries=bd2b1819...548a`; and
- every false license flag.

The real-tree tracer records exactly ten regular-file opens over seven
allowlisted metadata files: dataset manifest once, dataset rows once, sidecar
manifest once, train role twice, checkpoint-selection role twice,
probability-calibration role twice, and source index once. It records exactly
704 metadata validations of 352 selected source references, each validated
twice. All referenced-payload and excluded/protected open counters remain zero.

## Verification

Every command fixed OMP, OpenBLAS, MKL, and NumExpr threads to one; hid CUDA,
HIP, ROCr, and HSA devices; and disabled external pytest plugins.

```text
V5 focused author suite:                 19 passed in 1.60s
V5 exact V4 mismatch matrix:              6 passed in 0.04s
V1-V4 predecessor author suites:         76 passed in 9.24s
V1-V5 combined author suites:            95 passed in 10.68s
Frozen V4 independent BLOCK replay:       9 passed, 2 failed in 1.70s
Frozen V3 independent BLOCK replay:      14 passed, 2 failed in 2.51s
py_compile (V5 source and tests):        PASS
git diff --check (V5 source and tests):  PASS
```

## Required different-agent review

A reviewer other than `/root/v4_full_panel_v3_author` must independently:

1. rehash the V5 source, test, and this handoff plus all frozen V4 evidence;
2. inspect every chain row and prove that the original seven-field fingerprint,
   not only identity or type, is compared for named entries and descriptors at
   both pre-read and post-read boundaries;
3. replay the two exact V4 failures and independently probe filesystem-root,
   repository-root, ancestor, source-parent, and leaf continuity;
4. rerun the 19 focused and 76 predecessor author tests plus both historical
   BLOCK reproducers with threads capped and accelerators hidden;
5. reproduce all scientific identities, the ten-open allowlist, 704 metadata
   validations, and zero excluded/protected/referenced-payload opens; and
6. issue a separate PASS or BLOCK record without opening source payloads or
   granting downstream authority.

Until that review passes, V5 is author evidence only. It does not authorize raw
dataset construction, training, checkpoint selection, calibration, G2,
held-out evaluation, runtime, hardware, navigation, production, or promotion.
