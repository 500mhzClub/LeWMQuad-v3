# Shared JEPA V5 Raw Supervision Auditor V8 independent review

Date: 2026-07-13

Reviewer: `/root/raw_v8_auditor_reviewer`

Implementation author:
`/root/camera_v5_independent/camera_v7_pre_freeze_review/v7_review_artifact_schema`

Verdict: **BLOCK**

## Frozen candidate

| Role | Path | SHA-256 |
| --- | --- | --- |
| `auditor_source` | `lewm/datasets/go2_shared_jepa_v5_raw_supervision_auditor_v8.py` | `fb585b4ee9c860eb6a2c2814ff84000a07f8cb070496e530bfb75905e67e1d87` |
| `auditor_cli` | `scripts/audit_go2_shared_jepa_v5_raw_supervision_v8.py` | `13c1ebedc6864db21951e0545133664a70a24f1aa02b6082764f0426737f6fc2` |
| `auditor_test` | `lewm/tests/test_go2_shared_jepa_v5_raw_supervision_auditor_v8.py` | `4270c1a1350b8a7a0ef32daec5366cd719965e10776309ea299cd0e8172c8006` |

All three hashes matched before and after review. The explanatory author
handoff hash was
`ed3fdf3d2c9314e64b230997174936f9fedc282a224b0920ff05140d45f418d2`;
it is not a review candidate or authorization row.

## Governing closure

| Artifact | SHA-256 |
| --- | --- |
| Terminal-quiet successor amendment | `054de82d8648cd6be7edff01b82d549ec916700ebffad51698d4c2041edc6c88` |
| Existing-agent rebinding amendment | `392745c80ca2c6e7a103cca4a55c3614cd2c988de9a379fba950b0087df41698` |
| Reviewer-owned QA | `5fe390c3c3ca94bc6e3bce7d153aa86a475bf35e521e1259294b85e588bd229b` |

The Auditor V8 predecessor map has 69 rows, exactly equals Builder V8
`FROZEN_PARENT_HASHES`, and has canonical content SHA-256
`79fe832122ed335188357a59bad8a031cc235449ef17e6e19ac78de9d5aff669`.
All 69 files rehashed without mismatch. All nine reviewed V4 source rows and
all four fixed Builder V8 candidate rows also rehashed without mismatch.

The fixed authors and this reviewer are distinct as required. The nine source
roles, paths, order, schemas, review candidates, and two-reviewer separation
checks match the amendments.

## Blocking finding

### Final report identity accepts a recreated ancestor alias

`require_final_quiet` performs the second event drain, validates retained
ancestry, then calls the final report inventory/hash/destination helper and
returns without another event read. The report helper validates the report by
both the retained publication-parent descriptor and an absolute `Path.stat`.
That absolute lookup follows aliases in intermediate path components.

The reviewer-owned synthetic test
`test_block_final_report_identity_allows_ancestor_move_and_symlink_recreation`
injects one namespace change at entry to the second and final report helper,
after the final retained-ancestry validation:

1. Rename the synthetic repository ancestor while its descriptors remain open.
2. Recreate its canonical name as a directory symlink to the moved tree.
3. Run the unmodified final report inventory/hash/destination helper.

Both descriptor-relative and absolute report lookups then identify the same
unchanged, singly linked report inode. The helper passes, the transaction is
not poisoned, and `require_final_quiet` returns success. The ancestor events
remain queued because no event read follows this final helper.

This is a filesystem consistency/TOCTOU defect at
`go2_shared_jepa_v5_raw_supervision_auditor_v8.py:1933-1934`, together with the
intermediate-alias-following absolute lookups at lines 1759 and 1770. It
violates the amendment requirements that moving or recreating any retained
ancestor during either identity pass reject and that canonical recreation
never repair transaction poison. Auditor source approval therefore cannot be
granted.

The amendment's literal absolute guarantee is also not attainable with only a
finite, noncooperative userspace sequence of stat, hash, and inotify
observations against a concurrent process with equivalent namespace mutation
rights: every finite sequence has a last observation before return. The
confirmed candidate defect occurs earlier, between two final observations, and
is directly reproducible. A successor contract must either define a precise
linearization point with checks that close this ordering gap or require a
cooperative or kernel-enforced writer-exclusion boundary for an absolute
postcondition.

## Passing evidence

All dynamic tests used synthetic pytest temporary roots, one native math
thread, empty accelerator visibility, and disabled pytest plugin autoload.

| Check | Result |
| --- | --- |
| Frozen Auditor V8 author suite | `56 passed` |
| Retained Auditor V7 author, reviewer QA, and root preaudit | `31 passed` |
| Applicable retained Auditor V1/V2 suites | `63 passed`, `6 deselected` predecessor demonstrations |
| Auditor V8 reviewer defect reproducer | `1 passed` |
| Candidate and reviewer QA compile check | PASS |
| Candidate hashes after testing | PASS |

Source and test inspection otherwise confirmed the required two-drain shape,
V7 science AST parity, one/six-worker deterministic science bytes, strict
worker range and spawn policy, fixed phase-one/phase-two authority closure,
duplicate-key-free canonical review parsing, exact rename event checks,
modify/restore rejection, unrelated-event filtering, protected-name rejection,
foreign-destination preservation, inode-owned cleanup, and the absence of a
production callback, test hook, dynamic import, mutable authority registry,
legacy auditor import, Builder V8 import, or legacy exact entry.

## Authority boundary

This review grants no Auditor V8 source approval and no exact build, exact
audit, dataset-use, training, selection, calibration, G2, held-out, runtime,
navigation, hardware, production, promotion, retry, or deployment authority.
No canonical authorization, `.generated` tree, source payload, dataset, RGB,
checkpoint, exact output, G2, held-out, runtime, hardware, or production path
was opened. No exact build, exact audit, or accelerator execution was run.
