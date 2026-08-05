# Independent review: Shared JEPA V5 raw-supervision Auditor V9

Date: 2026-07-13

Reviewer: `/root/raw_v8_builder_reviewer`

Implementation author: `/root/camera_v5_independent/camera_v7_pre_freeze_review/v7_review_artifact_schema`

Governing amendment SHA-256: `6fba5de8d7f04d85bd87e084096ae269c3d3dd6368a6db0b0f8f149c1c5cf773`

Verdict: **PASS**

## Review boundary

This is a different-agent source review of the frozen Auditor V9 source, CLI, and author test. The reviewer differs from `/root`, both fixed V9 implementation authors, and the Builder V9 reviewer. The three candidate files were not modified.

The review used source inspection and CPU-only synthetic tests under temporary roots. It did not open or modify the canonical authorization, `.generated`, dataset or source payloads, RGB, checkpoints, exact outputs, G2, held-out data, runtime or navigation results, hardware evidence, production paths, or accelerator state. The author handoff was used as explanatory evidence only and is not an authority row.

## Frozen candidate

| Role | Path | SHA-256 |
|---|---|---|
| `auditor_source` | `lewm/datasets/go2_shared_jepa_v5_raw_supervision_auditor_v9.py` | `ebe0c6a31cf027b8b0bc049257079a5e0ab0493b12aabeb96bf50f02990bbc14` |
| `auditor_cli` | `scripts/audit_go2_shared_jepa_v5_raw_supervision_v9.py` | `76f0b2b29eff8df6905fed142cc622eb0fa8024c397a3c7efb54e58cc36f67ba` |
| `auditor_test` | `lewm/tests/test_go2_shared_jepa_v5_raw_supervision_auditor_v9.py` | `10951cc2e622281f72ec2a20114ccca184af7624a95fef4683c83dc6839992d1` |

The explanatory handoff `docs/lewm_go2_shared_jepa_v5_raw_supervision_auditor_v9_author_handoff_2026-07-13.md` rehashed to `819d1857bf315f775f45c4a16db994f333d7174c5c20f5cb762f93d04b30a3a5`.

## Scientific and authority findings

Auditor V9 retains the complete Auditor V8 scientific implementation. After mechanical `V9`/`v9` normalization, all 106 top-level definitions are identical except `_ClosedAuditPublicationTransaction`. Within that class, V9 adds `_stat_absolute_report_leaf` and changes only `_validate_report_inventory_and_destination` and `require_final_quiet`. The complete validation, join, population, sampling, shard, array, replay, provenance, and resource-control region is AST-identical to V8.

The retained constants remain 5,172 pairs, 10,344 endpoint references, 9,460 unique endpoints, 88 scene shards, three scene-disjoint roles, 24 audit samples, and eight arrays including the 64 x 64 raster labels. Synthetic one-worker and six-worker result bytes are identical.

`FROZEN_V9_PREDECESSOR_SHA256` has exactly 83 rows, equals the frozen Builder V9 parent map, and has canonical digest `76823317704cb35ad3342cb27c03c218816da89b56c294897a7eddd651cdd83e`. It binds the governing threat model, V9 amendment, V8 QA and canonical BLOCK, and the retained predecessor closure.

The authorization parser accepts exactly nine ordered role/path/hash objects. It requires both fixed implementation authors, two distinct non-root reviewers different from both authors, exact candidate bindings, canonical review records, and source-only review authority. The only production entry is keyword-only `execute_exact_audit_v9(*, authorization_sha256, workers)`. Worker values are exact non-boolean integers from one through six. The CLI imports only that V9 entry and exposes no alternate path or execution mode.

## Publication findings

The terminal transaction follows the amended five-step ordering. It drains pending protected events, performs the complete retained source/directory/report validation, drains events queued during that pass, repeats canonical ancestry and complete report inventory/hash/destination validation, and performs a third event drain as the literal publication linearization point.

The third drain is the final statement in `require_final_quiet`. The successful caller tail after it only copies the already prepared in-memory result, closes retained descriptors, and returns. No stat, read, hash, fsync, rename, unlink, write, chmod, or other namespace/content operation follows on the successful path.

Absolute report identity is resolved from the retained filesystem-root descriptor through every fixed intermediate component using `O_DIRECTORY | O_NOFOLLOW`. Each opened component is compared with the retained descriptor and fingerprint, and the report leaf is looked up without following links. Both symbolic-link replacement and real-directory identity substitution are rejected.

The frozen V8 ancestor-move/symlink reproducer still returns success under V8. The same timing is rejected by V9 because the third drain consumes the queued protected events. Reviewer QA also injects a report mutation during the third read itself; V9 poisons the transaction at `final publication linearization`.

Protected source, ancestry, report, destination, and candidate changes remain permanently poisoning even when bytes or names are restored. Unrelated-name churn remains filtered. The owned `RENAME_NOREPLACE` event sequence is exact, foreign destinations and replacements are preserved, and cleanup removes only the inode owned by the attempt.

## Verification

| Suite | Result |
|---|---|
| Frozen Auditor V9 focused author suite | `61 passed` |
| Frozen Auditor V8 author and QA suites | `57 passed` |
| Frozen Auditor V7 author, QA, and preaudit suites | `31 passed` |
| Applicable retained Auditor V1/V2 suites | `63 passed, 6 deselected` |
| Reviewer-owned Auditor V9 QA | `7 passed` |
| Combined V9 author plus reviewer QA rerun | `68 passed` |
| Source, CLI, author test, and reviewer QA compilation | PASS |
| Candidate and reviewer whitespace scan | PASS |

The 219 distinct applicable focused, retained, and reviewer cases passed. Native math threads were capped at one, accelerator visibility was empty, and third-party pytest plugin autoload was disabled.

Reviewer QA: `lewm/tests/test_go2_shared_jepa_v5_raw_supervision_auditor_v9_independent_qa.py`, SHA-256 `0434daa215ad6f487223042898564fc5af14a569dc754a1ed09083da729574f2`.

## Authority conclusion

The frozen Auditor V9 source candidate satisfies the amendment and is approved as source. This review record sets only `auditor_source_approved=true`. It does not itself authorize exact build or audit, dataset use, training, selection, calibration, G2, held-out evaluation, runtime, navigation, hardware, production, promotion, retry, or deployment. Any later exact authority requires the separate canonical nine-role authorization and its independently frozen SHA-256.
