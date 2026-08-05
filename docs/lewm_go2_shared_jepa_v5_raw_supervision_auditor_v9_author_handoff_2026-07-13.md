# Shared JEPA V5 raw-supervision Auditor V9 author handoff

Date: 2026-07-13

Implementation author: `/root/camera_v5_independent/camera_v7_pre_freeze_review/v7_review_artifact_schema`

Status: **FROZEN AUTHOR CANDIDATE; NO REVIEW OR EXACT AUTHORITY**

## Frozen contract

The implementation follows the source-free successor amendment:

| Artifact | SHA-256 |
| --- | --- |
| `docs/lewm_go2_shared_jepa_v5_raw_supervision_builder_auditor_v9_linearization_successor_amendment_2026-07-13.md` | `6fba5de8d7f04d85bd87e084096ae269c3d3dd6368a6db0b0f8f149c1c5cf773` |

The amendment creates an additive V9 namespace and fixes the Auditor V9
implementation author to the identity above. It grants no exact or downstream
authority.

## Frozen candidate

The Auditor V9 review candidate is exactly these three files in this order:

| Role | Artifact | SHA-256 |
| --- | --- | --- |
| `auditor_source` | `lewm/datasets/go2_shared_jepa_v5_raw_supervision_auditor_v9.py` | `ebe0c6a31cf027b8b0bc049257079a5e0ab0493b12aabeb96bf50f02990bbc14` |
| `auditor_cli` | `scripts/audit_go2_shared_jepa_v5_raw_supervision_v9.py` | `76f0b2b29eff8df6905fed142cc622eb0fa8024c397a3c7efb54e58cc36f67ba` |
| `auditor_test` | `lewm/tests/test_go2_shared_jepa_v5_raw_supervision_auditor_v9.py` | `10951cc2e622281f72ec2a20114ccca184af7624a95fef4683c83dc6839992d1` |

This handoff is explanatory only. It is not a tenth V9 authority row and is
not part of the Auditor V9 review candidate.

## V9 bindings

- The nine ordered authority roles and literal paths exactly match the frozen
  V9 amendment.
- The fixed Builder V9 role hashes are
  `2388c1138d9b03ea6e385cc0250c81a1869a40cab62507d02f709ef39197c664`,
  `f239a4ef7c067a71f991b30e14bd5c8632c31be3173780fc25b3d9801fff79ee`,
  `541d1957df0a3da18c2b529cd2d7ca721d7e657c8ebcced2a37931d502cab7bc`,
  and `b6cdf34fa933214e1bb603681f4638f2226e093dad42705445fd8084d6442efd`
  for source, CLI, test, and handoff.
- Builder V9's later independent PASS review file has SHA-256
  `c39eb2787c37f8cab064de75355b3af56971ef98209d329e4789eb383c1dc60f`
  and content SHA-256
  `49d8024ae48211cc4fc7d7c2fb674c7ddc7adb38abccace1eb8c6bbc4f10b0df`.
  It remains the separate `builder_review` authority row and is not added to
  the frozen predecessor map.
- `FROZEN_V9_PREDECESSOR_SHA256` exactly equals Builder V9
  `FROZEN_PARENT_HASHES`: 83 rows with canonical JSON SHA-256
  `76823317704cb35ad3342cb27c03c218816da89b56c294897a7eddd651cdd83e`.
- The predecessor map binds the V8 QA, V8 canonical BLOCK, governing threat
  model, and V9 amendment at their frozen hashes.
- Successful and failure report leaves are additive `.audit_v9.json` and
  `.audit_v9.failed.json` namespaces.
- The only exact API is keyword-only `execute_exact_audit_v9(*,
  authorization_sha256, workers)` with exact non-boolean workers in `[1, 6]`.

## Publication ordering

Auditor V9 is standalone and preserves Auditor V8 science, source closure,
worker policy, no-replace report publication, cleanup, and unaffected
transaction behavior. Its terminal sequence is:

1. reject pending protected events;
2. revalidate complete sources, directories, report inventory, report hash,
   destination identity, and retained ancestry;
3. reject events queued during that full pass;
4. repeat retained ancestry and complete report inventory/hash/destination
   validation; and
5. perform the final successful event read as the publication linearization
   point.

The final event read is the last filesystem observation in
`require_final_quiet`. The success path after it only copies the prepared
in-memory result, closes retained descriptors, and returns.

The absolute report lookup walks from the retained filesystem-root descriptor
through every fixed intermediate component with `O_NOFOLLOW`, matches every
opened directory to the retained chain, and performs a no-follow leaf lookup.
It therefore does not treat an intermediate symbolic link as canonical path
identity.

## Author verification

All tests used pytest temporary roots, disabled third-party pytest plugin
autoload, one native CPU thread, and empty accelerator visibility.

| Check | Result |
| --- | --- |
| Auditor V9 focused author suite | `61 passed` |
| Frozen Auditor V8 author suite | `56 passed` |
| Frozen Auditor V8 QA reproducer | `1 passed` |
| Frozen Auditor V7 author/QA/preaudit suites | `31 passed` |
| Applicable retained Auditor V1/V2 suites | `63 passed`, `6` predecessor-only cases deselected |
| V9 predecessor map vs Builder V9 frozen map | `83/83`, exact equality |
| One-worker/six-worker synthetic science bytes | exact equality |
| Terminal event-read and post-linearization AST checks | PASS |
| Intermediate-component no-follow checks | PASS |
| `py_compile` for source, CLI, and test | PASS |
| Source import, exact-entry, and fixed-path boundaries | PASS |
| Candidate whitespace check | PASS |

No canonical authorization, `.generated`, dataset, source payload, RGB,
checkpoint, exact output, G2, held-out, runtime, navigation, hardware,
production, or accelerator namespace was opened or changed during authoring.
No exact work was run.

These results grant no independent review, authorization, exact audit,
dataset-use, training, selection, calibration, G2, held-out, navigation,
runtime, hardware, production, promotion, retry, or deployment authority. A
different agent must independently review the exact frozen three-file
candidate and publish the fixed canonical V9 review JSON before any later
dual-review authorization is possible.
