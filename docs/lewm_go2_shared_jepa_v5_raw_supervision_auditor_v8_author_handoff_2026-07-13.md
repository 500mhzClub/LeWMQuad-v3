# Shared JEPA V5 raw-supervision Auditor V8 author handoff

Date: 2026-07-13

Implementation author: `/root/camera_v5_independent/camera_v7_pre_freeze_review/v7_review_artifact_schema`

Status: **FROZEN AUTHOR CANDIDATE; NO REVIEW OR EXACT AUTHORITY**

## Frozen contract

The implementation follows these frozen amendments in order:

| Artifact | SHA-256 |
| --- | --- |
| `docs/lewm_go2_shared_jepa_v5_raw_supervision_builder_auditor_v8_terminal_quiet_successor_amendment_2026-07-13.md` | `054de82d8648cd6be7edff01b82d549ec916700ebffad51698d4c2041edc6c88` |
| `docs/lewm_go2_shared_jepa_v5_raw_supervision_v8_existing_agent_identity_rebinding_amendment_2026-07-13.md` | `392745c80ca2c6e7a103cca4a55c3614cd2c988de9a379fba950b0087df41698` |

The rebinding amendment fixes the Auditor V8 implementation author to the
identity above. It changes no science, authority row, API, transaction, test,
resource, or review-separation requirement from the terminal-quiet amendment.

## Frozen candidate

The Auditor V8 review candidate is exactly these three files in this order:

| Role | Artifact | SHA-256 |
| --- | --- | --- |
| `auditor_source` | `lewm/datasets/go2_shared_jepa_v5_raw_supervision_auditor_v8.py` | `fb585b4ee9c860eb6a2c2814ff84000a07f8cb070496e530bfb75905e67e1d87` |
| `auditor_cli` | `scripts/audit_go2_shared_jepa_v5_raw_supervision_v8.py` | `13c1ebedc6864db21951e0545133664a70a24f1aa02b6082764f0426737f6fc2` |
| `auditor_test` | `lewm/tests/test_go2_shared_jepa_v5_raw_supervision_auditor_v8.py` | `4270c1a1350b8a7a0ef32daec5366cd719965e10776309ea299cd0e8172c8006` |

This handoff is explanatory only. It is not a tenth V8 authority role and is
not part of the Auditor V8 review candidate.

## V8 bindings

- The nine ordered authority roles and literal paths exactly match the frozen
  terminal-quiet amendment.
- The fixed Builder V8 role hashes are
  `f45533354c8b45b88f8eadb2126ec5eaf96fe1f57c21a9bfcd95a8855bfaaa35`,
  `f6471f1fa0ca7a13976f752a41ee9ddacbc76636e4d5fb0eee1ebf75bdaee72d`,
  `fc1f0cf3fc18bdbd1393be6a514bc04459f943f39b438ced78ebee30e7c57d9a`,
  and `9f4898e3620ac87c9a0145be103c4fdf397f727fe37d9f6ca306a0f50916156b`
  for source, CLI, test, and handoff.
- The implementation authors are fixed as
  `/root/raw_v7_successor_author/auditor_v7_author` and
  `/root/camera_v5_independent/camera_v7_pre_freeze_review/v7_review_artifact_schema`.
- `FROZEN_V8_PREDECESSOR_SHA256` contains exactly the same 69 rows as Builder
  V8 `FROZEN_PARENT_HASHES`; its canonical JSON SHA-256 is
  `79fe832122ed335188357a59bad8a031cc235449ef17e6e19ac78de9d5aff669`.
- Every review authority field outside its one source-approval field is false,
  including `retry_authorized`.
- Successful and terminal failure report leaves are additive
  `.audit_v8.json` and `.audit_v8.failed.json` namespaces.
- The only exact API is keyword-only `execute_exact_audit_v8(*,
  authorization_sha256, workers)` with exact non-boolean workers in `[1, 6]`.

## Terminal repair

Auditor V8 is standalone and retains Auditor V7 science and all unaffected
transaction methods. Its terminal success sequence now performs the required
first event drain, complete source/directory/report and retained-ancestry
validation, second event drain, and final retained-ancestry plus canonical
report inventory/hash/identity validation. The report remains bound through
one retained descriptor across the owned no-replace rename, with rename-time
ctime rebased only after its inode, name, fingerprint, link count, and content
are proved.

The V8 tests cover mutations before, during, and after both drains; both final
ancestry passes; source, candidate, and report modify-then-restore cases; both
report hash and identity passes; move-and-recreation cases; protected versus
unrelated named churn; foreign-destination preservation; the frozen V7 BLOCK
reproducer; exact rename ordering; and deterministic one/six-worker science
bytes. No callback, hook, sleep-based quiet period, retry loop, dynamic import,
legacy exact entry, or Builder V8/Auditor V7 production import was added.

## Author verification

All checks used one native CPU thread and empty accelerator visibility.

| Check | Result |
| --- | --- |
| Auditor V8 focused author suite | `56 passed` |
| Frozen Auditor V7 author, independent-QA, and root-preaudit suites | `31 passed` |
| Applicable retained Auditor V1/V2 suites | `63 passed`, `6` predecessor exploit demonstrations deselected |
| `py_compile` for source, CLI, and test | PASS |
| V8 predecessor map vs Builder V8 frozen map | `69/69`, exact equality |
| Source import and exact-entry boundary | PASS |
| Candidate whitespace check | PASS |

The attempted internal read-only helper review produced no usable result and
no artifact; it is not a formal V8 review and this handoff does not depend on
it.

No canonical authority, canonical dataset, audit output, source payload, RGB,
checkpoint, G2, held-out, runtime, navigation, hardware, production, or
accelerator namespace was opened or changed during authoring. Behavioral tests
used pytest temporary roots only.

These results grant no independent review, authorization, exact audit,
dataset-use, training, selection, calibration, G2, held-out, navigation,
runtime, hardware, production, promotion, retry, or deployment authority. A
different agent must independently review the exact frozen three-file
candidate and publish the fixed canonical V8 review JSON before any later
dual-review authorization is possible.
