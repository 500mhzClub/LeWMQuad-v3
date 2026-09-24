# Shared JEPA V5 raw-supervision Auditor V6 author handoff

Date: 2026-07-13

Implementation author: `/root/raw_auditor_author`

Status: **COMPILE-SAFE IMPLEMENTATION INPUT ONLY; BUILDER V6 BLOCKED; NO AUTHORITY**

## Frozen implementation candidate

| Role | Path | SHA-256 |
| --- | --- | --- |
| Auditor source | `lewm/datasets/go2_shared_jepa_v5_raw_supervision_auditor_v6.py` | `cf67c993427950c147860f9afe0e7661b2cb6841ccec27a867868cc34c7c00b8` |
| Auditor CLI | `scripts/audit_go2_shared_jepa_v5_raw_supervision_v6.py` | `de37e42d09d949ac5ca1cd8e4ebba2d32e757ef72cc769a151f814cc8fe84ffe` |
| Auditor author tests | `lewm/tests/test_go2_shared_jepa_v5_raw_supervision_auditor_v6.py` | `6cc84a493cb677437385efd3c00a8120b26748e8cabb2abd76d0f4825deaf764` |

These bytes were authored under the frozen V6 amendment:

`docs/lewm_go2_shared_jepa_v5_raw_supervision_builder_auditor_v6_authorization_successor_amendment_2026-07-13.md`

SHA-256: `09ced36b2eab16585c759e65f7eda844f76006b93de013e5f7057fb9a8e7a137`.

## Terminal Builder V6 BLOCK

Builder V6 was independently blocked while this auditor was being authored.
Its final quiet check did not retain or watch the canonical ancestors above
the immediate publication parent. An adversary could move that ancestor after
post-rename validation, recreate an empty canonical path, and let the builder
return success while the canonical dataset was absent.

| Evidence | SHA-256 |
| --- | --- |
| Independent QA | `2c74e3315be3443bab11a3b7896df4df29d8b233b634b7ab539123386bc0c89a` |
| Machine BLOCK file | `55d50a38f0c7d23e4ff537b124db3b9f24a24ea5b30413ff6be1ac381870c163` |
| Machine BLOCK content | `c639170b672180c8943e08efaff8d23063e8773488d1ff0f77beeb4ce44dd74b` |
| Review report | `3b00de780b8df6a98aec4db0b28f4cb651a9fa7b64be2a3333ae7093821cf7e6` |

The fixed V6 builder-review path now contains `BLOCK`, not `PASS`. Therefore
no valid dual-PASS V6 authorization can exist and this auditor cannot receive
an Auditor V6 PASS review for exact use. It is retained only as compile-safe
implementation evidence for an additive successor amendment.

## Implemented boundary

The V5 scientific audit engine remains owned directly by the standalone V6
module. V6 changes the authority and terminal publication boundary:

- exact authority is the fixed nine-row V6 map and requires the frozen Builder
  V6 candidate hashes;
- Builder V6's complete frozen-parent provenance map is required exactly;
- the only public exact entry is keyword-only
  `execute_exact_audit_v6(authorization_sha256, workers)`;
- native worker threads are one, accelerators are hidden, process start is
  `spawn`, and exact workers are limited to integers 1 through 6;
- a complete report candidate is written and fsynced before the final source
  pass;
- one nonblocking close-on-exec inotify transaction retains and hashes all
  authority, reviewed-source, metadata, raw-source, contract, complete dataset,
  and report-candidate leaves;
- all source/dataset directories and every canonical source/publication
  ancestor are retained and watched continuously through the final complete
  source pass, `renameat2(RENAME_NOREPLACE)`, post-rename validation, parent
  fsync, and final quiescence;
- only the characterized file event sequence is accepted: publication-parent
  `IN_MOVED_FROM`, matching-cookie `IN_MOVED_TO`, then candidate
  `IN_MOVE_SELF`;
- any mutation permanently poisons the attempt, including a byte-restored
  mutation, destination race, watch loss, queue anomaly, namespace change, or
  ancestor movement; and
- failure cleanup unlinks only a name whose device/inode identity still equals
  the retained attempt-owned candidate. Failure receipts remain fixed,
  non-authoritative, and cannot replace the primary exception.

The ancestor coverage is deliberately stronger than the blocked Builder V6
implementation and is exercised by the author test. It has not received a
different-agent review and makes no authority claim.

## Verification

All commands used one native CPU thread per library and empty accelerator
visibility.

| Check | Result |
| --- | --- |
| V6 source/CLI/test `py_compile` | PASS |
| V6 focused author suite | `23 passed` |
| Retained Auditor V1/V2 suites | `37 passed` |
| Combined applicable suite | `60 passed` |
| `git diff --check` for V6 source/CLI/test | PASS |

The invalidated Auditor V3 suite was also sampled: 42 tests pass and 23 fail
inside the frozen V3 module because its invalidation removed the legacy `_v1`
and `plan_v5` module bindings. Those known V3 structural failures do not
exercise or import the standalone V6 candidate and are not counted as a V6
gate.

The V6 tests use only synthetic temporary roots. They prove the fixed API and
role map, exact worker boundary, zero payload opens for absent authority, clean
owned publication, byte modify-then-restore poisoning, preservation of a
foreign destination, rejection of post-rename ancestor movement, and source
AST ordering with the transaction live before the final pass and rename.

## Prohibited actions

No exact build or audit was run. No canonical dataset, audit report, failure
receipt, authorization, development source payload, RGB payload, G2, held-out,
checkpoint, runtime, navigation, hardware, production, or promotion namespace
was opened or created. This handoff does not authorize review PASS, exact work,
dataset use, training, selection, calibration, G2, held-out evaluation,
runtime, navigation, hardware, production, or promotion.

The next legal action is an additive Builder/Auditor V7 amendment that binds
the Builder V6 BLOCK and may use these Auditor V6 bytes only as a non-candidate
implementation input.
