# Shared JEPA V5 Raw Supervision Auditor V7 independent review

Date: 2026-07-13

Reviewer: `/root/coordinator_v2_qa`

Implementation author: `/root/raw_v7_successor_author/auditor_v7_author`

Verdict: **BLOCK**

## Frozen candidate

| Role | Path | SHA-256 |
| --- | --- | --- |
| `auditor_source` | `lewm/datasets/go2_shared_jepa_v5_raw_supervision_auditor_v7.py` | `3550917e36d1401f8ad9c895afcf591b3226b2e0c5a09f4ad427d0b04bb1490e` |
| `auditor_cli` | `scripts/audit_go2_shared_jepa_v5_raw_supervision_v7.py` | `9940d35e4e33b628bf64c4947cb1f92a68e1413e20e63fd0b9080728a64f949e` |
| `auditor_test` | `lewm/tests/test_go2_shared_jepa_v5_raw_supervision_auditor_v7.py` | `6d123d39014fd9c3dc7b34d113e665861536010d79117a3004cb8ee1484e894f` |

The candidate bytes matched the author handoff and were not modified during
this review.

## Governing evidence

| Artifact | SHA-256 |
| --- | --- |
| V7 amendment | `ebeb552a89792b63f10c7d9ab5c9c9abd96d74d6ae7cf39f709f0657708798fc` |
| Auditor V7 author handoff | `1351a2641025735a3a96d50283a7119f2ce02f7c49578e656133b1c48a46fd21` |
| Builder V7 independent PASS review file | `85d1a111e10eaac865a80cebd97e771b39eaa47f6ebcf6ffe6716ed445a1ff46` |
| Root cross-binding preaudit | `fa2e76a1a9b2a7befac8096ea3c989bc2c87266f5d2454766d4e0f9f15b67e09` |
| Independent reviewer QA | `5d0ffc94070bdb60dfb9a90a062e5598931fa793855386e758192e6d4c3078c0` |

## Blocking finding

### Terminal event drain has no final identity revalidation

The V7 amendment requires the terminal publication boundary to drain events,
revalidate the complete ancestry/source/published inventory, drain again, and
then repeat the retained ancestry and canonical destination identity checks
immediately before success. It also requires rejection of mutations injected
during the terminal drains and final identity check.

Frozen Auditor V7 does not implement that sequence. Its
`_ClosedAuditPublicationTransaction.require_final_quiet` method validates the
bound inventory and retained chain once, performs one nonblocking event drain,
then returns. The drain is the method's final operation. There is no second
drain and no ancestry or canonical report identity check after it.

The independent synthetic test
`test_block_ancestor_move_after_terminal_drain_returns_success` makes this gap
deterministic. It completes the owned no-replace rename and post-rename checks,
lets the auditor's terminal drain report quiet, then moves the retained
repository ancestor and recreates the canonical parent before that terminal
call returns. Frozen Auditor V7 returns success even though its canonical audit
report path is absent. The queued self/parent events are never consumed and the
canonical namespace is never revalidated.

This permits an exact audit to claim terminal success for a report that is
reachable only through the moved alias and absent from the fixed canonical
path. It violates the V7 ancestor-closed publication and terminal-success
contracts, so the candidate cannot receive an Auditor V7 PASS review or enter
a dual-PASS V7 authorization.

Because the candidate is frozen, repair requires an additive successor. The
successor must implement the amendment's two-drain terminal sequence and final
ancestry/destination recheck, then add adversarial injections after each drain
and during each final identity check. Builder V7's terminal transaction already
shows the required ordering pattern.

## Passing evidence

The blocking result is narrow. Source inspection and tests confirmed:

- the fixed ordered nine-role V7 source map and exact Builder V7 bindings;
- distinct builder/auditor authors and reviewers, exact review schemas, and
  cross-binding agreement between Builder V7 and Auditor V7;
- pure phase-one validation before mapped-target opens and complete fixed
  phase-two predecessor/review rehashing;
- the keyword-only `execute_exact_audit_v7` boundary and exact non-boolean
  worker range `[1,6]`;
- empty accelerator visibility, one native CPU thread, spawn-worker policy,
  and worker authority revalidation;
- no Auditor V6, Builder V7, or legacy exact-entry import/call in the Auditor
  V7 production module;
- fixed science cardinalities of 5,172 pairs, 10,344 endpoint references,
  9,460 unique endpoints, and 88 scene shards, with the V6 science engine
  retained mechanically; and
- clean publication, modify-then-restore poisoning, foreign-destination
  preservation, and ancestor movement detected when it occurs before the
  current terminal drain.

Verification used one CPU thread per native library and empty CUDA/HIP/ROCr
visibility:

| Check | Result |
| --- | --- |
| Frozen Auditor V7 author tests + retained Auditor V1/V2 tests | `62 passed` |
| Root V7 preaudit | `2 passed` |
| Independent reviewer QA | `4 passed` |
| Combined review suite | `68 passed in 1.03s` |
| Candidate, CLI, author test, root preaudit, and reviewer QA `py_compile` | PASS |
| `git diff --check` on reviewed/reviewer files | PASS |

The reviewer QA intentionally passes while proving the frozen candidate's
blocking behavior; it asserts that the prohibited success is reproducible.

## Authority boundary

No exact build or audit was executed. No canonical authorization, development
source payload, dataset, audit output, RGB, checkpoint, G2, held-out, runtime,
navigation, hardware, production, or accelerator payload was opened or
created. All dynamic testing used synthetic temporary roots.

This review grants no Auditor V7 source approval and no exact build, exact
audit, dataset-use, training, selection, calibration, G2, held-out, runtime,
navigation, hardware, production, promotion, or retry authority. A subsequent
canonical machine review at the fixed V7 path records `BLOCK` with every
authority field false.
