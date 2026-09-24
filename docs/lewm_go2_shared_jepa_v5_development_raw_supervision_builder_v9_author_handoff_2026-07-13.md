# Shared JEPA V5 development raw-supervision Builder V9 author handoff

Date: 2026-07-13

Implementation author: `/root/raw_v7_successor_author/auditor_v7_author`

Status: **FROZEN AUTHOR CANDIDATE; INDEPENDENT REVIEW REQUIRED; NO EXACT OR DATA AUTHORITY**

## Contract

This candidate implements only the Builder V9 role established by the frozen
source-free amendment:

| Artifact | SHA-256 |
| --- | --- |
| `docs/lewm_go2_shared_jepa_v5_raw_supervision_builder_auditor_v9_linearization_successor_amendment_2026-07-13.md` | `6fba5de8d7f04d85bd87e084096ae269c3d3dd6368a6db0b0f8f149c1c5cf773` |

The amendment binds the governing scientific-execution threat model, the
independently passing Builder V8 candidate, and the terminal Auditor V8 BLOCK.
It establishes the final successful event drain as the publication
linearization point. Builder V8 already satisfies that finite transaction
contract, so Builder V9 is a strictly mechanical authority/provenance
successor with no construction or publication behavior change.

This handoff is not an authorization. No canonical authorization, `.generated`
payload, source data, RGB, checkpoint, dataset, audit output, exact execution,
G2, held-out, runtime, hardware, production path, or accelerator was opened or
changed during authoring or testing.

## Frozen candidate

| Role | Artifact | SHA-256 |
| --- | --- | --- |
| `builder_source` | `lewm/datasets/go2_shared_jepa_v5_raw_supervision_builder_v9.py` | `2388c1138d9b03ea6e385cc0250c81a1869a40cab62507d02f709ef39197c664` |
| `builder_cli` | `scripts/build_go2_shared_jepa_v5_development_raw_supervision_v9.py` | `f239a4ef7c067a71f991b30e14bd5c8632c31be3173780fc25b3d9801fff79ee` |
| `builder_test` | `lewm/tests/test_go2_shared_jepa_v5_raw_supervision_builder_v9.py` | `541d1957df0a3da18c2b529cd2d7ca721d7e657c8ebcced2a37931d502cab7bc` |

The fourth Builder V9 candidate role is this handoff. Its SHA-256 is computed
externally after freeze and must be bound verbatim by the different-agent
review and any later nine-row authorization source map.

No non-role V9 helper was added. Production-ineligible synthetic authority and
construction support is contained entirely within the reviewed test role.

## Mechanical successor

Builder V9 is standalone. It imports or calls no Builder V8, auditor, legacy
builder, test module, dynamic module loader, or legacy exact entry.

The author test normalizes only `V8`/`v8` identifiers and compares every
top-level class and function AST between Builder V8 and Builder V9. All 80
definitions are exactly equal after normalization. This includes every
science, authority-validation, worker, source-closure, filesystem, event,
cleanup, and publication-transaction body.

The only V9 changes are the fixed nine V9 paths, schemas, immutable capsule
type names, provenance labels, exact entry name, and CLI import. Fixed authors,
reviewer separation, downstream/retry denials, worker/resource rules, and the
canonical output namespace are unchanged from passing Builder V8.

`FROZEN_PARENT_HASHES` contains 83 rows: Builder V8's complete 69-row closure,
the V9 amendment, the governing threat model, five Builder V8 candidate/PASS
rows, and seven Auditor V8 candidate/QA/report/BLOCK rows. Its canonical sorted
compact JSON SHA-256 is
`76823317704cb35ad3342cb27c03c218816da89b56c294897a7eddd651cdd83e`.
All 83 predecessors and all nine reviewed V4 science files reproduce their
declared hashes.

## Preserved behavior

The only exact entry is keyword-only:

```text
execute_exact_build_v9(*, authorization_sha256: str, workers: int)
```

Workers remain exact non-boolean integers in `[1,6]`, use `spawn`, validate
the fixed V9 authority before task receipt and protected source use, expose no
accelerator, and use one native math thread.

The successful directory transaction retains its two validation passes and
three drains. `require_final_quiet` ends with the literal final publication-
identity drain; no later call exists in that method. On the success path only
the already prepared in-memory manifest is returned and retained descriptors
are closed. Every later consumer must independently reopen and hash the
canonical artifact under the governing threat model.

All retained science is unchanged: 5,172 pairs, 10,344 endpoint references,
9,460 unique endpoints, 88 development scenes, three scene-disjoint roles,
one schedule/raycast per endpoint, reviewed full-RPY camera geometry, eight
fixed arrays, `64 x 64` three-state rasters, strict joins, 354 source records,
zero forbidden opens, deterministic bytes, private staging, retained
descriptors, fsync, one `renameat2(RENAME_NOREPLACE)`, inode-owned cleanup, and
all downstream license denials.

## Author verification

All commands used one native CPU thread, empty accelerator visibility, and
disabled pytest plugin autoload. Filesystem tests were serialized.

```text
pytest -q lewm/tests/test_go2_shared_jepa_v5_raw_supervision_builder_v9.py
72 passed in 0.99s

pytest -q lewm/tests/test_go2_shared_jepa_v5_raw_supervision_builder_v8.py
69 passed in 0.71s
```

The V9 suite includes the complete retained V8 authority, worker, science,
transaction, mutation, event, ancestry, cleanup, and publication tests. New
proofs cover exact 80/80 AST equivalence, the 83-row closure, the final drain
as the transaction's last filesystem observation, rejection of an intermediate
canonical symlink, and full synthetic publication with requested worker counts
one and six producing identical manifest and complete tree bytes.

The source, CLI, and test pass `py_compile` and `git diff --check`. Negative
inspection finds no Builder V8 import/entry, auditor/test import, dynamic
import/eval/exec, mutable authority registry, alternate opener/path, exact
switch, skip, retry, visible accelerator assignment, or unrestricted
production injection.

## Independent review request

A reviewer distinct from `/root`, both fixed V9 implementation authors, and
the Auditor V9 reviewer must independently bind the exact source, CLI, test,
and this handoff. Review must rerun all 72 V9 and 69 retained V8 tests, reproduce
the complete hash closure and AST comparison, verify one/six-worker identical
bytes, inspect the final-drain ordering and canonical no-follow path handling,
and confirm the exact V9 authority boundary.

The only eligible review path and schema are:

```text
docs/lewm_go2_shared_jepa_v5_development_raw_supervision_builder_v9_independent_review_2026-07-13.json
lewm_go2_shared_jepa_v5_raw_supervision_builder_v9_independent_review_v1
```

A `PASS` may grant only `builder_source_approved=true`. Every exact-build,
exact-audit, dataset-use, training, selection, calibration, G2, held-out,
navigation, runtime, hardware, production, promotion, deployment, and retry
field remains false. Changed bytes or any finding requires another additive
successor.
