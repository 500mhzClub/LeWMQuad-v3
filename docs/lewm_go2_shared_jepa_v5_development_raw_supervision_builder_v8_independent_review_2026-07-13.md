# Shared JEPA V5 Raw Builder V8 independent review

Date: 2026-07-13

Reviewer: `/root/raw_v8_builder_reviewer`

Implementation author: `/root/raw_v7_successor_author/auditor_v7_author`

Verdict: **PASS for Builder V8 source only**

## Authority boundary

This review covers only the frozen Builder V8 source, CLI, author test, and
handoff under the terminal-quiet and identity-rebinding amendments. It does
not review Auditor V8 and cannot substitute for its separately required
different-agent review. It grants only `builder_source_approved=true`.

Exact build, exact audit, dataset use, training, selection, calibration, G2,
held-out, navigation, runtime, hardware, production, promotion, and retry all
remain false. No V8 authorization was created or validated, no exact entry was
called, and no `.generated` payload, dataset, RGB, label payload, checkpoint,
model output, accelerator, G2, held-out, runtime, hardware, or production path
was opened.

## Frozen candidate

| Role | Path | SHA-256 |
| --- | --- | --- |
| Terminal-quiet amendment | `docs/lewm_go2_shared_jepa_v5_raw_supervision_builder_auditor_v8_terminal_quiet_successor_amendment_2026-07-13.md` | `054de82d8648cd6be7edff01b82d549ec916700ebffad51698d4c2041edc6c88` |
| Identity-rebinding amendment | `docs/lewm_go2_shared_jepa_v5_raw_supervision_v8_existing_agent_identity_rebinding_amendment_2026-07-13.md` | `392745c80ca2c6e7a103cca4a55c3614cd2c988de9a379fba950b0087df41698` |
| `builder_source` | `lewm/datasets/go2_shared_jepa_v5_raw_supervision_builder_v8.py` | `f45533354c8b45b88f8eadb2126ec5eaf96fe1f57c21a9bfcd95a8855bfaaa35` |
| `builder_cli` | `scripts/build_go2_shared_jepa_v5_development_raw_supervision_v8.py` | `f6471f1fa0ca7a13976f752a41ee9ddacbc76636e4d5fb0eee1ebf75bdaee72d` |
| `builder_test` | `lewm/tests/test_go2_shared_jepa_v5_raw_supervision_builder_v8.py` | `fc1f0cf3fc18bdbd1393be6a514bc04459f943f39b438ced78ebee30e7c57d9a` |
| `builder_handoff` | `docs/lewm_go2_shared_jepa_v5_development_raw_supervision_builder_v8_author_handoff_2026-07-13.md` | `9f4898e3620ac87c9a0145be103c4fdf397f727fe37d9f6ca306a0f50916156b` |
| Independent QA | `lewm/tests/test_go2_shared_jepa_v5_raw_supervision_builder_v8_independent_qa.py` | `85e4f90a5be07d11d692f5a476cd093437e7bd1c0877758b832b6d141b88547f` |

All 69 frozen predecessor bindings and all nine reviewed V4 science-source
bindings rehash exactly. The candidate hashes still match the author handoff
after all review runs.

## Independent findings

### Complete authority before target open

The exact source map is the ordered, unique nine-role V8 map required by the
amendment. Phase one validates the exact authorization fields and content hash,
all nine role/path/hash rows, both nested review bindings, both fixed
implementation authors, distinct `/root/...` reviewers, and the complete
candidate bindings without calling a target opener. Missing, extra, reordered,
aliased, dot-dot, cross-reviewer, amendment-author, or implementation-author
inputs reject before any target read.

Only a completed immutable phase-one capsule can reach phase two. The
independent synthetic probe observed exactly the nine roles in canonical order
before predecessor/science closure revalidation. Both review records require
canonical duplicate-key-free JSON, `PASS`, matching file/content hashes, the
exact candidate, and an authority object with only the relevant source approval
true. `retry_authorized` and every downstream field are false.

### Mechanical V7 successor

An independent whole-module AST comparison normalizes only `V7`/`v7` version
identifiers. Every non-authority top-level statement, class, and function is
identical to independently passing Builder V7. The only excluded nodes are the
two rebound implementation identities, amendment paths/author, expanded frozen
predecessor closure, explicit retry denial, and `_review_binding` reviewer
separation required by the amendments. The normalized CLI is identical apart
from its V8 import/entry names and trailing whitespace.

Therefore the retained 5,172-pair/9,460-endpoint/88-scene science, V4 camera and
raycast semantics, eight arrays, strict joins, source ledger, worker bodies,
staging inventory, and output ordering are unchanged. The isolated retained V4
spawn probe reproduced byte-identical artifacts with one and six workers.

### Closed publication transaction

Builder V8 retains Builder V7's private staging, no-follow retained
descriptors, single-link leaves, complete source and staging watches, one
`renameat2(RENAME_NOREPLACE)`, exact owned-rename event sequence, inode-owned
cleanup, and parent fsync.

At terminal close it drains after the parent fsync, revalidates the retained
parent and complete ancestry, rehashes every source and published leaf,
rechecks the full inventory and destination inode, drains again, repeats the
retained ancestry and destination identity checks, and ends with the final
event drain. The independent QA modified and restored a source immediately
after the last destination identity lookup. The final drain observed the
mutation and permanently poisoned the transaction. Clean publication produced
the exact three terminal phases and succeeded.

The retained suites additionally cover mutation before/during/after rename and
each terminal validation, ancestor move-and-recreation, modification followed
by restoration, strict versus ancestry-only event filtering, watch loss and
queue overflow, foreign destination preservation, and matching-inode cleanup.

### Process and import boundary

The only production entry is keyword-only
`execute_exact_build_v8(*, authorization_sha256, workers)`. Workers are exact
non-boolean integers in `[1,6]`, use spawn, authorize in the initializer and
again before source use, set every native math thread selector to one, and
empty CUDA, HIP, ROCr, and GPU ordinal selectors.

Static and runtime inspection found no Builder V7 or auditor import, legacy
exact entry, test-module import, dynamic import, `eval`, `exec`, mutable
authority registry, callback, injected root/path/reader, alternate worker
target, or accelerator-visible assignment. The only submitted worker targets
are the fixed scene loader, source revalidator, and shard writer.

## Verification evidence

All required commands used one native math thread, disabled external pytest
plugins, and hid every accelerator selector.

```text
Builder V8 author + independent QA + retained V7: 147 passed
Retained V6/V5/V4 builder and metadata suites:    162 passed
Retained Builder V7 independent QA:                 4 passed
V4 one/six spawn byte-equivalence probe:             1 passed
Total required/retained checks:                    314 passed
py_compile for candidate and reviewer QA:           PASS
candidate, predecessor, and science rehash:         PASS
normalized whole-module/CLI equivalence:            PASS
V8 authorization/exact/data/GPU operation:          NONE
```

For transparency, an optional full run of the current V4 builder test file
reported 23 passes and two stale-state failures. Those two tests assert that
`docs/lewm_go2_observable_camera_ray_fit_v4_implementation_manifest_2026-07-12.json`
must be unreviewed, while that current manifest now literally records
`exact_fit_build_authorized_after_review=true` and
`exact_fit_audit_authorized_after_review=true`. They do not exercise Builder
V8, do not change the frozen V4 science source, and the required isolated
one/six-worker byte-equivalence test passes. This is repository test-maintenance
drift, not a V8 candidate finding.

## Decision

Builder V8 satisfies both frozen amendments and receives
`builder_source_approved=true` only. The canonical machine review is published
separately and last. Auditor V8 still requires its own distinct reviewer PASS;
two PASS reviews still require a separate nine-row authorization; and exact
work remains forbidden until a human separately supplies that authorization
file's frozen hash. Any change to the four candidate bytes invalidates this
PASS and requires another additive successor.
