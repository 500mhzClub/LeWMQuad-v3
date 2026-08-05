# Two-resolution navigation development integration V2 independent QA

Date: 2026-07-13

Status: **BLOCK**

This is a different-agent review of the final standalone V2 candidate. It does
not modify or supersede the frozen candidate source, author tests, or author
handoff. V2 grants no navigation-work, production, runtime, or hardware
authority while this block is open.

## Frozen candidate

- Source:
  `lewm/planning/two_resolution_navigation_development_integration_v2.py`
  - SHA-256:
    `5a1379ee47b81a5f400b967abf092ca32431d0a19097d880916820d0cc8bd3de`
- Author tests:
  `lewm/tests/test_two_resolution_navigation_development_integration_v2.py`
  - SHA-256:
    `a18434608a23ceaa58f975c208171258f9932af1afa058b63498448383497cca`
- Author handoff:
  `docs/lewm_go2_two_resolution_navigation_development_integration_v2_handoff_2026-07-13.md`
  - SHA-256:
    `9e188f5de337a6a821b2b27d879866769a629dffc16068a5a528228340f51008`

All three hashes reproduced before and after review. The candidate source and
author tests were not edited.

## Blocking finding

**Controller registration is outside the rollback boundary.**

The downstream transaction is protected by `try`/rollback through source line
1791. Controller-record construction, controller-registry insertion, and the
stored owner-seal update occur afterward at lines 1792-1797. A failure while
constructing `_ControllerRecordV2` therefore escapes without invoking
`transaction.restore(state)`.

The independent test injects exactly that late failure. It observes all four
parts of the stranded state:

1. the complete transaction-owner digest changed;
2. the exact V5 outcome is consumed;
3. the V2 controller registry contains no artifact; and
4. the coordinator's stored downstream-owner seal is stale.

This is not a cosmetic exception path. The operation has irreversibly changed
G3/G4/G5/posterior/router/waypoint state but has issued no exact-live controller
authority, and the same coordinator cannot continue normally. It violates the
candidate's claimed all-stage atomicity and retry guarantee.

Required successor behavior: construct the controller record within the
rollback envelope, and transactionally cover controller-registry insertion and
owner-seal assignment. Add explicit injected stages after record construction,
registry insertion, and seal assignment. Every stage must restore the exact
pre-call downstream digest, leave the outcome unconsumed, leave the controller
registry empty, restore the previous coordinator seal, and permit corrected
retry.

## Passed review surface

The remaining required properties passed independent inspection or execution:

- the exact semantic-target to task-object mapping is complete and one-to-one;
- physical manifest, scene, episode, both session/frame identities, snapshot,
  component, revisions, and projection source are exact-live bound;
- documented V1 target splice, scene splice, late validation, stranded outcome,
  and observer mutation failures are closed by V2's existing focused tests;
- claim inputs are validated before the downstream mutation chain begins;
- copied, deep-copied, serialized, replaced, cross-integration, and reused
  authority/controller/observer objects reject;
- the public V2 object has no instance dictionary, retained V1 engine, or public
  mutable-owner aliases;
- a separately constructed V1 mutation over shared owners is detected before a
  new exact outcome is consumed;
- the observer ledger is exactly empty, raw evaluator feedback is empty, the
  serialized observer callback is `None`, and observer results remain exact-live
  and single-use; and
- every exposed authority remains development-only with production and hardware
  authority exactly false.

Static review also confirmed that the rollback snapshot enumerates the mutable
G3 planner, G4 view/frontier, G5 outcome/evidence, reversible posterior, router,
and waypoint containers touched by the operation. The existing 18 injected
stages restore and retry successfully. The block is the uncovered final
coordinator-commit stage after those 18 stages.

## Reproduction

Independent QA:

```text
3 passed, 1 failed in 43.93s
```

The one failure is
`test_independent_qa_late_controller_registration_failure_is_atomic`.

Frozen V2 author suite:

```text
25 passed in 514.17s
```

Unchanged adjacent G3/G4/G5/posterior/router/waypoint/claim/observer suite:

```text
71 passed in 287.39s
```

Frozen V1 focused suite:

```text
4 passed in 97.70s
```

Frozen V1 adversarial review, deliberately reproduced to establish the
predecessor failures:

```text
5 failed, 1 passed in 87.96s
```

`py_compile` passed. The candidate and all nine documented downstream source
hashes reproduced. Runs used CPU only, one OpenMP/BLAS/native thread per process,
empty HIP/CUDA/ROCr visibility, and disabled external pytest plugin autoload.

## Scope boundary

No G2, held-out, runtime, hardware, checkpoint, model, dataset, production, or
navigation-result payload was opened. The machine-readable failure receipt is:

`docs/lewm_go2_two_resolution_navigation_development_integration_v2_independent_qa_block_2026-07-13.json`

The next candidate must be additive with new hashes. This frozen V2 remains
useful BLOCK evidence but cannot be promoted in place.
