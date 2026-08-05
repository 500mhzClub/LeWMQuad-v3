# Two-resolution navigation development integration V3 author handoff

Date: 2026-07-13

Status: **AUTHOR CANDIDATE PASS for independent review only; no independent,
navigation-work, production, runtime, hardware, or held-out authority**

## Result

This additive standalone V3 closes the sole blocking finding in the frozen V2
independent QA. V1, V2, their tests, and all V2 handoff/review/BLOCK artifacts
remain unchanged.

V2 protected every downstream G3/G4/G5/posterior/router/waypoint mutation but
left controller-record construction, controller-registry insertion, and stored
owner-seal assignment after its rollback boundary. An exception at record
construction therefore consumed the exact outcome and changed downstream state
without registering a controller artifact.

V3 changes only that commit protocol:

1. the complete transaction fingerprint now covers the V3 controller registry
   and stored downstream-owner seal as well as every downstream owner and exact
   append-only outcome ingress;
2. the rollback snapshot captures the exact controller-registry container and
   contents plus the prior stored seal;
3. controller-record construction, collision checking, registry insertion, and
   seal assignment now occur inside the same `BaseException` rollback envelope
   as downstream mutation; and
4. deterministic faults immediately after record construction, registry
   insertion, and seal assignment prove exact restoration and corrected retry.

V3 imports and directly coordinates the already-passed downstream owner APIs.
It does not import, instantiate, retain, expose, or call a V1 or V2 integration
engine.

## Candidate hashes

- V3 source:
  `lewm/planning/two_resolution_navigation_development_integration_v3.py`
  - SHA-256:
    `6d8b00aa8ffaa0117efc01baa218cadd299a871732e86d2751e51463520d6523`
- V3 focused/adversarial tests:
  `lewm/tests/test_two_resolution_navigation_development_integration_v3.py`
  - SHA-256:
    `d2af0e5a798ff6d186813d6054588e460cda37bb7989697261125a64d0265a54`

Any byte change requires new hashes and complete reruns.

## Frozen predecessor evidence

- V1 source:
  `9ba954c191321c629e01cbd8a447a9aff39cf41b35aef26a12f0f7262bd4a0a4`
- V1 tests:
  `9bf16a8cbb685bf07313f0ebb33df47211399198e25b3033ba4552c35a5ddf9c`
- V2 source:
  `5a1379ee47b81a5f400b967abf092ca32431d0a19097d880916820d0cc8bd3de`
- V2 tests:
  `a18434608a23ceaa58f975c208171258f9932af1afa058b63498448383497cca`
- V2 author handoff:
  `9e188f5de337a6a821b2b27d879866769a629dffc16068a5a528228340f51008`
- V2 independent QA test:
  `d746078e1190b51a14d1738214bb4bcf1e5e4e9dc6695adf8b5f5468a629d1f8`
- V2 independent BLOCK review:
  `abe943e963a09a83a64d4d9275dd474ed65e2b65dd4a86c343a46b5a15d93252`
- V2 machine BLOCK receipt:
  `46fecbb07a5cdaa66ded028fba8c02a106fa2673d570c4bae3043e0be56fd721`

The exact V2 independent QA was rerun and retained its expected result:

```text
3 passed, 1 failed in 41.53s
```

Its one failure is the frozen late-registration defect. No predecessor byte was
changed to make V3 pass.

## Atomicity evidence

The focused suite injects one-shot failures after each of 21 stages. The 18 V2
downstream stages remain intact, and V3 adds:

- `controller_record_construct`;
- `controller_registry_insert`; and
- `coordinator_seal_assign`.

For every stage, the test proves the complete pre/post transaction fingerprint
is identical, the outcome remains unconsumed, no new controller or observer is
registered, downstream evidence/posterior/route/waypoint state is uncommitted,
and the same coordinator and exact outcome succeed on corrected retry.

The frozen blocker is also reproduced directly by replacing the record
constructor with a deterministic exception. V3 catches it within the rollback
envelope, restores all state, and then succeeds after restoring the constructor.

A second three-stage matrix starts with one valid exact-live controller already
registered. Each V3 commit fault must preserve that exact registry row and seal,
leave the second outcome unused, keep the first artifact valid, and accept a
corrected second claim. All three cases pass.

## Other retained properties

V3 retains V2's exact semantic-target/task-object mapping; physical manifest,
scene, episode, session, frame, revision, snapshot, component, and projection
binding; preflight-before-mutation ordering; append-only outcome witness;
original-digest controller and observer registries; shared-owner interference
detection; lazy observer import; empty evaluator feedback/callback ledger; and
development-only fail-closed authority.

Copy, deep-copy, serialization, replacement, cross-integration, mutation with
rehash, and replay remain rejected. The public V3 object has no instance
dictionary, predecessor-engine attribute, or mutable downstream-owner alias.

## Verification

Final-byte V3 focused/adversarial suite:

```text
32 passed in 712.87s
```

Final-byte explicit frozen-blocker successor test:

```text
1 passed, 31 deselected in 21.75s
```

Unchanged adjacent G3/G4/G5/posterior/router/waypoint/claim/observer suite:

```text
71 passed in 283.14s
```

The three new empty-registry fault stages passed `3/3`; the three nonempty
prior-registry fault/retry cases passed `3/3`. `py_compile`, trailing-whitespace,
88-column, predecessor-import, and schema scans passed.

All executions were CPU-only with `OMP_NUM_THREADS=1`,
`OPENBLAS_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `NUMEXPR_NUM_THREADS=1`, empty
HIP/CUDA/ROCr visibility, and external pytest plugin autoload disabled.

## Scope and authority

No G2, held-out, runtime, hardware, production, checkpoint, model, dataset, or
navigation-result payload was opened. V3 exposes no production or hardware
authority.

This is an author handoff, not an independent PASS. A different agent must
freeze these exact source/test/handoff bytes, rerun the focused blocker and
adjacent evidence, adversarially review the complete transaction boundary, and
publish a separate PASS before V3 can grant navigation-work readiness.
