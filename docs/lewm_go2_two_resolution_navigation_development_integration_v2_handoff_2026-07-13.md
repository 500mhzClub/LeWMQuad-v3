# Two-resolution navigation development integration V2 author handoff

Date: 2026-07-13

Status: **AUTHOR CANDIDATE PASS for independent review only; no independent,
production, hardware, learned-runner, held-out, or navigation-execution authority**

## Result

This additive V2 closes the four blocking findings in the frozen V1 independent
review without changing V1. It is a standalone coordinator over the exact passed
G3/G4/G5/posterior/router/waypoint owners. It does not import, retain, expose, or
call a V1 navigation-integration engine.

The implementation adds:

1. one exact-live episode authority binding the original physical manifest,
   canonical complete task-object set, complete one-to-one semantic-target to
   task-object mapping, G3 snapshot/component, both canonical scene-named
   sessions, both frame IDs and hashes, both revisions, and projection source;
2. claim preflight of the exact authority, manifest, outcome, semantic/object
   mapping, canonical task set, episode/trace/event fields, tick/index, full pose,
   start cell, yaw, and pose variances before the destructive G5 chain;
3. an atomic owner transaction covering every mutable G3 planner, G4,
   outcome/evidence, posterior, router, and waypoint registry/counter touched by
   issuance, with exact rollback and digest verification on any exception;
4. an append-only exact-outcome ingress witness plus a downstream owner-state
   seal which rejects mutation through another coordinator before V2 accepts or
   consumes a new outcome;
5. an exact-live controller registry and original issuance digest; and
6. an exact-live observer-result registry and original issuance digest, with
   direct/replace/cross-integration/mutate-and-rehash/replay rejection.

The observer and canonical evaluator remain lazy imports after controller
sealing. Route and waypoint validation/consumption and every production/hardware
denial remain intact.

## Candidate hashes

- V2 integration source:
  `lewm/planning/two_resolution_navigation_development_integration_v2.py`
  - SHA-256:
    `5a1379ee47b81a5f400b967abf092ca32431d0a19097d880916820d0cc8bd3de`
- V2 focused/adversarial tests:
  `lewm/tests/test_two_resolution_navigation_development_integration_v2.py`
  - SHA-256:
    `a18434608a23ceaa58f975c208171258f9932af1afa058b63498448383497cca`

Any byte change requires new hashes and a new focused run.

## Frozen V1 evidence preserved

- V1 integration source:
  `9ba954c191321c629e01cbd8a447a9aff39cf41b35aef26a12f0f7262bd4a0a4`
- V1 focused tests:
  `9bf16a8cbb685bf07313f0ebb33df47211399198e25b3033ba4552c35a5ddf9c`
- V1 independent adversarial tests:
  `d3755aa349b2177192007ab5a9293a5bdcd79c9696b87964be1c72a85f5256a2`
- V1 independent BLOCK review:
  `677881b7ee825e979301b92828ebd26a9528f7eae404f567753b18fe6949eb99`

None of those four frozen artifacts was edited.

Exact downstream source hashes also remain unchanged:

- G3 V2 projection/planner:
  `3c858a89170f78a73f401c9534e231f24d6d91bb0469ea95eb00002158146107`
- G4 V2 frontier/viewpoint:
  `5c84e79e558f51b75b00cf2baa26d7860302d6e3912ac14432dfc010efdc4f82`
- two-grid G5 evidence:
  `f731b848f6b7ced3b07e11d4f9edca81daa8c66f083f9d503ed069809e38a9a2`
- reversible posterior V1:
  `6d17d06718df355893fa7a6f2f1f735fcf835933178e53c554f4d60181ae96c3`
- target router V2:
  `c8e071d239d1b9894028752fdc090cc2e1be9273f6f9de5a7c7b4d147741b6d2`
- world-waypoint V2:
  `9b710c6f6044bfefd3fd52bcdbb55a52f890b1fdc6c00629029bbf5a670e8fc1`
- canonical raw claim trace:
  `a41f1fa22f5a90503c82db459ccc9520af334173d416bac0b090308d69cc8fb3`
- observer wrapper:
  `1db940a49f01313b23c5d37699796b52da776a3a5c88bf3af1381d7d58103e30`
- canonical claim evaluator:
  `7ea003160ea03da6e989cb76124501b1e7de8571bf8586870b9c8dd7b42f04df`

## Atomicity evidence

The focused suite injects a one-shot exception after each of 18 fallible or
post-mutation stages: G4 view issuance, frontier generation/selection/validation,
G5 context/writer/evidence, posterior apply/validation, router issue/validation,
waypoint issue/validation, payload build, artifact construction, both single-use
consumptions, and final owner-seal computation.

For every stage it proves:

- the complete before/after owner-state digest is byte-identical;
- the exact outcome remains unconsumed;
- no V2 controller or observer object is registered;
- evidence, posterior, router, and waypoint state is uncommitted; and
- the same coordinator and exact outcome succeed on corrected retry.

The transaction includes the shared G3 planner component/frontier/path
registries, G4 state/candidate/score registries, source issuance and consumption,
context writer/evidence state, posterior mass/counts/chains/snapshots, router
plans/consumption, and waypoint receipts/consumption. Strong registry references
keep live object identities from being reused during their authority lifetime.

An additional adversarial probe constructs a separate V1 coordinator over the
same passed owners and triggers V1's known late-failure mutation. V2 detects the
changed shared-owner seal and rejects a new exact outcome without consuming it.
The public V2 object has no instance dictionary and no reachable V1 engine or
owner-registry attributes.

## Verification

Final-byte V2 focused/adversarial suite:

```text
25 passed in 497.80s
```

Unchanged frozen V1 focused suite:

```text
4 passed in 101.18s
```

Unchanged adjacent G3/G4/G5/posterior/router/waypoint/claim/observer suite:

```text
71 passed in 288.89s
```

Combined unchanged compatibility evidence: **75/75 passed**.

`py_compile` passed for the V2 source and focused tests. Static scans found no
V1 integration import/engine, accelerator/model/dataset surface, trailing
whitespace, or lines over 88 columns in the V2 candidate.

All pytest runs were CPU-only with `OMP_NUM_THREADS=1`,
`OPENBLAS_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `NUMEXPR_NUM_THREADS=1`, empty
HIP/CUDA/ROCr visibility, and external pytest plugin autoload disabled. No G2,
held-out scene, model checkpoint, dataset, accelerator, hardware, runtime, or
production input was opened.

## Author boundary

This document is an implementation-author handoff, not an independent review.
The V2 bytes are ready for a different agent to reproduce and adversarially
review. Until that review records a separate PASS, V1 remains frozen BLOCK
evidence and V2 grants no downstream navigation-work, learned-runner,
production-promotion, or hardware-execution authority.
