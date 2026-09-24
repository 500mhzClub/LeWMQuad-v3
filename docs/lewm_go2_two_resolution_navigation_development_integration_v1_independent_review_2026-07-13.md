# Two-resolution navigation development integration V1 independent review

Date: 2026-07-13

Verdict: **BLOCK**

The frozen V1 candidate correctly composes and seals its individual downstream
owners, but it is not an end-to-end navigation authority. Four independently
reproduced defects permit cross-target or cross-scene claim splicing, leave a
failed issuance partially committed, and permit observer-result substitution.
V1 is not approved as the development navigation coordinator.

## Frozen artifacts reviewed

- implementation:
  `lewm/planning/two_resolution_navigation_development_integration_v1.py`
  - SHA-256:
    `9ba954c191321c629e01cbd8a447a9aff39cf41b35aef26a12f0f7262bd4a0a4`
- candidate tests:
  `lewm/tests/test_two_resolution_navigation_development_integration_v1.py`
  - SHA-256:
    `9bf16a8cbb685bf07313f0ebb33df47211399198e25b3033ba4552c35a5ddf9c`
- implementation handoff:
  `docs/lewm_go2_two_resolution_navigation_development_integration_v1_handoff_2026-07-13.md`
  - SHA-256:
    `6366ae7c3c5cff438d554f86906d1d93aec5b10093bedc2974d12f85f37aa784`
- independent adversarial tests:
  `lewm/tests/test_two_resolution_navigation_development_integration_v1_independent_review.py`
  - SHA-256:
    `d3755aa349b2177192007ab5a9293a5bdcd79c9696b87964be1c72a85f5256a2`

The supplied source, candidate tests, and handoff hashes matched the review
request and were not edited.

The consumed owner-source hashes also match the handoff exactly:

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
- raw claim trace:
  `a41f1fa22f5a90503c82db459ccc9520af334173d416bac0b090308d69cc8fb3`
- observer wrapper:
  `1db940a49f01313b23c5d37699796b52da776a3a5c88bf3af1381d7d58103e30`
- canonical claim evaluator:
  `7ea003160ea03da6e989cb76124501b1e7de8571bf8586870b9c8dd7b42f04df`

## Blocking findings

### 1. Semantic target and task object are not bound

The G5 outcome, posterior, and route retain `outcome.target_id`, while claim
construction independently accepts caller-supplied `task_object_id`. V1 checks
that each value is internally retained but has no episode authority containing
the required one-to-one `target_id -> object_id` mapping. See source lines
688-715 for the target chain, line 729 for the independent object reference,
and lines 773-774 for the two unconnected artifact fields.

The independent probe replaced the valid `red` outcome with an exact live
`blue` outcome from the same issuer, then requested `beacon_red`. V1 issued the
trace instead of rejecting it. A direct observer probe credited the red beacon
and reported `all_targets_claimed=True`, despite the routed semantic target
being blue. This makes perception evidence for one task target usable as claim
authority for another.

The independent target-authority test fails because no binding error is raised.

### 2. G3 scene sessions are not bound to the claim manifest

V1 validates the exact current G3 snapshot and stores its two frame hashes, but
it independently accepts any exact `SceneManifest`. No check relates the G3
physical/configuration session identities to the claim manifest scene or its
hash. See source lines 668-686 for G3/G4 validation and lines 729-748 for claim
construction from the unrelated manifest.

The independent probe paired G3 sessions rooted at
`two-resolution-development-integration` with a manifest whose scene ID was
`foreign-physical-claim-scene`. V1 issued the trace. A direct observer probe
then reported `all_targets_claimed=True` for the foreign scene. Hashing both
halves into one artifact records the splice; it does not prove that the halves
belong together.

The independent scene-session test fails because no binding error is raised.

### 3. Late claim rejection partially commits the one-shot chain

Caller-controlled claim inputs are not fully validated before destructive G5
operations. V1 issues the context/evidence and mutates target memory at source
lines 688-700. Only later does it build the claim input at lines 729-748 and
construct the task-bound artifact at lines 750-784.

Supplying a non-manifest `task_object_id` therefore raises the expected late
`ValueError`, but only after the raw outcome has been consumed and its positive
evidence has entered target memory. The independent probe observed the raw
outcome hash in `_seen_raw_outcomes`, a positive posterior count of one, and a
`TwoResolutionTargetEvidenceReplayError` on the corrected retry.

Both independent atomicity tests fail: state is mutated after the rejected call,
and the exact single-use outcome is permanently stranded for that coordinator
attempt.

### 4. Observer results have no original-issuance authority

The controller trace has an exact-live registry plus an immutable original
digest, and the independent positive control confirms that it rejects both a
clone and a same-object nested mutation followed by rehashing. The observer
result does not have the equivalent boundary.

`TwoResolutionObserverClaimEvaluationV1.__post_init__()` only requires a
non-`None` capability and computes an unkeyed self-hash. Its `assert_integrity()`
only recomputes that same self-hash at source lines 450-467. The integration
records only the observed controller-artifact ID; it does not retain the exact
observer-result object or its original digest at lines 847-856.

The independent probe replaced `evaluated_claim_trace` on the exact result,
recomputed `content_sha256`, and called `assert_integrity()`. V1 accepted the
substitution. The observer-result adversarial test consequently fails because
no exact-live/original-issuance binding error is raised.

## Contract checks that passed

- Construction requires the exact projection, planner, view issuer, frontier
  planner, evidence issuer, target memory, router V2, and waypoint V2 owner
  chain. Mixed owner instances reject.
- The exact current G3 snapshot/component, both map frames, both revisions,
  projection supports, G4 state/candidate set/selection, G5 context/evidence,
  posterior, and target route are revalidated by their owners.
- The router V2 route and world-waypoint V2 receipt are validated and consumed
  exactly once before the controller artifact is exposed. Their frame,
  revision, path, waypoint, and authority-denial commitments remain intact.
- The controller trace exact-object registry and stored original digest reject
  a dataclass clone and a same-object nested mutation plus rehash.
- The controller path does not import the observer or canonical evaluator. The
  observer import remains lazy and occurs only after an exact controller trace
  is asserted.
- The claim task set is canonical, sorted, unique, manifest-contained, and
  hash-bound. Raw evaluator feedback remains `[]`.
- Observer execution is one-use per controller trace and requires the exact
  three-key, integer-zero access ledger. Missing, extra, boolean, or nonzero
  entries reject.
- Production construction, the production entrypoint, hardware execution, and
  promotion remain denied.

## Verification

All commands hid HIP, CUDA, and ROCr devices, disabled external pytest plugins,
and capped OMP/OpenBLAS/MKL/NumExpr threads to one. The focused and adjacent
runs were executed concurrently as independent CPU processes.

```text
frozen candidate focused suite:
  4 passed in 94.44s

adjacent frozen owner/evaluator suites:
  71 passed in 276.12s

independent adversarial suite:
  1 passed, 5 failed in 83.06s
  - controller clone and nested-rehash rejection: passed
  - semantic-target/task-object splice rejection: failed
  - G3-session/claim-manifest scene splice rejection: failed
  - rejected issuance leaves target memory unchanged: failed
  - rejected issuance permits corrected exact-outcome retry: failed
  - observer-result same-object mutation plus rehash rejection: failed

py_compile: passed
```

The `75/75` frozen candidate and adjacent PASS is retained. It demonstrates
that the defect is in coordinator-level composition rather than in the reviewed
G3/G4/G5/posterior/router/waypoint/evaluator owners.

No G2 input or output, held-out scene, dataset, model checkpoint, accelerator,
hardware, runtime, production input, or unfinished learned projection was
opened.

## Required additive V2 closure

Preserve V1 and this frozen review evidence. An additive V2 must:

1. require an exact-live episode authority that hash-binds the scene ID,
   physical manifest, canonical task-object set, and a one-to-one, complete
   `target_id -> object_id` mapping; reject a claim unless the routed posterior
   target maps exactly to the requested and claimed object;
2. bind both exact G3 map-session identities and frame hashes to that same
   manifest/scene authority, rejecting a foreign or merely hash-adjacent claim
   manifest rather than inferring compatibility from separate fields;
3. prevalidate the manifest, task set/mapping, trace/episode/event identities,
   tick/index, full-precision pose, and every other caller-controlled claim
   field before consuming an outcome or mutating target memory; order the
   remaining operations so a rejected preflight needs no rollback, leaves no
   reservation, and permits a corrected retry with the exact outcome;
4. retain every issued observer result in an exact-live registry with its
   original content digest, reject direct construction, clone/replace,
   same-object mutation plus rehash, cross-integration transfer, and replay, and
   expose an integration-owned assertion for the result;
5. preserve route and waypoint exact validation/consumption, lazy evaluator
   loading, the actually-empty ledger, controller-trace original-digest seal,
   and all production/hardware denials;
6. pass the `75` frozen focused/adjacent tests and all `6` independent tests,
   then receive a different-agent independent PASS before downstream use.

G2, held-out, learned-runner, production, hardware, and navigation execution
authority remain closed.
