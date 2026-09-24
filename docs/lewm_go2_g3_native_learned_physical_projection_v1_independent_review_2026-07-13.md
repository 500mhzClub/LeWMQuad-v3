# Go2 G3 native learned physical projection V1 independent review

Date: 2026-07-13

Verdict: **BLOCK**

The frozen development candidate is not approved for downstream integration.
Its normal projection/commit path and conservative geometry pass, but two
independently reproduced retraction-lifecycle defects violate the exact-object
and retractability requirements of the preregistered G3 plan.

## Frozen artifacts reviewed

- implementation:
  `lewm/planning/native_learned_physical_projection_v1.py`
  - SHA-256:
    `f8b149c685a4320ae938ff367edcf833047016250caae7699cddfe8026cc0634`
- candidate tests:
  `lewm/tests/test_native_learned_physical_projection_v1.py`
  - SHA-256:
    `1f47ee15e46be1e8d5407ffa6f39f753b2dba92d15be67af8217ab4e146b5661`
- implementation handoff:
  `docs/lewm_go2_g3_native_learned_physical_projection_v1_handoff_2026-07-13.md`
  - SHA-256:
    `caccd6204e394bd07e7c1f3d15b35775de20ac6fa2e17027d63efc5c326dbb2a`
- independent adversarial tests:
  `lewm/tests/test_native_learned_physical_projection_v1_independent_review.py`
  - SHA-256:
    `787b6d1ba10f24161ad355aef13a84e9891556d42d40693a02c803779b342ac3`

The three supplied candidate hashes exactly match the review request. The
candidate source, candidate tests, and handoff were not edited.

## Blocking findings

### 1. Retraction accepts a mutated committed package

`issue_retraction()` checks that the supplied object is the same object stored
in `_issued`, then calls its self-consistency check. It does not compare the
object's current content hash with the original issued content hash retained in
`_issued`. That comparison exists for ordinary commit in
`_assert_exact_package()`, but the retraction path bypasses it.

An adversarial test commits valid learned FREE evidence, mutates the committed
package pose, updates the nested pose binding, recomputes the nested and outer
unkeyed hashes, and supplies the same object to `issue_retraction()`. V1 accepts
it. This violates the required immutable instance-issued provenance and lets a
retraction record carry post-commit substituted pose/source content.

Relevant implementation: lines 2155-2161 accept exact object identity plus
self-consistency, while lines 2252-2256 show the missing original-issued-digest
comparison used on the commit path.

### 2. A stale retraction permanently strands active learned evidence

`issue_retraction()` adds the target observation to `_retraction_issued` before
the package commits. If another valid transaction advances the memory revision,
the issued retraction becomes permanently stale. Its commit correctly rejects,
but V1 never clears the reservation. A fresh retraction against the current
snapshot then always raises `NativeLearnedProjectionReplayError`, although the
original learned observation is still active in memory.

The independent reproducer confirms the active observation remains in
`learned_observation_ids` after the stale rejection and cannot be retracted.
This violates the plan's requirement that learned contributions remain exactly
retractable.

Relevant implementation: lines 2172-2175 reject any second issuance and line
2235 reserves at issue time; no failure or stale-package path releases it.

## Contract checks that passed

- Inputs are typed synthetic runner-issued raw ground logits/query geometry and
  ordered ray logits/depths. No caller label, aggregate metric, Torch, NumPy,
  checkpoint loader, file opener, accelerator, held-out, hardware, or navigation
  surface exists in this module.
- The adapter requires the exact frozen native `0.05 m` source geometry and
  exact `0.05 m` physical / `0.10 m` configuration frames, shared origin,
  `2:1` shapes, snapshot hashes, revisions, projection source, camera identity,
  checkpoint identity, G2 identity, calibration identity, and diagonal
  covariance envelope. Derived `0.10 m` or upsampled sources reject.
- FREE uses complete closed destination-square coverage for every registered
  uncertainty transform. OCCUPIED uses the closed point supercover union across
  transforms. OCCUPIED precedes FREE and remaining projected support is UNKNOWN.
- Raw outcomes and ordinary transaction commits enforce exact instance
  ownership, stored issued digests, one-use consumption, stale-snapshot
  rejection, adapter transfer rejection, and replay rejection.
- Independent serialization-reload and `object.__new__` package forgeries are
  rejected by the live-instance registries.
- Commit reconstructs the hidden `PhysicalEvidenceTransaction` and checks its
  exact digest before applying it atomically.
- Production runner/checkpoint/G2/calibration/adapter globals remain `None`.
  Synthetic constructors require the explicit fixture opt-in. Adapter,
  admission, projection receipt, and package surfaces all report development
  only, hardware false, and promotion false.

## Verification

All commands hid HIP, CUDA, and ROCr devices, disabled external pytest plugins,
and capped OMP/OpenBLAS/MKL/NumExpr threads to one.

```text
candidate + adjacent frozen suites:
  80 passed in 101.42s

independent adversarial suite:
  1 passed, 2 failed in 9.55s
  - serialized reload/object.__new__ rejection: passed
  - post-commit mutation rejection: failed (mutation accepted)
  - stale retraction recovery: failed (active evidence stranded)

py_compile: passed
git diff --check: passed at review time
```

No real V4/V5 checkpoint, G2 output, held-out scene, accelerator, hardware, or
navigation input was opened.

## Required successor closure

A V2 successor must preserve V1 and its frozen evidence, then:

1. compare a committed projection's current content hash with the immutable
   digest retained at original issuance before creating any retraction;
2. make retraction issuance retryable after a stale/uncommitted package, or
   eliminate issue-time target reservation and rely on exact package single use,
   current-snapshot validation, active-observation validation, and successful
   commit to close the target;
3. pass both independent adversarial tests plus the 80 candidate/adjacent tests;
4. receive a different-agent review before any downstream integration.

The real-runner/source-isolation, real G2 calibration, view-diversity,
traversal-correction, cold-start, promotion, hardware, and navigation gates
remain explicitly outside this synthetic V1 candidate and remain closed.
