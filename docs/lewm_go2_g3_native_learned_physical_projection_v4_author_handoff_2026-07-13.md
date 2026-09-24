# G3 Native Learned Physical Projection V4 Author Handoff

Date: 2026-07-13

Status: **AUTHOR COMPLETE; INDEPENDENT REVIEW REQUIRED**

This handoff freezes the additive V4 candidate that remediates the independent
V3 BLOCK. It is not an independent PASS and does not authorize production,
held-out access, or promotion.

## Frozen Inputs

- V3 independent review:
  `docs/lewm_go2_g3_native_learned_physical_projection_v3_independent_review_2026-07-13.md`
  - SHA-256: `edd29160d30095d5d8ab11c4dc74046bc2f1703010a43e1e01a31a0ae5c7ac82`
- V3 source:
  `lewm/planning/native_learned_physical_projection_v3.py`
  - SHA-256: `c472b4792279a20fd7085189ea53d3a6c7d2c33343d86cc9063c73eea42f136f`
- V3 focused tests:
  `lewm/tests/test_native_learned_physical_projection_v3.py`
  - SHA-256: `d5113b0873b40c700a3f0b752b7d30f92e1943d2597cae886e67769ac3f19129`

The V3 independent review found two decisive defects:

1. A reachable `_commit_core_v3` path did not perform the final immutable
   target and reservation binding checks, so a mutated target could be removed
   through that path after the public commit path rejected it.
2. A permanent duplicate-observation rejection left a LIVE reservation holding
   the target slot, making the failed package unrecoverable while also blocking
   a fresh exact issue.

## V4 Candidate

- Source:
  `lewm/planning/native_learned_physical_projection_v4.py`
  - SHA-256: `66486f70f0998502f36e16e496f1c76d11cd117176046e6de433db911473f16a`
- Focused tests:
  `lewm/tests/test_native_learned_physical_projection_v4.py`
  - SHA-256: `df9b89778adea21da70b89004da41a01354b7086dd25eab5961f3a5bb1e0abb2`

V4 is standalone and additive. It preserves V1, V2, and V3 unchanged and does
not import their adapter implementations. Production identities remain unset.

## Remediation

- `_commit_core_v4` is itself a complete guarded commit surface. It requires
  the exact LIVE reservation, exact active target, and unchanged target and
  retraction digests immediately before transaction commit.
- Every successful callable commit path consumes the reservation.
- Snapshot-binding failures mark the reservation STALE and release the target
  slot.
- A permanent duplicate-observation rejection, proven by the admission
  observation identity already being present in the physical memory, marks the
  reservation STALE and releases the target slot.
- Other `TransactionRejectedError` failures remain retryable because their
  permanence is not established.
- Bound and unbound V1, V2, and V3 commit surfaces reject V4 packages.

## Author Verification

CPU verification used one thread per numerical runtime with accelerators hidden.

- Focused V4 suite: **15 passed in 67.39s**.
- Adjacent regression panel: **73 passed in 164.39s**.
  - V4 focused tests
  - frozen V3 focused tests
  - revisioned physical-memory tests
  - two-resolution configuration projection tests
- `py_compile`: PASS.
- `git diff --check` on the V4 source and focused tests: PASS.

The focused suite includes direct core-path target mutation, permanent and
transient late rejection, exact retry/release behavior, replay/copy/transfer
rejection, atomic removal, geometry invariants, old-version cross-calls, source
isolation, and fail-closed production checks.

## Required Independent Review

A different agent must independently freeze these exact hashes and attempt to
falsify:

1. final target and reservation binding on every reachable commit path;
2. terminal release after provably permanent late rejection;
3. retry preservation after transient late rejection;
4. replay, copied-package, transferred-package, and old-version cross-call
   resistance;
5. atomicity of target removal relative to the physical-memory transaction;
6. fail-closed production and promotion state.

No promotion decision may cite this author handoff as an independent PASS.
