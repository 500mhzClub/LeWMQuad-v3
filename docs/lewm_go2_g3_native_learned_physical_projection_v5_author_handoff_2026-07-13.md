# G3 Native Learned Physical Projection V5 Author Handoff

Date: 2026-07-13

Status: **AUTHOR COMPLETE; INDEPENDENT REVIEW REQUIRED**

This handoff freezes the additive V5 candidate that remediates the independent
V4 BLOCK. It is not an independent PASS and does not authorize production,
held-out access, G2 execution, hardware execution, or promotion.

## Frozen Inputs

- V4 source:
  `lewm/planning/native_learned_physical_projection_v4.py`
  - SHA-256: `66486f70f0998502f36e16e496f1c76d11cd117176046e6de433db911473f16a`
- V4 focused tests:
  `lewm/tests/test_native_learned_physical_projection_v4.py`
  - SHA-256: `df9b89778adea21da70b89004da41a01354b7086dd25eab5961f3a5bb1e0abb2`
- V4 independent BLOCK record:
  `docs/lewm_go2_g3_native_learned_physical_projection_v4_independent_review_block_2026-07-13.json`
  - File SHA-256: `e52a0431c48d2105b4d0f8478699f05ddb0ec19b6ea1a993b29dbcec8da30f33`
- V4 independent review:
  `docs/lewm_go2_g3_native_learned_physical_projection_v4_independent_review_2026-07-13.md`
  - SHA-256: `659b4ad67691fda9e4ea1665ec76c5e290aad7a5af1bcef0fd7757686c90d491`

The V4 reviewer showed that an observation identity accepted and later
retracted is absent from `learned_observation_ids` but remains permanently
inadmissible. V4 therefore misclassified a duplicate retraction identity in
that state as transient, retained its LIVE reservation, and blocked a fresh
same-snapshot retraction indefinitely.

## V5 Candidate

- Source:
  `lewm/planning/native_learned_physical_projection_v5.py`
  - SHA-256: `5ccd22e83c83a4c41db11286d31d417fe7af5615ebd7e62e51d7719d5378eca1`
- Focused tests:
  `lewm/tests/test_native_learned_physical_projection_v5.py`
  - SHA-256: `e5f0d30b96d1da525ac004ded1eac6bcca96330657d92571594914c548a6d077`
- Shared memory source with the narrow history query:
  `lewm/planning/revisioned_physical_configuration_memory.py`
  - SHA-256: `bb05f957e0443e0c1e8405042b97c61948746a66040e84690e12b0a10887d483`
- Focused history-query tests:
  `lewm/tests/test_revisioned_physical_memory_seen_observation_ids.py`
  - SHA-256: `20860a1abca8848a5951481ce167da501420ce27ad21fba1c9821bc092459fa4`

V5 is standalone and additive. It preserves V1 through V4 unchanged, retains
the frozen V4 raw-outcome/calibration boundary, and does not import or compose
an older adapter. Production identities remain unset.

## Remediation

- `RevisionedPhysicalMemory.seen_observation_ids` returns a fresh immutable
  `frozenset` backed by the memory's existing serialized append-only identity
  history. It creates no new mutable authority or admission path.
- `_commit_core_v5` marks a rejected retraction terminal only when its exact
  admission observation identity is in `seen_observation_ids`. That proves the
  package can never succeed because observation identities are single-use even
  after their evidence is retracted.
- A permanent active duplicate and a permanent used-then-retracted duplicate
  both release the target slot and make the old package terminally stale.
- An unrelated injected `TransactionRejectedError` remains retryable; the exact
  package retains the sole live reservation and commits after the transient
  rejection is removed.
- V4 final-binding, direct-core, stale-snapshot, replay/copy/transfer,
  atomicity, geometry, contradiction/retraction, and fail-closed production
  protections are retained.
- Bound and unbound V1, V2, V3, and V4 surfaces reject V5 packages.

A normalized source comparison (`V5`/`v5` mapped back to `V4`/`v4`) differs
from frozen V4 only in the contract declaration for the history dependency and
the permanent-rejection classifier changing from active learned identities to
append-only seen identities, plus its explanatory comment.

## Author Verification

CPU verification used one thread per numerical runtime with all accelerators
hidden.

- Focused V5 plus history-query suite: **19 passed in 71.24s**.
- Decisive three-branch retry/release subset: **3 passed in 12.77s**.
- Adjacent compatibility panel: **80 passed**.
  - V5, history-query, and frozen V4 suites: **34 passed in 137.29s**.
  - Full revisioned-memory and two-resolution projection suites:
    **46 passed in 38.97s**.
- `py_compile` for both sources and both focused test files: PASS.
- Normalized V4-to-V5 semantic diff audit: PASS with only the intended delta.

The focused suite covers active and tombstoned duplicate identities, transient
retry with same-snapshot reservation exclusion, direct-core final binding,
stale release, replay/copy/serialization/transfer rejection, atomic removal,
geometry invariants, legacy cross-calls, source isolation, strict memory
roundtrip, and fail-closed production state.

## Required Independent Review

A different agent must independently freeze these exact hashes and attempt to
falsify:

1. terminal release for a duplicate identity that was accepted and retracted;
2. preservation of exact-package retry and reservation exclusivity for
   unrelated transient rejection;
3. immutability, append-only behavior, and strict roundtrip of the history
   query;
4. final target binding across public, direct-core, copied, transferred,
   replayed, and V1-V4 legacy call surfaces;
5. atomicity and the retained physical geometry invariants;
6. fail-closed production, hardware, held-out, and promotion state.

No promotion decision may cite this author handoff as an independent PASS.
