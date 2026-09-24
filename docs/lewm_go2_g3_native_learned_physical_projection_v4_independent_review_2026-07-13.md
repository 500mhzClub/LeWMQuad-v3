# Go2 G3 native learned physical projection V4 independent review

Date: 2026-07-13

Verdict: **BLOCK**

Reviewer: `/root/v4_full_panel_v3_author`

The frozen V4 candidate is not approved for downstream integration. Its final
target checks, direct-core checks, transient retry, identity/adapter isolation,
atomic memory transaction, geometry invariants, and production closure passed
independent probes. One reachable append-only identity-history case falsifies
the permanent-rejection release rule.

## Frozen artifacts reviewed

| Artifact | SHA-256 |
| --- | --- |
| `lewm/planning/native_learned_physical_projection_v4.py` | `66486f70f0998502f36e16e496f1c76d11cd117176046e6de433db911473f16a` |
| `lewm/tests/test_native_learned_physical_projection_v4.py` | `df9b89778adea21da70b89004da41a01354b7086dd25eab5961f3a5bb1e0abb2` |
| `docs/lewm_go2_g3_native_learned_physical_projection_v4_author_handoff_2026-07-13.md` | `79407230f17714634ec0cb492fbf822131bf4aa958b140ffa49e0b95d027cbce` |
| `lewm/tests/test_native_learned_physical_projection_v4_independent_review.py` | `e598ee440ca92810c6776f6fdc8492a01406d6b4803d6e929368bcd31170b312` |

Candidate and author-test bytes were not edited.

## Blocking finding

V4 catches a late `TransactionRejectedError` at source lines 1219-1231. It
marks a retraction reservation terminal only when the retraction observation
ID is currently present in `memory.learned_observation_ids`. That property is
an active-record view, not the memory's indelible observation-ID history.

`RevisionedPhysicalMemory` rejects any ID already in its private append-only
`_seen_observation_ids` set, including IDs whose learned evidence was later
retracted. Such an ID is permanently unusable but absent from
`learned_observation_ids`.

The independent reproducer exercised this without mutation or monkeypatching:

1. commit a valid V4 projection target;
2. commit a learned observation whose ID equals the next V4 retraction ID;
3. retract that colliding observation, leaving the ID indelibly used but no
   longer active;
4. issue the V4 retraction on a current snapshot;
5. observe the permanent `duplicate observation identity` rejection;
6. retry the old package and receive the same memory rejection, not terminal
   replay rejection;
7. attempt a fresh retraction on the same still-current snapshot and find the
   target slot still occupied by the impossible LIVE reservation.

The rejected commit did not change memory revision and the target remained
active, so a fresh package should have been available. The issue is solely the
incorrect permanent/transient classification boundary.

## Passed adversarial surfaces

The distinct review suite also confirmed:

- bound and unbound `commit` and `_commit_core_v4` all recheck the target's
  exact issued digest immediately before applying the transaction;
- an unrelated injected `TransactionRejectedError` leaves exactly the same
  package retryable and prevents a competing same-snapshot package;
- copy, deepcopy, pickle reconstruction, slot reconstruction, cross-adapter
  transfer, replay, and bound/unbound V1-V3 commit calls reject;
- successful removal is one physical-memory revision and preserves unrelated
  active learned evidence;
- production identities remain unset and every surface is development-only,
  hardware-forbidden, and promotion-forbidden.

## Verification

CPU commands disabled external pytest plugins, capped OMP, MKL, OpenBLAS, and
NumExpr to one thread, and hid HIP, CUDA, and ROCr devices.

```text
independent V4 review: 6 passed, 1 failed in 29.87s
frozen author V4 suite: 15 passed in 68.69s
py_compile independent test: passed
git diff --check independent test: passed
```

The one independent failure is the blocking tombstoned-ID case above. No G2,
held-out, runtime, hardware, production, or navigation-result data was opened.

The canonical BLOCK JSON has content SHA-256
`38f22ba9e230d2ba8c2370a8830025a394733c0b353645b177fb402876edd9f7`.

## Required closure

An additive successor needs a narrow immutable public view of the physical
memory's append-only observation identity history. A rejected retraction whose
exact observation ID is in that history must become terminal and release its
target slot, whether or not that ID remains active. An unrelated rejection
must remain retryable. Exception text is not an acceptable classifier.

This BLOCK grants no downstream integration, G2, held-out, navigation,
runtime, hardware, production, or promotion authority.
