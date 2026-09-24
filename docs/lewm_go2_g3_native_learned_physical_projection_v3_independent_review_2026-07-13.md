# Go2 G3 native learned physical projection V3 independent review

Date: 2026-07-13

Verdict: **BLOCK**

The frozen V3 candidate is not approved for downstream integration. Two
independent synthetic lifecycle probes show that its public transaction boundary
can be bypassed and that a permanent memory rejection does not restore the
retraction reservation state.

## Frozen artifacts reviewed

| Artifact | SHA-256 |
|---|---|
| `lewm/planning/native_learned_physical_projection_v3.py` | `c472b4792279a20fd7085189ea53d3a6c7d2c33343d86cc9063c73eea42f136f` |
| `lewm/tests/test_native_learned_physical_projection_v3.py` | `d5113b9c98ad88f42315ce326cc8bb2b12933b3fc37471419282886f32f19129` |
| `docs/lewm_go2_g3_native_learned_physical_projection_v3_author_handoff_2026-07-13.md` | `93cd66b03001abbf465053c1ae2277fa3c9daba8ee3332cd212e8d990a74722b` |
| `lewm/tests/test_native_learned_physical_projection_v3_independent_qa.py` | `4db838e6be6d9a77bc202679ef30d05b3a0dae435e148f15bbdf7bc1aa9ef3a8` |
| `lewm/tests/test_native_learned_physical_projection_v3_independent_review.py` | `2b1729fbf7c52df3caf286c77cf22417d093a8c4e89dffb813fe91ca6e3ced78` |

The implementation and test sources were not edited by this review.

## Blocking findings

### 1. Reachable internal commit bypasses final target binding

The public `commit()` checks reservation and active-target state before calling
`_commit_core_v3()` at source lines 1396-1411. The callable core at lines
1163-1193 validates only the retraction package and snapshot. It neither checks
the reservation's immutable target digest nor revalidates the exact active
target before applying the hidden transaction.

The independent probes issue a valid retraction, mutate and rehash the committed
target, and confirm public `commit()` rejects the changed target with `issued
content`. A direct call to `_commit_core_v3()` then succeeds and removes that
target from `learned_observation_ids`. A reachable method therefore bypasses the
advertised final pre-removal boundary.

### 2. Permanent duplicate identity rejection leaves a LIVE reservation

`commit()` transitions a reservation to terminal STALE only for
`SnapshotBindingError` at lines 1410-1417. A memory-level
`TransactionRejectedError("duplicate observation identity")` escapes without
calling `_mark_reservation_terminal_v3()`, so the target remains indexed by the
same LIVE reservation.

The QA probe pre-seeds the exact retraction observation identity. The first
commit rejects without changing memory and the target remains active. Retrying
the old package repeats `TransactionRejectedError`; issuing a replacement
against the same still-current snapshot raises
`NativeLearnedProjectionReplayError`. Creating a new projection snapshot can
proactively stale that record, but it does not repair the missing terminal
failure cleanup or the occupied same-snapshot reservation.

## Verification

Both independent files were run directly with `PYTHONPATH=.`. External pytest
plugins were disabled, HIP/CUDA/ROCr/HSA device visibility was empty, and
OMP/OpenBLAS/MKL/NumExpr threads were capped at one.

```text
independent QA suite:
  7 passed, 2 failed in 27.31s
  - terminal duplicate-identity cleanup: failed
  - reachable internal commit target binding: failed

independent review suite:
  5 passed, 2 failed in 22.54s
  - reachable internal commit target binding: failed independently
  - standalone-shape probe: test-harness AttributeError from reading an
    unmangled `__slots__` spelling; this is not a candidate pass or lifecycle
    finding

aggregate:
  16 tests collected; 12 passed, 4 failed
```

The second suite's late-memory-rejection probe obtains a new projection snapshot
before replacement issuance, so proactive stale detection makes that weaker
probe pass. The QA suite preserves the exact current snapshot and exposes the
LIVE reservation left by the permanent rejection.

All test inputs were synthetic development fixtures. No G2, held-out, runtime,
hardware, production, or navigation-result data was used.

## Required closure

A successor must make every reachable commit entry enforce the same immutable
target/reservation checks, and must terminally release a reservation after a
non-retryable memory rejection while preserving old-package replay rejection.
The successor requires fresh independent lifecycle tests and review before any
integration.
