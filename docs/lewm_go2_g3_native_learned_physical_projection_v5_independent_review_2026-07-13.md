# Go2 G3 native learned physical projection V5 independent review

Date: 2026-07-13

Verdict: **PASS**

This is an independent review of the additive V5 development candidate. The
review used only synthetic fixtures and read-only candidate artifacts. It did
not access held-out data, exact simulator labels, G2 payloads, hardware, or a
production runtime, and it does not authorize any of those surfaces.

## Frozen candidate

- `lewm/planning/native_learned_physical_projection_v5.py`
  - SHA-256: `5ccd22e83c83a4c41db11286d31d417fe7af5615ebd7e62e51d7719d5378eca1`
- `lewm/planning/revisioned_physical_configuration_memory.py`
  - SHA-256: `bb05f957e0443e0c1e8405042b97c61948746a66040e84690e12b0a10887d483`
- `lewm/tests/test_native_learned_physical_projection_v5.py`
  - SHA-256: `e5f0d30b96d1da525ac004ded1eac6bcca96330657d92571594914c548a6d077`
- `lewm/tests/test_revisioned_physical_memory_seen_observation_ids.py`
  - SHA-256: `20860a1abca8848a5951481ce167da501420ce27ad21fba1c9821bc092459fa4`
- `docs/lewm_go2_g3_native_learned_physical_projection_v5_author_handoff_2026-07-13.md`
  - SHA-256: `6fb25e5af95b5794a45e67c1167c7c618fe9fd5e9aab22fbd83d37bd2da661cc`

## Independent artifact

- `lewm/tests/test_native_learned_physical_projection_v5_independent_review.py`
  - SHA-256: `dcf0b8c1d8c3a00d5e1f0d5e99838fd1da1416cfb4f53ef18acfcc0215b0ac77`

## Findings

No blocking defect was reproduced.

1. `seen_observation_ids` returns an immutable view of append-only accepted
   identity history. Querying it does not alter canonical serialization, old
   views do not change after later commits, and strict deserialize/serialize
   roundtrip preserves the exact bytes and history.
2. A retraction identity colliding with either active evidence or evidence that
   was accepted and later retracted is classified as permanently unusable. The
   rejected package becomes terminal, the memory transaction is atomic, and a
   fresh package can reserve and remove the original target.
3. An unrelated `TransactionRejectedError` does not consume or release the
   package. The exact package retains the exclusive reservation and succeeds
   after the injected rejection is removed.
4. Public, direct-core, bound, and unbound V5 commit entry points all recheck
   the final target binding. Mutating the target after issue is rejected before
   a memory revision or target removal.
5. Bound and unbound V1 through V4 surfaces cannot issue or commit V5
   authority. V5 remains standalone rather than composing an older adapter.
6. Successful retraction removes exactly the selected observation in exactly
   one memory revision while preserving unrelated active evidence. The retained
   physical-memory and two-resolution projection suites preserve the geometry
   and atomicity contracts.
7. All production identities remain unset, the production accessor fails
   closed, and all reviewed V5 surfaces remain development-only with hardware
   execution and production promotion unauthorized.

## Verification

CPU tests ran with one thread per numerical runtime and all accelerator devices
hidden.

- Frozen author V5 plus history-query suites: **19 passed in 71.38s**.
- Independent adversarial suite: **9 passed in 49.55s**.
- Combined retained panel, including the independent suite, frozen V4/V5,
  history query, full revisioned memory, and two-resolution projection:
  **89 passed in 226.88s**.
- Independent test `py_compile`: PASS.
- Independent test `git diff --check`: PASS.

## Decision

The V5 development candidate closes the V4 retired-identity reservation leak
without weakening retry behavior, final binding, version isolation, atomic
removal, or fail-closed production state. It passes this independent review for
the stated synthetic development boundary only.
