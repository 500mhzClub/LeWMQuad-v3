# Batched retained floor-patch query preparation

The completed receipt-copy replay reduced hold median controller time by about
6.7% and full-prefix total time by about 2.1%; every post-warmup call still
exceeded 100 ms. The completed earlier profile also identifies retained patch
coverage as a cost distinct from the already integrated current-frame index
cache. This preparation targets that historical-query calculation.

Implemented `BatchedRetainedFloorPatches` in
`lewm/batched_retained_floor_patch_development.py`. It inherits the original
append method and storage, batches small projection arrays across at most
32 historical frames, leaves pixel-prefix images in their original storage,
and queries them chronologically. It preserves the first complete witness
for every centre, all depth/pixel limits, floating-point margins, integer
prefix tests, copied witness data and negative qualification flags. An unused
later projection error falls back to chronological evaluation, preserving
early termination and errors on frames that actually remain necessary.

Validation:

- Initial component suite: 42 tests passed in 0.45 seconds.
- Final suite: 47 tests passed in 0.54 seconds, including three-axis rotations,
  representable values adjacent to depth thresholds, 4,096-frame histories,
  partial and missing floor pixels, mixed earliest witnesses, original query
  rejection messages, output isolation and unused/required arithmetic errors.
- Session 95273: source-only discovery and verification passed 1,977 bindings,
  inheriting the completed paired-copy source identity. No controller, model,
  sensor observation, benchmark process or native scene was launched by this
  preflight.

New source bindings:

| Path | SHA-256 |
| --- | --- |
| lewm/batched_retained_floor_patch_development.py | d16244f88818ade6dd9a07f2aeebadb9fadc99f6f2a303c8d9cd5fcf64121e5d |
| lewm/tests/test_batched_retained_floor_patch_development.py | 68ef3e5655eb7940b186bbdb4e855b2d46c73a4d9a47f635f012181fbd492768 |
| docs/go2_batched_retained_floor_patch_v1_2026-09-10.md | 1a5d6d30ee288c71d8ecf1107b92ce1ea8cdd51828c60157b9b5d5398fc469e6 |
| docs/go2_receipt_copied_anchored_prefix_result_2026-09-10.md | efbafd6a97d0afa9fdec91bcba888d5626b73d0c1b5dc692444ff34712a504ec |
| docs/go2_receipt_copied_anchored_prefix_verification_2026-09-10.json | 904022748b8bd0a7bd51e46f10b9b1b6e8fc4431b50bdc2c6b3ddcc4d152a34b |

This candidate is not installed in any controller. Next work is separately
named controller integration, exact normalization of implementation-only
identity, full raw decision/public-input/retained-state comparison, and paired
unprofiled measurement on the fixed original prefix. Do not infer a speedup
from test duration or synthetic projection comparisons. The native comparison
batch and its queued policy interventions remain unchanged.
