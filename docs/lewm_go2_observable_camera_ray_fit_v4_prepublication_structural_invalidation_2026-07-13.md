# V4 N5 prepublication structural invalidation

Date: 2026-07-13

Status: **terminal prepublication structural invalidation; no execution authority**

## Decision

The immutable `development_fit_v2` seed `20260710`, `N=5` result is not a
valid predecessor for canonical metric finalization. The frozen validator was
not changed. The result, checkpoint, reservation, and completion artifacts were
not changed.

The matched evaluation stores a total loss of
`0.27940133213996887`. The frozen equal-weight formula applied to its four
stored components gives `0.27940132907242515`. The signed difference is
`+3.067543719037502e-09`, which exceeds the frozen absolute tolerance of
`1e-09` with zero relative tolerance. The frozen validator therefore raises:

`ValueError: V4 matched evaluation losses are inconsistent`

The wrong-RGB evaluation is internally consistent: stored
`2.0213493436574934`, computed `2.021349344518967`, signed difference
`-8.614735591550016e-10`, within tolerance.

## Full validation

A CPU-only diagnostic copied the immutable result in memory, replaced only
`$.evaluation.matched_rgb.losses.total` with the computed value, recomputed the
copy's enclosing content hash, and called the complete frozen result validator.
That counterfactual copy passed. This establishes that the frozen validator
detects no independent downstream failure after the single replacement.

The counterfactual is diagnostic only. It does not authorize mutation,
republication, metric finalization, or checkpoint use.

## Secondary finding

Two read-only reproductions also returned stable sorted-depth byte commitments
that differ from the immutable result while preserving the exact numeric
metrics:

- matched: immutable
  `a8ec842a10766b724b9ee4835c0e6866ce4b2323ccb7c33757c9f9d04ac20326`,
  recomputed
  `6014597b1c286c42e5e7caa0643a98141b9545809c325a40763c82caf99d9f08`;
- wrong-RGB: immutable
  `6ec4af60dd8f684bf6ef74339e4e439e7235d1a5fdf632aca0b79e77e95e1c86`,
  recomputed
  `1e161762ff2158664cee260ff65b903864e14cce3c7bc09a405336140eee5ec8`.

These are recorded as secondary observations, not as an exception to the
primary structural failure. Their descriptive reason is
`stable_rerun_sorted_depth_byte_commitment_replacement_with_exact_numeric_metrics`.

## Immutable evidence

- reservation file/content:
  `f5926ee9006df8d163a2d1a17882d82124608ddce319ea0fb5e80fcfe2c2a8aa` /
  `699b4e95ed05cb13a79fe6af8507fae5d987af9ff1977b0e4684f32742aa4943`;
- checkpoint file/declared content:
  `f1739c742f9c19d5e17753da504a547254eb6e1997bb1ac4eca8b188bbf1dcf0` /
  `589060417903167bbf9ce7605c906b25cd802edd73b79ec607c77403c6df305a`;
- result file/content:
  `39030bb7928a6b078b03156dc9e14fb206c60c73ab2acac88bfd307c5a65bbfa` /
  `8c38e13f411a5cd9b03362cb5ac98379875065f284a75ac894706944ff252b61`;
- completion file/content:
  `4fb9b5629f039ac16692ec6e171a8188f3bf8b7d052ac8cde26b8ac86c10f6af` /
  `48022dca829a73b7cbd3b665ac7679807825a9aefd56a48e752ae07e6eaa336f`.

The frozen gate source is
`aa51413edfea10a2d7c04b034033c83c78c27b1c08d2be1413f5917dc32e36ad`.
The CPU-only diagnostic source is
`e5903f8e4979b2dfe4c81440f617c2a06d603a5cf9e9f6552b9d80861682fef3`.
Its test source is
`1a357ddd32767048bcd38a1d5f292732592691b043392604cf9de519fafb1811`.
The diagnostic output content hash is
`6de176104c587d91bcd74e20d1de6a91f4c725c42441c5e046e700b3c1546f9e`.

The machine-readable terminal record is bound as:

- file SHA-256:
  `1744a50badd6c9f5c1ef4c8c3cbd05f8c0fc8acff4fbbf066e40e1f7de24f560`;
- content SHA-256:
  `7bdaae6ebb13b7d90290dfe07f5d48f403d29cad977f4a56c9ac7b8cfbcb8602`.

## Authority boundary

This record authorizes no checkpoint use, metric receipt, stage finalization,
later rung, second seed, training, G2, held-out access, runtime use, promotion,
or publication claim. The never-reviewed V3 verifier/finalizer drafts were
removed; they produced no receipt or gate and have no authority.

A fresh full-panel successor requires a separate reviewed amendment that binds
the exact machine-readable terminal record. It must not mutate or canonically
finalize this invalid N5 result.
