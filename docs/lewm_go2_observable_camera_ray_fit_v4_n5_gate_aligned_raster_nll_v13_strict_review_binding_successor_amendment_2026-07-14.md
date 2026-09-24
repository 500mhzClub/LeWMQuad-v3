# Observable Camera-Ray Fit V4 N5 Gate-Aligned Raster-NLL V13 Strict Review-Binding Successor Amendment

Date: 2026-07-14

Amendment author: `/root`

Status: **source construction and different-agent review only; no exact authority**

## Trigger and terminal V12 source review

The source-free Camera V12 amendment froze the scientifically justified
gate-aligned raster-NLL successor before any V12 source existed. Its file
SHA-256 is
`77de8c69b1bef69ab3d1b976567eb20371f53d47d81af757ef8c7fdaade93c1b`.
The implementation author then froze this source-only closure:

| Role | Path | SHA-256 |
|---|---|---|
| retained V12 model/loss | `lewm/models/observable_camera_ray_evidence_v4_gate_aligned_raster_nll_v12.py` | `735563f811c5d7b9efb9e37dca8348825a8467bd0a059f83ab94d41d45d57662` |
| V12 policy | `lewm/benchmarks/go2_observable_camera_ray_fit_v4_n5_gate_aligned_raster_nll_v12.py` | `ad8a77c4f201f00891e7e6b45c395966eaa8f3723a3b2720d26eeb0b1ca23fc6` |
| V12 trainer | `scripts/train_go2_observable_camera_ray_fit_v4_n5_gate_aligned_raster_nll_v12.py` | `91018ecd28483fbbc3399eea70d720a9b327e7e03b4920dbe349ca9b81603d54` |
| V12 verifier | `scripts/verify_go2_observable_camera_ray_fit_v4_n5_gate_aligned_raster_nll_v12.py` | `f8814836c1073f13c563ba11035f806a0faa70be9a0d44b7d3e900350b1a8baf` |
| V12 executor | `scripts/execute_go2_observable_camera_ray_fit_v4_n5_gate_aligned_raster_nll_v12.py` | `4e4c45c85827ad4db6e65a4f02557fd6c5b1e9d97ada4ac4577cb0b6b099b521` |
| V12 synthetic proof | `lewm/tests/n5_gate_aligned_raster_nll_v12_synthetic_execution.py` | `1cbcb80d3f6bec5b9ce536d6b4fa9bad645d170a4be6b4e8d1b261ad5f5dc453` |
| V12 science tests | `lewm/tests/test_go2_observable_camera_ray_fit_v4_n5_gate_aligned_raster_nll_v12.py` | `98a11ec91865ff106dd943a6b6468ca227018d92db4a346fcbbe9497a7d8d099` |
| V12 lifecycle tests | `lewm/tests/test_go2_observable_camera_ray_fit_v4_n5_gate_aligned_raster_nll_v12_lifecycle.py` | `77b5d05373613220a0de1d78236659f0c038e9f1a91f3a4efbf5cbcaa73936c1` |
| V12 handoff | `docs/lewm_go2_observable_camera_ray_fit_v4_n5_gate_aligned_raster_nll_v12_implementation_handoff_2026-07-14.md` | `21d4858035225e2454a3e7fec3e71fb8571e4d69e7a592c5822c4a435b17b0b9` |

The eligible different-agent review is the canonical BLOCK at
`docs/lewm_go2_observable_camera_ray_fit_v4_n5_gate_aligned_raster_nll_v12_independent_review_2026-07-14.json`,
file SHA-256
`076855183730bcff58b507d8fde6c613a023b633681c7516daaf0d80b5e27158`
and canonical content SHA-256
`4a56c46ede9482f72b5ae304734e12a706d8f7075873b4e5de135f9fa6cc289d`.

The reviewer independently rehashed every candidate and predecessor, passed
the V12 suite `202/202`, retained V11 suite `190/190`, direct isolated-child
smoke, compilation, and whitespace checks. It found one fail-closed schema
defect in V12 policy `preflight_source_review`: nested source and proof binding
objects were accessed through `.get("path")` and `.get("file_sha256")` without
requiring their exact key set. A canonical, self-consistent review could add
`"unexpected_authority_claim": true` inside either nested binding and still
pass preflight. This violates the V12 amendment's required rejection of extra
review/source-hash schema values.

No V12 exact attempt, reservation, output root, checkpoint, RGB/data open, GPU
use, or downstream authority exists. The V12 BLOCK is terminal for those
source bytes.

## Preserved scientific contract

V13 is a source-only additive governance successor. It preserves the complete
V12 scientific experiment byte-for-value:

- fresh model initialization; no V11/V12 checkpoint open, hash, copy, load,
  warm start, comparison, repair, or reuse;
- seed `20260710`, exact N5 subset/targets/mappings and train-only role;
- retained hierarchical model, rasterizer, four V11 losses, and additive exact
  `0.25 * derived_raster_cell_nll` gate-aligned objective;
- exact gather, float32-epsilon clamp, negative log, all-cell mean, class/family
  non-gating diagnostics, and native/compatibility schemas;
- batch five, 4,000 updates, 20,000 exposures, AdamW LR/weight decay `1e-4`,
  float32/no autocast, clip norm `1`, final-update-only selection, 41 frozen
  diagnostics, and unchanged schedule hash;
- matched and cyclic wrong-RGB controls, independent verifier, isolated child,
  transaction, resource, failure, cleanup, and publication semantics; and
- the exact retained 26 checks, all thresholds, arithmetic, pass/fail meaning,
  and false later-rung/downstream/retry licenses.

V13 may not change a tensor operation, loss value, gradient, model parameter,
data mapping, schedule, metric, gate, threshold, diagnostic arithmetic,
publication field meaning, or hardware contract.

## Sole permitted correction

Every nested source or proof binding accepted by V13 source-review preflight
must be a plain `dict` whose key set is exactly:

```text
{"path", "file_sha256"}
```

The check must run before either value is consumed and before any data, RGB,
checkpoint, GPU, exact output, or governed source target is opened. It must
reject a missing key, any extra key, mapping subclass, non-string value,
noncanonical path, malformed digest, changed digest, duplicate path, or
source/proof role mismatch. The original V12 source-review object must remain
unmodified.

This exact-key check must cover every nested item in both the successor-source
and successor-proof binding classes. No filtering, normalization, key dropping,
truthiness coercion, permissive `.get`, or compatibility fallback is allowed.

## V13 source and proof namespace

Fixed implementation author:
`/root/camera_v12_gate_aligned_implementer`.

The retained model/loss source remains the exact V12 file and hash above. The
new production closure is:

1. `lewm/benchmarks/go2_observable_camera_ray_fit_v4_n5_gate_aligned_raster_nll_v13.py`;
2. `scripts/train_go2_observable_camera_ray_fit_v4_n5_gate_aligned_raster_nll_v13.py`;
3. `scripts/verify_go2_observable_camera_ray_fit_v4_n5_gate_aligned_raster_nll_v13.py`;
4. `scripts/execute_go2_observable_camera_ray_fit_v4_n5_gate_aligned_raster_nll_v13.py`.

The proof closure is:

1. `lewm/tests/n5_gate_aligned_raster_nll_v13_synthetic_execution.py`;
2. `lewm/tests/test_go2_observable_camera_ray_fit_v4_n5_gate_aligned_raster_nll_v13.py`;
3. `lewm/tests/test_go2_observable_camera_ray_fit_v4_n5_gate_aligned_raster_nll_v13_lifecycle.py`;
4. `docs/lewm_go2_observable_camera_ray_fit_v4_n5_gate_aligned_raster_nll_v13_implementation_handoff_2026-07-14.md`.

Canonical review:
`docs/lewm_go2_observable_camera_ray_fit_v4_n5_gate_aligned_raster_nll_v13_independent_review_2026-07-14.json`.

The only possible exact output root is
`.generated/go2_observable_camera_ray_fit_v4/n5_gate_aligned_raster_nll_v13/`.

## Required author and reviewer proof

All source tests are CPU-only with every accelerator selector hidden and use
only synthetic temporary roots. They must not open canonical RGB/data,
`.generated` payloads, V11/V12 checkpoints, or GPUs.

The author and future reviewer must:

1. rehash the V12 amendment, complete V12 closure/handoff, and canonical BLOCK;
2. mechanically compare V13 and V12 production ASTs after version renaming and
   permit only the strict nested-binding key-set correction and authority path;
3. prove a valid minimal source-review object still passes unchanged;
4. add one extra key independently to a nested source binding, recompute every
   enclosing canonical content hash, and require rejection;
5. repeat the attack for a nested proof binding and require rejection;
6. test missing keys, mapping subclasses, nonstrings, malformed/changed hashes,
   duplicate paths, noncanonical paths, and source/proof role swaps;
7. spy all data, RGB, checkpoint, GPU, output, and source-target openers and
   require zero calls on every invalid review;
8. rerun all `202` V12 tests, all `190` retained V11 tests, the exact loss/
   gradient/parity/diagnostic proofs, actual isolated verifier child, all 26
   unchanged checks, and lifecycle fault injections;
9. prove fresh init and V11/V12 checkpoint exclusion; and
10. run compilation, whitespace, source-identity, absence, and no-authority
    checks before freezing the handoff.

The V13 reviewer must have a `/root/` path and differ from `/root`, the
amendment author, fixed implementation author, V12 reviewer, and future exact
execution agent. The implementation author may not self-review.

## One-attempt lifecycle and non-authority

Only a canonical different-agent V13 review `PASS` binding every exact frozen
byte may authorize one future fresh V13 N5 attempt. The exact execution agent
must differ from implementation author and reviewer. The attempt must run once
on discrete GPU0 R9700 only, keep the Raphael iGPU at zero use, and remain
serialized with every `.generated` mutator.

No retry is permitted under success, numeric failure, runtime failure,
verification failure, publication failure, timeout, or interruption. A full
unchanged 26-check numerical pass may license only a later source-free ladder
design/review; it does not authorize later-rung execution, full Shared-JEPA
training, checkpoint use, held-out, G2, navigation, runtime, hardware,
production, promotion, or deployment.

This amendment grants only V13 source construction and different-agent review.
It grants no exact execution, checkpoint use, retry, data/RGB open, GPU use,
later-rung work, training, selection, calibration, held-out, G2, navigation,
runtime, hardware, production, promotion, or deployment authority.
