# V9 Content-Adaptive Dense Local Token Lift — Preimplementation Amendment

- Date: 2026-07-29, after preregistration commit
  `4704347c8085d038a34d6d498d518e1ad25f96f8` and before completed V9 source,
  tests, data access, GPU work, or model output.
- Purpose: remove one mathematically inert parameter and make the already
  selected 5x5 local-attention mechanism mechanically exact. This amendment
  changes no data, target, loss, seed, schedule, cap, gate, retry rule, or
  scientific hypothesis.

## Superseding attention definition

- Supersede the preregistration's statement that all four projections are
  biased. The query, value, and output projections are biased 64-to-64 linear
  maps. The key projection is a bias-free 64-to-64 linear map. A shared key
  bias would add the same scalar to every valid logit and cancel exactly under
  softmax, so retaining it would create a permanently zero-gradient tensor.
- The new online attention inventory is exactly seven parameter tensors and
  16,576 scalar parameters: Q weight/bias, K weight, V weight/bias, and O
  weight/bias. The target inventory is identical, detached, and frozen.
- Attention contains only those four projections, a four-head reshape with
  head width 16, multiplication by `1/sqrt(16)`, invalid-support masking,
  softmax across the 25 supports, weighted value summation, the output
  projection, and residual addition to the sampled centre token. There is no
  LayerNorm, FFN, positional or relative bias, learned temperature, extra
  residual gate, dropout, prototype, per-cell query, or other parameter.

## Exact sampling and schema boundary

- Every support is sampled with
  `torch.nn.functional.grid_sample(mode="bilinear", padding_mode="zeros",
  align_corners=False)`. Invalid proposed support coordinates are replaced by
  the exact normalized coordinate `(2.0,2.0)` before the packed grid sample.
- The inherited config field `samples_per_cell=4` remains unchanged only as a
  legacy clean-V4 config identity; V9 does not use it to size its support.
  V9 binds separate constants `support_side=5` and `support_count=25`.
- The centre support is valid whenever the inherited ground anchor is visible.
  For an invisible/all-invalid cell, the implementation may temporarily
  unmask only the zero-valued centre slot to keep softmax finite, but it must
  overwrite the resulting cell with inherited null evidence before any BEV
  consumer or receipt latent. Reported validity remains false for all 25
  supports on that cell.
- The V9 sampling receipt is separately versioned and reports latent,
  inherited `anchor_in_frustum`, `[B,64,64,25]` support validity,
  `[B,64,64]` cell validity, `[B,64,64,25,2]` safe support grids,
  fixed `[25,2]` token offsets, and `[B,64,64,4,25]` per-head attention
  weights. It has no learned-offset field and does not claim V4's four-sample
  receipt schema.
- The schemas preserved unchanged from V4 are semantic logits, predicted
  latents, swept-survival outputs, final evaluation metrics, and final result
  semantics. The exact retained visibility behavior is the boolean
  `anchor_in_frustum` mask plus fixed invalid semantic logits; there is no
  learned visibility logit.

## Exact initialization

- Save the caller CPU RNG state. Construct Q, K, V, and O in that order. Then
  use a private CPU `torch.Generator` seeded exactly `20260729` to apply
  Xavier-uniform gain 1.0 to Q, K, V, and O weights in that order. Set every
  present bias to exact zero. Restore the caller CPU RNG state exactly.
- No other tensor is initialized or changed by this seed. Every inherited
  tensor named by the preregistration remains bit-exact to the clean V4
  construction.
- The original requirement now applies to the exact seven-tensor inventory:
  every online Q/K/V/O parameter tensor must have a finite nonzero gradient by
  update 2, must be optimizer-owned exactly once, and must have an identical
  target counterpart with no gradient.

All other terms and stopping rules in the committed V9 preregistration remain
in force.
