# V18 spatial-token delay-line Joint-JEPA V1 — overflow-safe route-norm integrity replacement V3

Date: 2026-07-31

Status: preregistered, source-only, execution denied pending implementation review, a certified narrow export, and a fresh one-shot authority.

## Reason for V3

The consumed batch-schema integrity replacement V2 fixed the inherited nested
V25→V24→V23→V21 physical builder and reached the real update-1 training
computation. Both B4 physical microbatches and all eight B2 memory
microbatches completed their registered forward and `torch.autograd.grad`
calls. The runner then terminated before parameter-gradient assignment or an
optimizer step with `V13 gradient norm or scale is nonfinite`.

The independently reviewed terminal result is
`docs/lewm_go2_v18_spatial_token_delay_line_causal_convolution_joint_jepa_v1_update_zero_gate_timing_integrity_replacement_v2_terminal_numerical_failure_result_2026-07-31.json`
at commit `dfa9693135fa59973c5dbd930a596fce32c1003b`.
The file is 14,344 bytes with SHA-256
`519385bdf2b61b68671086b5509a3d28b736bb349d5616de5dc11b1467048c8a`
and content hash
`67c7e27294ac671aeb6c65bb93d3321abda42c8e92942534bfbf582e1373719a`.
It records zero atomically completed updates and source-infers from the frozen
control flow and exact failure site that 24 presentations were attempted and
14 gradient calls returned before the route-norm failure.

The inherited helper squares and sums accumulated route gradients in float32.
An exact source-only CPU synthetic graph of the frozen persistence-initialized
H4 masked predictor alone, using the registered
`(full + 0.5 × masked) / 8` loss but no encoder, real batch, or eight-batch
accumulation, produced finite individual raw gradient tensors with maximum
magnitude `2.003031701502789e20`, an infinite legacy float32 norm, and finite float64
norm `1.2587330137152002e21`. The runtime receipt did not name its failing
route, so V3 must retain route-specific fail-closed checks rather than assume
that every runtime gradient is finite.

## Exactly authorized correction

Authorize exactly one fresh, science-identical V3 that:

- adds a V18-local overflow-safe route-norm helper without changing the shared
  frozen V13/V25 helper;
- for each named route, rejects any nonfinite individual raw gradient tensor
  with the route name in the error;
- computes the same mathematical L2 norm in float64 and the same
  `min(1, reciprocal(max(norm, float32_tiny)))` route scale, preserving the
  frozen helper's float32-tiny floor and exact applied scale 1 for a finite
  zero norm, then casts only the applied scale to the common raw
  route-gradient dtype;
- records route name, raw-gradient finiteness, maximum absolute raw gradient,
  preclip norm, and applied scale in every successful route receipt; any
  failure message must record the route, failing stage, raw-gradient
  finiteness, maximum magnitude when defined, and norm or scale when computed;
- routes all six V18 gradient buffers through that local helper and changes no
  gradient recipient or combination order;
- adds a CPU-only exact persistence/checkerboard/H4 regression using the
  registered full plus 0.5-weight masked loss divided by eight, proving finite
  raw gradients and legacy float32 norm overflow; it must also prove the
  overflow-safe norm and scale are finite, the scale is positive and equals
  `1 / norm` for this norm-above-one case, and the clipped route's float64 L2
  norm is within `1e-6` of one;
- adds a compact normal-range reference regression proving the new helper's
  norm and applied scale equal the frozen float32 helper within `1e-7` when
  the legacy reduction is finite;
- adds a tiny injected-NaN and injected-Infinity CPU regression proving the
  helper fails closed and includes the exact route name in its error;
- changes only the V3 schema, document, certified-source, authority,
  output-root, and experiment-arm identities needed for a fresh lifecycle;
- starts from a fresh initialization and new output root, without using any
  V1/V2 checkpoint, metric, recovery, optimizer, RNG, or model state.

Do not use `nan_to_num`, silently clamp nonfinite raw gradients, alter the
registered losses, or replace exact route normalization with a new gradient
policy. The correction is numerical integrity for the already registered
unit-L2 route clipping, not a new optimizer or objective.

## Science that must remain identical

V3 must preserve exactly the V2:

- model class, parameters, encoder, representation, memory predictor,
  persistence initialization, normalization epsilon, and checkerboard mask;
- K4 spatial-token FIFO, local causal depthwise Conv3D reader, action FiLM,
  shared recursive H4 prediction, and EMA target;
- data identities, train/selection splits, RGB loaders, seed `20260731`, and
  schedule order;
- optimizer, learning rates, parameter groups, gradient recipients,
  mathematical per-route unit-L2 clipping, EMA coefficient, and update order;
- physical objectives and full plus 0.5-weight masked-current memory JEPA
  losses;
- 16 memory and 8 physical presentations per update, one optimizer step, and
  one EMA step;
- wrong-action, reset, reverse, shuffle, persistence, and HOLD controls;
- update-zero gate timing, first absolute noncollapse enforcement at update
  250, and all update-250, update-500, and terminal thresholds;
- observation updates 0, 100, 250, 500, 750, and 1000; snapshots at 250, 500,
  750, and 1000;
- stage-A cap 500, terminal cap 1000, memory cap 16,000, physical cap 8,000,
  and combined cap 24,000 presentations;
- no probability-calibration, G2, navigation, held-out, or sealed access.

## Terminal rules

- If any individual raw route gradient is nonfinite, stop and close this V3;
  do not hide it with norm precision, clamping, or a retry.
- If the corrected named route is still numerically invalid, do not create a
  V4 norm-only integrity replacement. Any successor must materially change the
  mechanism implicated by the named failure; a learned-mask or zero-safe
  normalization successor is appropriate only if the memory route is
  implicated. An unrelated independently established infrastructure fault may
  be corrected only through a fresh reviewed lifecycle.
- If unchanged absolute noncollapse fails at update 250, close this exact
  delay-line mechanism.
- Passing update 250 does not guarantee continuation; every later frozen gate
  remains conjunctive.

## Current authority

This document grants source implementation and CPU-only synthetic-test
authority only. It grants no dataset/RGB payload, GPU, training, checkpoint,
recovery, navigation, probability-calibration, G2, held-out, sealed,
production, promotion, retry, resume, or execution authority.
