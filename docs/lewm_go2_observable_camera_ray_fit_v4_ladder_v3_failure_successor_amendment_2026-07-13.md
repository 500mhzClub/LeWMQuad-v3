# Observable camera-ray fit V4 ladder-v3 failure-successor amendment

Date: 2026-07-13  
Status: frozen implementation candidate; independent byte review required

## Purpose

The first ladder-v2 N5 attempt for seed `20260710` consumed its one permitted
`development_fit_v1` reservation and terminated without publishing a
checkpoint or result. The failure was caused by an exact-warning parser defect,
not a data, model, schedule, threshold, or resource-policy finding. This
amendment creates one clean successor namespace without modifying, deleting,
or reusing the consumed V1 attempt.

## Immutable V1 lineage

The only terminal V1 attempt is
`.generated/go2_observable_camera_ray_fit_v4/development_fit_v1/attempts/seed_20260710/n5`.
Its inventory is exactly `reservation.json` and `failed.json`:

- reservation file SHA-256:
  `115e3a4e0ad7db7f5bd6b01c7ddde29d79563600ffb84ef77a0c585f009e854e`;
- reservation content SHA-256:
  `ca458f9371a211017f1b7a710b41508e2219a1afe19516ace2553a8eaa4d15dd`;
- failure file SHA-256:
  `6eb1becc195165e5fb49c1d222cac301f4169f301a48245d23a2b8213363af48`;
- failure content SHA-256:
  `7c1fe8f1ea73d8caef33debd9076bc3ddcacfaf337ec2a0000cec64f678c21e4`.

The failure receipt is terminal, reports `execution_failure` / `runtime`,
binds the reservation above, and records partial-artifact removal. There is no
V1 checkpoint, result, completion, gate, or metric-verification artifact. V1
is read-only historical evidence and must never be used as a V2 output root.

## V2 execution namespace

The successor contract is `observable_camera_ray_fit_v4_ladder_v3`. All new
attempts, gates, and metric receipts live only under
`.generated/go2_observable_camera_ray_fit_v4/development_fit_v2`.
Every V2 reservation binds the immutable V1 lineage above. Each original
seed/rung pair receives exactly one fresh V2 attempt; a V2 attempt directory is
never removed or reused after reservation.

The ladder remains `(5, 16, 32, 320)` for seeds `(20260710, 20260711)`. Dataset,
target partitions, model, parameter count, steps, batch sizes, optimizer,
learning rate, weight decay, loss weights, thresholds, ordering, seed gates,
wrong-RGB control, resource policy, and all trainer/metric license fields are
unchanged. No result-derived tuning is authorized.

## Warning normalization

The kernel inventory remains exactly `grid_sampler_2d_backward_cuda` and
`scatter_add_cuda_kernel`. An observed warning is accepted only when it is an
otherwise byte-exact member of the existing allowlist, optionally followed by
exactly one trailer of this form:

` (Triggered internally at /pytorch/aten/src/ATen/Context.cpp:<positive ASCII decimal>.)`

The parser removes only that trailer for allowlist comparison. It retains both
the raw warning and normalized warning, plus the parsed numeric source line.
Changed kernel names, warning text, punctuation, spacing, path, filename,
non-decimal or zero/leading-zero line numbers, duplicate trailers, and any
other suffix remain fatal. This amendment does not permit new warning kernels
or warning bodies.

## Authorization boundary

This amendment grants no execution by itself. It does not authorize G2,
held-out, runtime, promotion, aggregation, checkpoint reuse, or any widening of
the already reviewed narrow development-fit and metric-verification licenses.
The complete V2 source map, authorization, and review metadata require an
independent byte review before V2 N5 may be launched.
