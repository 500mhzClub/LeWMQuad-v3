# Shared JEPA V5 output/loss correction candidate

Date: 2026-07-13

Status: **PASS for final model/one-shot source bytes; no checkpoint or gate licensed**

## Failed gate

The first V5 source review correctly closed one-shot inference/finalization, but
a later integration audit found two model-contract defects:

1. the output contract described pixel-ray tensors as `(B,H,W,D)` although the
   V4 head and runtime tensors are `(B,D,H,W)`; and
2. the joint-training API accepted only the derived 0.10 m hierarchical raster
   loss, allowing training to discard the three raw V4 source/ray objectives
   needed to preserve calibrated 0.05 m physical evidence.

## Candidate mechanism

- The output schema is versioned to
  `lewm_go2_shared_observable_camera_ray_jepa_output_v2` and records both pixel
  tensors as `(B,D,Hray,Wray)`.
- `ObservableCameraRayV4FrameSupervisionV5` carries raw pixel-hit/range and
  ground-support labels plus the derived-raster labels for each frame.
- Joint training reconstructs, inside the model, the exact four-equal V4
  objective for current and next frames:
  `ordered_first_hit_nll`, `target_bin_offset_smooth_l1`,
  `ground_clear_distance_state_balanced_bce`, and
  `derived_raster_hierarchical_bce`.
- The old hierarchical-only method remains a diagnostic, but its result is not
  accepted by `combine_joint_losses`.
- Ground visibility must exactly match the model's deterministic calibration
  before any complete V4 loss is constructed.

## Candidate identities and tests

- model source SHA-256:
  `b438295d7ec5cb0897cc953a229f461da7fca16322c4c936555d37833a36e4b9`
- model test SHA-256:
  `848aa8be369b89c973a4da916f9c7abeff47eca12aceb4304cf612ed4d53227b`
- unchanged one-shot core SHA-256:
  `32ddaa83a1120c6b4610863020b4ff4d6dda94b1f8d37dafa2eb5b7740781a2f`
- unchanged one-shot test SHA-256:
  `1ae41c32a1c9c4a9dbf91c5941d1519acb301f70c8418b9b2caf91a6ec3eb798`
- targeted regression result: `3/3` passed
- combined V5 model/one-shot result: `51/51` passed in `6.92 s`
- bytecode compilation: passed

Tests used one CPU thread per native numerical library. No dataset, G2, G3,
held-out input, GPU, checkpoint, or production identity was opened or changed.
All six V5 production identities remain `None`.

## Review rule

This candidate could replace the earlier V5 model-source identity only after a
different reviewer reproduced the runtime shapes, loss arithmetic, gradients,
fail-closed input behavior, and unchanged one-shot boundary. The review below
satisfies that source rule; production identity binding remains a later step.

## Different-agent final-byte review

The final bytes passed independent review at the identities above. With GPU
visibility disabled and native numerical threads capped, the combined suite
passed `51/51` in `6.84 s`; bytecode compilation also passed. The reviewer
independently reproduced `(B,D,H,W)` runtime and metadata order, exact per-frame
`0.25` weighting of all four V4 components, nonzero component-specific and
shared-encoder gradients, rejection of an actual hierarchical-only loss
package, and fail-closed behavior for changed ground visibility, noncanonical
no-hit distance, and invalid raster labels. No alternate joint-training
callsite exists outside the reviewed source/tests.

The one-shot core/test bytes were unchanged and all six production identities
were runtime-asserted `None`. This PASS closes source readiness only. It does
not authorize data access, training, G2/G3 input, checkpoint publication,
runtime use, held-out access, or promotion.

## Later lifecycle reopening

The model/output/loss PASS above remains unchanged. A later first-principles
integration audit found an execution-order cycle in the separately reviewed
one-shot authority: it required hashes of future runner/finalizer outputs and
could not publish the G2-qualified G3 candidate before G3. The one-shot hash in
this document is therefore historical rather than the current lifecycle
candidate. The staged replacement and its new independent-review requirement
are recorded in
[`lewm_go2_shared_jepa_v5_staged_lifecycle_candidate_2026-07-13.md`](lewm_go2_shared_jepa_v5_staged_lifecycle_candidate_2026-07-13.md).
