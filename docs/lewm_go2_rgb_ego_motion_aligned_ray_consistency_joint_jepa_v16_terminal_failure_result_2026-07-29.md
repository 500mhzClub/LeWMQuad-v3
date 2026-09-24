# V16 terminal infrastructure-failure result

Date: 2026-07-29

Status: the original V16 attempt is consumed and terminal. It did not test the
ego-motion-aligned ray-consistency hypothesis.

- Preregistration commit: `2792343e14d3376add9d6adbda7f29346a3e9e29`.
- Execution-authority commit: `29706d7ee968e83d5f01938845e7a7b2bc3ffedc`.
- Exact attempt root:
  `.generated/go2_rgb_ego_motion_aligned_ray_consistency_joint_jepa_v16/attempt_v1`
  under the certified source root.
- The immutable failure receipt is `failure.json`, SHA-256
  `aff6f4d21e7c43e970bf452583af36262f1c7c5d7c4668f71e6904682b90c0fc`,
  byte count `1173`, content SHA-256
  `533ff26e7fa4dd48158917deba7dcae64a8e72782e8d47e58b7d4e390a15e789`.
- Terminal status:
  `FAIL_EXCEPTION_TERMINAL_REVIEW_RECOVERY_MILESTONE` at
  `train_update_1`, exception type `ValueError`.
- The exception-message SHA-256
  `9d0a8542eeef7e9cbaca31ac20af5086142eab97506a7876a3608b4088f86faf`
  exactly identifies `current_evidence floating evidence must be float32`.
- Initialization and the update-0 observation completed. The trace has one
  initialization row, SHA-256
  `5caa9a83046c2b1198933b0d7abd711b19250d4a220f70c9cfcdbfa9e3ee683f`,
  byte count `2786`. The update-0 metric has SHA-256
  `bf4863edf538f0fa88a983be49ce0b3f45f642be0c5e46e16da7abf7e248a8af`,
  byte count `8023`.
- No optimizer step, EMA update, recovery checkpoint, development checkpoint,
  continuation decision, or scientific gate occurred. The predictor forward
  preceding the failing validation grants no completed-update evidence.
- Runtime custody closed cleanly. The terminal access receipt has SHA-256
  `5eaf029f025483f255a58d7f1fb856023d87330bd680a8063c96c8196da4363e`,
  byte count `450258`. Probability calibration, G2, navigation, held-out, and
  sealed material remained unopened.

## Root cause

The V14 learned tensors used by the new loss are float32: first-hit hazards,
within-bin offsets, and ground-survival logits. Existing V14 query-geometry
metadata are intentionally float64: ground-query UV coordinates and target
distances. The V16 helper incorrectly applied a float32 schema assertion to
both the learned tensors it consumes and the unused geometry metadata.

A source-only production-shape reproduction generated the exact V14 auxiliary
evidence dtypes and reproduced the mismatch without opening scientific data.
The correction is therefore an obvious schedule-schema adapter fix: validate
the three learned tensors as float32 while preserving the existing float64
query metadata. It changes no model tensor operation, loss, data, seed,
schedule, threshold, optimizer, or 1,000-update / 16,000-presentation cap.

The consumed attempt is not retryable or resumable. At most one separately
preregistered, science-identical integrity replacement may use the narrow
schema correction and a fresh output root.
