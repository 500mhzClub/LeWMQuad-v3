# V15 extended-horizon integrity replacement V1 preregistration

Date: 2026-07-29

Status: preregistered integrity replacement only; no replacement reservation,
training, checkpoint, qualification, calibration, G2, navigation, or held-out
access has occurred.

## Trigger and completed predecessor

- The original V15 attempt is terminal and cannot be retried or resumed. Its
  complete independently audited result is frozen in commit
  `51cfeb7fd5dbc1743bf043d21f350937755c0647` at
  `docs/lewm_go2_rgb_unified_ray_survival_joint_jepa_v15_extended_horizon_terminal_failure_result_2026-07-29.json`,
  with SHA-256
  `03ec7227f1a4deb072b5f059568d6655648bd0587f3c0991978f0d2555a4842d`
  and byte count `4399`.
- That attempt published exactly one initialization trace row and the update-0
  observation, then terminally failed at `train_update_1` before its first
  joint-training update. Completed accounting is zero updates, zero
  presentations, zero optimizer steps, zero EMA steps, zero backward calls,
  and zero predictor forwards. No checkpoint or success receipt exists.
- The immutable exception-message SHA-256
  `2d6a68d12cbf21023c3afe53179efb3bc53d7b5b1ae572e4f86447a032ce24a6`
  exactly identifies `V13 training microbatch validator is absent`.
- The source cause is complete: the V15 training adapter exported the frozen
  V13 public `__all__`, while the frozen executor separately obtains the
  private callable `_validate_microbatches_v13` before every update. The
  private callable was present in the privately loaded frozen training module
  but absent from the adapter namespace.

The original attempt therefore did not test the longer-horizon hypothesis.
This is a distinct science-identical integrity replacement, not a retry or
resume of that consumed attempt.

## Sole permitted implementation change

- Expose the exact already-loaded frozen callable with the equivalent binding
  `_validate_microbatches_v13 = _training._validate_microbatches_v13` in the
  V15 training adapter.
- Do not copy, wrap, alter, intercept, or add behavior to that callable. Do not
  add it to the scientific public `__all__`. Its object identity and function
  globals must remain those of the privately loaded frozen V13 training core.
- Update only replacement evidence selectors, preregistration identity, and
  the fresh output root needed to distinguish the replacement attempt.

Before authority, a focused CPU synthetic regression must invoke the real
frozen executor microbatch-validation path with the repaired adapter and prove
both exact callable identity and rejection of a malformed batch. Import and
tests remain source-only and may not open scientific data or query a GPU.

## Frozen scientific identity

Preserve the original V15 preregistration
`af0f786841b1404d1f42542b507ad198ee574250` exactly, including:

- the exact V14 RGB unified ray-survival joint-JEPA model and parameter counts;
- constructor, schedule, execution, projection, and bootstrap seeds;
- fresh N320 initialization with no predecessor state or checkpoint;
- train and checkpoint-selection roles, labels, RGB, camera metadata, and
  authority-bound runtime inputs;
- float32 AdamW, learning rates, betas, epsilon, weight decay, route-wise
  clipping, four `B=4` microbatches, losses, weights, optimizer, and EMA;
- the same 16,000-presentation frozen base schedule repeated exactly once in
  memory, with one continuous model/optimizer/EMA/accounting/RNG trajectory;
- observations at updates `0`, `100`, `400`, `1000`, `1400`, and `2000`;
- the unchanged V14 update-400 gate, the V15 update-1,400 feasibility gate
  with twelve freshly recomputed controls, and the unchanged final gate at
  update 2,000;
- the exact cap of 2,000 updates and 32,000 presentations; and
- checkpoint publication only after a complete update-2,000 pass.

No model, architecture, tensor operation, data, schedule element, seed, loss,
coefficient, threshold, metric, control, initialization, optimizer, EMA,
observation, stopping rule, accounting multiplier, or scientific
interpretation may change.

## One-shot replacement identity

- Schema/evidence prefix:
  `lewm_go2_rgb_unified_ray_survival_joint_jepa_v15_extended_horizon_integrity_replacement_v1`.
- Exact fresh attempt root:
  `.generated/go2_rgb_unified_ray_survival_joint_jepa_v15_extended_horizon_integrity_replacement_v1/attempt_v1`.
- There is exactly one integrity-replacement attempt. The root must initially
  be absent. Retry and resume are false.
- The original V15 output is closed. Only its committed source-only terminal
  result may be used as replacement evidence; no original runtime artifact,
  model state, tensor, or checkpoint may be consumed.
- Any source, authority, reservation, accounting, custody, exception, or
  scientific gate failure is terminal and publishes no failed checkpoint.
  No second automatic integrity replacement is preregistered.

## Authority boundary

Implementation, focused regression, recursive source closure, independent
source review, narrow clean-export certification, and one-shot execution
authority must be frozen before reservation. Until a complete update-2,000
development pass, probability calibration, G2, navigation, held-out, sealed,
production, promotion, deployment, retry, and resume remain forbidden.
