# V18 object-space height-volume integrity replacement V1 preregistration

Date: 2026-07-30

Status: preregistered science-identical integrity replacement only. No
replacement reservation, scientific input access, GPU training, checkpoint,
qualification, calibration, G2, navigation, or held-out access has occurred.

## Trigger and terminal predecessor

- The original V18 attempt is terminal and cannot be retried or resumed. Its
  complete failure audit is frozen in commit
  `0c0b0804730028bdd5dadf4e5180685b4bc33e0e` at
  `docs/lewm_go2_rgb_object_space_height_volume_joint_jepa_v18_terminal_failure_result_2026-07-30.json`,
  SHA-256
  `a04b90e31298d5aa0a0764478ce4794d21f5120b981c6f16edac5fda103ee66f`,
  byte count `5828`.
- The attempt completed schedule validation and model/optimizer construction,
  then failed while resolving the pre-update structural wrong-RGB probe. It
  completed zero observations, updates, presentations, optimizer steps, EMA
  steps, backward calls, and predictor forwards. It published no metric,
  checkpoint, or success artifact and therefore did not test the V18
  hypothesis.
- Its immutable exception-message SHA-256
  `fb849bcba3193b7d0c26a5ea1130ae7472b22b781f51902f1ed7113486af59d6`
  exactly identifies
  `module 'scripts.execute_go2_rgb_object_space_height_volume_joint_jepa_v18' has no attribute 'registered_wrong_rgb_mapping_v13'`.
- Two independent full compatibility-surface audits found exactly two missing
  executor-adapter aliases: the observed
  `registered_wrong_rgb_mapping_v13` helper and
  `flatten_physical_metrics_v13`, which the same frozen runtime would next
  require during update-0 physical evaluation. Both exact functions already
  exist in the privately loaded, reviewed V13/V14 engine.

This is a distinct science-identical integrity replacement, not a retry or
resume of the consumed original attempt.

## Sole permitted implementation correction

- Export the exact already-loaded private-engine callable
  `registered_wrong_rgb_mapping_v13` from the V18 executor adapter, with a V18
  alias permitted for inspection.
- Export the exact already-loaded private-engine callable
  `flatten_physical_metrics_v13` from the V18 executor adapter, with a V18
  alias permitted for inspection.
- Do not copy, wrap, intercept, or alter either callable. Object identity and
  function globals must remain those of the privately loaded frozen engine.
- Add one source-only focused regression that enumerates every executor API
  attribute directly consumed by the inherited custody runtime and proves
  the two repaired helpers have exact private-engine identity.
- Change only replacement evidence selectors, preregistration identity,
  schema prefix, and fresh output/source roots needed to distinguish the
  replacement attempt.

No model code, tensor operation, architecture, data field, loss, coefficient,
seed, schedule element, optimizer setting, EMA operation, observation,
metric, control, threshold, stopping rule, or accounting multiplier may
change.

## Frozen scientific identity

Preserve original V18 preregistration commit
`5522b226e845907b091ff98ebac3b6f6315a4ca7` exactly, including:

- the learned RGB unified ray field and explicit eight-height object-space
  volume as the sole JEPA perception state;
- the 64-by-64, 0.1-m XY grid, eight registered height centres, five voxel
  input channels, 8-channel 3D residual volume, height-major 64-channel
  flattening, and all parameter counts and initialization;
- the jointly trained online encoder, volume representation, semantic decoder,
  action-conditioned predictor, and stop-gradient EMA target;
- the same N320 initialization, train and checkpoint-selection roles, labels,
  RGB, camera metadata, and authority-bound runtime inputs;
- constructor, schedule, experiment, bootstrap, projection, and volume seeds;
- four microbatches of four, float32 AdamW, parameter groups, learning rates,
  clipping, losses and weights, one optimizer step, and one EMA step per
  completed update;
- observations at updates `0`, `100`, `400`, and `1000`;
- the exact update-100 health checks, the five-check update-400 falsification
  gate, and the unchanged V14 final gate;
- exactly 1,000 maximum updates and 16,000 maximum presentations; and
- checkpoint publication only after a complete passing update-1000 result.

There is no model or data retry, hyperparameter search, architecture change,
automatic resume, extension, or gate reinterpretation.

## One-shot replacement identity

- Schema/evidence prefix:
  `lewm_go2_rgb_object_space_height_volume_joint_jepa_v18_integrity_replacement_v1`.
- Exact fresh attempt root:
  `.generated/go2_rgb_object_space_height_volume_joint_jepa_v18_integrity_replacement_v1/attempt_v1`.
- Exact clean source root:
  `/home/andrewknowles/Workspace/LeWMQuad-v3-v18-object-space-height-volume-integrity-replacement-v1-source`.
- There is exactly one replacement attempt. Both roots must initially be
  absent. Retry and resume are false.
- The original V18 output is closed. Only its committed terminal failure
  result may be used as source-only identity evidence; no original runtime
  artifact, model state, tensor, or checkpoint may be consumed.
- Any replacement source, authority, reservation, custody, exception, or gate
  failure is terminal and publishes no failed checkpoint. No second integrity
  replacement is preregistered.

## Authority boundary

Implementation, focused regression, recursive source closure, independent
source review, narrow clean-export certification, and one-shot execution
authority must be frozen before reservation. Until a complete update-1000
development pass, probability calibration, G2, navigation, held-out, sealed,
production, promotion, deployment, retry, resume, and extension remain
forbidden.
