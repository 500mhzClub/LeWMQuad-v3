# V18 object-space height-volume integrity replacement V2 preregistration

Date: 2026-07-30

Status: preregistered science-identical integrity replacement only. No V2
reservation, scientific input access, GPU training, checkpoint, qualification,
calibration, G2, navigation, or held-out access has occurred.

## Trigger and terminal predecessors

- The original V18 attempt and integrity replacement V1 are terminal. Neither
  may be retried or resumed.
- The original attempt's complete failure audit remains frozen in commit
  `0c0b0804730028bdd5dadf4e5180685b4bc33e0e` at
  `docs/lewm_go2_rgb_object_space_height_volume_joint_jepa_v18_terminal_failure_result_2026-07-30.json`,
  SHA-256
  `a04b90e31298d5aa0a0764478ce4794d21f5120b981c6f16edac5fda103ee66f`,
  byte count `5828`.
- Replacement V1's complete failure audit is frozen in commit
  `8acbb240a59c22d65ab5082a81596dcc24de86ee` at
  `docs/lewm_go2_rgb_object_space_height_volume_joint_jepa_v18_integrity_replacement_v1_terminal_failure_result_2026-07-30.json`,
  SHA-256
  `4b6246a94bd5ce8265807d255620deb46fb32406dfc1c92c396e87aa42bc8dc8`,
  byte count `7828`, content SHA-256
  `3d864460c94a22408bf4c52d0f57c7dc5b50ab61ceefdb3196e169ccb3b3d560`.
- Replacement V1 passed model construction, initialization, source and input
  rehashing, structural integrity, and update-0 observation. It then built all
  four first-update microbatches but failed before the inherited microbatch
  validator and before `joint_training_update_v13`.
- It completed zero training updates, zero completed training presentations,
  zero backward calls, zero optimizer steps, zero EMA steps, and zero training
  predictor forwards. Its update-0 values are an untrained diagnostic only; no
  scientific gate was reached and no checkpoint was published.
- Its immutable exception-message SHA-256
  `2d6a68d12cbf21023c3afe53179efb3bc53d7b5b1ae572e4f86447a032ce24a6`
  exactly identifies `V13 training microbatch validator is absent`.
- Two independent exhaustive caller-surface audits found all 17 executor API
  attributes, all 21 model-module requirements plus 11 model methods, and all
  25 public training attributes present. The sole missing runtime attribute is
  the private `_validate_microbatches_v13` bridge. An in-memory exact-object
  bridge passed the inherited engine guard, and the identical repair has a
  frozen V15 precedent.

This is a distinct science-identical integrity replacement, not a retry or
resume of either consumed attempt.

## Sole permitted implementation correction

- Add exactly
  `_validate_microbatches_v13 = _base._validate_microbatches_v13` to the V18
  training adapter after its frozen public re-export.
- Do not wrap, copy, rename, intercept, or mutate the callable. Preserve exact
  object identity and the private frozen V13 function globals.
- Keep `_validate_microbatches_v13` private and outside the V18 `__all__`.
- Add source-only focused regressions proving exact identity, exclusion from
  `__all__`, successful traversal of the real inherited engine validator with
  four valid synthetic microbatches, rejection of malformed input, and
  completeness of the exact 26-name inherited training caller surface.
- Change only replacement evidence selectors, preregistration identity,
  schema prefix, and fresh output/source roots needed to distinguish V2.

No model code, tensor operation, architecture, data field, loss, coefficient,
seed, schedule element, optimizer setting, EMA operation, observation, metric,
control, threshold, stopping rule, or accounting multiplier may change. No
additional compatibility export is permitted by this preregistration.

## Frozen scientific identity

Preserve original V18 preregistration commit
`5522b226e845907b091ff98ebac3b6f6315a4ca7` and replacement V1
preregistration commit `402f61522d59943e0def9df0b90ebf785867d366`
exactly, including:

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
- the exact update-100 health checks, five-check update-400 falsification gate,
  and unchanged V14 final gate;
- exactly 1,000 maximum updates and 16,000 maximum presentations; and
- checkpoint publication only after a complete passing update-1000 result.

There is no model or data retry, hyperparameter search, architecture change,
automatic resume, extension, or gate reinterpretation.

## One-shot V2 identity

- Schema/evidence prefix:
  `lewm_go2_rgb_object_space_height_volume_joint_jepa_v18_integrity_replacement_v2`.
- Exact fresh attempt root:
  `.generated/go2_rgb_object_space_height_volume_joint_jepa_v18_integrity_replacement_v2/attempt_v1`.
- Exact clean source root:
  `/home/andrewknowles/Workspace/LeWMQuad-v3-v18-object-space-height-volume-integrity-replacement-v2-source`.
- There is exactly one V2 attempt. Both roots must initially be absent. Retry
  and resume are false.
- Both predecessor outputs are closed. Only their committed terminal failure
  results and preregistrations may be used as source-only identity evidence;
  no predecessor runtime artifact, model state, tensor, or checkpoint may be
  consumed.
- Any V2 source, authority, reservation, custody, exception, or gate failure
  is terminal and publishes no failed checkpoint. No V3 integrity replacement
  is preregistered.

## Authority boundary

Implementation, focused regression, recursive source closure, independent
source review, narrow clean-export certification, and one-shot execution
authority must be frozen before reservation. Until a complete passing
update-1000 development result, probability calibration, G2, navigation,
held-out, sealed, production, promotion, deployment, retry, resume, and
extension remain forbidden.
