# RGB Post-Action Projective-Support Corridor Joint-JEPA V1 — Label-Preflight V3 Schedule-Schema Adapter Amendment

Date: 2026-07-28
Status: AUTHORIZED SCIENCE-IDENTICAL INTEGRITY SUCCESSOR
Governing preregistration commit: `8a52adb77d30cb98a6dd086037e6f7c296d76d63`
Reviewed V2 adapter source-freeze commit: `2788ee7ab7eb68c7f9e424f875fa87624e1eff8a`

## Terminal V2 preparation receipt

V2 reserved its exact label-preflight root and stopped while validating schedule metadata, before label materialization, geometry joins, source-scene or RGB access, tensors, model construction, GPU work, training, calibration, or selection.

- Reservation: `.generated/go2_post_action_projective_support_labels_v2/reservation.json`, 2,362 bytes, file SHA-256 `48eaec32a56bf0f872c0141ed359f2b673653c71bc76b5db96f4cf040b4bb165`, content SHA-256 `2cce455b5bf302cd4b43a263caf9b427b8b9512f1388b2eeb00dbf655939e803`.
- Failure: `.generated/go2_post_action_projective_support_labels_v2/failure.json`, 2,417 bytes, file SHA-256 `4fd4e3ec067564a423e8dba41a75862df5b3c5051d4ae2a3ca8b015936a18ecd`, content SHA-256 `7b3cd79f76924ad12907303ca1d214bf260ace9d64c63bed5fa5814a71e74528`.
- Failure phase: `prepare_execution_binding`; error: `frozen presentation schedule identity changed`.
- Access ledger: exactly one open each for the raw manifest, pairs, endpoints, audit, and frozen schedule; all geometry, source-scene, RGB, checkpoint, runtime-output, tensor/GPU/training, G2, navigation, held-out, sealed, and production counters remained zero.

The V2 root is terminal. It must not be removed, replaced, resumed, or reused.

## Exact adapter defect

The frozen schedule file itself passed its exact file SHA-256 `08f54578febbc182d936a999d6cf86263b8cd03a5f640da064c1538dd53dc270` and content SHA-256 `274c0cbd9a87cbbc5bbc3123fff046f02ac3555014b5ec750d4a32b552650a15` bindings. Its owning schema is `lewm_go2_shared_jepa_v5_matched_training_v4_schedule_v1`; it stores the ordered train-pair identity as `ordered_pair_ids_sha256`, rather than embedding an `ordered_train_pair_ids` list. The V2 loader incorrectly required `lewm_go2_shared_jepa_v5_full_training_v4_schedule_v1` and that nonexistent embedded list.

The tracked pure adapter `lewm/benchmarks/go2_shared_jepa_v5_multires_probe_v2_schedule.py` and prior terminal audits establish that this is a schema-only V4-to-V1 representation difference. They bind the unchanged seed `20260713`, 4,262 train pairs, 128,000 presentations, exact index identity `a6f4fda5eb570336fb360631af3629832cccbe4cba21bdbb325dcb8a21963663`, exact ordered-pair identity `74b90f10347a89d2151c4f65f76d6fc3c6a94fb3e8caa350d2a92e934e80840a`, and exact first-16,000 prefix `3f7b5799e855c3d218dcc62428f26ae0f9577c0dd4b04af5156d439a6f81e528`.

## Authorized V3 correction

Exactly one label-preflight V3 successor is authorized on fresh write-once V3 paths. It may make only these operational changes:

1. Bind and validate the exact terminal V1 and V2 reservation/failure receipts before reserving V3.
2. Accept only the exact frozen matched-training V4 schedule schema and complete field set.
3. Validate its exact file/content hashes, seed, dimensions, full index hash, and registered schedule identity hashes.
4. Reconstruct the ordered train-pair IDs from the already validated raw pair index and require their canonical hash to equal the schedule's `ordered_pair_ids_sha256`.
5. Return the exact first 16,000 indices only after proving they are an unchanged prefix; do not rewrite, regenerate, reorder, filter, resample, replace, or extend them.
6. Use new V3 label reservation, builder-binding, preflight-receipt/failure, source-manifest/review, and final execution-binding paths.

No model, data, role partition, input, label geometry, train-only prior, wrong-RGB mapping, seed, schedule index, initialization, optimizer, loss, coefficient, threshold, control, calibration/selection order, update accounting, or cap may change. Preserve the joint `S + P + Q + R` objective from update 1, maximum 1,000 updates and 16,000 presentations, and one training attempt with no retry or resume.

The V3 source must pass the existing recursive source review and focused synthetic suite before metadata preparation. A V3 preparation or materialization failure is terminal and does not itself authorize a V4. Training remains unauthorized until the V3 model-free label/oracle preflight passes and a fresh exact execution binding is written.
