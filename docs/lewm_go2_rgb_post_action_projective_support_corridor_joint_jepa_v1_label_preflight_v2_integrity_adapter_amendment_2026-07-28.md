# RGB Post-Action Projective-Support Corridor Joint-JEPA V1 — Label-Preflight V2 Integrity Adapter Amendment

Date: 2026-07-28  
Status: AUTHORIZED SCIENCE-IDENTICAL INTEGRITY SUCCESSOR  
Governing preregistration commit: `8a52adb77d30cb98a6dd086037e6f7c296d76d63`  
Reviewed V1 source-freeze commit: `af6b30fc1d159aaf6de27eb0bdceaf63adf3d637`

## Terminal V1 preparation receipt

The first label-preflight preparation reserved its exact V1 root and stopped before labels, RGB, tensors, GPU, model construction, training, calibration, or selection.

- Reservation: `.generated/go2_post_action_projective_support_labels_v1/reservation.json`, 2,356 bytes, file SHA-256 `93cae83c890ffaa71aca7791306055be87927f89d60b4c3db99acd11c1f2806c`, content SHA-256 `d314e741fefc900df06f3a24e8d6a1700a0cb1d74c8748de93291249cd6b3041`.
- Failure: `.generated/go2_post_action_projective_support_labels_v1/failure.json`, 2,409 bytes, file SHA-256 `b01baa82df0b0f44e0c1f6b856cd99399daa154ce2f81c4481f5498388fda141`, content SHA-256 `ffc40032c6dea96307255c9557cff377386571fee5545d79e487e39af30947da`.
- Failure phase: `prepare_execution_binding`; error: `raw manifest identity or population changed`.
- Access ledger: exactly one `raw_manifest_opens`; every pair, endpoint, audit, schedule, geometry, source-scene, RGB, checkpoint, runtime-output, GPU/training, G2, navigation, held-out, sealed, and production counter remained zero.

The V1 root is terminal. It must not be removed, replaced, resumed, or reused.

## Exact integrity defect

The V1 adapter incorrectly used the earlier metadata-plan endpoint digest
`8130e961b7b5c04944b178fa4f73c1fa157776f7702ab5cdc213cf16c922f698`
as the raw endpoint-index order. The frozen V13 raw manifest and the canonical
sequence of parsed `endpoints.jsonl` row `content_sha256` values both bind the
raw endpoint-index order digest
`ab21c1a89b37ef60a056de390d59d3983705ab2e40de061d0cb163d1837e850f`.
The tracked reviewed matched-training loader and its negative representation
test establish that `8130…f698` is a plan digest and must be rejected for the
raw endpoint index.

## Authorized V2 correction

Exactly one label-preflight V2 successor is authorized on new write-once V2 paths.
It may make only these operational changes:

1. Validate the raw manifest field against `ab21…850f`.
2. Independently reconstruct the parsed endpoint JSONL row-content sequence and validate it against the same raw-index digest `ab21…850f`; explicitly reject the plan digest `8130…f698` for either raw representation.
3. Bind the exact terminal V1 reservation and failure receipts before reserving the V2 root.
4. Use new V2 label reservation, builder-binding, preflight-receipt/failure, source-manifest/review, and final execution-binding paths so no V1 artifact is overwritten or reused.

No other scientific or data change is authorized. Preserve exactly the same model and fresh initialization, N320 encoder input, raw data and role partitions, RGB/raster inputs, counterfactual geometry and labels, train-only action prior, wrong-RGB mapping, seed, 16,000-presentation schedule prefix, update/microbatch accounting, joint `S + P + Q + R` loss from update 1, calibration and selection order, thresholds, controls, 1,000-update/16,000-presentation cap, and one training attempt with no retry or resume.

The V2 source must pass the same recursive source review and synthetic suite before metadata preparation. A V2 preparation or materialization failure is terminal; it does not authorize a V3. Training remains unauthorized until the V2 model-free label/oracle preflight passes and a new exact execution binding is written.
