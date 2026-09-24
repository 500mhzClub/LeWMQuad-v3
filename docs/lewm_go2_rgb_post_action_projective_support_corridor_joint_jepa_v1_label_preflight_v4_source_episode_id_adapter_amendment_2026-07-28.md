# RGB Post-Action Projective-Support Corridor Joint-JEPA V1 — Label-Preflight V4 Source Episode-ID Adapter Amendment

Date: 2026-07-28
Status: AUTHORIZED SCIENCE-IDENTICAL FINAL INFRASTRUCTURE SUCCESSOR
Governing preregistration commit: `8a52adb77d30cb98a6dd086037e6f7c296d76d63`
Reviewed V3 adapter source-freeze commit: `77cd78f3940a2f1ff7d4b28198d114d65f422b6b`

## Terminal V3 materialization receipts

V3 passed metadata preparation and the exact frozen schedule-prefix check, then stopped during the first source-scene join before any label bundle, RGB read, tensor, model, GPU work, training, calibration, or selection.

- Reservation: `.generated/go2_post_action_projective_support_labels_v3/reservation.json`, 2,362 bytes, file SHA-256 `387c7dc37fa3f34fc048e3bab64a82196811689ddd5fbd8648ad017f182bb28e`, content SHA-256 `22fa973b1ac0afb6b8f1ef8a0d3fe7f2da75e275fbd338f0e91385b592ed4627`.
- Builder claim: `.generated/go2_post_action_projective_support_labels_v3/builder_claim.json`, 504 bytes, file SHA-256 `f451a9105cb3cf9baf8035fda7d04530d6044ecc0ae8a898adf4de447732fea9`, content SHA-256 `b96c94c1aebbe862f04361414338e6fa38a58fe17db71b1c5cb64e16ef680e92`.
- Builder failure: `.generated/go2_post_action_projective_support_labels_v3/failure.json`, 2,551 bytes, file SHA-256 `998a5bca429ba2db13dc2996aadd57ff64d3cedef3f3c00420786040f3aa73d8`, content SHA-256 `86a57a2ec562e9395b967778fa9133e11e3b1711acae4846b855130745a6271e`.
- Preflight failure: `.generated/go2_post_action_projective_support_labels_v3_preflight_failure.json`, 2,585 bytes, file SHA-256 `6eb23a50388a4a10f755dee494848cbfb7750045e84beb900f091adbc26465d7`, content SHA-256 `ad0536d7aba6544c797913b7e993a3e900c2ae443b9da6f7ba2771bfff21164e`.
- Builder execution binding: `docs/lewm_go2_post_action_projective_support_labels_v3_execution_binding_2026-07-28.json`, 111,848 bytes, file SHA-256 `ada9f377db4f3adf6fe6e796bc5f8410f01a69c4a6ecb271ee353435fe2944d7`, content SHA-256 `12a5c9ccc2c001f9116e8bfafb31c4029e62cd91fc999e580eda16124a6534bb`.
- Failure: `source episode_id must be a nonempty string` in `materialize_and_publish_manifest_last` / `materialize_label_bundle`.
- Builder access ledger: exactly one open of each authorized metadata/source artifact needed to reach the join; `rgb_opens`, checkpoint/runtime-output/G2/navigation/held-out/sealed/production opens, and all training activity remained zero.

The V3 root and receipts are terminal and must not be removed, replaced, resumed, or reused.

## Exact adapter defect and reviewed authority

The source frame did contain an episode ID. Native V04 source frames encode `episode.episode_id` as a strict non-boolean integer; paired/raw supervision metadata intentionally stores the same identity as a nonempty decimal string. The V3 join incorrectly required the native source value itself to be a string.

Two existing reviewed loaders establish the canonical representation bridge:

- `lewm/datasets/go2_attitude_sidecar.py` requires an exact integer source episode ID, converts it with `str(source_episode_id)`, and compares it to the exact string dataset row while separately checking reset count and episode step.
- `scripts/audit_go2_n32_camera_frustum_observability.py` performs the same string-normalized equality for the V04 source-frame join.

The complete join audit found no other conflict. Frame index, environment index, timestamp, reset count, episode step, manifest identity, image commitment, base position, and stored yaw already match. Numeric `episode.scene_id` is a rollout-local source identity and must not be compared to the textual maze scene ID; source `split` must not be equated to dataset role. The adapter must not add either invalid comparison or newly require camera/quaternion fields that this labeler does not consume.

## Authorized V4 correction

Exactly one label-preflight V4 successor is authorized on fresh write-once V4 paths. It may make only these operational changes:

1. Bind and validate the exact terminal V1, V2, and V3 receipt sets before reserving V4.
2. Require `episode.episode_id` in a selected native source frame to be an exact non-boolean nonnegative integer.
3. Normalize only that value with its canonical decimal `str(...)` representation before comparing it to the existing exact string pair/endpoint identity and reconstructing the unchanged endpoint hash.
4. Add focused synthetic acceptance for integer-source/string-index equality and rejection of a string source episode ID, boolean, negative integer, or mismatched decimal string.
5. Use new V4 label reservation, builder-binding, preflight-receipt/failure, source-manifest/review, and final execution-binding paths.

No source payload is rewritten. No model, data, role partition, input, label geometry, train-only prior, wrong-RGB mapping, seed, schedule index, initialization, optimizer, loss, coefficient, threshold, control, calibration/selection order, update accounting, or cap may change. Preserve the joint `S + P + Q + R` objective from update 1, maximum 1,000 updates and 16,000 presentations, and one training attempt with no retry or resume.

The V4 source must pass the existing recursive source review and focused synthetic suite before metadata preparation. If V4 metadata preparation or model-free materialization fails, stop and reassess the integration rather than automatically authorizing V5. Training remains unauthorized until the V4 model-free label/oracle preflight passes and a fresh exact execution binding is written.
