# Geometry-Anchored Joint-JEPA V3 scalar-tensor state-hash integrity replacement

Date: 2026-07-27

Status: preregistered for source implementation, CPU-only synthetic preflight,
source closure, and independent review. Execution still requires a distinct
one-attempt authorization.

## Decision

V2 is consumed and closed. Its committed terminal audit is
`docs/lewm_go2_rgb_geometry_anchored_deformable_bev_lift_joint_jepa_v2_runtime_import_integrity_replacement_terminal_audit_2026-07-27.json`
at commit `20b13fe3100d96e8d17b65da49261d1388d5015f`, raw SHA-256
`184ba6c10e2c37fec12608bf56ba97fc345ab180c3f001989588a630dde9bb5e`,
content SHA-256
`5cfff834da4b0c0667ebc6e282abdba651dfa58a766db0174d00d32c2510ea51`,
and 8,922 bytes.

V2 proved that the isolated import-lifetime correction works. It validated the
authorized development metadata, schedule, N320 gate/checkpoint and one visible
R9700, then constructed and moved the model to the GPU. It loaded zero training
pairs, made zero RGB/raster requests or physical reads, and ran zero
presentations, objectives, backward calls, optimizer updates, EMA updates, or
observations. It wrote no checkpoint or trace. The result is operational
evidence, not a scientific or mechanism result.

The failure occurred in the write-only state-integrity digest before update 0.
Frozen V1/V2 `_tensor_state_sha256` calls `tensor.view(torch.uint8)`. PyTorch
cannot dtype-view a zero-dimensional float tensor to a different element size.
The unchanged BEV lift intentionally contains persistent scalar float camera
buffers, including `camera_near_m`, so initial online/target state hashing
raised `RuntimeError: self.dim() cannot be 0 to view Float as Byte`.

Standing user authority explicitly permits an obvious operational correction
when the scientific idea has not been tested or rejected. V3 is exactly one
fresh science-identical integrity replacement, not a V2 retry or resume.

## Sole implementation delta

V3 changes only the byte-view adapter inside `_tensor_state_sha256`:

- V2: `tensor.view(torch.uint8)`
- V3: `tensor.reshape(-1).view(torch.uint8)`

The tensor is already detached, copied to CPU and made contiguous. Flattening
therefore changes no tensor value or byte order. It gives a scalar a one-element
dimension so the byte view is defined. For every non-scalar tensor, V3 must
produce exactly the frozen V2 digest contribution. No digest is a model input,
loss, gradient, gate value, or optimizer input.

No other code or science change is allowed. V3 inherits the corrected V2 import
root lifetime and V2-labelled complete failure receipts unchanged.

Before source review can pass, CPU-only synthetic tests must prove:

- scalar float, integer and boolean buffers hash without error;
- mixed scalar/non-scalar mappings match an independent raw-byte reference;
- all-non-scalar mappings are digest-identical under frozen V2 and V3 adapters;
- a complete freshly constructed unchanged V1 model, including its scalar
  camera and counter buffers, can be state-hashed for predictor, online encoder,
  target encoder, online BEV lift, target BEV lift and full state;
- online/target equality and all initialized bytes remain unchanged; and
- caller CPU RNG and `sys.path` are restored exactly.

These preflights may import Torch and construct the unchanged model on CPU from
synthetic encoder state. They may not open generated inputs, the N320
checkpoint, runtime outputs, traces, accelerators, navigation, held-out, sealed,
or rejected material.

## Frozen science and authority

V3 preserves every V1/V2 scientific variable: RGB-only model and geometry,
exact N320 initialization, seeds `20260712` and `20260713`, parameter draw order
and bytes, target hard sync and EMA `0.996`, data rows/roles/mappings/hashes,
schedule/order/batch/microbatch, semantic warmup updates 1-400, genuine joint
JEPA updates 401-1000, all losses/weights/reductions, AdamW groups and clipping,
all observation/gate thresholds, one attempt, 1,000 updates, 16,000
presentations, and 30 active GPU minutes.

The frozen science-contract SHA-256 remains
`f839076bf7f9db9e9f211703323436f4b607cca21e2e60fb228e4d174c699fa3`.
Model, objective, optimizer, schedule and gate hashes remain respectively
`595d91a6fc9ae985378ff480780bf7ad5a9beeb3c7f35ab012c010bb74162f39`,
`93c73c1f1a91de70699f634821159d4d544431b45faa469202016fa0b9fd7ba8`,
`2bb70f943838b656540b3dac3b6e0f30bb384547180270274abfc5077e264b34`,
`bc0ad45c06171cff7533fbfcb054e5afecf6086de0a58060c35cb5ca0256c2e3`,
and `0c485c0bccb88873c0ff76a1061a315420b6c27c4865b259d3b4c6f374862bd0`.

The corrective V2 source freeze is commit
`1ff023d306f63d0651639f699038d538f8f6336d`. Its 79-source manifest has raw
SHA-256 `270f64c520e7a0193f73a81e7a4cad9c62db162ddbcfbb840302e79c15ba004e`,
content SHA-256
`90b0f6feca7d987d6a547f201d55784c52e5ea2646dc339f9d4ff63f81cb2d0a`,
and source-bindings SHA-256
`8c017968de1eb2b1077d970f407917631db1c167774b7385552ad6a0e0020403`.
The zero-finding V2 review is commit
`fb73acb0acd19ab29a6826ddbec393a6d2913f80`, raw SHA-256
`d9855b4454d45fd10f6947ba1dfb66c37549e1270681d479533a2976f2bc17a3`,
content SHA-256
`fa2bcee13e1abf0cdf985ba17586e3cd16b4ccd2cfd91e2a95d051a061bacdb5`.
The consumed V2 authorization is commit
`7fd4a0718c45c1b088780732d42e1e9756c092e6`, raw SHA-256
`ca77161bceace3b20a7642ccb9b71cd382396a5a71c6592ea3c52d32e00fe908`,
content SHA-256
`027e4a65be73301c41b55bcdcd422705c0474a6a017d6b2b7728d525cd45d740`.
It grants no V3 execution authority.

## V3 lifecycle

The experiment ID is
`geometry_anchored_deformable_bev_lift_joint_jepa_v3_scalar_tensor_state_hash_integrity_replacement`.
Its sole fresh output root is
`.generated/go2_rgb_geometry_anchored_deformable_bev_lift_joint_jepa_v3_scalar_tensor_state_hash_integrity_replacement/attempt_v1`.
It must be absent before authorization and reservation. Any reservation consumes
it. V1 and V2 roots, checkpoints and traces are never V3 inputs and may not be
reopened.

Implementation is a lean five-file overlay on the frozen 79-source V2 closure:
one contract, runner wrapper, launcher wrapper, closure checker and focused test
module. There is no new model file. The exact V3 closure is 84 sources.

Execution requires a committed V3 source manifest, independent source/science
review, distinct independent authorization and exact absent-root check. A pass
authorizes only a separate terminal audit and scientific decision—not
checkpoint use, navigation, G2, held-out/sealed access, promotion, deployment,
retry, resume, second seed, or another scientific successor.

No generated input, checkpoint, runtime output, trace, GPU, navigation,
held-out, sealed, or rejected material was opened to write this preregistration.
