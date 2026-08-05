# Go2 categorical-radial N32 V4 smoke result

Date: 2026-07-11

Status: non-authoritative smoke passed wiring and access control; authoritative
seed 20260710 is licensed under the frozen V4 binding.

## Frozen implementation reference

- implementation manifest:
  `docs/lewm_go2_categorical_radial_n32_v4_implementation_manifest_2026-07-11.md`;
- manifest SHA-256:
  `6f1f936efeca1e684e394e2a1680002b5ba719d4d24c27694c01821455926ffc`;
- transitive source map: 41 entries, canonical JSON SHA-256
  `fe136d8543a9664417e65ec8e07f052875f9903b5913ac915eb1ec6d68791800`.

## Successful smoke

- artifact:
  `.generated/go2_categorical_radial_n32/v4/smoke_seed_20260710.json`;
- file SHA-256:
  `c7a44accb9b65c4dafd81cf1a9882b305a2036cd48204c392fce66517de1d1de`;
- canonical content SHA-256:
  `eb9476613d4075922af54f0576f9488b5552ca331931bd80c25943a2e760b58e`;
- schema: `lewm_go2_categorical_radial_n32_v4_smoke_result_v1`;
- device: discrete `AMD Radeon AI PRO R9700` through `HIP_VISIBLE_DEVICES=0`;
- seed: 20260710;
- updates/evaluations: 3/3;
- decision: `non_authoritative_smoke`, favorable false, aggregation false;
- all runtime, G2, G3, promotion, and candidate licenses: false.

The smoke is deliberately too short to test learning. Its purpose is to prove
that the frozen model, optimizer, loss, evaluation, provenance, and access
ledger execute together.

## Access audit

The successful process records:

- fit: 320 image decodes, 20 label-shard NPZ opens, 1,200 target requests;
- same-scene holdout: unauthorized, zero byte opens, decodes, label opens,
  model calls, or model outputs;
- cross-scene holdout: unauthorized, zero byte opens, decodes, label opens,
  model calls, or model outputs;
- checkpoint selection, probability calibration, physical non-train, and G2:
  zero image/label opens and zero model outputs;
- holdout payload and holdout-check fields: null;
- current physical dataset role `train` governs access; legacy rollout split is
  provenance only and did not filter, rank, calibrate, or select rows.

Canonical-content recomputation matches the stored content hash. The smoke
uses the frozen 41-entry source map and cannot occupy an authoritative result
path. The torch-free finalizer explicitly rejects smoke-schema artifacts as
authoritative evidence.

## Failed device-selection attempt

Before the successful process, the same immutable smoke command was attempted
with `HIP_VISIBLE_DEVICES=1`. That device was the unsupported integrated
`AMD Radeon Graphics` adapter and failed with `hipErrorInvalidDeviceFunction`
while moving the freshly initialized model to the device.

By runner order, the failed process had already performed its two integrity
hash passes over the same authorized 320 fit images and 20 fit shards. It had
not called the dataset, decoded an image, opened an NPZ label array, performed
an optimizer update, or emitted a model output. It opened no holdout,
checkpoint-selection, probability-calibration, physical-nontrain, G2, or
sealed payload. It created no result file. This attempt changes no model,
schedule, source, evidence, or selection decision.

## Authorization

The pre-smoke implementation commitments remain unchanged. This evidence
licenses exactly one authoritative seed-20260710 execution at:

`.generated/go2_categorical_radial_n32/v4/seed_20260710_result.json`

with the registered 2,000-update schedule. Seed 20260711 remains forbidden
unless the strict torch-free finalizer finds seed 20260710 fully favorable and
its immutable file hash is supplied before seed-11 device/model construction.
