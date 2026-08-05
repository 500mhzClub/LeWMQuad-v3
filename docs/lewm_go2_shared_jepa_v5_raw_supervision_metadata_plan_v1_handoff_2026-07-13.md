# Shared JEPA V5 raw-supervision metadata plan V1 handoff

Date: 2026-07-13

Status: **author complete; different-agent review required**

## Scope

This additive metadata-only stage implements the exact pair, endpoint, and
attitude join required by the frozen development raw-supervision
preregistration. It does not raycast, decode RGB, open label shards, train,
select, calibrate, inspect G2 payloads, or grant downstream authority.

Frozen parent preregistration:

- `docs/lewm_go2_shared_jepa_v5_development_raw_supervision_preregistration_2026-07-13.md`
- SHA-256 `07a51661f7d86391bda8974799a881287ccace8083fadf396e5c01b6345ed3bb`

## Implementation

- `lewm/datasets/go2_shared_jepa_v5_raw_supervision_plan.py`
- SHA-256 `e7ab8727b0d93d3fd8f9e2a3ab5cfdc4f9199e18b8d0a7f5a1f7dc0b5dc0c18e`

The implementation:

- reads the exact paired manifest and complete row metadata by frozen file
  hashes;
- uses G2 row metadata only to exclude its 469 transitions;
- opens exactly the train, checkpoint-selection, and probability-calibration
  attitude sidecars through the existing fail-closed role loader;
- never requests or opens the G2 sidecar;
- joins each development pair to exactly one role-matched sidecar row;
- defines endpoint identity as the complete preregistered role/scene/episode/
  environment/step/frame/timestamp/image-hash tuple;
- deduplicates only that complete identity and rejects conflicting metadata for
  one identity;
- retains primitive, relative SE(2), global row, scene, family, label-shard
  commitment, and both endpoint references in each pair record; and
- emits content hashes and an explicit zero-access ledger without publishing
  a dataset or granting a build/training license; and
- independently reduces the frozen 96-row rendered source index to the exact
  88 planned development scenes, reproducing all five source-inventory hashes
  while retaining zero referenced-payload opens.

## Exact metadata result

The author ran the frozen metadata loader CPU-only with every native thread
capped at one and all accelerator visibility disabled. It reproduced:

| Role | Pairs | Endpoint instances | Unique endpoints | Scenes |
|---|---:|---:|---:|---:|
| train | 4,262 | 8,524 | 7,777 | 72 |
| checkpoint_selection | 495 | 990 | 924 | 8 |
| probability_calibration | 415 | 830 | 759 | 8 |
| total | 5,172 | 10,344 | 9,460 | 88 |

Observed plan identities:

- plan content SHA-256:
  `8004ab0d3aa6a2f5d576ba0ff4d6a75f50899152e542dc62b8d6e35f614921a3`;
- ordered pair SHA-256:
  `76810dba883f3aaffb92fccb593d382daf7edca74a9bb5559a977e7e88b7b5ea`;
- ordered endpoint SHA-256:
  `8130e961b7b5c04944b178fa4f73c1fa157776f7702ab5cdc213cf16c922f698`.

The observed ledger records zero G2 sidecar, G2 geometry/label payload, label
shard payload, RGB byte/decode, checkpoint/model-output, runtime/navigation,
held-out/sealed, hardware, and production opens.

## Tests

- `lewm/tests/test_go2_shared_jepa_v5_raw_supervision_plan.py`
- SHA-256 `e2b49e660292ff99a7794ff3c761f9563a9e182b2889aec3a4e94b835c4be56c`

Author verification:

```text
9 passed in 1.45s
py_compile: PASS
git diff --check: PASS
```

The focused tests cover exact endpoint deduplication, rejection of image-hash
only identity, conflicting endpoint metadata, sidecar join mutation, G2
sidecar exclusion, exact frozen development counts, metadata-only source
selection, missing-scene rejection, and reproduction of the 88-scene source
inventory.

## Remaining authority

Different-agent review must reproduce the joins and access boundary before
this plan can license a builder. The later builder must still be separately
implemented and reviewed, reuse the frozen V4 raycast/raster semantics, publish
through private staging plus exclusive atomic rename, and receive its own
independent dataset audit before training.
