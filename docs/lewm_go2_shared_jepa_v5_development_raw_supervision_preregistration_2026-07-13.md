# Shared JEPA V5 development raw-supervision preregistration

Date: 2026-07-13

Status: **frozen before implementation or role-payload access**

## Objective

Build the missing paired raw camera-ray supervision for shared V5 training,
checkpoint selection, and probability calibration. The dataset must cover the
full development role population rather than only the 160-transition V4 fit
panel, while leaving the G2 role payload unopened.

This artifact supplies labels and exact pair joins only. It authorizes no
training, checkpoint selection, calibration fit, G2 run, runtime use,
navigation evaluation, hardware use, or promotion.

## Frozen parent identities

- paired navigation manifest:
  `.generated/go2_paired_navigation/geometry_v3_physical_v1/dataset/dataset_manifest.json`,
  file SHA-256
  `ed927cceaedb56ff68334af5109381466740850554048127bb72f04da59f7180`;
- paired row index:
  `.generated/go2_paired_navigation/geometry_v3_physical_v1/dataset/rows.jsonl`,
  file SHA-256
  `187b92f0f311718cf3da098f252da89a992071ea800406bbfff382809085caac`;
- role assignment SHA-256:
  `016c5f872c493065ee4c38fb612fb76958728b37a64987b80d7c0d2736616a02`;
- geometry contract file/content SHA-256:
  `e7d0627d1de259c6e01dabe142aa55e69fed3e75c9c745974d437d7682d40a52` /
  `e06830cbffa67dedec4c20ecd3c1fb9873fe814f212bfa09ec0f160b6514d0ca`;
- render audit file/content SHA-256:
  `9a045dff82fb82adbbb89d10cb4dc0063297805038b000e5f6cd53816e995a9a` /
  `c9280ed4cab9ff54f7d8684835b8448886209a8cc50eba3588519c34572a6358`;
- attitude sidecar manifest file/content SHA-256:
  `6fafa417b4f724a0fdf32cfde5740025c3117e4c0b43231fe9ebe94bd9eff529` /
  `6f1ef7d9ac0c55a42182c3e2c75909f00ab37fffa460aadb549d5cd60d278c1a`.

The exact allowed sidecar role files are:

- train:
  `6cd47d0d679ace897f5b5d8e5c2f11eabab01930904666161eec3792fd9ab6d6`;
- checkpoint selection:
  `4ed434d04afc94b7b82050f5e9fafc900cc03c33a2d847f9784410f8f76f65de`;
- probability calibration:
  `3e5c10e6c15969eb30fbf38bbdb7b47d5fafe25bf14c5547f07ac609b79d91ae`.

The G2 sidecar file, G2 label payloads, and G2 RGB bytes are forbidden.

## Exact role population

All parent transitions in the three development roles are retained, without
subsampling, balancing, backfill, or result-derived exclusion:

| Role | Scenes | Transitions | Endpoint instances | Unique endpoint identities |
|---|---:|---:|---:|---:|
| train | 72 | 4,262 | 8,524 | 7,777 |
| checkpoint_selection | 8 | 495 | 990 | 924 |
| probability_calibration | 8 | 415 | 830 | 759 |
| total | 88 | 5,172 | 10,344 | 9,460 |

An endpoint identity is the exact tuple of role, scene, episode, environment,
episode step, frame index, timestamp, and image SHA-256. Repeated endpoint
instances may share one immutable label payload only when that complete tuple
is identical. Pair records retain both endpoint references, primitive, exact
relative SE(2), source global row, label-shard identity, and scene/family role.
No join by image hash alone is allowed.

## Label contract

For every unique allowed endpoint, build the same raw V4 supervision used by
the frozen fit contract:

- calibrated camera origin, body-frame forward/right/up basis, and body-frame
  ground-plane height;
- ordered pixel first-hit mask and metric first-hit distance on the canonical
  `84 x 112` ray lattice;
- five-support ground-query in-frustum and clear-to-target booleans on the
  native `128 x 128`, `0.05 m` source lattice;
- deterministic derived three-state `64 x 64`, `0.10 m` raster;
- exact evidence and raster content hashes.

Geometry, raycasting, obstacle inclusion, camera calibration, distance
conventions, support offsets, and derived rasterization must call the reviewed
V4 implementations. Reimplementing or approximating those semantics is not
authorized.

The published pair index must independently prove that every allowed parent
transition joins to exactly two role-matched endpoint records and that no G2
endpoint, cross-role frame, orphan endpoint, missing pair, duplicate identity,
or changed primitive/delta entered the artifact.

## Output and execution

The sole output namespace is:

`.generated/go2_shared_observable_camera_ray_jepa_v5/development_raw_supervision_v1`

Publication uses a temporary directory followed by exclusive atomic rename.
The final manifest must inventory every file, byte count, file hash, ordered
role/pair/endpoint hash, input source hash, implementation source hash, and a
complete access ledger. A separately implemented auditor must reconstruct all
joins, hashes, counts, target partitions, and a deterministic sample of raw
raycasts before the dataset can be used.

CPU execution uses at most six scene workers. Every worker sets
`OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, and
`NUMEXPR_NUM_THREADS=1`. GPU use is forbidden for construction and audit.
The builder must not decode RGB; it carries only already-bound RGB paths and
SHA-256 commitments for later role-controlled loading.

## Access boundary

Reading row-level role metadata solely to exclude G2 is allowed and must be
counted. The following remain exactly zero:

- G2 sidecar byte opens;
- G2 scene/source geometry payload opens;
- G2 label-shard payload opens;
- G2 RGB byte opens or decodes;
- checkpoint, model-output, runtime, held-out, sealed, hardware, physical
  executor/reset, navigation-result, or production-promotion opens;
- writes outside the new output namespace.

Source implementation and an independent audit implementation must pass
different-agent review and be hash-frozen before exact construction. A failed
or interrupted publication grants no partial dataset authority and must leave
an explicit failure receipt outside the final immutable namespace.
