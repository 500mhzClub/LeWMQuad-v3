# Go2 Dynamic-Cartesian Attitude Sidecar Result

Date: 2026-07-11

## Outcome

The metadata-only attitude sidecar was built and published successfully for all
96 bound source scenes. The sidecar supplies the exact current and next
world-frame base quaternion and stored base yaw needed by the dynamic
cell-square projection while preserving the frozen dataset role split.

Published manifest:

- Path: `.generated/go2_attitude_sidecar/dynamic_cartesian_v1/manifest.json`
- File SHA-256: `6fafa417b4f724a0fdf32cfde5740025c3117e4c0b43231fe9ebe94bd9eff529`
- Content SHA-256: `6f1ef7d9ac0c55a42182c3e2c75909f00ab37fffa460aadb549d5cd60d278c1a`
- Source scenes: 96
- Train rows: 4,262
- Checkpoint-selection rows: 495
- Probability-calibration rows: 415
- Sealed G2-evaluation rows: 469

The train, checkpoint-selection, and probability-calibration role files were
independently loaded through the strict role loader and matched their declared
hashes and counts. The G2 role file was not opened after publication, no G2
access receipt exists, and the build ledger records zero G2 role-file reopens.

## Frozen Implementation

The authoritative build used the independently reviewed implementation
precommit:

- Implementation-manifest file SHA-256:
  `3c2fa57a0bb230bdbba0bea92045abafd93370b46a424dac579da702aa79ac64`
- Implementation content SHA-256:
  `d09719886598bb584e0822cd19b1cd75c2b9ebefa9e1dd90367594396bd1269c`
- Implementation source-map SHA-256:
  `7930e4760990add21241ea54b6508831203b5cf9d5bcf14d7aa2b2a4dd6ba161`

The focused sidecar suite passed 39 tests. The combined dynamic model, sidecar,
trainer-compatibility, and runtime-compatibility suites passed 157 tests.

The final join fails closed across:

- Named-scene hash and frozen dataset role
- Frames-file path and SHA-256
- Scene-manifest SHA-256 and source split
- One stable, nonnegative, exact-integer frame-internal scene ID
- Exact-integer frame episode ID matched to the canonical string dataset ID
- Environment, reset, episode step, frame index, and timestamp
- Current and next endpoints independently
- Injective transition and row identities

The access ledger records a metadata-only build with zero image, label-shard,
depth, or model-artifact byte opens. Publication used private staging followed
by no-replace hard links, with the manifest published last.

## Failed-Closed First Attempt

The first authorized production build stopped before publication because the
synthetic fixture had modeled frame episode scene and episode IDs as strings,
whereas the production frame metadata stores exact integers. The output and
staging directories were absent after the failure.

Before rerunning, the join was corrected to keep the named dataset scene
namespace separate from the frame-internal numeric namespace. Production-shaped
mutation tests were added for boolean, string, mixed, and negative numeric scene
IDs; frame and row episode-ID type mismatches; manifest and split mismatches;
source-index scene-hash mismatch; dataset-source role mismatch; and source-scene
count mismatch. An independent review then authorized the new frozen hashes
listed above.

## Next Gate

Use only the authorized development roles to run the 320-frame fit-panel
projection audit. Every Torch float32 dynamic visibility/support decision must
match the stdlib float64 cell-square reference before the N32 occupancy-only
training runner is frozen.
