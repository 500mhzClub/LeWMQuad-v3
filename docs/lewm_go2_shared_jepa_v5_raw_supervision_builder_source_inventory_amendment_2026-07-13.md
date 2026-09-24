# Shared JEPA V5 raw-supervision builder source-inventory amendment

Date: 2026-07-13

Status: **frozen before builder source review or development payload access**

## Purpose

The parent raw-supervision preregistration binds the paired rows, role split,
geometry contract, render audit, and development attitude sidecars. This
amendment freezes the remaining metadata-only reduction from the complete
96-scene rendered source index to the exact 88 development scenes that the
raycast builder may dereference.

It changes no role, row, endpoint, label, raycast, raster, training, selection,
calibration, or evaluation contract. It grants no data-build or learned-output
authority.

## Frozen source index

- path:
  `.generated/go2_paired_navigation/geometry_v3_physical_v1/source_index/go2_navigation_sources_v04.jsonl`;
- file SHA-256:
  `11b9a669324cc7630ba072138983f2dd0daf0d0a4e12596a1204f665eb208a6c`;
- total metadata rows: `96`;
- selected development metadata rows: `88`;
- excluded G2 metadata rows: `8`.

Selection is exact membership in the full endpoint scene set emitted by the
metadata plan, never path pattern, family count, source split, or source-index
position. The selected set contains 72 train, 8 checkpoint-selection, and 8
probability-calibration scenes. G2 source-index metadata may be parsed only to
exclude it; its frames, scene manifest, render plan, summary, labels, and RGB
must remain unopened.

## Canonical inventory hashes

Every list below is ordered by literal `scene_id` and hashed as canonical JSON
with sorted keys and compact separators.

- scene/role records (`scene_id`, `role`):
  `f967364a2869f9f87a4c2c1c0053616e263464be53691283909a8f910b94ed5b`;
- source-frame records (`scene_id`, absolute path, file SHA-256):
  `7512a041d2f163cc8978eee1a261951162b2ccd2020414325504e41eac9c623d`;
- source-scene-manifest records (`scene_id`, absolute path, file SHA-256,
  semantic/content SHA-256):
  `2bc5f468eeba3f44b1f428b0145c9e63ec84d08d1c21e8a2870227e12e0c44c5`;
- render-plan records (`scene_id`, absolute path, file SHA-256):
  `0359078471ac3f85aa704f44012a44ec9f3c1c2fd6f61ce1628533fc1c2a36e4`;
- render-summary records (`scene_id`, absolute path, file SHA-256):
  `bd2b181973e3023df0825200657d0d2895f71804134100f60234363503be548a`.

The parent render audit remains bound by file/content SHA-256
`9a045dff82fb82adbbb89d10cb4dc0063297805038b000e5f6cd53816e995a9a` /
`c9280ed4cab9ff54f7d8684835b8448886209a8cc50eba3588519c34572a6358`.
The physical geometry contract remains bound by file/content SHA-256
`e7d0627d1de259c6e01dabe142aa55e69fed3e75c9c745974d437d7682d40a52` /
`e06830cbffa67dedec4c20ecd3c1fb9873fe814f212bfa09ec0f160b6514d0ca`.

## Builder requirements

The exact builder must:

1. reproduce the reviewed metadata plan and every inventory hash above before
   opening a selected source payload;
2. reject any missing, extra, repeated, cross-role, or G2 scene before
   dereferencing a frames, manifest, plan, or summary path;
3. reuse the reviewed V4 camera composition, full-RPY rendered-box parity,
   raycast, ground-support, and rasterization functions rather than
   reimplementing their semantics;
4. raycast each of the 9,460 exact unique endpoints once, while retaining all
   10,344 pair endpoint references;
5. use at most six spawned CPU scene workers, with all four native thread
   variables fixed to one and no GPU visibility;
6. open no RGB or parent physical-label shard payload;
7. rehash every opened source payload after use and revalidate all parent
   metadata immediately before publication;
8. publish the complete dataset through a private sibling staging directory
   and one exclusive atomic rename, with directory fsync ordering and an
   explicit external failure receipt for an interrupted or failed build; and
9. leave training, checkpoint selection, calibration, G2, runtime, hardware,
   production, and promotion authority false.

An independently implemented auditor must reconstruct the complete pair and
endpoint indexes, byte inventories, role boundaries, and a deterministic
precommitted sample of raw V4 evidence from original development geometry.
The builder's own validation cannot authorize dataset use.

## Access statement

This inventory reduction opened only the frozen paired manifest/row metadata,
the three allowed development sidecars, and the source-index bytes. It did not
open a source frames file, source scene manifest, render plan, render summary,
label shard, RGB, checkpoint/model output, G2 sidecar or payload, runtime,
navigation result, held-out/sealed input, hardware, or production artifact.
