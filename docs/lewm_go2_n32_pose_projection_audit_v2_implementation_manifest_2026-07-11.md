# Go2 N32 pose-projection audit V2 implementation manifest

Date: 2026-07-11

Status: frozen after the role-namespace correction and tests, before any
pose-audit result.

## Governing documents

- original audit binding:
  `c959c45737b9242ef667772af4c7b72effcbb39ae687f5ee28226e38cd63854a`
- fit-panel access amendment:
  `56f29c4f2eb05c726b0b4461352fe89da2639b86bf9341ec3072958720cf7c6d`
- superseded, unexecuted train-source filter proposal:
  `35c0de28a795d6b5c246548f5d773326b3f137310c0ec9a840b3e7bf1d302e1d`
- governing role-namespace amendment and sole command authorization:
  `ae17eb856c5329e8c5dfa5e4339306ef19e60c53c5f67d43746b268be9cc3370`

The role-namespace amendment retains all 160 current physical-training
transitions and 320 unique frame records, with 64 frames in each of five
families. Legacy source splits are exact committed provenance only and cannot
alter inclusion.

## Input and output state

- fit-only panel file SHA-256:
  `77d84e242d75b81fd2b96f086e9cf5df72f0a907e1fe7ce24fc48bbc5d514037`
- fit-only panel content SHA-256:
  `8e44dd0238077120e97fd06b4550d6504627066c7e8ddfdfbd138fd7504ee7a8`
- immutable result path:
  `.generated/go2_n32_pose_projection_audit/v1/result.json`
- result state when this manifest was written: absent

The first V1 attempt wrote no result. It failed closed on the legacy/current
role ambiguity before scanning a source `frames.jsonl`. The historical V1
implementation manifest remains an exact record of that attempted source.

## Source freeze

- pure geometry module
  `lewm/benchmarks/go2_n32_pose_projection_audit.py`:
  `8835fbecc798c1cc3dd7a17b07821677a893854e1a1af0c3073c1bded9a07ac6`
- V2 authoritative runner `scripts/audit_go2_n32_pose_projection.py`:
  `3b422299eaa8d81c2397301e4981ce92a0c60731c39179d52bf7645868b674a3`
- adversarial tests `lewm/tests/test_go2_n32_pose_projection_audit.py`:
  `8f47bcf85ba9eaf4c0e7614a8838dfa838fa6b18c91f1e0ea378f9a00af0b951`
- fit-only extractor `scripts/extract_go2_n32_pose_fit_panel.py`:
  `f9f4a15f37deff8571dff800fb21c4d50f12cdaa76d68416d9a6b22b8cf4b4bb`

The runner's local executable source closure remains the runner and the pure
geometry module. Both are hashed before and after metadata access.

## Role and access hardening

The V2 runner:

- requires the newest amendment hash before any metadata access and rejects the
  three older governing hashes;
- hashes all four governing documents before and after the audit;
- validates all fit rows as current `physical_dataset_role=train` and rejects
  any current physical non-train row;
- binds each summary path to its exact file hash and exact legacy source split;
- passes that frozen legacy split to source-episode validation without using it
  for inclusion, ranking, calibration, or selection;
- requires legacy frame-record counts `train=244`, `test_hard=14`,
  `test_id=32`, and `val=30`, with current physical train `320` and current
  physical non-train `0`;
- retains exact summary/source allowlists, before/after hashes, exact-once frame
  matching, source-code hashes, and exclusive atomic output;
- never opens the original monolithic panel, RGB, labels, checkpoints, model
  outputs, physical selection/calibration/G2 payloads, or sealed data; and
- records that legacy source split was not used for inclusion.

## Verification

The combined focused suite passed before the result existed:

```text
lewm/tests/test_go2_n32_pose_projection_audit.py
lewm/tests/test_categorical_radial_perception.py
lewm/tests/test_go2_categorical_radial_factorization.py

74 passed in 1.07s
```

Coverage includes a complete synthetic successful 160-transition/320-frame
runner with legacy `test_hard` provenance, all older-authorization rejections,
all four governing-document mutation branches, exact role/count reconciliation,
summary/source mutation, exact-once matching, and inclusive ordering boundaries.

This metadata audit can only order the next N32 representation experiment. It
cannot pass N32, G2, or a runtime gate.
