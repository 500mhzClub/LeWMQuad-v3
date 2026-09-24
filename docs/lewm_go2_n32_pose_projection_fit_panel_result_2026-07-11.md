# Go2 N32 pose-audit fit-panel extraction result

Date: 2026-07-11

Status: completed; this is a provenance result, not a research result.

## Authorization

The extraction was authorized by
`docs/lewm_go2_n32_pose_projection_fit_panel_amendment_2026-07-11.md`, whose
file SHA-256 is
`56f29c4f2eb05c726b0b4461352fe89da2639b86bf9341ec3072958720cf7c6d`.
The extractor source SHA-256 was
`f9f4a15f37deff8571dff800fb21c4d50f12cdaa76d68416d9a6b22b8cf4b4bb`.

## Output

- path: `.generated/go2_n32_pose_projection_audit/v1/fit_panel.json`
- schema: `lewm_go2_n32_pose_projection_fit_panel_v1`
- file SHA-256:
  `77d84e242d75b81fd2b96f086e9cf5df72f0a907e1fe7ce24fc48bbc5d514037`
- canonical content SHA-256:
  `8e44dd0238077120e97fd06b4550d6504627066c7e8ddfdfbd138fd7504ee7a8`
- fit transitions copied: 160
- fit frames represented: 320

The output binds the immutable source panel by file SHA-256
`c3f44c6b1147efbb6a5fbc2294c6431c72e25da877cab6884972d25c1ffdb16c`
and content SHA-256
`f3e5198b81ac48c06f6c8e4b21e8bf24d62200e3830b1d6685d949a668349d5f`.

## Access result

The extraction ledger records one source-panel parse and two source-panel byte
opens for before/after hashing. It copied zero non-fit rows and opened zero RGB
bytes, label-shard bytes, model checkpoints or outputs, G2 payloads, and sealed
manifests or payloads.

The authoritative pose-audit runner must consume only this fit-only artifact.
It must never open the monolithic source panel. This extraction cannot pass
N32 or G2 and cannot license runtime promotion.
