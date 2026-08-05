# Go2 N32 pose-audit fit-panel access amendment

Date: 2026-07-11

Status: frozen before fit-only panel extraction and before pose-audit output.

## Reason

The original pose-audit binding named the immutable N32 panel JSON but also
forbade holdout metadata access. That source JSON is monolithic: its `panels`
object contains fit, same-scene-holdout, and cross-scene-holdout rows. A runner
cannot parse only `panels.fit` without first reading bytes that encode the two
other panels.

The authoritative pose audit must therefore not open the monolithic panel.
This amendment authorizes one metadata-only preflight extractor to verify the
original panel and publish a new immutable fit-only artifact. This is a
protocol correction before any pose-audit result, not a model-selected change.

## Authorized extraction

Input:

- `.generated/go2_physical_micro_overfit/patch7_v1/panel.json`;
- file SHA-256
  `c3f44c6b1147efbb6a5fbc2294c6431c72e25da877cab6884972d25c1ffdb16c`;
- canonical content SHA-256
  `f3e5198b81ac48c06f6c8e4b21e8bf24d62200e3830b1d6685d949a668349d5f`;
- fit rows SHA-256
  `5a75e202c8f7a803aafaad093c7f474137dd2e69f50ecdb7fb4e97765afb659d`.

Output:

- `.generated/go2_n32_pose_projection_audit/v1/fit_panel.json`;
- schema `lewm_go2_n32_pose_projection_fit_panel_v1`;
- source identity and hashes;
- registered family order;
- exactly the original `panels.fit` object and no other panel;
- explicit source-access ledger and canonical content hash.

The extractor must verify the original file hash before parsing, recompute its
canonical content hash, verify the fit row/frame counts and fit-row hash, copy
only the fit object, rehash the original after parsing, and publish with
exclusive atomic creation. It may decode the monolithic metadata once for this
split. It must not open images, label shards, model checkpoints/outputs, G2,
non-train payloads, or sealed data. Its output is not a research result and
cannot pass any gate.

## Superseding audit input

After extraction, the pose-audit runner must bind the fit-only artifact's file
and content hashes in source before execution and must never open the original
panel. It must also bind this amendment alongside the original audit binding.

All other geometry, record, source-frame, summary, decision, output, and access
rules in
`docs/lewm_go2_n32_pose_projection_audit_binding_2026-07-11.md` remain
unchanged.
