# V13 camera-evidence joint-JEPA integrity replacement V3 preregistration

Date: 2026-07-29

Status: **selected science-identical operational replacement**.

## Reason and scope

- Integrity replacement V2 terminated during update 1 with zero completed
  forward passes, zero optimizer updates, zero EMA updates, and zero scientific
  presentations. Its complete audit is
  `docs/lewm_go2_rgb_camera_evidence_bottleneck_joint_jepa_v13_integrity_replacement_v2_failure_audit_2026-07-29.json`.
- The exact exception was
  `dense features and ground queries must share a device`. The inherited V1
  base builder placed RGB and labels on `cuda:0`, but the fourteen V13 camera
  tensors added afterward remained on CPU.
- The otherwise valid trace file also contained one invalid internal row hash.
  Its nested schedule-prefix mapping used integer keys before JSON converted
  them to strings and changed their canonical sort order after reload.
- This replacement may change only those two operational representations,
  focused tests, source/provenance bindings, certified paths, and the fresh
  output root. It may not change scientific scope.

## Exact permitted repairs

- In `_build_one_microbatch_v13`, move every tensor in the completed
  fourteen-entry camera addition mapping to `runtime.device` before merging it
  with the inherited V1 batch. Preserve every value, dtype, shape, key, and key
  order. Keep raw caches and `_stack_camera_rows_v13` unchanged.
- In `validate_schedule_v13`, return the three `prefix_sha256` receipt keys as
  JSON strings (`100`, `400`, and `1000`). Keep schedule generation, prefix
  values, comparison points, and verification unchanged.
- Focused source-only tests must prove that all fourteen additions share the
  base RGB device without changing values, dtypes, or shapes, and that every
  emitted trace row validates its content hash after JSON serialization and
  parsing.
- Before execution authority, one real synthetic V13 joint training update on
  the bound accelerator runtime must complete forward, backward, optimizer,
  and EMA accounting with four microbatches of four examples. It may not open
  generated scientific data. The complete focused V13 suite and isolated
  accelerator-hidden import smoke must also pass.

## Frozen science

- The V13 model source remains exact SHA-256
  `ac46ff9cd604e003a300a2f78704d8b58e2e43dec4aae713cdd01ca23f2dbd03`
  (35,982 bytes). The V13 training source remains exact SHA-256
  `f194d0451855d49c5338bfd06f66f8b13b0312a71662d24e140cb151bbccbe08`
  (32,558 bytes).
- Architecture, RGB-only inference boundary, 40 FREE planes, 64 first-hit
  OCCUPIED planes, sole 64-channel JEPA bottleneck, predictor, survival head,
  losses, exact C/N gradient balancing, optimizer, initialization, target hard
  sync and EMA, data, labels, role populations, schedule, seed, controls,
  observation updates, thresholds, and physical gates are unchanged.
- Runtime inputs remain the same exact Raw-V13 manifest/audit, N320 gate and
  checkpoint, 16,000-presentation schedule, and swept-label manifest/train/
  checkpoint-selection files. Probability calibration remains closed.

## Execution contract

- Exact output root:
  `.generated/go2_rgb_camera_evidence_bottleneck_joint_jepa_v13_integrity_replacement_v3/attempt_v1`.
- Exactly one fresh attempt; no retry or resume. The root must be absent before
  an immutable reservation is created.
- Maximum 1,000 updates and 16,000 presentations, with unchanged observation
  points `0/100/400/1000`, unchanged update-400 falsification, and unchanged
  update-1000-only checkpoint eligibility.
- Execution requires a fresh committed source manifest, independent review,
  clean-export custody exception and certification, execution binding, and
  canonical one-shot authority. Neither earlier V13 authority grants access.
- G2, navigation, held-out, sealed, production, deployment, and promotion
  remain unauthorized. The existing V4 30-scene sealed benchmark remains
  unopened.
