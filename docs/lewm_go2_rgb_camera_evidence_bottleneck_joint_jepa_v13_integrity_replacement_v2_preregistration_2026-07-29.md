# V13 camera-evidence joint-JEPA integrity replacement V2 preregistration

Date: 2026-07-29

Status: **selected science-identical operational replacement**.

## Reason and scope

- Integrity replacement V1 terminated before training with zero optimizer
  updates and zero scientific presentations. Its complete audit is
  `docs/lewm_go2_rgb_camera_evidence_bottleneck_joint_jepa_v13_integrity_replacement_v1_failure_audit_2026-07-29.json`.
- The exact terminal exception was
  `expected np.ndarray (got numpy.float32)`: the raw loader selected the
  scalar `ground_plane_z_body_m.f4` row and passed that NumPy scalar to
  `torch.from_numpy`.
- Source-only reproduction also proved that the following fixed structural
  check would compare an online train/gradient-mode attention path against an
  EMA eval/no-gradient path. Identical state then differs only by floating
  execution dispatch, while deterministic eval/no-gradient execution is
  bit-identical.
- This replacement may change only those two operational adapters, their
  focused tests, source/provenance bindings, certified paths, and the fresh
  output root. It may not change scientific scope.

## Exact permitted repairs

- In the reviewed raw row loader, convert the selected row to an ndarray
  before `torch.from_numpy`. Existing ndarray rows remain unchanged; the
  scalar ground row becomes a zero-dimensional float32 tensor and stacks to
  the already required `(B,)` schema.
- In the initial structural-integrity check only, save the model training
  mode, run all comparison encodes in eval mode under `torch.no_grad()`, and
  restore the saved mode in `finally`. Exact equality remains required; no
  tolerance may replace it.
- Focused source-only tests must exercise the real scalar loader path and the
  real V13 model. They must prove preserved dtype/value/shape, unchanged
  model state and RNG, no gradients, restored online training mode, and EMA
  targets remaining in eval mode.
- Before execution authority, a fresh clean export must run those tests plus
  the complete V13 focused suite and an isolated accelerator-hidden runtime
  import smoke. No generated scientific input may be opened by these tests.

## Frozen science

- The V13 model source remains exact SHA-256
  `ac46ff9cd604e003a300a2f78704d8b58e2e43dec4aae713cdd01ca23f2dbd03`
  (35,982 bytes). The V13 training source remains exact SHA-256
  `f194d0451855d49c5338bfd06f66f8b13b0312a71662d24e140cb151bbccbe08`
  (32,558 bytes).
- Architecture, RGB-only inference boundary, 40 FREE planes, 64 first-hit
  OCCUPIED planes, sole 64-channel JEPA bottleneck, predictor, survival head,
  losses, exact C/N gradient balancing, optimizer, initialization, target
  hard sync and EMA, data, labels, role populations, schedule, seed,
  controls, observation updates, thresholds, and physical gates are unchanged.
- Runtime inputs remain the same exact Raw-V13 manifest/audit, N320 gate and
  checkpoint, 16,000-presentation schedule, and swept-label manifest/train/
  checkpoint-selection files. Probability calibration remains closed.

## Execution contract

- Exact output root:
  `.generated/go2_rgb_camera_evidence_bottleneck_joint_jepa_v13_integrity_replacement_v2/attempt_v1`.
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
