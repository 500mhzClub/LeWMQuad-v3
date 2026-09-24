# V13 camera-evidence joint-JEPA integrity replacement V1 preregistration

Date: 2026-07-29

Status: **selected science-identical operational replacement**.

## Reason and scope

- V13 attempt V1 ended before scientific payload, GPU query, checkpoint load,
  or training because its clean export omitted a dynamically read source
  contract. The terminal audit is
  `docs/lewm_go2_rgb_camera_evidence_bottleneck_joint_jepa_v13_attempt_v1_terminal_failure_audit_2026-07-29.md`.
- This replacement may change only source-closure discovery, certified-source
  import activation, evidence/authority paths, provenance constants, and the
  fresh output root needed to execute the already reviewed V13 program.
- The V13 model source must remain byte-identical to SHA-256
  `ac46ff9cd604e003a300a2f78704d8b58e2e43dec4aae713cdd01ca23f2dbd03`
  (35,982 bytes), and the V13 training source must remain byte-identical to
  SHA-256
  `f194d0451855d49c5338bfd06f66f8b13b0312a71662d24e140cb151bbccbe08`
  (32,558 bytes).
- Model architecture, RGB-only inference boundary, 40 FREE planes, 64
  first-hit OCCUPIED planes, 64-channel bottleneck, predictor, survival head,
  Camera loss, joint JEPA losses, exact C/N gradient balancing, optimizer,
  initialization, target hard sync and EMA, data, labels, role populations,
  schedule, seed, controls, observation updates, thresholds, and physical
  gates are frozen exactly as V13.

## Required integrity repair

- Add
  `lewm/benchmarks/go2_shared_jepa_v5_multires_probe_v3.py` to the forced
  dynamic source closure.
- Include the local `lewm_worlds` package closure reached by the frozen label
  evaluator, and activate its certified package root under isolated Python.
- The replacement manifest must recursively bind all resulting local Python
  sources. Every exported source byte must match a frozen source-and-review
  commit and an independently reviewed path/SHA-256/byte-count inventory.
- Before any execution authority is committed, the clean export must pass the
  closure checker and an isolated, accelerator-hidden import of the complete
  runtime module sequence through the label evaluator. This smoke test may
  import libraries and source only; it may not open generated inputs, labels,
  RGB, checkpoints, runtime outputs, or query hardware.

## Execution contract

- Exact output root:
  `.generated/go2_rgb_camera_evidence_bottleneck_joint_jepa_v13_integrity_replacement_v1/attempt_v1`.
- Exactly one fresh attempt; no retry or resume. The root must be absent before
  an immutable reservation is created.
- Maximum 1,000 updates and 16,000 presentations, with unchanged observation
  points `0/100/400/1000`, unchanged update-400 falsification, and unchanged
  update-1000-only checkpoint eligibility.
- Runtime inputs are the same exact Raw-V13 manifest/audit, N320 gate and
  checkpoint, 16,000-presentation schedule, and swept-label manifest/train/
  checkpoint-selection files bound by the consumed V13 authority. Probability
  calibration remains closed.
- Execution requires the same isolated Python/Torch/ROCm/NumPy/Pillow
  fingerprint and exactly one visible AMD Radeon AI PRO R9700 with
  34,208,743,424 bytes.
- A fresh committed source review, clean-export custody exception and
  certification, execution binding, and canonical one-shot authority are all
  required before reservation. The prior V13 authority grants no replacement
  access.
- G2, navigation, held-out, sealed, production, deployment, and promotion
  remain unauthorized. The existing V4 30-scene sealed benchmark remains
  unopened.
