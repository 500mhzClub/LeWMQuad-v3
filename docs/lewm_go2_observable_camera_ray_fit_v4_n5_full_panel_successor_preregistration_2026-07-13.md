# Observable camera-ray fit V4 N5 full-panel successor preregistration

Date: 2026-07-13

Status: **frozen before successor training or metric inspection**

## Trigger

The immutable ladder-v3 seed `20260710`, `N=5` checkpoint reproduces every
reported numeric loss, confusion count, quantile, and gate metric exactly in
fresh GPU0 inference. Its unchanged numeric gate is expected to fail. The only
reproduction differences are two byte commitments over sorted depth-error
vectors; a narrow independently reviewed V3 verifier/finalizer successor will
record those original and stable rerun commitments without changing a number
or threshold.

The failed fit used batch size one for 1,000 optimizer updates. Each of the
five frames was seen approximately 200 times, but every update optimized one
family/frame in isolation. The recorded trace oscillates between examples and
the final aggregate model is materially worse than the best instantaneous
single-frame loss. This is an optimizer-interference diagnosis, not evidence
for changing the task, labels, model, losses, or gates.

## One authorized experiment

After the V3 finalizer publishes the immutable N5 numeric failure, one fresh
development-only attempt is permitted under a new output namespace:

`.generated/go2_observable_camera_ray_fit_v4/n5_full_panel_v1/attempts/seed_20260710/n5`

The attempt is fixed as follows:

- selected panel: the same exact frozen seed-`20260710`, `N=5` subset;
- model: a fresh `ObservableCameraRayEvidenceV4Model` initialization;
- seed: `20260710`;
- optimizer: AdamW;
- optimizer updates: `400`;
- training batch: all `5` selected frames in one update;
- per-update order: the existing seeded concatenated-randperm schedule;
- learning rate: `1e-4`;
- weight decay: `1e-4`;
- precision: float32, no autocast;
- gradient clipping: global norm `1.0` after the full-panel loss backward;
- objective: the same four raw/derived V4 losses with weights exactly `0.25`
  each;
- checkpoint selection: final update only; no best-step, validation, restart,
  warm start, or result-derived selection;
- evaluation: the same matched-RGB and wrong-RGB evaluations, batch size one;
- device: GPU0, AMD Radeon AI PRO R9700; GPU1/iGPU is forbidden;
- CPU/native threads: one per decoder worker, at most five RGB workers.

This gives every optimizer update simultaneous evidence from every represented
family. It uses 2,000 frame exposures, exactly twice the failed attempt, while
reducing optimizer updates from 1,000 to 400. The change is deliberately only
the optimization exposure boundary and budget.

## Unchanged scientific gate

The model must pass the existing frozen `N=5` thresholds without any edit,
rounding exception, calibration change, class reweighting, or excluded cell.
The verifier must independently rerun exact inference from the published
checkpoint. It must retain the wrong-RGB controls and all source, input,
checkpoint, result, output, GPU, and access-ledger bindings.

A structurally valid numeric failure is terminal for this attempt. It does not
license a retry. A pass licenses design and independent review of the later-rung
full-panel/accumulation schedule; it does not itself license `N=16`, G2,
runtime, held-out, hardware, or production promotion.

## Authority and preservation

The immutable ladder-v3 N5 artifacts and its V3 failure receipt/gate remain
read-only lineage. The successor must use exclusive creation and publish an
explicit completion or failure receipt. Source implementation and a
different-agent review must be hash-frozen before the command can execute.

No G2, held-out, runtime, navigation benchmark, physical executor/reset,
promotion, or sealed input may be opened by implementation, review, training,
verification, or finalization of this experiment.
