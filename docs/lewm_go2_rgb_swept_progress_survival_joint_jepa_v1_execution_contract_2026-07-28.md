# RGB Swept-Progress Survival Joint-JEPA V1 — Execution Contract

- Status: frozen before GPU execution on 2026-07-28.
- Purpose: falsify one action-conditioned perception mechanism, not qualify navigation.
- Reviewed executor commit: `dfc4b27953029273363d223af3e1f95d03e0a0a5`.
- Executor SHA-256: `870022fc84ad391c97c3fe06da83357d8575408a7d57874aa0aac118ace9deb2`.
- Independent critical-path source review: PASS; no result-invalidating blocker found.

## Frozen experiment

- Input is RGB only. The N320 encoder initializes the online encoder; the geometry-anchored BEV lift, action predictor, EMA target encoder, semantic objective, and shared swept-progress survival head train jointly from update 1.
- The survival target is exact immediate primitive feasibility followed by 15 conditional 0.1 m swept-footprint segments, for a maximum predicted progress of 1.5 m for each of nine action primitives.
- No component is trained separately or post hoc, and no latent is detached from the joint loss.
- Loss is `S + P + survival + ranking`: semantic loss normalized by `log(3)`, executed-action EMA latent persistence, at-risk survival BCE normalized by `log(2)`, and unequal-prefix ranking `softplus(-8 * margin) / log(2)`.
- AdamW is frozen at encoder `1e-4`, lift/semantic `3e-4`, predictor/survival head `3e-4`, betas `(0.9, 0.999)`, epsilon `1e-8`, weight decay `1e-4`, with independent representation and predictor clipping at `1.0`.
- Four microbatches of four form one update. The cap is exactly 1,000 updates and 16,000 presentations, with one optimizer step and one EMA step per update.
- Fresh-component constructor seed: `20260712`. Execution, stochastic, and bootstrap seed: `20260728`.
- Schedule prefix SHA-256: `3f7b5799e855c3d218dcc62428f26ae0f9577c0dd4b04af5156d439a6f81e528`.

## Frozen inputs and geometry

- Complete development label-manifest file SHA-256: `edc0df8c796f97d3f91c8c3796e9795a4355dceac79770b91de382132fe8e1d3`; content SHA-256: `6e0ea572612cdf94cb6dd91dffb90e50c828053617f69b42307161c958700c03`; byte count: `5914`.
- Label preflight passed over all 88 development scenes: 3,546 informative train states, 337 calibration states, and 399 selection states. The frozen schedule contains 13,310 informative presentations of 16,000.
- Accepted N320 checkpoint file/content SHA-256: `ece874b53941e841fffc61b724a86d4383b881549afa453b746dd5d68aba11b0` / `9dcca536943f89acfd7d463fdab591e19a030ef3dc8f3f19a050b1b10025fc2b`.
- Accepted N320 gate file/content SHA-256: `4943b4060e88296503c09fc714e55e40fd762527cfccb70a3a341f0df800efe6` / `76ce5ab703560d171f7c84684b90eed18e8b4cdcc2d8ed3eff6d48496f4de67b`.
- Predicted-next/post-action mask tensor SHA-256: `11ae5e26b182da85c8a7ca866ee4914c72b5b84b8b601dd807903097d754485c`.
- Coordinate-matched current-frame persistence mask tensor SHA-256: `c4b8c475032433e448cd7df9decfead2c0800426219098f45306a0540154d2ff`.

## Decision rule

- Selection is evaluated only after the terminal update. Calibration diagnostics do not alter weights, thresholds, or selection.
- Semantic floors are balanced accuracy `0.80`, free recall `0.85`, occupied recall `0.70`, unknown recall `0.90`, and rough-scene occupied recall `0.65`.
- On informative selection states, chosen/oracle prefix utility must be at least `0.85` overall and `0.70` in every family; zero-prefix selection must be at most `0.05` overall and `0.20` in every family; unequal-pair concordance must be at least `0.75` overall and `0.60` in every family.
- The full arm must beat each of coordinate-matched persistence, shuffled predicted-action slots, wrong RGB, and the train action-mean prior: positive equal-scene mean delta, positive 95% paired-scene bootstrap lower bound, and positive family delta in at least six of eight families.
- All gates are conjunctive. Progress calibration and MAE are reported but not thresholded.
- A PASS authorizes preparation of a matched no-JEPA training arm before any causal JEPA claim. A FAIL closes this mechanism at this cap.

## Execution boundary

- Exact device: one visible `AMD Radeon AI PRO R9700`, `34208743424` bytes; run with `HIP_VISIBLE_DEVICES=0` in the reviewed ROCm environment.
- Exact output root: `.generated/go2_rgb_swept_progress_survival_joint_jepa_v1/attempt_v1`; it must be absent at launch and is write-once.
- Once update 1 begins, there is no retry or resume. A purely mechanical failure before update 1 may receive one science-identical correction with a complete failure receipt.
- No G2, navigation, held-out, sealed-benchmark, production, or produced-checkpoint access is authorized by this run.
- Command: `HIP_VISIBLE_DEVICES=0 /home/andrewknowles/.local/share/lewmquad-v12-runtime-rocm711/bin/python -I -B scripts/execute_go2_rgb_swept_progress_survival_joint_jepa_v1.py`.
