# RGB Swept-Progress Survival Joint-JEPA V4 Residual Local Semantic Decoder — Execution Binding

- Status: frozen before V4 GPU execution.
- Preregistration commit: `9f9ab784b4bfa827585ec095f2a7f7a30333480a`.
- Source commit: `aaa47a138d0eeb78aa20d9524e67f813f7a74a41`.
- Independent design and final critical-path implementation reviews: PASS.
- Full relevant V1/V2/V3/V4 compatibility suite: 64 passed.
- Independent focused V4 suite: 14 passed; executor `--help` smoke test: PASS.

## Frozen V4 source

- V4 model SHA-256: `1c5a26f02a856d9a84903063c53bf23095142d86885787556b09388c508711ef`.
- V4 model test SHA-256: `05e2783eeeffbe231b9e1128aae4695d5a6f695ea566ca64f0336bbf730763b2`.
- V4 executor SHA-256: `243ef91ccec4e1fcdfa5a0c3f112bf4c645f46ba7de8692c1dddcb47f87c9f40`.
- V4 executor test SHA-256: `712a7666a7ff1fb610c0d9a6e5013125db9ac543d98580e652c71d61e89fb021`.
- V4's sole scientific delta is the preregistered 37,123-parameter residual local semantic decoder: the inherited base `Conv2d(64,3,1)` plus `Conv2d(64,64,3,padding=1)` -> exact GELU -> zero-initialized `Conv2d(64,3,1)`.
- The inherited base-head object and visibility mask are preserved. Initial masked semantic logits are bitwise equal to V3. Decoder seed is exactly `20260713`, isolated from and restored to the caller RNG.
- All decoder parameters enter the existing lift/semantic optimizer and clipping group exactly once. The EMA target remains encoder plus BEV lift only.

## Frozen training and runtime

- V4 reuses the frozen V3 coefficient-`0.5` training core and all V3/V1 data, labels, accepted N320 encoder input, masks, optimizer rules, schedule, seeds, losses, controls, metrics, gates, and evaluation helpers.
- The accepted N320 encoder checkpoint is the only model-state input. Rejected V1/V2/V3 output roots and checkpoints are neither named nor read.
- Training is joint from update 1. The zero-gated output projection receives gradient on the first backward pass; the local 3x3 convolution is expected to unlock after the first optimizer step rather than forming a separate training stage.
- Exact schedule prefix; 1,000 updates / 16,000 presentations; four size-four microbatches and one optimizer/EMA step per update.
- Exact device: one visible `AMD Radeon AI PRO R9700`, selected with `HIP_VISIBLE_DEVICES=0`.
- Fresh write-once output root: `.generated/go2_rgb_swept_progress_survival_joint_jepa_v4_residual_local_semantic_decoder/attempt_v1`; confirmed absent before launch.
- Checkpoint and trace are written after terminal training and before evaluation. They remain development-only, unqualified, and non-resumable.
- Exit code `2` with a complete `result.json` is a scientific gate failure, not an execution failure.
- Once update 1 begins, no retry, resume, extension, intermediate selection, or decoder variant is authorized. Failure closes this decoder mechanism.
- A full V4 pass authorizes only the identical-decoder matched no-JEPA causal control. It does not authorize G2, navigation, sealed, held-out, production, deployment, or promotion access.
- Exact command: `HIP_VISIBLE_DEVICES=0 /home/andrewknowles/.local/share/lewmquad-v12-runtime-rocm711/bin/python -I -B scripts/execute_go2_rgb_swept_progress_survival_joint_jepa_v4_residual_local_semantic_decoder.py`.
