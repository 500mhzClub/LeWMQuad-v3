# RGB Swept-Progress Survival Joint-JEPA V5 Near-Field Hazard Ranking — Execution Binding

- Status: frozen before V5 GPU execution.
- Preregistration commit: `7fe075d752b5d14c539eaed213c9f28510659c79`.
- Frozen interpretation commit: `23f7388b68897a9db876909fd4ebd5b3f0bae52b`.
- Source commit: `f01a66f12f6cac0da69b86c3668484a7b771d0bc`.
- Independent final source review: PASS.
- Full relevant V1–V5 compatibility suite: 64 passed. Independent focused V3/V4/V5 suite: 29 passed. Executor help smoke test: PASS.

## Frozen V5 source

- Training core SHA-256: `c2d21d556396a23a04726b3aa54695d176e1a70c51cf4824e14f1314142c35e7`.
- Training-core test SHA-256: `d687a36604e50b0452845526a7bee6d65751858db4fc916efdd7ac8e21d6a8cb`.
- Executor SHA-256: `897ba65058e41496a5cc391e10e0f482bac953cf72b36c006d73492aa1a4d77b`.
- Executor test SHA-256: `dc62002e3e2733b733793e6b431a93e9a23b2a348a5ebbbe3e0b06002fe660c5`.
- The sole scientific delta from V4 is the coefficient-one, parameter-free near-field hazard-ranking loss `H` frozen in the preregistration and interpretation. Total loss is `S+P+U+R+O+H`; inherited `O` remains coefficient `0.5`.
- `H` is trained jointly from update one through the existing V4 semantic decoder, BEV lift, and RGB encoder. The action-conditioned JEPA predictor remains jointly trained through its inherited objectives; there is no detached head or separate predictor stage.

## Frozen execution

- Fresh initialization is the exact accepted N320 encoder-only path plus the unchanged V4 model and constructor seeds. No V4 candidate, original V4 runtime, no-persistence control, or rejected checkpoint is named or read.
- Data, labels, action order, masks, optimizer groups and hyperparameters, clipping, seed, schedule, controls, metrics, complete 24-check V4 full-arm development gate, and terminal checkpoint selection remain unchanged.
- Cap: exactly 1,000 optimizer/EMA updates, 4,000 microbatch graphs/backward calls, and 16,000 presentations. No retry, resume, extension, intermediate selection, coefficient variant, or margin/range variant.
- Hazard receipts cover every microbatch, eligible current/next sample counts, complete-pair counts, per-update `H`, and ten 100-update windows. JEPA progress-ranking receipts remain separate.
- Exact device: one visible `AMD Radeon AI PRO R9700`, selected with `HIP_VISIBLE_DEVICES=0`.
- Fresh write-once output root `.generated/go2_rgb_swept_progress_survival_joint_jepa_v5_near_field_hazard_ranking/attempt_v1` was confirmed absent immediately before this binding.
- Checkpoint and trace are written after terminal training and before evaluation. They remain development-only, unqualified, and non-resumable.
- Exit code `2` with a complete `result.json` is a scientific full-arm gate failure, not an execution failure.
- A full-arm pass stages only a separately frozen V5-specific refit of the unchanged physical-calibration protocol. It does not itself qualify the checkpoint or authorize G2, navigation, held-out, sealed, production, deployment, or promotion access.
- Exact command: `HIP_VISIBLE_DEVICES=0 /home/andrewknowles/.local/share/lewmquad-v12-runtime-rocm711/bin/python -I -B scripts/execute_go2_rgb_swept_progress_survival_joint_jepa_v5_near_field_hazard_ranking.py`.
