# RGB Swept-Progress Survival Joint-JEPA V3 Half Occupied-Safety Auxiliary — Execution Binding

- Status: frozen before V3 GPU execution.
- Preregistration commit: `2b917fcd4d8e4115f15b57d4fc26691a39c37328`.
- Source commit: `5543a3a25bc9de0519165e8006aba3faff597ef1`.
- Independent design and critical-path implementation reviews: PASS.
- Focused V1/V2/V3 compatibility suite: 51 passed.
- Executor `--help` smoke test: PASS.

## Frozen source

- Refactored V2 training core SHA-256: `6f76dd5b098ff360a3ada5bbb18f74a13342f3a5212e871da6db8f5f3a5bb1bf`.
- V3 training core SHA-256: `7cab73752593b12b638b55710714ff956a2441e92df2fe775902472a7b69a8cb`.
- V3 training-core test SHA-256: `48376097957911eadf1c40db3b2e28cb1ead0b7e93384a605841f17e5a273852`.
- V3 executor SHA-256: `164e2baf53f2a882ef18eabeee99ae4b2c27a7d8d799543c798f24a49782b182`.
- V3 executor test SHA-256: `a291adfcf9d42e16db40f17c477827bb9db280fe8f6054d7862a271fd67d4c7f`.
- The V2 edit is a behavior-preserving private extraction. Its public entry points retain coefficient `1.0`; the historical V2 result remains bound to source commit `78d0ac331f5c1214344aafc6c981fec3a38c2b8b`.
- V3's sole scientific delta is occupied-safety coefficient `1.0 -> 0.5`; `L=S+P+U+R+0.5*A_occ` from update 1. No other science or gate change is authorized.

## Runtime binding

- V3 directly reuses the frozen V2/V1 data, labels, accepted N320 encoder input, model, masks, optimizer, schedule, seeds, controls, metrics, gates, and evaluation helpers.
- The accepted N320 encoder checkpoint is the only model-state input. Rejected V1/V2 output roots and checkpoints are neither named nor read.
- Same constructor, execution, and bootstrap seeds; exact schedule prefix; 1,000 updates / 16,000 presentations; one optimizer and EMA step per update.
- Exact device: one visible `AMD Radeon AI PRO R9700`, selected with `HIP_VISIBLE_DEVICES=0`; it was idle before launch.
- Fresh write-once output root: `.generated/go2_rgb_swept_progress_survival_joint_jepa_v3_half_occupied_safety_aux/attempt_v1`; confirmed absent before launch.
- Checkpoint and trace are written after terminal training and before evaluation. They remain development-only, unqualified, and non-resumable.
- Exit code `2` with a complete `result.json` is a scientific gate failure, not an execution failure.
- Once update 1 begins, no retry or resume is authorized. This is the final occupied-safety coefficient attempt regardless of outcome.
- A V3 pass authorizes only the preregistered matched no-JEPA causal control. It does not authorize G2, navigation, sealed, held-out, production, deployment, or promotion access.
- Exact command: `HIP_VISIBLE_DEVICES=0 /home/andrewknowles/.local/share/lewmquad-v12-runtime-rocm711/bin/python -I -B scripts/execute_go2_rgb_swept_progress_survival_joint_jepa_v3_half_occupied_safety_aux.py`.
