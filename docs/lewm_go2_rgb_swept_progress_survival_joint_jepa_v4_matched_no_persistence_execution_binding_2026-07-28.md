# RGB Swept-Progress Survival Joint-JEPA V4 Matched No-Persistence Control — Execution Binding

- Status: frozen before control GPU execution.
- Preregistration commit: `3dd4ca0680347f0a7f35d42d387781ecf53b1685`.
- Pre-runtime clarification commit: `8cd4486ff8fc5e82dbfb745da1ed8d4b3a4101b1`.
- Source commit: `4d55f6b68ac4edfa8aef93fdb3b2e4c7666f09e2`.
- Independent design and final result-invalidating source reviews: PASS.
- Full relevant V1/V2/V3/V4/control compatibility suite: 80 passed.
- Executor `--help` smoke test: PASS.

## Frozen control source

- No-persistence training core SHA-256: `90b66a5e4bdc7e6634db57d6852d9b3c5a187581d67a80ce81bf95fb371c34ab`.
- Training-core test SHA-256: `1cb39173c8fa389abe38897ea0409b927ed7717deaa4516412d89ce0d405f647`.
- Control executor SHA-256: `f1e6a74c070d2db018cad120e4dcbc764f5432e4ebff1d88f179db079ad09cfd`.
- Executor test SHA-256: `684a52056fc45cbf6d04e0c9a1ff963e0add0048138acec4d38c9859809f5e69`.
- The sole treatment delta is direct backward membership: full V4 uses `S+P+U+R+O`; the control uses `S+U+R+O`. `P` remains computed through the identical forward and is traced diagnostically; it cannot affect a gradient or update.
- Trace keys are exactly `S`, `P_diagnostic`, `U`, `R`, `O`, `L_full_diagnostic`, and `L_backward`.

## Matching and runtime

- The control reconstructs the exact frozen V4 model twice on CPU from the accepted N320 encoder input and frozen seeds. It requires identical canonical complete-state receipts, target hard-sync count `1`, EMA count `0`, and an empty exact optimizer receipt before training.
- The four update-1 backward calls must reproduce the frozen full-V4 `S/P/U/R/O` component means exactly before the first optimizer step; mismatch consumes the write-once attempt and fails closed.
- V4 model/decoder, RGB/data/labels, schedule, seeds, optimizer groups and hyperparameters, clipping, masks, `S/U/R/O` with `O=0.5`, all forwards, EMA, evaluator, gates, controls, bootstrap, and 1,000-update / 16,000-presentation cap remain unchanged.
- Full-V4 comparison uses only the embedded immutable eight-family utility reference with canonical SHA-256 `8ba8d6126e922f6a36038304e3444d0d21ee69350fef4acd3828265754810e1e`; no V4 runtime artifact is reopened.
- Positive treatment requires full-V4-minus-control mean utility delta `>0`, 10,000-draw paired bootstrap lower bound `>0`, and at least 6/8 positive families.
- Historical same-attempt/serialized-clone mechanics are explicitly superseded for this bounded diagnostic. The result cannot establish causal generalization.
- Exact device: one visible `AMD Radeon AI PRO R9700`, selected with `HIP_VISIBLE_DEVICES=0`; no KFD compute process was present before launch.
- Fresh write-once output root: `.generated/go2_rgb_swept_progress_survival_joint_jepa_v4_matched_no_persistence/attempt_v1`; confirmed absent before launch.
- Terminal checkpoint and trace precede evaluation. The control checkpoint is development-only, diagnostic-only, unqualified, non-resumable, and cannot replace V4.
- There is no retry, resume, extension, alternate seed/loss, intermediate selection, warm start, or result-conditioned intervention.
- Neither treatment outcome authorizes G2, navigation, sealed, held-out, production, deployment, promotion, or final-evaluation access.
- Exact command: `HIP_VISIBLE_DEVICES=0 /home/andrewknowles/.local/share/lewmquad-v12-runtime-rocm711/bin/python -I -B scripts/execute_go2_rgb_swept_progress_survival_joint_jepa_v4_matched_no_persistence.py`.
