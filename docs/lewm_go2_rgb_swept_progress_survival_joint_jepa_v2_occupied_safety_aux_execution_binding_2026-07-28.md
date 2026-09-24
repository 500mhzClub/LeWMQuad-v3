# RGB Swept-Progress Survival Joint-JEPA V2 Occupied-Safety Auxiliary — Execution Binding

- Status: frozen before V2 GPU execution.
- Preregistration commit: `4ba71294dd8da510a85e82c7e60945a89378d568`.
- Source commit: `78d0ac331f5c1214344aafc6c981fec3a38c2b8b`.
- Independent design and critical-path implementation reviews: PASS.
- Focused V1/V2 compatibility suite: 29 passed.

## Frozen source

- V2 training core SHA-256: `10c28145ca332fbfaee9e2474e5701e6289e16f4e6388eb55b7c7ac4992b855f`.
- V2 executor SHA-256: `88e294ae1d4ccf01577ca16469defa008d5227b312b72af22ca6d9b5cfd7c0a8`.
- V2 training-core test SHA-256: `b760481b9a07835ba9285be1afef5e0d7cbc0607e4aa771384aca98fe769c5e1`.
- V2 executor test SHA-256: `6b1e2b03913d43ed1e88b7dee1e1b4784788b1c838f1e14c89879baa6ab859eb`.
- Sole delta remains the coefficient-one normalized occupied-vs-rest auxiliary `A_occ`; `L=S+P+U+R+A_occ` from update 1. No other science or gate changes are authorized.

## Runtime binding

- V2 directly reuses the frozen V1 label/N320/raw/schedule/model/mask/optimizer/control/gate helpers bound by `docs/lewm_go2_rgb_swept_progress_survival_joint_jepa_v1_execution_contract_2026-07-28.md`.
- The accepted N320 encoder checkpoint is the only model-state input. The rejected V1 output root and checkpoint are not named by the executor and must not be read.
- Same constructor, execution, and bootstrap seeds; exact schedule prefix; 1,000 updates / 16,000 presentations; one optimizer and EMA step per update.
- Exact device: one visible `AMD Radeon AI PRO R9700`, `34208743424` bytes, selected with `HIP_VISIBLE_DEVICES=0`.
- Fresh write-once output root: `.generated/go2_rgb_swept_progress_survival_joint_jepa_v2_occupied_safety_aux/attempt_v1`; confirmed absent before launch.
- Checkpoint and trace are written after terminal training and before evaluation. They remain development-only, unqualified, and non-resumable.
- Exit code `2` with a complete `result.json` is a scientific gate failure, not an execution failure.
- Once update 1 begins, no retry or resume is authorized. No G2, navigation, sealed, held-out, production, deployment, or promotion access is authorized.
- Exact command: `HIP_VISIBLE_DEVICES=0 /home/andrewknowles/.local/share/lewmquad-v12-runtime-rocm711/bin/python -I -B scripts/execute_go2_rgb_swept_progress_survival_joint_jepa_v2_occupied_safety_aux.py`.
