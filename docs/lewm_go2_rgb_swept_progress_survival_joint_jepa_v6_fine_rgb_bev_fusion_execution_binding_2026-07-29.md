# RGB Swept-Progress Survival Joint-JEPA V6 Fine RGB BEV Fusion — Execution Binding

- Status: frozen before the single V6 GPU execution.
- Preregistration commit: `cc9ec66d796b37724e0a9e15d737813817e95265`.
- Model commit: `407da6be6fa7e52c08844aedc883520d636b15b8`.
- Runner/executor source commit: `b8624910642e13fb12bb970e70e3cd96027f6699`.
- Independent implementation review found no remaining source blocker.
- Full relevant regression suite: 82 passed. Focused combined V6 source suite: 14 passed.

## Frozen V6 source

- V6 model SHA-256: `c30034ec73c378022eac3f8154c593ffdbe5e8d6ac4624053cca36f3206fc8f6`.
- V6 model test SHA-256: `28a78f3cae505d87bf6827a37b2fb4131ebd393182d8a0290a7d5a5eadbbdd79`.
- V6 training-wrapper SHA-256: `a2af42c433aa1ac981fc88968917e371a98d85adbcba56593f1cbcb736feb09d`.
- V6 training-wrapper test SHA-256: `f1318c9414fe9d3224d1c3675d3f8bb7458e7c28631407bcb61e930bebb4d574`.
- V6 executor SHA-256: `6eeac0c38494fa4f9d05df383341f3731ced07dedef42758ccb27c3df195e9c1`.
- V6 executor test SHA-256: `c7d6c296029b2b85f7d38d46cf14e2f0fd4acd7f7c2f94cd3b8005cd7da93516`.
- The sole scientific delta from clean V4 is the preregistered 12,256-parameter fine-RGB branch. It preserves a learned `112x112` RGB feature map, samples it at the exact inherited four geometry grids with the exact inherited weights, and adds its initially zero residual to the inherited BEV latent.
- The fused latent is consumed jointly by the residual semantic decoder, action-conditioned JEPA predictor, and online/EMA persistence path. All online branch parameters enter the inherited lift/semantic optimizer group exactly once; the exact target copy remains frozen and follows inherited hard-sync/EMA behavior.
- V6 keeps the V4 losses `S+P+U+R+O`, with `O` coefficient `0.5`. V5 loss `H` is absent. Initial zero-projection parity, branch unlock activity, parameter partition, and target-gradient isolation are receipted.

## Frozen execution

- Fresh initialization is the exact accepted N320 encoder-only path plus the unchanged V4 constructor path and seeds, with isolated fine-branch seed `20260714`. No V4 candidate, original V4 runtime, no-persistence control, V5 runtime, or rejected checkpoint is named or read.
- Data, labels, action order, masks, optimizer rules and hyperparameters, clipping, schedule, controls, metrics, complete unchanged V4 24-check full-arm development gate, and terminal checkpoint selection remain unchanged.
- Cap: exactly 1,000 optimizer/EMA updates, 4,000 microbatch graphs/backward calls, and 16,000 presentations. No retry, resume, extension, intermediate selection, channel-width variant, alternate tap, normalization, or fusion variant.
- ROCm prelaunch receipt identified GPU `0` as `AMD Radeon AI PRO R9700` with `34,208,743,424` bytes total VRAM. The exact launch restricts the process to that device with `HIP_VISIBLE_DEVICES=0`.
- Fresh write-once output root `.generated/go2_rgb_swept_progress_survival_joint_jepa_v6_fine_rgb_bev_fusion/attempt_v1` was confirmed absent immediately before this binding.
- Checkpoint and trace are written after terminal training and before evaluation. They remain development-only, unqualified, and non-resumable.
- Exit code `2` with a complete `result.json` is a scientific full-arm gate failure, not an execution failure. Such a failure closes V6 without calibration, tuning, retry, or checkpoint access.
- A full-arm pass stages only a separately source-frozen, one-shot V6-specific refit of the unchanged reviewed physical-calibration protocol. It does not itself qualify the checkpoint or authorize G2, navigation, held-out, sealed, production, deployment, or promotion access.
- Exact command: `HIP_VISIBLE_DEVICES=0 /home/andrewknowles/.local/share/lewmquad-v12-runtime-rocm711/bin/python -I -B scripts/execute_go2_rgb_swept_progress_survival_joint_jepa_v6_fine_rgb_bev_fusion.py`.
