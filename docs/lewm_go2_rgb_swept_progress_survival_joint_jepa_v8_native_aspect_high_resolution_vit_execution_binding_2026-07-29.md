# RGB Swept-Progress Survival Joint-JEPA V8 — Native-Aspect High-Resolution ViT Execution Binding

- Status: frozen before the single V8 GPU execution.
- Preregistration commit:
  `b17599fa1bb49017178f45d0e1a4c83ac8bb9314`.
- Exact model, loader/executor, training-wrapper, and focused-test commit:
  `0cd50cee09e7e4fbdfc696001e13c1b41a6e6772`.
- The source commit is the direct child of the preregistration commit. All six
  V8 source/test paths were clean and byte-identical to that commit at review.
- Three independent read-only reviews passed with zero material science,
  integrity, no-repeat, custody, or execution blockers.
- Focused V8 model/wrapper/executor suite: 18 passed. Complete V1--V8
  swept-progress model/wrapper/executor regression suite: 138 passed.

## Frozen source hashes

- Preregistration SHA-256:
  `2eea464bc46a2ef8e95a29c36214febe06dc2539b77c80b9d6acf0c82216f0a2`.
- Model SHA-256:
  `696211b7c1f18898e0d82e1a66a5cf4276b4de4d4d034f9796a64cbc50c8d6ad`.
- Model-test SHA-256:
  `d5845761e24c6899e12c5a7c0dd37f87320916985c6fb503e7054d73016c1705`.
- Training-wrapper SHA-256:
  `f3a98bef89221f264f55e77ef99959a4e695231ed0fe8cd1eb0258af15e9f79e`.
- Training-wrapper-test SHA-256:
  `0de0dcf564857b5b010c0c690f4afb46190bc63e08de5150a9cb6cdd3e184ee7`.
- Executor SHA-256:
  `6a3e3ad718769fc4689603b3ef7706606ced6403cbe66ee23a95f4d47909a89d`.
- Executor-test SHA-256:
  `764ff3fc7720f7347bbba964c4372deec346ccb0f6d1c961485cdd5de3c25e60`.

## Frozen scientific execution

- V8 decodes the same bound RGB bytes directly at native `224x168`, with no
  resize, crop, pad, upscale, augmentation, role change, or label change. It
  returns normalized float32 `[3,168,224]`.
- The accepted N320/V4 patch-7 projection, CLS token, six global transformer
  blocks, final norm, and every non-positional encoder tensor are exact. Only
  the spatial positions are deterministically migrated on CPU float32 from
  `16x16` to `24x32` using bicubic interpolation with
  `align_corners=False`, `antialias=False`, and row-major flattening. The CLS
  position remains bit-exact.
- The online encoder has exactly `2,845,824` parameters and returns CLS plus
  768 spatial tokens as `[B,769,192]`.
- The rectangular lift retains every V4 state tensor. Its proposed normalized
  grid uses exact V4 arithmetic, `2*tanh(raw)*(2/16)`, while reporting native
  token-cell offsets as `tanh(raw)*[4,3]`. Initial grids, validity masks, and
  weights were independently checked bit-exact against V4.
- Online/target copies, one initial hard-sync count, `0.996` EMA, optimizer
  partitions, V4 residual semantic decoder, action predictor, survival head,
  and the complete `S+P+U+R+O` objective with `O=0.5` remain unchanged. All
  components train jointly from update one; there is no separately trained
  encoder, predictor, or readout.
- Data roles, labels, schedule, action order, masks, seeds, optimizer rules,
  controls, bootstrap, metrics, terminal selection, and all 24 V4 development
  gates are unchanged.
- Cap: exactly 1,000 optimizer/EMA updates, 4,000 size-four microbatch graphs
  and backward calls, and 16,000 presentations. The executor invokes training
  exactly once and permits no retry, resume, extension, alternate seed,
  intermediate selection, or square-resolution follow-up.
- A complete result with exit code `2` is a valid scientific full-arm failure
  and closes V8. Only a 24/24 pass stages a separately frozen, unchanged V4
  physical calibration; it does not qualify the checkpoint or open G2.

## Prelaunch state and command

- Immediately before this binding, the write-once output root
  `.generated/go2_rgb_swept_progress_survival_joint_jepa_v8_native_aspect_high_resolution_vit/attempt_v1`
  was absent and was not a symlink.
- Prelaunch inspection found exactly one visible
  `AMD Radeon AI PRO R9700` with `34,208,743,424` bytes total memory.
- The isolated executor help/import preflight passed without runtime data or
  checkpoint access.
- Exact command:
  `HIP_VISIBLE_DEVICES=0 /home/andrewknowles/.local/share/lewmquad-v12-runtime-rocm711/bin/python -I -B scripts/execute_go2_rgb_swept_progress_survival_joint_jepa_v8_native_aspect_high_resolution_vit.py`.
- Checkpoint and trace, if written, are development-only, unqualified, and
  non-resumable. No rejected checkpoint, held-out, sealed, navigation,
  production, deployment, promotion, or G2 access is authorized.
